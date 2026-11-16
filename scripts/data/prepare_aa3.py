"""AbdomenAtlas 3.0 (RadGPT, Bassi et al., ICCV 2025; HF AbdomenAtlas/AbdomenAtlas3.0Mini) IID test split (926 scans, the
paper's 10% test set) on our 1 mm RAS grid, for zero-shot tumour detection in the liver, pancreas and kidney
(extaudit/aa3_det.py, 2026-10-27). AbdomenAtlas 3.0 tumour masks were made by AI and revised by radiologists, who added
missed tumours and removed false positives in every scan, so an organ without a lesion mask is a checked negative, and
positives and negatives come from the same cohort.
Raw data: benchmarks/AbdomenAtlas3.0/IID_test/<BDMAP id>/{ct.nii.gz, segmentations/*.nii.gz} (download_iid_test.sh).

Labels per organ o in {liver, pancreas, kidney} (masks liver_lesion, pancreatic_lesion, kidney_lesion): positive = at least
one 26-connected lesion component with largest extent >= 10 mm; negative = no lesion voxel in the
organ; organs whose lesions are all < 10 mm are left out of that organ's test (ambiguous). The organ must be in the field
of view (its AbdomenAtlas organ mask non-empty). One scan per patient is not guaranteed by the release (no patient ids):
each scan counts as one patient in the bootstrap.

Stages (resumable; the raw files are only read):
  resample  (RAS, 1 mm, HU trilinear int16; masks nearest) -> {W}/aa3/1mm/img/<id>.nii.gz and
            {W}/aa3/1mm/lab/<id>.nii.gz (uint8: 1 liver lesion, 2 pancreatic lesion, 3 kidney lesion)
  ts        TotalSegmentator v2 'total' fast (scripts/data/lesion_geometry.py; the harness field of view and the gated encoders
            use its ids) -> {W}/aa3/1mm/seg/<id>.nii.gz
  finalize  {W}/aa3/cases.parquet (per-organ labels, lesion size, contrast phase) + harness set
            {W}/extaudit/aa3_set.parquet

    python scripts/data/prepare_aa3.py --stage resample --workers 16
    <TotalSegmentator env>/bin/python scripts/data/prepare_aa3.py --stage ts --device gpu
    python scripts/data/prepare_aa3.py --stage finalize --workers 16
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import nibabel as nib
import numpy as np

RAW = os.environ.get('AA3_ROOT', '')   # AbdomenAtlas 3.0 download root (holds TrainTestIDS/, IID_test/, AbdomenAtlas3.0MiniWithMeta.csv)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from organic.workroot import work_root                              # noqa: E402

W = work_root()
MIN_EXTENT_MM = 10.0
LESION = {'liver': ('liver_lesion', 1, ['liver']), 'pancreas': ('pancreatic_lesion', 2, ['pancreas']),
          'kidney': ('kidney_lesion', 3, ['kidney_left', 'kidney_right'])}


def out_dirs(W):
    d = {k: os.path.join(W, 'aa3', '1mm', k) for k in ('img', 'lab', 'seg')}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


def cases():
    ids = [l.strip() for l in open(f'{RAW}/TrainTestIDS/IID_test.csv').read().splitlines()[1:] if l.strip()]
    return ids


def ready(c):
    """raw case fully extracted: CT and all lesion / organ masks present and not modified in the last 2 minutes."""
    fs = [f'{RAW}/IID_test/{c}/ct.nii.gz'] + [f'{RAW}/IID_test/{c}/segmentations/{m}.nii.gz'
                                               for m, _, org in LESION.values() for m in [m] + org]
    return all(os.path.exists(f) and time.time() - os.path.getmtime(f) > 120 for f in fs)


def resample_one(c):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    d = out_dirs(W)
    fi, fl = os.path.join(d['img'], f'{c}.nii.gz'), os.path.join(d['lab'], f'{c}.nii.gz')
    if os.path.exists(fi) and os.path.exists(fl):
        return dict(case=c, skipped=True)
    try:
        t0 = time.time()
        im = nib.load(f'{RAW}/IID_test/{c}/ct.nii.gz')
        imc = nib.as_closest_canonical(im)
        a = np.ascontiguousarray(np.asarray(imc.dataobj, dtype=np.float32))
        lab = np.zeros(a.shape, np.uint8)
        for m, v, _ in LESION.values():
            mk = nib.load(f'{RAW}/IID_test/{c}/segmentations/{m}.nii.gz')
            if not np.allclose(im.affine, mk.affine, atol=1e-2):
                # 2 of 926 release masks (BDMAP_00000118 pancreatic_lesion, BDMAP_00002636 liver_lesion) carry a header
                # with no origin and a flipped y sign; their voxels lie on the CT grid (94% / 100% inside the organ
                # mask as stored, 0% / 98% flipped), so the CT header is used for them
                assert mk.shape == im.shape, f'image / {m} grids differ'
                print(f'{c}: {m} header replaced by the CT header', flush=True)
                mk = nib.Nifti1Image(np.asarray(mk.dataobj), im.affine)
            lab[np.asarray(nib.as_closest_canonical(mk).dataobj) > 0.5] = v
        zoom = np.asarray(imc.header.get_zooms()[:3], float)
        n_old = np.asarray(a.shape)
        size = tuple(int(x) for x in np.maximum(1, np.round(n_old * zoom)).astype(int))
        hu = F.interpolate(torch.from_numpy(a)[None, None], size=size, mode='trilinear', align_corners=False)[0, 0]
        hu = np.clip(np.rint(hu.numpy()), -32768, 32767).astype(np.int16)
        lab1 = F.interpolate(torch.from_numpy(lab)[None, None].float(), size=size, mode='nearest-exact')[0, 0].numpy()
        sc = n_old / np.asarray(size)
        S = np.eye(4)
        S[:3, :3] = np.diag(sc)
        S[:3, 3] = 0.5 * sc - 0.5
        aff = imc.affine @ S
        for arr, path, dt in ((hu, fi, np.int16), (lab1.astype(np.uint8), fl, np.uint8)):
            img = nib.Nifti1Image(arr, aff)
            img.set_data_dtype(dt)
            img.set_qform(aff, code=1)
            img.set_sform(aff, code=1)
            nib.save(img, path + '.tmp.nii.gz')
            os.replace(path + '.tmp.nii.gz', path)
        return dict(case=c, zooms=zoom.tolist(), shape_1mm=list(size), seconds=time.time() - t0)
    except Exception as ex:                                           # noqa: BLE001
        return dict(case=c, error=str(ex)[:300])


def finalize_one(c):
    from scipy import ndimage
    d = out_dirs(W)
    lab = np.asarray(nib.load(os.path.join(d['lab'], f'{c}.nii.gz')).dataobj)
    row = dict(case=c, shape=list(lab.shape))
    for o, (m, v, org) in LESION.items():
        present = any(np.asarray(nib.load(f'{RAW}/IID_test/{c}/segmentations/{x}.nii.gz').dataobj).any() for x in org)
        cc, n = ndimage.label(lab == v, structure=np.ones((3, 3, 3)))
        ext = [float(max(s.stop - s.start for s in sl)) for sl in ndimage.find_objects(cc)] if n else []
        big = [e for e in ext if e >= MIN_EXTENT_MM]
        row[f'{o}_label'] = ('absent' if not present else 'positive' if big else 'negative' if not n else 'small_only')
        row[f'{o}_n'] = len(big)
        row[f'{o}_max_mm'] = max(ext, default=0.0)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['resample', 'ts', 'finalize'])
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--device', default='gpu')
    ap.add_argument('--loop', action='store_true', help='resample / ts: keep going until all 926 cases are done')
    a = ap.parse_args()
    cs = cases()
    print(f'{len(cs)} AbdomenAtlas 3.0 IID test scans', flush=True)
    d = out_dirs(W)
    while True:
        if a.stage == 'resample':
            todo = [c for c in cs if not os.path.exists(os.path.join(d['lab'], f'{c}.nii.gz')) and ready(c)]
            with Pool(a.workers) as p:
                for r in p.imap_unordered(resample_one, todo):
                    print(r, flush=True)
            left = sum(not os.path.exists(os.path.join(d['lab'], f'{c}.nii.gz')) for c in cs)
        elif a.stage == 'ts':
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import lesion_geometry as G
            G.run_ts(W, a.device, cs, out_dirs)         # skips cases without a resampled image yet
            left = sum(not os.path.exists(os.path.join(d['seg'], f'{c}.nii.gz')) for c in cs)
        else:
            import pandas as pd
            with Pool(a.workers) as p:
                t = pd.DataFrame(p.map(finalize_one, cs))
            meta = pd.read_csv(f'{RAW}/AbdomenAtlas3.0MiniWithMeta.csv').set_index('BDMAP ID')
            t['contrast'] = t.case.map(meta['contrast'])
            t.to_parquet(os.path.join(W, 'aa3', 'cases.parquet'), index=False)
            s = pd.DataFrame(dict(index=np.arange(len(t)), volume=t.case + '.nii.gz', patient_id=t.case,
                                  img=[os.path.join(d['img'], f'{c}.nii.gz') for c in t.case],
                                  seg=[os.path.join(d['seg'], f'{c}.nii.gz') for c in t.case], lat_pairs='[]'))
            s.to_parquet(os.path.join(W, 'extaudit', 'aa3_set.parquet'), index=False)
            for o in LESION:
                print(o, t[f'{o}_label'].value_counts().to_dict())
            return
        print(f'{time.strftime("%T")} {a.stage}: {left} cases left', flush=True)
        if not a.loop or left == 0:
            return
        time.sleep(300)


if __name__ == '__main__':
    main()
