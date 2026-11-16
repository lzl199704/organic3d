"""Public CT files -> the 1 mm RAS NIfTI + TotalSegmentator v2 mask layout every other script expects.

    <dst_root>/1mm/img/<rel>.nii.gz       HU, int16, 1 mm isotropic, RAS (nibabel as_closest_canonical), NOT clipped
    <dst_root>/1mm/ts_seg/<rel>.nii.gz    TotalSegmentator v2 'total' multilabel mask on the same grid (uint8)
(sub-folders configurable: RAD-ChestCT uses --img_sub lung_nii/img --seg_sub lung_nii/seg, see scripts/prepare_radchest.py)

<rel> is the source path relative to --src_dir without '.nii.gz', so the folder structure is kept:
  CT-RATE  (HF ibrahimhamamci/CT-RATE, dataset/{train,valid}/<patient>/<scan>/<volume>.nii.gz)
           --src_dir <CT-RATE>/dataset/train (and .../valid)  ->  1mm/img/<patient>/<scan>/<volume>.nii.gz
  Merlin   (Stanford AIMI, <study>.nii.gz)                     ->  1mm/img/<study>.nii.gz
  RAD-ChestCT: convert the released .npz volumes to NIfTI first (RAD-ChestCT documentation), then as above.

Resampling: reorient to RAS, new size = round(old size *
spacing), HU trilinear (align_corners=False), rounded to int16; affine adjusted so voxel centres stay aligned.
TotalSegmentator: `totalsegmentator(img, seg, ml=True, fast=..., device=...)`.

Note: the 1 mm volumes and masks used for the paper's CT-RATE and Merlin numbers came from a pre-existing conversion
with the same conventions (1 mm, RAS, TotalSegmentator v2 ids); this script reproduces those conventions from the
public files, so voxel values can differ slightly. CT-RATE: if your copy stores raw values, pass --rescale_csv
(CT-RATE metadata: VolumeName, RescaleSlope, RescaleIntercept); the printed HU check (median HU in air / lungs /
liver) should look like CT.

    python scripts/prepare_1mm.py --src_dir /path/to/CT-RATE/dataset/valid --dst_root /path/to/CT-RATE --stage resample --workers 8
    <TotalSegmentator env>/bin/python scripts/prepare_1mm.py --dst_root /path/to/CT-RATE --stage ts --device gpu
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import nibabel as nib
import numpy as np


def resample_one(job):
    """(src, dst_img, slope, intercept) -> 1 mm RAS int16 NIfTI (1 mm RAS, trilinear HU)."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    src, fi, slope, inter = job
    if os.path.exists(fi):
        return dict(src=src, skipped=True)
    try:
        t0 = time.time()
        im = nib.load(src)
        imc = nib.as_closest_canonical(im)
        a = np.ascontiguousarray(np.asarray(imc.dataobj, dtype=np.float32))     # canonical flips give negative strides
        if slope is not None:
            a = a * float(slope) + float(inter)
        zoom = np.asarray(imc.header.get_zooms()[:3], float)
        n_old = np.asarray(a.shape)
        n_new = np.maximum(1, np.round(n_old * zoom)).astype(int)
        size = tuple(int(x) for x in n_new)
        hu = F.interpolate(torch.from_numpy(a)[None, None], size=size, mode='trilinear', align_corners=False)[0, 0].numpy()
        hu = np.clip(np.rint(hu), -32768, 32767).astype(np.int16)
        sc = n_old / n_new                       # new voxel i' centre = old continuous index (i' + 0.5) * n/n' - 0.5
        S = np.eye(4)
        S[:3, :3] = np.diag(sc)
        S[:3, 3] = 0.5 * sc - 0.5
        aff = imc.affine @ S
        img = nib.Nifti1Image(hu, aff)
        img.set_data_dtype(np.int16)
        img.set_qform(aff, code=1)
        img.set_sform(aff, code=1)
        os.makedirs(os.path.dirname(fi), exist_ok=True)
        nib.save(img, fi + '.tmp.nii.gz')
        os.replace(fi + '.tmp.nii.gz', fi)
        return dict(src=src, axcodes=''.join(nib.aff2axcodes(im.affine)), zooms=zoom.tolist(), shape_1mm=list(size),
                    seconds=time.time() - t0)
    except Exception as ex:                                           # noqa: BLE001
        return dict(src=src, error=str(ex)[:300])


def run_ts(dst_root, device, fast, img_sub='1mm/img', seg_sub='1mm/ts_seg'):
    """TotalSegmentator v2 'total' on every 1 mm image without a mask; lock files let several workers share the list."""
    from totalsegmentator.python_api import totalsegmentator
    imgs = sorted(glob.glob(os.path.join(dst_root, img_sub, '**', '*.nii.gz'), recursive=True))
    lock = os.path.join(dst_root, '.ts_lock')
    os.makedirs(lock, exist_ok=True)
    done = 0
    for src in imgs:
        rel = os.path.relpath(src, os.path.join(dst_root, img_sub))
        dst = os.path.join(dst_root, seg_sub, rel)
        key = rel.replace(os.sep, '__')
        if os.path.exists(dst):
            continue
        try:
            fd = os.open(os.path.join(lock, key), os.O_CREAT | os.O_EXCL)
            os.close(fd)
        except FileExistsError:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        t0 = time.time()
        try:
            totalsegmentator(src, dst + '.tmp.nii.gz', ml=True, fast=fast, device=device, quiet=True,
                             nr_thr_resamp=2, nr_thr_saving=2)
            os.replace(dst + '.tmp.nii.gz', dst)
            done += 1
            print(f'{rel} {device} {time.time() - t0:.0f} s', flush=True)
        except Exception as ex:                                       # noqa: BLE001
            print('ERROR', rel, str(ex)[:300], flush=True)
            os.remove(os.path.join(lock, key))
    print(f'TotalSegmentator worker ({device}): {done} volumes', flush=True)


def hu_check(dst_root, n=20, seed=0, img_sub='1mm/img', seg_sub='1mm/ts_seg'):
    """Median HU inside TotalSegmentator structures for a random sample (organic.volio.hu_checks rules)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from organic.volio import hu_checks
    segs = sorted(glob.glob(os.path.join(dst_root, seg_sub, '**', '*.nii.gz'), recursive=True))
    rng = np.random.default_rng(seed)
    for s in rng.choice(segs, size=min(n, len(segs)), replace=False) if segs else []:
        img = os.path.join(dst_root, img_sub, os.path.relpath(s, os.path.join(dst_root, seg_sub)))
        ok, med = hu_checks(np.asarray(nib.load(img).dataobj, np.float32), np.asarray(nib.load(s).dataobj).astype(np.uint8))
        print('OK ' if ok else 'BAD', os.path.relpath(img, dst_root), med, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['resample', 'ts', 'check'], required=True)
    ap.add_argument('--src_dir', help='resample: folder searched recursively for *.nii.gz')
    ap.add_argument('--dst_root', required=True, help='dataset root of the env config (merlin_root / ctrate_root / ...)')
    ap.add_argument('--rescale_csv', default=None, help='optional CSV: VolumeName, RescaleSlope, RescaleIntercept')
    ap.add_argument('--img_sub', default='1mm/img', help='image sub-folder below --dst_root')
    ap.add_argument('--seg_sub', default='1mm/ts_seg', help='mask sub-folder below --dst_root')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', default='gpu', help="ts: 'gpu' or 'cpu'")
    ap.add_argument('--fast', action='store_true', help='ts: TotalSegmentator --fast (3 mm model)')
    a = ap.parse_args()
    if a.stage == 'resample':
        srcs = sorted(glob.glob(os.path.join(a.src_dir, '**', '*.nii.gz'), recursive=True))
        resc = {}
        if a.rescale_csv:
            import pandas as pd
            t = pd.read_csv(a.rescale_csv)
            resc = {r.VolumeName: (r.RescaleSlope, r.RescaleIntercept) for r in t.itertuples()}
        jobs = []
        for s in srcs:
            rel = os.path.relpath(s, a.src_dir)
            sl, ic = resc.get(os.path.basename(s), (None, None))
            jobs.append((s, os.path.join(a.dst_root, a.img_sub, rel), sl, ic))
        print(len(jobs), 'volumes', flush=True)
        with Pool(a.workers) as p:
            for k, r in enumerate(p.imap_unordered(resample_one, jobs)):
                if 'error' in r or k % 100 == 0:
                    print(k, r, flush=True)
    elif a.stage == 'ts':
        run_ts(a.dst_root, a.device, a.fast, a.img_sub, a.seg_sub)
    else:
        hu_check(a.dst_root, img_sub=a.img_sub, seg_sub=a.seg_sub)


if __name__ == '__main__':
    main()
