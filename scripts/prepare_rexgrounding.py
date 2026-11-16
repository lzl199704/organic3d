"""ReXGroundingCT lesion masks (Baharoon et al. 2025: radiologist segmentations of report findings in CT-RATE scans) on
our 1 mm grid, for the lesion-level grounding test (extaudit/core.py --task grounding).

Masks are (F, H, W, D) on the ORIGINAL CT-RATE voxel grid (dataset/{train,valid}_fixed NIfTI), one channel per report
finding. Each finding is mapped to the 1 mm RAS grid of our volume by world coordinates (nearest neighbour, sampled
backwards inside the lesion's bounding box so thick original slices leave no gaps).

Sanity check printed per category: median HU inside the mapped lesions vs the scan's lung (nodules, consolidation and
effusion must be denser than lung, emphysema darker); a shifted or flipped mapping would erase that contrast.

    python scripts/prepare_rexgrounding.py --env configs/env/local.yaml --workers 32
Writes {work_root}/rexground/masks/<volume>.npz (k -> flat 1 mm voxel indices) and {work_root}/rexground/findings.parquet.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from organic.config import abspath, load_config, read_manifest

REX = None   # ReXGroundingCT release (dataset.json, segmentations/): --rex_root, default <ctrate_root>/ReXGroundingCT
CATEGORY = {'1a': 'bronchial wall thickening', '1b': 'bronchiectasis', '1c': 'emphysema', '1d': 'septal / reticular',
            '1e': 'small airway / micronodules', '1f': 'air trapping / other airway', '2a': 'linear atelectasis',
            '2b': 'consolidation / atelectasis', '2c': 'ground glass / mosaic', '2d': 'nodules', '2e': 'pleural',
            '2f': 'honeycombing', '2g': 'bleb / pneumothorax', '2h': 'other patterns'}
LUNG_IDS = [10, 11, 12, 13, 14]


def one(job):
    vol, entries, img1, seg1, orig, mask_path, out = job
    try:
        i1, o = nib.load(img1), nib.load(orig)
        M = np.linalg.inv(o.affine) @ i1.affine                     # 1 mm voxel -> original voxel
        Minv = np.linalg.inv(M)
        shp1 = i1.shape
        mk = np.asarray(nib.load(mask_path).dataobj)
        if mk.ndim == 3:
            mk = mk[None]
        hu = np.asarray(i1.dataobj, dtype=np.float32)
        seg = np.asarray(nib.load(seg1).dataobj).astype(np.uint8)
        lung = np.isin(seg, LUNG_IDS)
        lung_hu = float(np.median(hu[lung])) if lung.any() else np.nan
        res, rows = {}, []
        for k, txt, cat in entries:
            if k >= mk.shape[0]:
                continue
            pts = np.argwhere(mk[k] > 0)
            if not len(pts):
                continue
            # 1 mm bounding box of the lesion (+1 voxel), then sample backwards
            corners = np.array([[a, b, c, 1] for a in (pts[:, 0].min(), pts[:, 0].max()) for b in (pts[:, 1].min(), pts[:, 1].max())
                                for c in (pts[:, 2].min(), pts[:, 2].max())], float)
            q = (Minv @ corners.T).T[:, :3]
            lo = np.clip(np.floor(q.min(0)) - 1, 0, np.array(shp1) - 1).astype(int)
            hi = np.clip(np.ceil(q.max(0)) + 2, 0, np.array(shp1)).astype(int)
            g = np.stack(np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)], indexing='ij'), -1).reshape(-1, 3)
            src = np.rint((M[:3, :3] @ g.T).T + M[:3, 3]).astype(int)
            ok = np.all((src >= 0) & (src < np.array(mk.shape[1:])), axis=1)
            hit = np.zeros(len(g), bool)
            hit[ok] = mk[k][src[ok, 0], src[ok, 1], src[ok, 2]] > 0
            v = g[hit]
            if not len(v):
                continue
            flat = np.ravel_multi_index(v.T, shp1).astype(np.int64)
            res[f'k{k}'] = flat
            in_lung = float(lung.ravel()[flat].mean())
            rows.append(dict(volume=vol, k=k, text=txt, category=cat, n_vox=len(flat), frac_in_lung=in_lung,
                             lesion_hu=float(np.median(hu.ravel()[flat])), lung_hu=lung_hu))
        np.savez_compressed(out, shape=np.asarray(shp1), **res)
        return rows, None
    except Exception as ex:                                           # noqa: BLE001
        return [], f'{vol}: {str(ex)[:200]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--only_missing', action='store_true')
    ap.add_argument('--rex_root', default=None, help='ReXGroundingCT folder (default <ctrate_root>/ReXGroundingCT)')
    a = ap.parse_args()
    cfg = load_config(os.path.join(os.path.dirname(__file__), '../configs/stageA.yaml'), env=a.env)
    out = os.path.join(cfg.paths.work_root, 'rexground')
    global REX
    REX = a.rex_root or os.path.join(cfg.paths.ctrate_root, 'ReXGroundingCT')
    os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    d = json.load(open(os.path.join(REX, 'dataset.json')))
    m = read_manifest(cfg, 'ctrate').set_index('volume')
    jobs, info = [], []
    for split in ('train', 'val'):
        for x in d[split]:
            v = x['name']
            if v not in m.index:
                continue
            r = m.loc[v]
            pid, scan = v.split('_')[0] + '_' + v.split('_')[1], '_'.join(v.split('_')[:3])
            orig = os.path.join(cfg.paths.ctrate_root, 'dataset', 'valid_fixed' if v.startswith('valid') else 'train_fixed', pid, scan, v)
            entries = [(int(k), x['findings'][k], x['categories'][k]) for k in x['findings']]
            jobs.append((v, entries, abspath(cfg, 'ctrate', r.img_rel), abspath(cfg, 'ctrate', r.seg_rel), orig,
                         os.path.join(REX, 'segmentations', v), os.path.join(out, 'masks', v.replace('.nii.gz', '.npz'))))
            info.append(dict(volume=v, rex_split=split, our_split=r.split, patient=r.patient_id))
    jobs = jobs[: a.limit or None]
    prev = None
    if a.only_missing:                                      # keep earlier results, map only scans without a mask file
        prev = pd.read_parquet(os.path.join(out, 'findings.parquet'))
        keep = set(prev.volume)
        jobs = [j for j in jobs if j[0] not in keep]
        prev = prev.drop(columns=['category_name', 'rex_split', 'our_split', 'patient'], errors='ignore')
    print(len(jobs), 'scans with a mask and a volume in our manifest', flush=True)
    rows, errs = [], []
    with Pool(a.workers) as p:
        for i, (r, e) in enumerate(p.imap_unordered(one, jobs, chunksize=2)):
            rows += r
            if e:
                errs.append(e); print('ERROR', e, flush=True)
            if i % 200 == 0:
                print(f'{i}/{len(jobs)}', flush=True)
    t = pd.concat([prev, pd.DataFrame(rows)]) if prev is not None else pd.DataFrame(rows)
    t = t.merge(pd.DataFrame(info), on='volume', how='left')
    t['category_name'] = t.category.map(CATEGORY)
    t.to_parquet(os.path.join(out, 'findings.parquet'), index=False)
    print(f'{len(t)} findings in {t.volume.nunique()} scans, {len(errs)} errors; splits {t.drop_duplicates("volume").our_split.value_counts().to_dict()}')
    t['contrast'] = t.lesion_hu - t.lung_hu
    print(t.groupby('category_name')[['n_vox', 'frac_in_lung', 'lesion_hu', 'lung_hu', 'contrast']].median().round(1).to_string())


if __name__ == '__main__':
    main()
