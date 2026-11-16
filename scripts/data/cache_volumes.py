"""Precompute a 2 mm cache so 24 GPUs are not limited by NIfTI decompression.

Per volume: anatomy-centred crop of (192 + 2*MARGIN)^3 at 2 mm (margin = jitter room), image in HU as int16,
organ-group map as uint8, saved as {work_root}/cache/{dataset}/{volume}.npz (compressed). Training then takes
a random 192^3 window (jitter up to MARGIN*2 mm); evaluation takes the centre.

    python scripts/data/cache_volumes.py --env configs/env/local.yaml --workers 48 [--splits train,val,test]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import nibabel as nib
from organic.config import load_config, abspath, read_manifest
from organic.volio import read_image, read_mask, image_space, sample_raw_on_mask_grid, hu_checks
from organic.data import SIZE, SPACING, CENTER_ORGANS, _LUT, _crop_bounds, _crop_pad, crop_center
from organic.vocab import ORGAN_ID, BACKGROUND

MARGIN = 8   # voxels at 2 mm = 16 mm jitter each way
_cfg = None


def _init(cfg):
    global _cfg
    torch.set_num_threads(1)
    _cfg = cfg


def _one(args):
    cfg_root, dataset, volume, img_rel, seg_rel, out = args
    if os.path.exists(out):
        return 0.0, volume, True, {}
    t = time.time()
    seg_path = abspath(_cfg, dataset, seg_rel)
    raw_space = image_space(_cfg, dataset) == 'raw'
    if raw_space:                      # original scan + 1 mm mask (layout img_space: raw): mask gives grid and position
        import nibabel as nib
        snii = nib.load(seg_path)
        seg = np.asarray(snii.dataobj).astype(np.uint8)
    else:
        seg = read_mask(seg_path)
    organ1 = _LUT[seg]
    center = crop_center(organ1, dataset)
    full = SIZE + 2 * MARGIN
    b1 = _crop_bounds(center, seg.shape, int(full * SPACING))
    seg_c = _crop_pad(seg, b1, 0)[1::2, 1::2, 1::2]                 # raw TotalSegmentator ids at the 2 mm positions
    organ = np.ascontiguousarray(_LUT[seg_c])
    if raw_space:
        # the 1 mm path averages voxels 2k and 2k+1 of the crop, i.e. samples mask-grid position start + 2k + 0.5
        vol = torch.from_numpy(sample_raw_on_mask_grid(abspath(_cfg, dataset, img_rel), snii.affine, seg.shape,
                                                       [b[0] for b in b1], 2.0, 0.5, (full,) * 3))
    else:
        img = read_image(abspath(_cfg, dataset, img_rel))
        img_c = _crop_pad(img, b1, -1000.0)
        vol = F.interpolate(torch.from_numpy(img_c)[None, None], size=(full,) * 3, mode='trilinear', align_corners=False)[0, 0]
    ok, med = hu_checks(vol.numpy(), seg_c)
    if not raw_space:
        ok = True          # lab 1 mm images are aligned with their masks: the check is recorded only, never fails the job
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out + '.tmp.npz', img=vol.round().clamp(-1024, 3071).to(torch.int16).numpy(), organ=organ)
    os.replace(out + '.tmp.npz', out)
    return time.time() - t, volume, ok, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--splits', default='train,val,test')
    ap.add_argument('--datasets', default='ctrate,merlin', help='comma list of datasets to cache')
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(os.path.join(os.path.dirname(__file__), '../../configs/stageA.yaml'), env=a.env)
    jobs = []
    for d in a.datasets.split(','):
        m = read_manifest(cfg, d).drop_duplicates('volume')
        m = m[m.split.isin(a.splits.split(','))]
        if d == 'ctrate':   # training uses one reconstruction per scan; test keeps every volume
            m = pd.concat([m[m.split == 'test'], m[m.split != 'test'].groupby('scan_id', group_keys=False).head(1)])
        if a.limit:
            m = m.head(a.limit)
        for r in m.itertuples():
            jobs.append((None, d, r.volume, r.img_rel, r.seg_rel, os.path.join(cfg.paths.work_root, 'cache', d, f'{r.volume}.npz')))
    print(len(jobs), 'volumes to cache', flush=True)
    os.makedirs(os.path.join(cfg.paths.work_root, 'cache'), exist_ok=True)
    t0, done, secs, bad = time.time(), 0, [], []
    log = open(os.path.join(cfg.paths.work_root, 'cache', 'hu_checks.tsv'), 'a')
    with Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        for s, vol, ok, med in pool.imap_unordered(_one, jobs, chunksize=4):
            done += 1; secs.append(s)
            if med:
                log.write(f'{vol}\t{int(ok)}\t{med}\n')
            if not ok:
                bad.append((vol, med))
                print('HU CHECK FAILED', vol, med, flush=True)
            if done >= 200 and len(bad) > 0.02 * done:
                raise SystemExit(f'stopping: {len(bad)} of {done} volumes fail the image-mask HU check (image and mask misaligned?)')
            if done % 500 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f'{done}/{len(jobs)}  {el / 60:.1f} min elapsed, ETA {(len(jobs) - done) * el / done / 60:.1f} min', flush=True)
    log.close()
    print('median seconds per volume per worker', float(np.median([s for s in secs if s > 0] or [0])))
    print(f'HU check: {len(bad)} of {done} volumes failed (log: cache/hu_checks.tsv)')


if __name__ == '__main__':
    main()
