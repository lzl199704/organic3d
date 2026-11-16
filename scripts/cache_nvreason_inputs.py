"""CPU job: NV-Reason-CT inputs for the audit sample, in the model's own geometry (LPS, 2 mm, anatomy crop),
image in HU (int16) and organ groups (uint8), so GPU time is spent only on the model.

    python scripts/cache_nvreason_inputs.py --env configs/env/local.yaml --limit 1000 --workers 24
Writes {work_root}/cache_nv/{dataset}/{volume}.npz; existing files are skipped.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from organic.audit_nvreason import REGION, NVPrep, audit_sample
from organic.config import abspath, load_config
from organic.volio import image_space

_prep = None


def _init():
    global _prep
    import torch
    torch.set_num_threads(1)
    _prep = NVPrep()


def _one(job):
    img, seg, region, out, raw = job
    if os.path.exists(out):
        return 0.0
    t = time.time()
    hu, organ = _prep.load(img, seg, region, raw_space=raw)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out + '.tmp.npz', hu=hu[0].round().clamp(-1024, 3071).short().numpy(), organ=organ.numpy().astype(np.uint8))
    os.replace(out + '.tmp.npz', out)
    return time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--datasets', default='merlin,ctrate')
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    cfg = load_config(os.path.join(os.path.dirname(__file__), '../configs/stageA.yaml'), env=a.env)
    os.environ.setdefault('HF_HOME', cfg.paths.hf_home)
    jobs = []
    for d in a.datasets.split(','):
        m = audit_sample(cfg, d, a.limit)
        for r in m.itertuples():
            jobs.append((abspath(cfg, d, r.img_rel), abspath(cfg, d, r.seg_rel), REGION[d],
                         os.path.join(cfg.paths.work_root, 'cache_nv', d, f'{r.volume}.npz'), image_space(cfg, d) == 'raw'))
    print(len(jobs), 'volumes', flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for _ in ex.map(_one, jobs, chunksize=2):
            done += 1
            if done % 100 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f'{done}/{len(jobs)} {el / 60:.1f} min, ETA {(len(jobs) - done) * el / done / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
