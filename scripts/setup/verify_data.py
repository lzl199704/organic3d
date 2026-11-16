"""Check that every image and mask referenced by the manifests exists under the env's dataset roots.

    python scripts/setup/verify_data.py --env configs/env/local.yaml [--splits train,val,test] [--workers 32]
Exits non-zero and lists examples if anything is missing.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from organic.config import abspath, load_config, read_manifest

ap = argparse.ArgumentParser()
ap.add_argument('--env', required=True)
ap.add_argument('--splits', default='train,val,test,rexgrounding_holdout')
ap.add_argument('--workers', type=int, default=32)
a = ap.parse_args()
cfg = load_config(os.path.join(os.path.dirname(__file__), '../../configs/stageA.yaml'), env=a.env)
bad = 0
for d in ('merlin', 'ctrate'):
    m = read_manifest(cfg, d)
    m = m[m.split.isin(a.splits.split(','))]
    paths = [abspath(cfg, d, p) for p in pd.concat([m.img_rel, m.seg_rel])]
    with ThreadPoolExecutor(a.workers) as ex:
        ok = list(ex.map(os.path.exists, paths))
    miss = [p for p, o in zip(paths, ok) if not o]
    bad += len(miss)
    print(f'{d}: {len(m)} volumes, {len(paths)} files, missing {len(miss)}', flush=True)
    for p in miss[:5]:
        print('   missing:', p)
sys.exit(1 if bad else 0)
