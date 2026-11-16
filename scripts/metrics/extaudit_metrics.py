"""Ladder metrics for a released encoder from its extaudit shards (extaudit/core.py), with the same metric and output
code as our models (evaluate_stageA.metrics / write_outputs), so the tables compare like with like.

    python scripts/metrics/extaudit_metrics.py --env configs/env/local.yaml --name CT-CLIP --dataset ctrate
Reads {work_root}/extaudit/out/{name}/{dataset}_shard*.npz; writes {work_root}/eval/{name}/{dataset}_ladder.csv,
_scores.npz and _summary.json (no organ-slot swap or block tests: those need our query slots / are ours only).
CT-RATE is scored on the official v2 test set (organic/ctrate_v2.py; 3,002 volumes) unless --keep_no_chest.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from organic.config import load_config
from organic.ctrate_v2 import no_chest_volumes
from organic.evaluate_stageA import write_outputs
from organic.vocab import FINDING_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--dataset', default='ctrate')
    ap.add_argument('--fills', default='fat,soft')
    ap.add_argument('--out_name', default=None, help='eval/ folder name (default: --name)')
    ap.add_argument('--keep_no_chest', action='store_true', help='CT-RATE: keep the 3 v2-excluded brain series')
    ap.add_argument('--root', default=None, help='harness root (default {work_root}/extaudit)')
    a = ap.parse_args()
    cfg = load_config(os.path.join(os.path.dirname(__file__), '../../configs/stageA.yaml'), env=a.env)
    root = a.root or os.path.join(cfg.paths.work_root, 'extaudit')
    fs = sorted(glob.glob(os.path.join(root, 'out', a.name, f'{a.dataset}_shard*.npz')))
    assert fs, f'no shards for {a.name} {a.dataset}'
    parts = [dict(np.load(f, allow_pickle=False)) for f in fs]
    conds = list(parts[0]['conds'])
    idx = np.concatenate([p['index'] for p in parts])
    order = np.argsort(idx)
    idx = idx[order]
    st = pd.read_parquet(os.path.join(root, f'{a.dataset}_set.parquet')).set_index('index')
    keep = np.ones(len(idx), bool)
    if a.dataset.startswith('ctrate') and not a.keep_no_chest:
        drop = set(st.index[st.volume.isin(no_chest_volumes(cfg.paths.work_root))])
        keep = ~np.isin(idx, list(drop))
        st = st.drop(index=list(drop))
    order, idx = order[keep], idx[keep]
    missing = len(st) - len(idx)
    lab = st.loc[idx, [f'label_{f}' for f in FINDING_NAMES]].to_numpy(float)
    res = {}
    for c in conds:
        res[c] = {'delta': torch.from_numpy(np.concatenate([p[f'delta__{c}'] for p in parts])[order]),
                  'feat': torch.from_numpy(np.concatenate([p[f'feat__{c}'] for p in parts])[order].astype(np.float32)),
                  'nvox': torch.from_numpy(np.concatenate([p[f'nvox__{c}'] for p in parts])[order])}
    res['index'] = idx
    res['patient'] = st.loc[idx, 'patient_id'].astype(str).values
    lat = np.concatenate([p['lat'] for p in parts])
    res['lat'] = lat[np.isin(lat[:, 0], idx)]
    errs = sum(len(p['errors']) for p in parts)
    print(f'{a.name} {a.dataset}: {len(idx)} volumes ({missing} missing, {errs} load errors), {len(fs)} shards')
    write_outputs(os.path.join(cfg.paths.work_root, 'eval', a.out_name or a.name), a.out_name or a.name, a.dataset,
                  res, lab, conds, a.fills.split(','))


if __name__ == '__main__':
    main()
