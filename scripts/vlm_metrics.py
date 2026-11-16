"""Ladder metrics for an external VLM from its extaudit/vlm_core.py shards, with the same metric code as NV-Reason-CT,
CT-CHAT, MedGemma and our stage-B models (evaluate_stageB.metrics). Wrong patient = the next scan's answers.

    python scripts/vlm_metrics.py --env configs/env/local.yaml --name M3D-LaMed --datasets ctrate,merlin
Writes {work_root}/evalB/{name}/{dataset}_ladder.csv and summary.json.

--set full (2026-10-25): the full test sets (<dataset>_set.parquet: CT-RATE v2 3,005 rows -> 3,002 after the non-chest
drop, Merlin 5,109) = the audit run merged with the run on the remaining scans (scripts/vlm_full.py: by volume, keyed by
the full-set index); writes {dataset}_ladder_full.csv and summary_full.json (merged with the datasets already in it),
so the audit1000 outputs stay untouched. Default (--set audit1000) is unchanged.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from organic.config import load_config
from organic.evaluate_stageB import metrics
from organic.ctrate_v2 import no_chest_volumes
from organic.vocab import FINDING_NAMES
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlm_full  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--datasets', default='ctrate,merlin')
    ap.add_argument('--set', default='audit1000', choices=['audit1000', 'full'])
    a = ap.parse_args()
    full = a.set == 'full'
    set_file = (lambda d: f'{d}_set.parquet') if full else (lambda d: f'{d}_audit1000_set.parquet')
    cfg = load_config(os.path.join(os.path.dirname(__file__), '../configs/stageA.yaml'), env=a.env)
    root = os.path.join(cfg.paths.work_root, 'extaudit')
    out = os.path.join(cfg.paths.work_root, 'evalB', a.name)
    os.makedirs(out, exist_ok=True)
    sname = 'summary_full.json' if full else 'summary.json'
    summary = json.load(open(os.path.join(out, sname))) if full and os.path.exists(os.path.join(out, sname)) else {}
    for d in a.datasets.split(','):
        if full:
            cat = vlm_full.load(a.name, d)
            if cat is None:
                print('full set incomplete for', d); continue
            finds = list(cat.pop('findings')); print(d, 'audit part', cat.pop('n_audit'), 'rest part', cat.pop('n_rest'))
        else:
            fs = sorted(glob.glob(os.path.join(root, 'out_vlm', a.name, f'{d}_shard*.npz')))
            if not fs:
                print('no shards for', d); continue
            Z = [dict(np.load(f)) for f in fs]
            finds = list(Z[0]['findings'])
            cat = {k: np.concatenate([z[k] for z in Z]) for k in Z[0] if k.startswith(('P__', 'D__', 'nvox__', 'index', 'feat'))}
        key = 'D__' if any(k.startswith('D__') for k in cat) else 'P__'      # logit difference when saved (no ties)
        o = np.argsort(cat['index']); cat = {k: v[o] for k, v in cat.items()}
        if d == 'ctrate':                       # CT-RATE v2: the brain series are not scored (organic/ctrate_v2.py)
            st0 = pd.read_parquet(os.path.join(root, set_file(d)))
            bad = st0['index'][st0.volume.isin(no_chest_volumes(cfg.paths.work_root))].to_numpy()
            keep = ~np.isin(cat['index'], bad); cat = {k: v[keep] for k, v in cat.items()}
        n = len(cat['index'])
        res = {k[3:].split('@')[0]: torch.tensor(v) for k, v in cat.items() if k.startswith(key)}
        res['wrong'] = res['clean'][np.roll(np.arange(n), 1)]
        res.update({'nvox_' + k[6:].split('@')[0]: torch.tensor(v) for k, v in cat.items() if k.startswith('nvox__')})
        res['feat'] = torch.tensor(cat['feat']); res['index'] = torch.tensor(cat['index'])
        res['gen_index'] = []; res['texts'] = {'clean': [], 'wrong': []}
        res['js'] = [FINDING_NAMES.index(f) for f in finds]
        st = pd.read_parquet(os.path.join(root, set_file(d)))
        L = st[[f'label_{f}' for f in FINDING_NAMES]].to_numpy(np.float32)          # row i = index i (metrics indexes it)
        tab, summ = metrics(res, L, d)
        tab.to_csv(os.path.join(out, f'{d}_ladder_full.csv' if full else f'{d}_ladder.csv'))
        summary[d] = summ
        print('SUMMARY', json.dumps({'run': a.name, 'dataset': d, 'set': a.set, 'n': n, **summ}), flush=True)
    with open(os.path.join(out, sname), 'w') as fh:
        json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
