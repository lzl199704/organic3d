"""Ladder metrics for a VLM from its extaudit/vlm_core.py shards on the full test sets (<dataset>_set.parquet: CT-RATE
v2 3,005 rows -> 3,002 after the non-chest drop, Merlin 5,109; scripts/metrics/vlm_scores.py), with the same metric
code for every VLM (evaluate_stageB.metrics). Wrong patient = another patient's answers (finding_reports.partner: the
scan half the list away in patient order; on CT-RATE the neighbouring scan is often the same patient).

    python scripts/metrics/vlm_metrics.py --env configs/env/local.yaml --name M3D-LaMed --datasets ctrate,merlin
Writes {work_root}/evalB/{name}/{dataset}_ladder.csv and summary.json (merged with the datasets already in it).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from organic.config import load_config
from organic.evaluate_stageB import metrics
from organic.ctrate_v2 import no_chest_volumes
from organic.vocab import FINDING_NAMES
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlm_scores  # noqa: E402
from finding_reports import partner, patient  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--datasets', default='ctrate,merlin')
    a = ap.parse_args()
    cfg = load_config(os.path.join(os.path.dirname(__file__), '../../configs/stageA.yaml'), env=a.env)
    root = os.path.join(cfg.paths.work_root, 'extaudit')
    out = os.path.join(cfg.paths.work_root, 'evalB', a.name)
    os.makedirs(out, exist_ok=True)
    sname = os.path.join(out, 'summary.json')
    summary = json.load(open(sname)) if os.path.exists(sname) else {}
    for d in a.datasets.split(','):
        cat = vlm_scores.load(a.name, d)
        if cat is None:
            print('no complete run for', d); continue
        finds = list(cat.pop('findings'))
        key = 'D__' if any(k.startswith('D__') for k in cat) else 'P__'      # logit difference when saved (no ties)
        st = pd.read_parquet(os.path.join(root, f'{d}_set.parquet'))
        if d == 'ctrate':                       # CT-RATE v2: the brain series are not scored (organic/ctrate_v2.py)
            bad = st['index'][st.volume.isin(no_chest_volumes(cfg.paths.work_root))].to_numpy()
            keep = ~np.isin(cat['index'], bad); cat = {k: v[keep] for k, v in cat.items()}
        n = len(cat['index'])
        res = {k[3:].split('@')[0]: torch.tensor(v) for k, v in cat.items() if k.startswith(key)}
        vols = st.set_index('index').loc[cat['index'], 'volume'].to_numpy()
        res['wrong'] = res['clean'][partner(np.array([patient(v, d) for v in vols]))]
        res.update({'nvox_' + k[6:].split('@')[0]: torch.tensor(v) for k, v in cat.items() if k.startswith('nvox__')})
        res['feat'] = torch.tensor(cat['feat']); res['index'] = torch.tensor(cat['index'])
        res['gen_index'] = []; res['texts'] = {'clean': [], 'wrong': []}
        res['js'] = [FINDING_NAMES.index(f) for f in finds]
        L = st[[f'label_{f}' for f in FINDING_NAMES]].to_numpy(np.float32)          # row i = index i (metrics indexes it)
        tab, summ = metrics(res, L, d)
        tab.to_csv(os.path.join(out, f'{d}_ladder.csv'))
        summary[d] = summ
        print('SUMMARY', json.dumps({'run': a.name, 'dataset': d, 'n': n, **summ}), flush=True)
    with open(sname, 'w') as fh:
        json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
