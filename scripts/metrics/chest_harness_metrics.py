"""Metrics of the harness chest rung (extaudit/chest.py, --task chest; 2026-10-24) with the stage-A chest evaluator's own
metric functions (organic/encoder/evaluate_chest.py metrics_lobes, metrics_edits), so encoders scored in the harness and in
organic.encoder.evaluate_chest use one rule. Labels: extaudit/ctrate_chest_set.parquet.

    python scripts/metrics/chest_harness_metrics.py --names CT-CLIP,COLIPRI,...
Writes {W}/eval/<name>/ctrate_chest_harness.json and {W}/eval/chest_harness_summary.csv (one row per model).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from organic.encoder.evaluate_chest import metrics_edits, metrics_lobes      # noqa: E402
from organic.vocab import FINDING_NAMES                              # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from organic.workroot import work_root                              # noqa: E402

W = work_root()


def load(name):
    fs = sorted(glob.glob(f'{W}/extaudit/out/{name}/chest_shard*.npz'))
    if not fs:
        return None
    Z = [np.load(f, allow_pickle=True) for f in fs]
    ns = {int(os.path.basename(f).split('of')[1].split('.')[0]) for f in fs}
    assert len(ns) == 1 and len(fs) == ns.pop(), f'{name}: incomplete shards {fs}'
    lobes = sum((json.loads(str(z['lobes'])) for z in Z), [])
    idx = np.concatenate([z['edit_index'] for z in Z])
    return dict(lobes=lobes, index=idx, nvox=np.concatenate([z['edit_nvox'] for z in Z]),
                delta=np.concatenate([z['edit_delta'] for z in Z]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', required=True)
    a = ap.parse_args()
    st = pd.read_parquet(f'{W}/extaudit/ctrate_chest_set.parquet').set_index('index')
    lab = st[[f'label_{f}' for f in FINDING_NAMES]].to_numpy(np.float32)
    pos = {int(i): j for j, i in enumerate(st.index)}
    rows = []
    for name in a.names.split(','):
        r = load(name)
        if r is None:
            print(f'{name}: no chest output'); continue
        recs = [dict(labels=lab[pos[int(i)]].tolist(), delta=d, nvox=v.tolist())
                for i, d, v in zip(r['index'], r['delta'], r['nvox'])]
        out = dict(model=name, n_scans=len(recs), lobes=metrics_lobes(r['lobes']), edits=metrics_edits(recs))
        os.makedirs(f'{W}/eval/{name}', exist_ok=True)
        json.dump(out, open(f'{W}/eval/{name}/ctrate_chest_harness.json', 'w'), indent=1)
        lo, ed = out['lobes'], out['edits']
        row = dict(model=name, n_scans=len(recs), n_sent=lo['n_sent'], lobe_img_top1=lo['img_top1'],
                   lobe_img_side=lo['img_side'], lobe_text_top1=lo['text_top1'], lobe_margin_adj=lo['img_margin_adj'])
        for e, v in ed.items():
            row.update({f'{e}_spec': v['specificity'], f'{e}_own_dauroc': v['own_d_auroc'], f'{e}_n': v['n_scans']})
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}))
    p = f'{W}/eval/chest_harness_summary.csv'
    old = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    new = pd.DataFrame(rows)
    if not old.empty:
        old = old[~old.model.isin(new.model)]
    pd.concat([old, new], ignore_index=True).to_csv(p, index=False)


if __name__ == '__main__':
    main()
