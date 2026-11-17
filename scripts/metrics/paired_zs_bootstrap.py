"""Paired patient-level bootstrap of the macro zero-shot AUROC difference, ours vs baselines, on one finding set.

Reads {work_root}/eval/{name}/{dataset}_scores.npz (per-scan scores written by extaudit_metrics / organic.encoder.evaluate) and
{name}/{dataset}_ladder.csv (finding order). Per finding, scans where either model has no score (NaN, e.g. RADAR's
segmenter did not find the organ) or no label are dropped, so both models are scored on the same scans.
  python scripts/metrics/paired_zs_bootstrap.py --env configs/env/local.yaml --ours enc4_organic3d \
      --names RADAR-plus-ft,RADAR-plus,Jolia-long --findings radar
--findings radar = the findings every RADAR checkpoint scores (as the abdominal table); all = every finding with labels.
Writes {work_root}/eval/paired_zs_{dataset}_{findings}.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from organic.config import load_config
from organic.vocab import FINDING_NAMES


def load(W, name, d):
    s = np.load(os.path.join(W, 'eval', name, f'{d}_scores.npz'), allow_pickle=True)
    f = pd.read_csv(os.path.join(W, 'eval', name, f'{d}_ladder.csv'), index_col=0)
    order = np.argsort(s['index'])
    return (s['index'][order], s['patient'][order].astype(str), s['labels'][order], s['delta__clean'][order], f)


def macro(lab, sa, sb, cols, rows):
    out = []
    for j in cols:
        y, a, b = lab[rows, j], sa[rows, j], sb[rows, j]
        m = np.isfinite(y) & (y >= 0) & np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10 or y[m].min() == y[m].max():
            out.append((np.nan, np.nan))
            continue
        out.append((roc_auc_score(y[m], a[m]), roc_auc_score(y[m], b[m])))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--ours', default='enc4_organic3d')
    ap.add_argument('--names', required=True)
    ap.add_argument('--dataset', default='merlin')
    ap.add_argument('--findings', default='radar')
    ap.add_argument('--n_boot', type=int, default=2000)
    a = ap.parse_args()
    W = load_config(os.path.join(os.path.dirname(__file__), '../../configs/stageA.yaml'), env=a.env).paths.work_root
    io, po, lo, so, fo = load(W, a.ours, a.dataset)
    names = a.names.split(',')
    if a.findings == 'radar':
        sets = [set(pd.read_csv(os.path.join(W, 'eval', r, f'{a.dataset}_ladder.csv'), index_col=0)
                    .zero_shot.dropna().index) for r in ('RADAR-plus', 'RADAR-plus-ft')]
        keep = set.intersection(*sets)
    else:
        keep = set(fo.zero_shot.dropna().index)
    cols = [FINDING_NAMES.index(f) for f in fo.index if f in keep]
    pats = np.unique(po)
    p2rows = {p: np.flatnonzero(po == p) for p in pats}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([p2rows[p] for p in rng.choice(pats, len(pats))]) for _ in range(a.n_boot)]
    out = []
    for n in names:
        ib, pb, lb, sb, fb = load(W, n, a.dataset)
        assert (ib == io).all(), f'{n}: scan order differs'
        full = np.arange(len(io))
        pt = macro(lo, so, sb, cols, full)
        ok = np.isfinite(pt).all(1)
        d0 = pt[ok, 0].mean() - pt[ok, 1].mean()
        bd = []
        for rows in boots:
            b = macro(lo, so, sb, [c for c, k in zip(cols, ok) if k], rows)
            g = np.isfinite(b).all(1)
            bd.append(b[g, 0].mean() - b[g, 1].mean())
        bd = np.array(bd)
        lo95, hi95 = np.percentile(bd, [2.5, 97.5])
        r = dict(ours=a.ours, baseline=n, n_findings=int(ok.sum()), ours_zs=pt[ok, 0].mean(), base_zs=pt[ok, 1].mean(),
                 diff=d0, diff_lo=lo95, diff_hi=hi95, p_le0=float((bd <= 0).mean()),
                 findings_ours_better=int((pt[ok, 0] > pt[ok, 1]).sum()))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        out.append(r)
    pd.DataFrame(out).to_csv(os.path.join(W, 'eval', f'paired_zs_{a.dataset}_{a.findings}.csv'), index=False)


if __name__ == '__main__':
    main()
