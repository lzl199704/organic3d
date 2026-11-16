"""Balanced side accuracy on unflipped scans (2026-11-03; extaudit/side.py). Per sentence naming one side: correct =
score(sentence) > score(same sentence, side swapped). Balanced side accuracy = mean of the accuracy on left-side and on
right-side sentences: 50% for chance and for any model that ignores side (a fixed text preference scores 100 on one
side and 0 on the other). Plain accuracy (= the old harness lat_acc on ctrate / merlin) is kept as a check.
Patient-level bootstrap 95% CIs; paired patient bootstrap of the difference --ref minus every other model.

    python scripts/metrics/side_metrics.py
Writes {W}/eval/side_summary.csv and side_paired.csv (sets ctrate, merlin, rex, aa3).
"""
import argparse
import glob
import os
import sys
import re

import numpy as np
import pandas as pd

from boot_utils import groups

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from organic.workroot import work_root                              # noqa: E402

W = work_root()
NAMES = ('CT-CLIP,CT-VocabFine,fVLM,ViSD-Boost,HLIP,ItemizedCLIP,COLIPRI,enc1_finetune,Jolia,SPECTRE,Merlin,RADAR-plus,'
         'RADAR-plus-ft,enc2_queries,enc3_routing,enc4_organic3d')
LEFT, RIGHT = re.compile(r'\bleft\b', re.I), re.compile(r'\bright\b', re.I)


def load(name, s):
    fs = sorted(glob.glob(os.path.join(W, 'extaudit', 'out', name, f'side_{s}_shard*.parquet')))
    if not fs:
        return None
    n = {int(f.rsplit('of', 1)[1].split('.')[0]) for f in fs}
    if len(n) != 1 or len(fs) != n.pop():
        print(f'{name} {s}: incomplete shards'); return None
    t = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    if 'side' not in t:
        l, r = t.text_true.str.contains(LEFT), t.text_true.str.contains(RIGHT)
        t['side'] = np.where(l & ~r, 'left', np.where(r & ~l, 'right', ''))
    if 'patient' not in t:
        st = pd.read_parquet(os.path.join(W, 'extaudit', f'{s}_set.parquet'), columns=['volume', 'patient_id'])
        t = t.merge(st.rename(columns={'patient_id': 'patient'}), on='volume', how='left')
    t = t[t.side != ''].copy()
    t['ok'] = (t.s_true > t.s_swap).astype(float)
    return t


def bal(ok, left):
    return 0.5 * (ok[left].mean() + ok[~left].mean())


def boot(fn, mem, B, rng):
    bs = []
    for _ in range(B):
        idx = np.concatenate([mem[k] for k in rng.integers(0, len(mem), len(mem))])
        bs.append(fn(idx))
    return np.nanquantile(bs, [0.025, 0.975])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', default=NAMES)
    ap.add_argument('--ref', default='enc4_organic3d')
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows, pr = [], []
    for s in ('ctrate', 'merlin', 'rex', 'aa3'):
        T = {}
        for name in a.names.split(','):
            t = load(name, s)
            if t is None:
                continue
            T[name] = t
            ok, left = t.ok.to_numpy(), (t.side == 'left').to_numpy()
            mem = groups(t.patient.astype(str).values)
            lo, hi = boot(lambda i: bal(ok[i], left[i]), mem, a.boot, rng)
            rows.append(dict(model=name, set=s, n=len(t), n_left=int(left.sum()), n_patients=len(mem), acc=ok.mean(),
                             acc_left=ok[left].mean(), acc_right=ok[~left].mean(), bal=bal(ok, left), bal_lo=lo,
                             bal_hi=hi, ties=float((t.s_true == t.s_swap).mean())))
        if a.ref not in T:
            continue
        key = ['volume', 'j']
        R = T[a.ref]
        for name, t in T.items():
            if name == a.ref:
                continue
            m = R[key + ['ok', 'side', 'patient']].merge(t[key + ['ok']], on=key, suffixes=('', '_b'))
            x, y, left = m.ok.to_numpy(), m.ok_b.to_numpy(), (m.side == 'left').to_numpy()
            lo, hi = boot(lambda i: bal(x[i], left[i]) - bal(y[i], left[i]), groups(m.patient.astype(str).values), a.boot, rng)
            pr.append(dict(set=s, ref=a.ref, model=name, n=len(m), diff=bal(x, left) - bal(y, left), lo=lo, hi=hi))
    sm, p = pd.DataFrame(rows), pd.DataFrame(pr)
    sm.to_csv(os.path.join(W, 'eval', 'side_summary.csv'), index=False)
    p.to_csv(os.path.join(W, 'eval', 'side_paired.csv'), index=False)
    if len(sm):
        print(sm.pivot(index='model', columns='set', values='bal').mul(100).round(1).to_string())
        print(sm.pivot(index='model', columns='set', values='acc').mul(100).round(1).to_string())
    if len(p):
        print(p.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
