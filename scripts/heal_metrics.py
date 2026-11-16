"""Lesion erasure (LE) on the heal sets of scripts/prepare_heal.py (2026-11-03, Table 1b): same protocol and metric as
scripts/boot_utils.py. Per lesion with a valid same-organ control: hit = (s_clean - s_les) > (s_clean - s_ctrl).
Patient-level bootstrap 95% CIs; paired patient bootstrap of the hit difference --ref minus every other model.
Organ groups: aa3 liver / pancreas / kidney (left + right) / all; rex all (left lung, right lung, both lungs as extra rows).

    python scripts/heal_metrics.py --ds aa3
Writes {W}/eval/heal_<ds>_summary.csv and heal_<ds>_paired.csv.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

from boot_utils import boot, groups

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workroot import work_root                              # noqa: E402

W = work_root()
NAMES = ('CT-CLIP,CT-VocabFine,fVLM,ViSD-Boost,HLIP,ItemizedCLIP,COLIPRI,enc1_finetune,Jolia,SPECTRE,Merlin,'
         'RADAR-plus,RADAR-plus-ft,enc4_organic3d')


def load(name, ds):
    fs = sorted(glob.glob(os.path.join(W, 'extaudit', 'out', name, f'heal_{ds}_shard*.parquet')))
    if not fs:
        return None
    n = {int(f.rsplit('of', 1)[1].split('.')[0]) for f in fs}
    if len(n) != 1 or len(fs) != n.pop():
        print(f'{name}: incomplete shards {fs}')
        return None
    t = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    t['organ_g'] = t.organ.replace({'left_kidney': 'kidney', 'right_kidney': 'kidney'})
    t['margin'] = (t.s_clean - t.s_les) - (t.s_clean - t.s_ctrl)
    t['hit'] = t.margin > 0
    return t


def subsets(t, ds):
    yield 'all', t
    for o in (('liver', 'pancreas', 'kidney') if ds == 'aa3' else ('left_lung', 'right_lung', 'lungs')):
        yield o, t[t.organ_g == o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True, choices=['aa3', 'rex'])
    ap.add_argument('--names', default=NAMES)
    ap.add_argument('--ref', default='enc4_organic3d')
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    n_set = len(pd.read_parquet(os.path.join(W, 'heal', a.ds, 'findings.parquet')))
    T, rows = {}, []
    for name in a.names.split(','):
        t = load(name, a.ds)
        if t is None:
            print('no complete output for', name); continue
        if len(t) != n_set:
            print(f'WARNING {name}: {len(t)} rows, set has {n_set}')
        T[name] = t
        v = t[t.ctrl_valid]
        for organ, h in subsets(v, a.ds):
            if len(h) < 10:
                continue
            mem = groups(h.patient.values)
            x = h.hit.to_numpy(float)
            lo, hi = boot(x, mem, a.boot, rng)
            rows.append(dict(model=name, organ=organ, n_lesions=len(h), n_patients=len(mem), hit=x.mean(), hit_lo=lo,
                             hit_hi=hi, ties=float((h.margin == 0).mean())))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(W, 'eval', f'heal_{a.ds}_summary.csv'), index=False)
    pr = []
    if a.ref in T:
        key = ['volume', 'k']
        R = T[a.ref][T[a.ref].ctrl_valid]
        for name, t in T.items():
            if name == a.ref:
                continue
            m = R[key + ['hit', 'organ_g', 'patient']].merge(t[key + ['hit']], on=key, suffixes=('', '_b'))
            for organ, h in subsets(m, a.ds):
                if len(h) < 10:
                    continue
                d = h.hit.to_numpy(float) - h.hit_b.to_numpy(float)
                lo, hi = boot(d, groups(h.patient.values), a.boot, rng)
                pr.append(dict(ref=a.ref, model=name, organ=organ, n=len(h), diff=d.mean(), lo=lo, hi=hi))
    p = pd.DataFrame(pr)
    p.to_csv(os.path.join(W, 'eval', f'heal_{a.ds}_paired.csv'), index=False)
    if len(s):
        print(s.pivot(index='model', columns='organ', values='hit').mul(100).round(1).to_string())
    if len(p):
        print(p[p.organ == 'all'].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
