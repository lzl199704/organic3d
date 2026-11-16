"""Patient-level bootstrap CIs for the stage-A ladder, and paired comparisons between runs (ladder v2).

Reads {work_root}/eval/<run>/<dataset>_scores.npz written by evaluate_stageA (per-volume scores of every condition).

    python scripts/metrics/ladder_stats.py --env configs/env/local.yaml --runs enc3_routing,enc4_organic3d --datasets merlin,ctrate
    python scripts/metrics/ladder_stats.py ... --compare enc4_organic3d:enc3_routing      # paired differences + BH over findings

Per run and dataset: macro zero-shot AUROC and the texture erasure effect (target minus control, fat fill), each with a
95% percentile CI from resampling patients (B = --boot). Per comparison: per-finding difference in zero-shot AUROC and in
erasure effect, its CI, a two-sided bootstrap p-value and Benjamini-Hochberg q-values across findings.
Writes {work_root}/eval/stats_<dataset>.csv and stats_compare_<a>_vs_<b>_<dataset>.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from organic.config import load_config
from organic.evaluate_stageA import EVAL_ORGANS, cond_name, organ_ids
from organic.train_stageA import auroc
from organic.vocab import CTRATE_FINDINGS, FINDING_NAMES, MERLIN_FINDINGS, SIZE_FINDINGS, control_organ_ids, finding_organ_ids

FAM = {'merlin': MERLIN_FINDINGS, 'ctrate': CTRATE_FINDINGS}


def load(W, run, d):
    z = np.load(os.path.join(W, 'eval', run, f'{d}_scores.npz'), allow_pickle=True)
    order = np.argsort(z['index'])                              # align runs by volume index
    return {k: z[k][order] for k in z.files if k not in ('lat',) and len(z[k]) == len(z['index'])}


def finding_stats(z, d, f, rows):
    """Zero-shot AUROC and texture erasure effect of finding f on the volume subset `rows`."""
    j = FINDING_NAMES.index(f)
    y = z['labels'][rows, j]
    if np.nansum(y) < 5 or np.nansum(1 - y) < 5:
        return np.nan, np.nan
    d0 = z['delta__clean'][rows, j]
    a0 = auroc(y, d0)
    tgt, ctrl = set(finding_organ_ids(f)), set(control_organ_ids(f))
    t, c = [], []
    for o in EVAL_ORGANS[d]:
        key = f'delta__{cond_name(o)}@fat'
        if key not in z:
            continue
        keep = z[f'nvox__{cond_name(o)}@fat'][rows] > 0
        yk = np.where(keep, y, np.nan)
        dz = auroc(yk, z[key][rows, j]) - auroc(yk, d0)
        ids = set(organ_ids(o))
        if tgt and ids & tgt:
            t.append(dz)
        elif ids <= ctrl:
            c.append(dz)
    eff = (np.nanmean(t) - np.nanmean(c)) if t and f not in SIZE_FINDINGS else np.nan
    return a0, eff


def boot_rows(patients, B, rng):
    """B resamples of volume rows, drawn by patient (all volumes of a drawn patient)."""
    up, inv = np.unique(patients, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(up))]
    for _ in range(B):
        pick = rng.integers(0, len(up), len(up))
        yield np.concatenate([members[k] for k in pick])


def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = ~np.isnan(p)
    if ok.any():
        ps = p[ok]; o = np.argsort(ps); r = ps[o] * len(ps) / (np.arange(len(ps)) + 1)
        q[np.where(ok)[0][o]] = np.minimum.accumulate(r[::-1])[::-1].clip(max=1)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--runs', default='')
    ap.add_argument('--datasets', default='merlin,ctrate')
    ap.add_argument('--compare', default='', help='a:b[,c:d]  paired comparisons on the same test volumes')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--work_root', default=None, help='read eval/ from here instead of the env work_root (e.g. results pulled from a cluster)')
    a = ap.parse_args()
    W = a.work_root or load_config(os.path.join(os.path.dirname(__file__), '../../configs/stageA.yaml'), env=a.env).paths.work_root
    rng = np.random.default_rng(0)
    for d in a.datasets.split(','):
        fs = FAM[d]
        rows = []
        for run in [r for r in a.runs.split(',') if r]:
            z = load(W, run, d); n = len(z['index'])
            pt = [finding_stats(z, d, f, np.arange(n)) for f in fs]
            bs = []
            for rr in boot_rows(z['patient'], a.boot, rng):
                s = [finding_stats(z, d, f, rr) for f in fs]
                bs.append((np.nanmean([x[0] for x in s]), np.nanmean([x[1] for x in s])))
            bs = np.array(bs)
            rows.append(dict(run=run, n=n, zero_shot=np.nanmean([x[0] for x in pt]),
                             zs_lo=np.percentile(bs[:, 0], 2.5), zs_hi=np.percentile(bs[:, 0], 97.5),
                             effect_texture=np.nanmean([x[1] for x in pt]),
                             eff_lo=np.percentile(bs[:, 1], 2.5), eff_hi=np.percentile(bs[:, 1], 97.5)))
        if rows:
            t = pd.DataFrame(rows).set_index('run')
            t.to_csv(os.path.join(W, 'eval', f'stats_{d}.csv'))
            print(f'== {d}\n' + t.round(3).to_string())
        for pair in [p for p in a.compare.split(',') if p]:
            ra, rb = pair.split(':')
            za, zb = load(W, ra, d), load(W, rb, d)
            assert np.array_equal(za['index'], zb['index']), f'{ra} and {rb} were not evaluated on the same volumes'
            n = len(za['index'])
            pa = [finding_stats(za, d, f, np.arange(n)) for f in fs]
            pb = [finding_stats(zb, d, f, np.arange(n)) for f in fs]
            diffs = []
            for rr in boot_rows(za['patient'], a.boot, rng):
                sa = [finding_stats(za, d, f, rr) for f in fs]; sb = [finding_stats(zb, d, f, rr) for f in fs]
                diffs.append([[y[0] - x[0], y[1] - x[1]] for x, y in zip(sa, sb)])
            D = np.array(diffs)                                              # (B, F, 2)
            out = []
            for k, f in enumerate(fs):
                for m, name in enumerate(['zero_shot', 'effect']):
                    v = D[:, k, m]; v = v[~np.isnan(v)]
                    est = pb[k][m] - pa[k][m]
                    p = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())) if len(v) else np.nan
                    out.append(dict(finding=f, metric=name, diff=est, lo=np.percentile(v, 2.5) if len(v) else np.nan,
                                    hi=np.percentile(v, 97.5) if len(v) else np.nan, p=p))
            o = pd.DataFrame(out)
            for name in ['zero_shot', 'effect']:
                sel = o.metric == name
                o.loc[sel, 'q'] = bh(o.loc[sel, 'p'])
            o.to_csv(os.path.join(W, 'eval', f'stats_compare_{ra}_vs_{rb}_{d}.csv'), index=False)
            for name in ['zero_shot', 'effect']:
                s = o[o.metric == name].dropna(subset=['diff'])
                print(f'== {d} {rb} minus {ra}, {name}: mean {s["diff"].mean():+.3f}; '
                      f'q<0.05 better in {int(((s.q < 0.05) & (s["diff"] > 0 if name == "zero_shot" else s["diff"] < 0)).sum())}, '
                      f'worse in {int(((s.q < 0.05) & (s["diff"] < 0 if name == "zero_shot" else s["diff"] > 0)).sum())} of {len(s)} findings')


if __name__ == '__main__':
    main()
