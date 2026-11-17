"""The paper's two main VLM tables on the full test sets (CT-RATE v2 3,002 scans, Merlin 5,109).

Table 2, finding classification: each finding asked as a yes/no question on the clean scan.
  AUROC                 threshold-free, from the answer score (P(Yes) or logit(Yes) - logit(No))
  Acc / BACC / F1       of the answer the model actually gives: Yes when P(Yes) >= 0.5 (logit difference >= 0)
  Macro over findings with >= 10 positives and >= 10 negatives (the ladder rule, organic.vlm.evaluate.metrics).
Table 3, finding-based reports: the report states every finding whose score clears its cross-fitted F1-max threshold
  (scripts/metrics/finding_reports.py: 5 patient folds, threshold fit on the other four).
  Acc / BACC / precision / recall / F1, macro over findings with >= 1 positive (organic.metrics.clinical_f1 rule);
  F1_wrong = the report of another patient scored against this one; attributable F1 (IA-F1) = F1 - F1_wrong.
Patient-level bootstrap 95% CIs; paired bootstrap of the reference model (--ref) minus each model on the same scans.
Free-text report metrics are computed separately (scripts/metrics/report_metrics_std.py).

Score source: harness shards {W}/extaudit/out_vlm/<name>/<dataset>_shard*.npz of one run on the whole test set
(scripts/metrics/vlm_scores.py), re-ordered to the test-set order by volume.

    python scripts/metrics/main_tables.py [--names vlm4_organic3d,...] [--ref vlm4_organic3d] [--boot 1000]
Writes {W}/eval/table_vlm_classification.csv, table_vlm_reports.csv, table_vlm_paired.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from organic.vocab import FINDING_NAMES
from organic.data.ctrate_v2 import no_chest_volumes
from finding_reports import VLMS, W, crossfit, from_harness, partner, patient


def auroc(y, s):
    y = np.asarray(y, bool)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def confusion(p, y):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum()
    fn = ((p == 0) & (y == 1)).sum(); tn = ((p == 0) & (y == 0)).sum()
    sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    return dict(acc=(tp + tn) / max(1, tp + tn + fp + fn), bacc=(sens + spec) / 2, prec=tp / max(1, tp + fp),
                rec=sens, f1=2 * tp / max(1, 2 * tp + fp + fn))


def macro(pred, Y, score=None, min_pos=1, min_neg=0):
    """pred, Y (n, F) with NaN = unknown; score (n, F) for AUROC. Macro over eligible findings."""
    out = {}
    for j in range(Y.shape[1]):
        ok = ~np.isnan(Y[:, j]) & ~np.isnan(pred[:, j])
        y, p = Y[ok, j], pred[ok, j]
        if y.sum() < min_pos or (1 - y).sum() < min_neg:
            continue
        c = confusion(p, y)
        if score is not None:
            c['auroc'] = auroc(y, score[ok, j])
        for k, v in c.items():
            out.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}


def aligned(src, vols):
    """Re-order an (P, nvox, findings, labels, volume) source to the harness volume order."""
    P, nv, finds, L, vol = src
    pos = {v: i for i, v in enumerate(vol)}
    if not all(v in pos for v in vols):
        return None
    o = np.array([pos[v] for v in vols])
    return P['clean'][o], list(finds), L[o]


def answer(S):
    """The model's own yes/no answer from its score (probabilities in [0, 1], else logit differences)."""
    thr = 0.5 if np.nanmin(S) >= 0 and np.nanmax(S) <= 1 else 0.0
    return (S >= thr).astype(float)


def boot_ci(fn, pats, B, rng):
    up = np.unique(pats)
    rows_of = {p: np.where(pats == p)[0] for p in up}
    est = fn(np.arange(len(pats)))
    bs = [fn(np.concatenate([rows_of[p] for p in rng.choice(up, len(up))])) for _ in range(B)]
    keys = est.keys()
    return {k: (est[k], *np.nanquantile([b.get(k, np.nan) for b in bs], [0.025, 0.975])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--names', default=None, help='comma-separated harness names (default: the paper\'s VLMs)')
    ap.add_argument('--ref', default='vlm4_organic3d', help='reference model of the paired differences')
    a = ap.parse_args()
    names = a.names.split(',') if a.names else VLMS
    t1, t2, pr = [], [], []
    for d in ('ctrate', 'merlin'):
        st = pd.read_parquet(f'{W}/extaudit/{d}_set.parquet').sort_values('index')
        if d == 'ctrate':                       # CT-RATE v2: the brain series are not scored (organic/data/ctrate_v2.py)
            st = st[~st.volume.isin(no_chest_volumes(W))]
        vols = st.volume.to_numpy()
        pats = np.array([patient(v, d) for v in vols])
        srcs = {n: from_harness(n, d) for n in names}
        S = {}
        for name, s in srcs.items():
            if s is None:
                print(f'{name} {d}: no per-scan scores'); continue
            al = aligned(s, vols)
            if al is None:
                print(f'{name} {d}: scans differ from the test set, skipped'); continue
            S[name] = al
        if not S:
            continue
        ref_finds = S[a.ref][1] if a.ref in S else next(iter(S.values()))[1]
        cols = [FINDING_NAMES.index(f) for f in ref_finds]
        for name, (P, finds, L) in S.items():
            assert finds == ref_finds, (name, finds)
            Y = L[:, cols]
            A = answer(P)
            T = crossfit(P, Y, pats)
            C = (P >= T).astype(float); C[np.isnan(T)] = np.nan
            w = partner(pats)
            cls = boot_ci(lambda i: macro(A[i], Y[i], P[i], 10, 10), pats, a.boot, np.random.default_rng(0))
            t1.append(dict(model=name, dataset=d, n=len(P), yes_rate=float(np.nanmean(A)),
                           **{k: v[0] for k, v in cls.items()},
                           **{f'{k}_lo': v[1] for k, v in cls.items()}, **{f'{k}_hi': v[2] for k, v in cls.items()}))

            def rep(i):
                r = macro(C[i], Y[i])
                r['f1_wrong'] = macro(C[i][partner(pats[i])], Y[i])['f1']
                r['attributable'] = r['f1'] - r['f1_wrong']
                return r
            rp = boot_ci(rep, pats, a.boot, np.random.default_rng(0))
            t2.append(dict(model=name, dataset=d, n=len(P), **{k: v[0] for k, v in rp.items()},
                           **{f'{k}_lo': v[1] for k, v in rp.items()}, **{f'{k}_hi': v[2] for k, v in rp.items()}))
            S[name] = (P, A, C, Y)
        if a.ref in S:
            P0, A0, C0, Y = S[a.ref]
            for name, (P, A, C, _) in S.items():
                if name == a.ref:
                    continue

                def diff(i):
                    x = macro(A0[i], Y[i], P0[i], 10, 10); y = macro(A[i], Y[i], P[i], 10, 10)
                    u = macro(C0[i], Y[i]); v = macro(C[i], Y[i])
                    return dict(auroc=x['auroc'] - y['auroc'], bacc=x['bacc'] - y['bacc'], f1=x['f1'] - y['f1'],
                                report_f1=u['f1'] - v['f1'], report_bacc=u['bacc'] - v['bacc'])
                for k, (e, lo, hi) in boot_ci(diff, pats, a.boot, np.random.default_rng(1)).items():
                    pr.append(dict(dataset=d, ref=a.ref, model=name, metric=k, diff=e, lo=lo, hi=hi))
    out = f'{W}/eval'
    t1, t2, pr = pd.DataFrame(t1), pd.DataFrame(t2), pd.DataFrame(pr)
    t1.to_csv(f'{out}/table_vlm_classification.csv', index=False)
    t2.to_csv(f'{out}/table_vlm_reports.csv', index=False)
    pr.to_csv(f'{out}/table_vlm_paired.csv', index=False)
    pd.set_option('display.width', 250)
    print(t1[['model', 'dataset', 'n', 'auroc', 'acc', 'bacc', 'f1', 'yes_rate']].round(3).to_string(index=False))
    print(t2[['model', 'dataset', 'n', 'acc', 'bacc', 'prec', 'rec', 'f1', 'f1_wrong', 'attributable']]
          .round(3).to_string(index=False))
    print(pr.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
