"""The paper's two main VLM tables (2026-10-21), on the same 1,000 audit scans per dataset (Track B: the first 200).

Table 1, finding classification: each finding asked as a yes/no question on the clean scan.
  AUROC                 threshold-free, from the answer score (P(Yes) or logit(Yes) - logit(No))
  Acc / BACC / F1       of the answer the model actually gives: Yes when P(Yes) >= 0.5 (logit difference >= 0)
  Macro over findings with >= 10 positives and >= 10 negatives (the ladder rule, evaluate_stageB.metrics).
Table 2, reports:
  Track A, composed     the report states every finding whose score clears its cross-fitted F1-max threshold
                        (scripts/report_tracks.py: 5 patient folds, threshold fit on the other four)
  Track B, generated    free-text report (each model's own prompt, greedy), labels read by one extractor per dataset
                        (CT-RATE: official RadBERT classifier; Merlin: frozen rule labeller v4; scripts/trackB_metrics.py)
  Acc / BACC / precision / recall / F1, macro over findings with >= 1 positive (organic.ladder.clinical_f1 rule);
  F1_wrong = the report of another patient scored against this one; attributable F1 = F1 - F1_wrong.
Patient-level bootstrap 95% CIs; paired bootstrap of ours (--ref) minus each model on the same scans.

Score sources: harness shards {W}/extaudit/out_vlm/<name>/ (ours, Hulu-Med, OmniCT, RadFM, M3D-LaMed, NV-Reason-CT_h);
audit score files (NV-Reason-CT on the cluster, MedGemma); CT-CHAT $CTCHAT_SCORES on CT-RATE, its harness run on Merlin. All re-ordered to the
harness scan order by volume.

    python scripts/main_tables.py [--boot 1000] [--skip_trackB]
Writes {W}/eval/table_vlm_classification.csv, table_vlm_reports.csv, table_vlm_paired.csv.

--set full (2026-10-25): the full test sets (CT-RATE v2 3,002 after the non-chest drop, Merlin 5,109) = audit run +
run on the remaining scans merged by volume (scripts/vlm_full.py), the Table 2 models (report_tracks.HARNESS_FULL),
no Track B; --ref defaults to vlm4_organic3d; writes table_vlm_classification_full.csv, table_vlm_reports_full.csv,
table_vlm_paired_full.csv. Default unchanged.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from organic.vocab import FINDING_NAMES
from organic.ctrate_v2 import no_chest_volumes
from report_tracks import HARNESS_FULL, W, crossfit, from_audit, from_ctchat, from_harness, partner, patient

HARNESS = ['vlm4_organic3d', 'vlm3_dependence', 'vlm1_plain', 'Hulu-Med-7B', 'OmniCT-7B', 'RadFM', 'M3D-LaMed', 'NV-Reason-CT_h']
AUDITS = {'NV-Reason-CT': 'NV-Reason-CT', 'MedGemma-1.5-4B': 'MedGemma-1.5-4B_scores'}
TRACKB = ['vlm4_organic3d', 'vlm3_dependence', 'vlm1_plain', 'NV-Reason-CT', 'MedGemma-1.5-4B', 'Hulu-Med-7B', 'OmniCT-7B',
          'CT-CHAT', 'RadFM', 'M3D-LaMed']


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
    ap.add_argument('--ref', default=None, help='default vlm3_dependence (audit1000) / vlm4_organic3d (full)')
    ap.add_argument('--skip_trackB', action='store_true')
    ap.add_argument('--set', default='audit1000', choices=['audit1000', 'full'])
    a = ap.parse_args()
    full = a.set == 'full'
    a.ref = a.ref or ('vlm4_organic3d' if full else 'vlm3_dependence')
    if full:
        a.skip_trackB = True
    t1, t2, pr = [], [], []
    for d in ('ctrate', 'merlin'):
        st = pd.read_parquet(f'{W}/extaudit/{d}_set.parquet' if full else
                             f'{W}/extaudit/{d}_audit1000_set.parquet').sort_values('index')
        if d == 'ctrate':                       # CT-RATE v2: the brain series are not scored (organic/ctrate_v2.py)
            st = st[~st.volume.isin(no_chest_volumes(W))]
        vols = st.volume.to_numpy()
        pats = np.array([patient(v, d) for v in vols])
        if full:
            srcs = {n: from_harness(n, d, full=True) for n in HARNESS_FULL}
        else:
            srcs = {n: from_harness(n, d) for n in HARNESS}
            srcs.update({n: from_audit(x, d) for n, x in AUDITS.items()})
            srcs['CT-CHAT'] = from_ctchat(d) if d == 'ctrate' else from_harness('CT-CHAT', d)   # Merlin: harness 2026-10-24
        S = {}
        for name, s in srcs.items():
            if s is None:
                print(f'{name} {d}: no per-scan scores'); continue
            al = aligned(s, vols)
            if al is None:
                print(f'{name} {d}: scans differ from the {a.set} set, skipped'); continue
            S[name] = al
        ref_finds = S[a.ref][1]
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
            t2.append(dict(model=name, dataset=d, track='A_composed', n=len(P), **{k: v[0] for k, v in rp.items()},
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
        if not a.skip_trackB:
            import trackB_metrics as tb
            s = tb.setup(d)
            pb, Yb, cb = s['pats'], s['Y'], s['cols']
            for name in TRACKB:
                r = tb.load_local(name, d) or tb.load_cluster({'NV-Reason-CT': 'NV-Reason-CT'}.get(name, ''), d)
                if r is None or len(r) < tb.N:
                    print(f'track B {name} {d}: missing'); continue
                pred = tb.extract([r[i] or '' for i in range(tb.N)], d)[:, cb]
                Yc = Yb[:, cb]
                w = np.roll(np.arange(tb.N), tb.N // 2)

                def gen(i):
                    r_ = macro(pred[i], Yc[i])
                    r_['f1_wrong'] = macro(pred[i][np.roll(np.arange(len(i)), len(i) // 2)], Yc[i])['f1']
                    r_['attributable'] = r_['f1'] - r_['f1_wrong']
                    return r_
                rp = boot_ci(gen, pb, a.boot, np.random.default_rng(0))
                t2.append(dict(model=name, dataset=d, track='B_generated', n=tb.N, **{k: v[0] for k, v in rp.items()},
                               **{f'{k}_lo': v[1] for k, v in rp.items()}, **{f'{k}_hi': v[2] for k, v in rp.items()}))
    out = f'{W}/eval'
    sfx = '_full' if full else ''
    t1, t2, pr = pd.DataFrame(t1), pd.DataFrame(t2), pd.DataFrame(pr)
    t1.to_csv(f'{out}/table_vlm_classification{sfx}.csv', index=False)
    t2.to_csv(f'{out}/table_vlm_reports{sfx}.csv', index=False)
    pr.to_csv(f'{out}/table_vlm_paired{sfx}.csv', index=False)
    pd.set_option('display.width', 250)
    print(t1[['model', 'dataset', 'n', 'auroc', 'acc', 'bacc', 'f1', 'yes_rate']].round(3).to_string(index=False))
    print(t2[['model', 'dataset', 'track', 'n', 'acc', 'bacc', 'prec', 'rec', 'f1', 'f1_wrong', 'attributable']]
          .round(3).to_string(index=False))
    print(pr.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
