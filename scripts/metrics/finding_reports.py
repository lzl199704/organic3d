"""Finding-based reports: answer-composed structured reports, the same rule for every VLM (docs/metrics.md, "Report level").

A model's report for a scan states every finding whose answer score (P(Yes), or the logit difference when saved) clears
that finding's threshold. Thresholds are cross-fitted identically for every model: 5 folds of patients, on each fold the
F1-maximising threshold from the other four. Metrics per model and dataset (full test sets: CT-RATE v2 3,002 scans,
Merlin 5,109):
  f1            clinical macro F1 of the report vs the labels (findings with >= 1 positive)
  f1_wrong      F1 of another patient's report scored against this patient's labels (a report written from priors
                keeps its F1 here)
  attributable  f1 - f1_wrong (IA-F1): the part of the report's correctness that comes from this scan
  retraction    of findings correctly stated on the clean scan, the share withdrawn after erasing the finding's own
                organ minus the share withdrawn after erasing a control organ (organic.vlm.evaluate target / control rule;
                scans where the organ was not found are skipped); non-size findings only (= OEH answer test)
Patient-level bootstrap 95% CIs (thresholds held fixed).

Sources: harness shards {W}/extaudit/out_vlm/<name>/<dataset>_shard*.npz of one run on the whole test set
(scripts/metrics/vlm_scores.py; labels from <dataset>_set.parquet). W = work root (organic/workroot.py).

    python scripts/metrics/finding_reports.py [--names vlm4_organic3d,...] [--boot 500]
Writes {W}/eval/finding_reports.csv.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from organic.data.ctrate_v2 import no_chest_volumes
from organic.encoder.evaluate import organ_ids
from organic.metrics import clinical_f1
from organic.vocab import FINDING_NAMES, SIZE_FINDINGS, control_organ_ids, finding_organ_ids

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlm_scores                                                   # noqa: E402
from organic.workroot import work_root                              # noqa: E402

W = work_root()
VLMS = ['vlm4_organic3d', 'vlm1_plain', 'Hulu-Med-7B', 'OmniCT-7B', 'MedGemma-1.5-4B',
        'M3D-LaMed', 'RadFM', 'CT-CHAT', 'NV-Reason-CT_h']           # the Table 2 rows


def patient(v, d):
    return re.sub(r'_[a-z]+_\d+\.nii\.gz$', '', v) if d == 'ctrate' else v


def from_harness(name, d, partial=False):
    cat = vlm_scores.load(name, d, partial)
    if cat is None:
        return None
    finds = [str(x) for x in cat['findings']]
    key = 'D__' if any(k.startswith('D__') for k in cat) else 'P__'
    cat = {k: v for k, v in cat.items() if k.startswith((key, 'nvox__', 'index'))}
    st = pd.read_parquet(f'{W}/extaudit/{d}_set.parquet').set_index('index').loc[cat['index']]
    P = {k[3:].split('@')[0]: v.astype(float) for k, v in cat.items() if k.startswith(key)}
    nv = {k[6:].split('@')[0]: v for k, v in cat.items() if k.startswith('nvox__')}
    return P, nv, finds, st[[f'label_{f}' for f in FINDING_NAMES]].to_numpy(np.float32), st.volume.to_numpy()


def f1max_thr(p, y):
    c = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 49)))
    tp = lambda t: ((p >= t) & (y == 1)).sum()
    f = [2 * tp(t) / max(1, 2 * tp(t) + ((p >= t) & (y == 0)).sum() + ((p < t) & (y == 1)).sum()) for t in c]
    return c[int(np.argmax(f))]


def crossfit(P, Y, pats, k=5, seed=0):
    """Folds of patients (reconstructions of one CT-RATE patient never straddle a fit and its held-out fold)."""
    up, inv = np.unique(pats, return_inverse=True)
    fold = np.random.default_rng(seed).integers(0, k, len(up))[inv]
    T = np.full(P.shape, np.nan)
    for j in range(P.shape[1]):
        ok = ~np.isnan(Y[:, j])
        for q in range(k):
            tr = ok & (fold != q)
            if Y[tr, j].sum() > 0:
                T[fold == q, j] = f1max_thr(P[tr, j], Y[tr, j])
    return T


def partner(pats):
    """Wrong-patient pairing: each row is scored with the report of a row half the list away in patient order, moved on
    until the partner is a different patient."""
    order = np.argsort(pats, kind='stable')
    n = len(order)
    w = np.empty(n, int)
    for r, i in enumerate(order):
        s = n // 2
        while pats[order[(r + s) % n]] == pats[i] and s < n - 1:
            s += 1
        w[i] = order[(r + s) % n]
    return w


def scores(pred, Y, cols, erased, w):
    """pred (n, F) 0/1/NaN; erased: list of (finding col j, is_target, retracted (n,) bool, valid (n,) bool)."""
    full = np.full((len(pred), len(FINDING_NAMES)), np.nan); full[:, cols] = pred
    fullw = np.full_like(full, np.nan); fullw[:, cols] = pred[w]
    Yf = np.full_like(full, np.nan); Yf[:, cols] = Y
    f1, f1w = clinical_f1(full, Yf, cols), clinical_f1(fullw, Yf, cols)
    tg = [r[ok].mean() for j, t, r, ok in erased if t and ok.sum() >= 5]
    ct = [r[ok].mean() for j, t, r, ok in erased if not t and ok.sum() >= 5]
    ret = (np.mean(tg) - np.mean(ct)) if tg and ct else np.nan
    return f1, f1w, f1 - f1w, ret


def evaluate(src, d, boot, rng):
    P, nv, finds, L, vol = src
    if d == 'ctrate':                           # CT-RATE v2: the brain series are not scored (organic/data/ctrate_v2.py)
        k = ~np.isin(vol, list(no_chest_volumes(W)))
        P, nv = {c: v[k] for c, v in P.items()}, {c: v[k] for c, v in nv.items()}
        L, vol = L[k], vol[k]
    cols = [FINDING_NAMES.index(f) for f in finds]
    Y = L[:, cols]
    pats = np.array([patient(v, d) for v in vol]); up = np.unique(pats)
    T = crossfit(P['clean'], Y, pats)
    pred = (P['clean'] >= T).astype(float); pred[np.isnan(T)] = np.nan
    n = len(pred)
    w = partner(pats)
    asserted = (pred == 1) & (Y == 1)
    erased = []
    for c, Pc in P.items():
        if not c.startswith('rem_'):
            continue
        ids = set(organ_ids(c[4:].split('+') if '+' in c else c[4:]))
        present = nv.get(c, np.ones(n)) > 0
        for jj, f in enumerate(finds):
            if f in SIZE_FINDINGS:                  # OEH answer test: non-size findings only, as the score test
                continue                            # (erasure keeps the outline, so size stays readable)
            tgt, ctl = set(finding_organ_ids(f)), set(control_organ_ids(f))
            is_t = bool(tgt and ids & tgt)
            if not is_t and not ids <= ctl:
                continue
            erased.append((jj, is_t, Pc[:, jj] < T[:, jj], asserted[:, jj] & present & ~np.isnan(T[:, jj])))
    est = scores(pred, Y, cols, erased, w)
    rows_of = {p: np.where(pats == p)[0] for p in up}
    bs = []
    for _ in range(boot):
        idx = np.concatenate([rows_of[p] for p in rng.choice(up, len(up))])
        er = [(j, t, r[idx], ok[idx]) for j, t, r, ok in erased]
        bs.append(scores(pred[idx], Y[idx], cols, er, partner(pats[idx])))
    bs = np.array(bs)
    lo, hi = np.nanquantile(bs, 0.025, axis=0), np.nanquantile(bs, 0.975, axis=0)
    out = dict(n=n)
    for i, k in enumerate(('f1', 'f1_wrong', 'attributable', 'retraction')):
        out[k], out[f'{k}_lo'], out[f'{k}_hi'] = est[i], lo[i], hi[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--names', default=None, help='comma-separated harness names (default: the paper\'s VLMs)')
    a = ap.parse_args()
    names = a.names.split(',') if a.names else VLMS
    rows = []
    for d in ('ctrate', 'merlin'):
        for name in names:
            s = from_harness(name, d)
            if s is None:
                print(f'{name} {d}: no per-scan scores yet'); continue
            rows.append(dict(model=name, dataset=d, **evaluate(s, d, a.boot, np.random.default_rng(0))))
    t = pd.DataFrame(rows)
    t.to_csv(f'{W}/eval/finding_reports.csv', index=False)
    pd.set_option('display.width', 220)
    for d, g in t.groupby('dataset', sort=False):
        print('==', d)
        print(g[['model', 'n', 'f1', 'f1_lo', 'f1_hi', 'f1_wrong', 'attributable', 'attributable_lo', 'attributable_hi',
                 'retraction', 'retraction_lo', 'retraction_hi']].sort_values('f1', ascending=False).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
