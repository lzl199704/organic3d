"""Report-level Track A: answer-composed structured reports, the same rule for every VLM (docs/metrics.md, "Report level").

A model's report for a scan states every finding whose answer score (P(Yes), or the logit difference when saved) clears
that finding's threshold. Thresholds are cross-fitted identically for every model: 5 folds of patients, on each fold the
F1-maximising threshold from the other four. Metrics per model and dataset (1,000 audit scans):
  f1            clinical macro F1 of the report vs the labels (findings with >= 1 positive)
  f1_wrong      F1 of another patient's report scored against this patient's labels (a report written from priors
                keeps its F1 here)
  attributable  f1 - f1_wrong: the part of the report's correctness that comes from this scan
  retraction    of findings correctly stated on the clean scan, the share withdrawn after erasing the finding's own
                organ minus the share withdrawn after erasing a control organ (evaluate_stageB target / control rule;
                scans where the organ was not found are skipped); non-size findings only (= OEH answer test)
Patient-level bootstrap 95% CIs (thresholds held fixed).

Sources: harness VLMs {W}/extaudit/out_vlm/<name>/{d}_shard*.npz; audits that save per-scan scores
{W or M}/evalB/<name>/{d}_scores.npz (NV-Reason-CT, MedGemma); CT-CHAT $CTCHAT_SCORES (glob, default
{W}/evalB/CT-CHAT/scores_shard*.npz; per-scan P(Yes) shards of a CT-CHAT scorer run in CT-CHAT's own environment).
W = work root, M = cluster work root (scripts/workroot.py: $ORGANIC3D_WORK_ROOT / $ORGANIC3D_CLUSTER_WORK_ROOT).

    python scripts/report_tracks.py [--boot 500]
Writes {W}/eval/report_trackA.csv.

--set full (2026-10-25): the full test sets = audit run + run on the remaining scans, merged by volume
(scripts/vlm_full.py; labels from <dataset>_set.parquet), for the Table 2 models (HARNESS_FULL); writes
report_trackA_full.csv. Default unchanged.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from organic.ctrate_v2 import no_chest_volumes
from organic.evaluate_stageA import organ_ids
from organic.ladder import clinical_f1
from organic.vocab import CTRATE_FINDINGS, FINDING_NAMES, MERLIN_FINDINGS, SIZE_FINDINGS, control_organ_ids, finding_organ_ids

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workroot import cluster_work_root, work_root                              # noqa: E402

W = work_root()
CTCHAT_SCORES = os.environ.get('CTCHAT_SCORES', f'{W}/evalB/CT-CHAT/scores_shard*.npz')
M = cluster_work_root()
FAM = {'ctrate': CTRATE_FINDINGS, 'merlin': MERLIN_FINDINGS}
HARNESS = ['vlm4_organic3d', 'vlm3_dependence', 'vlm1_plain', 'Hulu-Med-7B', 'OmniCT-7B', 'RadFM', 'M3D-LaMed', 'NV-Reason-CT_h']
AUDITS = {'NV-Reason-CT': 'NV-Reason-CT', 'MedGemma-1.5-4B': 'MedGemma-1.5-4B_scores'}
HARNESS_FULL = ['vlm4_organic3d', 'vlm1_plain', 'Hulu-Med-7B', 'OmniCT-7B', 'MedGemma-1.5-4B',
                'M3D-LaMed', 'RadFM', 'CT-CHAT', 'NV-Reason-CT_h']      # --set full: the Table 2 rows, all from the harness


def patient(v, d):
    return re.sub(r'_[a-z]+_\d+\.nii\.gz$', '', v) if d == 'ctrate' else v


def from_harness(name, d, full=False):
    if full:
        import vlm_full
        cat = vlm_full.load(name, d)
        if cat is None:
            return None
        finds = [str(x) for x in cat['findings']]
        key = 'D__' if any(k.startswith('D__') for k in cat) else 'P__'
        cat = {k: v for k, v in cat.items() if k.startswith((key, 'nvox__', 'index'))}
    else:
        fs = sorted(glob.glob(f'{W}/extaudit/out_vlm/{name}/{d}_shard*.npz'))
        if not fs:
            return None
        Z = [dict(np.load(f, allow_pickle=True)) for f in fs]
        finds = list(Z[0]['findings'])
        key = 'D__' if any(k.startswith('D__') for k in Z[0]) else 'P__'
        cat = {k: np.concatenate([z[k] for z in Z]) for k in Z[0] if k.startswith((key, 'nvox__', 'index'))}
        o = np.argsort(cat['index']); cat = {k: v[o] for k, v in cat.items()}
    st = pd.read_parquet(f'{W}/extaudit/{d}_set.parquet' if full else
                         f'{W}/extaudit/{d}_audit1000_set.parquet').set_index('index').loc[cat['index']]
    P = {k[3:].split('@')[0]: v.astype(float) for k, v in cat.items() if k.startswith(key)}
    nv = {k[6:].split('@')[0]: v for k, v in cat.items() if k.startswith('nvox__')}
    return P, nv, finds, st[[f'label_{f}' for f in FINDING_NAMES]].to_numpy(np.float32), st.volume.to_numpy()


def from_audit(dirname, d):
    for base in (W, M):
        p = f'{base}/evalB/{dirname}/{d}_scores.npz'
        if os.path.exists(p):
            z = dict(np.load(p, allow_pickle=True))
            P = {k: v.astype(float) for k, v in z.items() if k == 'clean' or k.startswith('rem_')}
            nv = {k[5:]: v for k, v in z.items() if k.startswith('nvox_')}
            return P, nv, FAM[d], z['labels'].astype(np.float32), z['volume'].astype(str)
    return None


def from_ctchat(d):
    fs = sorted(glob.glob(CTCHAT_SCORES))
    if d != 'ctrate' or not fs:
        return None
    Z = [dict(np.load(f)) for f in fs]
    cat = {k: np.concatenate([z[k] for z in Z]) for k in Z[0]}
    o = np.argsort(cat['index']); cat = {k: v[o] for k, v in cat.items()}
    st = pd.read_parquet(f'{W}/extaudit/ctrate_audit1000_set.parquet').set_index('index').loc[cat['index']]
    P = {k[3:]: v.astype(float) for k, v in cat.items() if k.startswith('P__')}
    nv = {k[6:]: v for k, v in cat.items() if k.startswith('nvox__')}
    return P, nv, CTRATE_FINDINGS, st[[f'label_{f}' for f in FINDING_NAMES]].to_numpy(np.float32), st.volume.to_numpy()


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
    if d == 'ctrate':                           # CT-RATE v2: the brain series are not scored (organic/ctrate_v2.py)
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
    ap.add_argument('--set', default='audit1000', choices=['audit1000', 'full'])
    ap.add_argument('--names', default=None, help='--set full: comma-separated harness names (default: the paper\'s VLMs)')
    a = ap.parse_args()
    full = a.set == 'full'
    names = a.names.split(',') if a.names else HARNESS_FULL
    rows = []
    for d in ('ctrate', 'merlin'):
        if full:
            srcs = {n: from_harness(n, d, full=True) for n in names}
        else:
            srcs = {n: from_harness(n, d) for n in HARNESS}
            srcs.update({n: from_audit(x, d) for n, x in AUDITS.items()})
            srcs['CT-CHAT'] = from_ctchat(d) if d == 'ctrate' else from_harness('CT-CHAT', d)   # Merlin: harness 2026-10-24
        for name, s in srcs.items():
            if s is None:
                print(f'{name} {d}: no per-scan scores yet'); continue
            rows.append(dict(model=name, dataset=d, **evaluate(s, d, a.boot, np.random.default_rng(0))))
    t = pd.DataFrame(rows)
    t.to_csv(f'{W}/eval/report_trackA_full.csv' if full else f'{W}/eval/report_trackA.csv', index=False)
    pd.set_option('display.width', 220)
    for d, g in t.groupby('dataset', sort=False):
        print('==', d)
        print(g[['model', 'n', 'f1', 'f1_lo', 'f1_hi', 'f1_wrong', 'attributable', 'attributable_lo', 'attributable_hi',
                 'retraction', 'retraction_lo', 'retraction_hi']].sort_values('f1', ascending=False).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
