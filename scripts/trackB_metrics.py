"""Report-level Track B (docs/metrics.md, "Report level"): free-text reports of every VLM on the same 200 scans per
dataset (the first 200 of the audit sample = evalB volume_index 0..199 of the stage-B evaluator), each model's own report prompt,
greedy; scored with ONE extractor per dataset for every model:
  CT-RATE  the CT-RATE authors' RadBERT text classifier (scripts/trackB_ctrate_classifier.py), on the generated report
           and on the reference report (Findings + Impression) of the same scan. Main truth = the classifier's labels
           of the reference report; f1_released = vs the released multi_abnormality_labels (identical on these 200).
  Merlin   our frozen rule labeller v4 (scripts/label_rules.py, rule layer only, via
           organic.ladder.merlin_report_labels; hedged = not stated) on the generated report; truth = the Merlin test
           reference labels of the manifest (label_mr_*, uncertain = NaN; the paper's reference labels are not
           distributed: supply your own, see scripts/prepare_manifests.py). The reference report through the same rules is the extractor ceiling.
Metrics: clinical macro F1 (organic.ladder.clinical_f1: findings with >= 1 positive among the scans); f1_wrong = each
report scored against another patient's labels (the scan n/2 positions on, as Track A; same pairing for all models);
attributable = f1 - f1_wrong. Patient-level bootstrap 95% CIs. Template check: share of distinct reports, share equal
to the model's most frequent report, empty reports, mean words, mean findings stated per report.

Sources: {W}/evalTrackB/<name>/{d}_reports_shard*.jsonl (extaudit/vlm_report.py); {M}/evalB/<run>/{d}_reports.json
(the stage-B evaluator's generations: NV-Reason-CT and our stage-B runs as generated on the cluster, 256 new tokens).
W = work root, M = cluster work root (scripts/workroot.py: $ORGANIC3D_WORK_ROOT / $ORGANIC3D_CLUSTER_WORK_ROOT).

    python scripts/trackB_metrics.py [--boot 1000]
Writes {W}/eval/report_trackB.csv and {W}/eval/report_trackB_per_finding.csv.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from organic.ladder import clinical_f1, merlin_report_labels
from organic.vocab import CTRATE_FINDINGS, FINDING_NAMES, MERLIN_FINDINGS
from report_tracks import patient
import trackB_ctrate_classifier as ctc

from workroot import cluster_work_root, work_root

W = work_root()
M = cluster_work_root()
N = 200
FAM = {'ctrate': CTRATE_FINDINGS, 'merlin': MERLIN_FINDINGS}
LOCAL = ['vlm4_organic3d', 'vlm1_plain', 'NV-Reason-CT', 'MedGemma-1.5-4B', 'Hulu-Med-7B', 'OmniCT-7B',
         'CT-CHAT', 'RadFM', 'RadFM-reportprompt', 'M3D-LaMed']
CLUSTER = {'NV-Reason-CT': 'NV-Reason-CT'}
EXTRACTOR = {'ctrate': 'CT-RATE RadBERT classifier (official)', 'merlin': 'rule labeller v4 (frozen)'}


def load_local(name, d):
    fs = sorted(glob.glob(f'{W}/evalTrackB/{name}/{d}_reports_shard*.jsonl'))
    out = {}
    for f in fs:
        with open(f) as fh:
            for l in fh:
                if l.strip():
                    r = json.loads(l); out[int(r['index'])] = r['text']
    return out or None


def load_cluster(run, d):
    p = f'{M}/evalB/{run}/{d}_reports.json'
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    return {int(i): t for i, t in zip(r['volume_index'], r['clean']) if int(i) < N} or None


def extract(texts, d):
    """(n, len(FINDING_NAMES)) 0/1 labels of the dataset's findings (other dataset NaN), one extractor per dataset."""
    out = np.full((len(texts), len(FINDING_NAMES)), np.nan)
    if d == 'ctrate':
        out[:, [FINDING_NAMES.index(f) for f in CTRATE_FINDINGS]] = (ctc.classify(texts) >= 0.5).astype(float)
        return out
    return merlin_report_labels(texts)


def scores(pred, Y, cols, w):
    f1 = clinical_f1(pred, Y, cols)
    f1w = clinical_f1(pred[w], Y, cols)
    return f1, f1w, f1 - f1w


def per_finding(pred, Y, cols):
    out = {}
    for j in cols:
        y, p = Y[:, j], pred[:, j]
        ok = ~np.isnan(y) & ~np.isnan(p)
        y, p = y[ok], p[ok]
        tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
        out[FINDING_NAMES[j]] = dict(pos=int(y.sum()), stated=int(p.sum()),
                                     f1=(2 * tp / max(1, 2 * tp + fp + fn)) if y.sum() else np.nan)
    return out


def evaluate(texts, Y, pats, cols, boot, rng, extra_truth=None):
    pred_all = texts
    n = len(pred_all)
    w = np.roll(np.arange(n), n // 2)
    est = scores(pred_all, Y, cols, w)
    up = np.unique(pats)
    rows_of = {p: np.where(pats == p)[0] for p in up}
    bs = []
    for _ in range(boot):
        idx = np.concatenate([rows_of[p] for p in rng.choice(up, len(up))])
        bs.append(scores(pred_all[idx], Y[idx], cols, np.roll(np.arange(len(idx)), len(idx) // 2)))
    bs = np.array(bs)
    lo, hi = np.nanquantile(bs, 0.025, axis=0), np.nanquantile(bs, 0.975, axis=0)
    out = {}
    for i, k in enumerate(('f1', 'f1_wrong', 'attributable')):
        out[k], out[f'{k}_lo'], out[f'{k}_hi'] = est[i], lo[i], hi[i]
    if extra_truth is not None:
        out['f1_released'] = clinical_f1(pred_all, extra_truth, cols)
    return out


def text_stats(texts, pred, cols):
    c = collections.Counter(texts)
    return dict(n_empty=int(sum(not t.strip() for t in texts)), distinct=len(c) / len(texts),
                top_template=c.most_common(1)[0][1] / len(texts), mean_words=float(np.mean([len(t.split()) for t in texts])),
                mean_stated=float(np.nansum(pred[:, cols], 1).mean()))


def setup(d):
    """Scans, patients, truth labels (and released labels for CT-RATE) and all report sources of one dataset."""
    st = pd.read_parquet(f'{W}/extaudit/{d}_audit1000_set.parquet').head(N)
    assert (st['index'].to_numpy() == np.arange(N)).all()
    vols = st.volume.to_numpy()
    pats = np.array([patient(v, d) for v in vols])
    w = np.roll(np.arange(N), N // 2)
    print(f'== {d}: {N} scans, {len(np.unique(pats))} patients, same-patient wrong pairs: {(pats == pats[w]).sum()}')
    cols = [FINDING_NAMES.index(f) for f in FAM[d]]
    L = st[[f'label_{f}' for f in FINDING_NAMES]].to_numpy(np.float64)
    if d == 'ctrate':
        ref_texts = ctc.reference_texts(vols)
        Y = extract(ref_texts, d)                       # classifier labels of the reference reports (main truth)
        rel = np.full_like(Y, np.nan); rel[:, cols] = ctc.released_labels(vols)
        assert np.allclose(np.nan_to_num(rel[:, cols], nan=-1), np.nan_to_num(L[:, cols], nan=-1)), 'audit labels != released'
        print(f'   classifier(reference) == released labels: {(Y[:, cols] == rel[:, cols]).mean():.4f}')
    else:
        mf = pd.read_parquet(f'{W}/manifests/merlin.parquet', columns=['volume', 'report']).set_index('volume')
        ref_texts = mf.loc[vols, 'report'].fillna('').tolist()
        Y, rel = L, None
    srcs = {'reference report (extractor ceiling)': ('reference', {i: t for i, t in enumerate(ref_texts)})}
    for name in LOCAL:
        r = load_local(name, d)
        if r is not None:
            srcs[name] = ('local', r)
    for name, run in CLUSTER.items():
        if name in srcs:
            name = f'{name} (cluster)'
        r = load_cluster(run, d)
        if r is not None:
            srcs[name] = ('cluster', r)
    return dict(vols=vols, pats=pats, cols=cols, Y=Y, rel=rel, srcs=srcs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    rows, pf = [], []
    for d in ('ctrate', 'merlin'):
        s = setup(d)
        pats, cols, Y, rel, srcs = s['pats'], s['cols'], s['Y'], s['rel'], s['srcs']
        for name, (src, r) in srcs.items():
            if len(r) < N:
                print(f'   {name}: only {len(r)}/{N} reports, skipped'); continue
            texts = [r[i] or '' for i in range(N)]
            pred = extract(texts, d)
            res = evaluate(pred, Y, pats, cols, a.boot, np.random.default_rng(0), rel)
            rows.append(dict(model=name, dataset=d, n=N, **res, extractor=EXTRACTOR[d], source=src,
                             **text_stats(texts, pred, cols),
                             reports_sha1=hashlib.sha1('\n'.join(texts).encode()).hexdigest()[:10]))
            for f, v in per_finding(pred, Y, cols).items():
                pf.append(dict(model=name, dataset=d, finding=f, **v))
    t = pd.DataFrame(rows)
    os.makedirs(f'{W}/eval', exist_ok=True)
    t.to_csv(f'{W}/eval/report_trackB.csv', index=False)
    pd.DataFrame(pf).to_csv(f'{W}/eval/report_trackB_per_finding.csv', index=False)
    pd.set_option('display.width', 250)
    for d, g in t.groupby('dataset', sort=False):
        print('==', d, EXTRACTOR[d])
        cols_show = ['model', 'n', 'f1', 'f1_lo', 'f1_hi', 'f1_wrong', 'attributable'] + \
                    (['f1_released'] if 'f1_released' in g and g.f1_released.notna().any() else []) + \
                    ['distinct', 'top_template', 'n_empty', 'mean_words', 'mean_stated']
        print(g[cols_show].sort_values('f1', ascending=False).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
