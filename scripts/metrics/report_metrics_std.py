"""Standard report-generation metrics, the protocol of NV-Reason-CT Table 4 / CT-RATE report papers, on the free-text
reports of every VLM on the full test sets (extaudit/vlm_report.py; our answer-then-write runs: the text after 'Report:').

CT-RATE (v2 test: 3,002 volumes)
  clinical  official RadBERT 18-label classifier on the generated report (threshold 0.5) vs the released labels:
            positive-class macro F1 over the 18 labels (primary, as NV-Reason-CT / CT-CHAT), macro precision / recall,
            CRG (Hamamci et al., MIDL 2025: dataset-level TP/FN/FP over all labels x reports, w_TP = w_FN = (T-A)/2A,
            w_FP = 1, CRG = S_max / (2 S_max - s); an all-normal or all-abnormal report set scores 1/3)
  text      vs the reference Findings_EN: BLEU-4, ROUGE-L, METEOR (COCO-caption implementations: PTB tokenizer, corpus
            BLEU, Java METEOR 1.5), BERTScore and RadGraph-F1 (RadEval), GREEN with --green (RadEval, 7B judge, slow)
Merlin (test: 5,109 scans)
  text      vs the reference FINDINGS section: the same text metrics (RadEval protocol of the Merlin report papers)
  clinical  our frozen rule labeller v4 (organic.ladder.merlin_report_labels) vs the audited QA labels where present:
            macro F1 over findings with >= 1 positive, and CRG (not a published Merlin standard; reported as ours)
Patient bootstrap 95% CIs for F1 and CRG (--boot); text metrics are corpus-level.

    PATH=<java 17>/bin:$PATH <report-eval env>/bin/python scripts/metrics/report_metrics_std.py \
        --names vlm4_organic3d,NV-Reason-CT,Hulu-Med-7B,... [--datasets ctrate,merlin] [--green]
Sources: {W}/eval_reports/<name>/<dataset>_reports_shard*.jsonl. Writes {W}/eval/report_std.csv (rows of other models
and datasets already in it are kept).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from organic.vocab import CTRATE_FINDINGS, FINDING_NAMES, MERLIN_FINDINGS

from organic.workroot import work_root                              # noqa: E402

W = work_root()


def load(name, d):
    """index -> report text. '<name>+list': our answer-then-write output with its finding list kept in front of the
    report (the whole reply, as NV-Reason-CT's reports keep their <answer> list); '<name>' alone: the report body."""
    base, with_list = (name[:-5], True) if name.endswith('+list') else (name, False)
    fs = sorted(glob.glob(f'{W}/eval_reports/{base}/{d}_reports_shard*.jsonl'))
    out = {}
    for f in fs:
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                out[int(r['index'])] = ((r.get('list', '') + ' ') if with_list else '') + (r['text'] or '')
    return out


def crg(pred, Y):
    """CRG over all (report, label) cells with a defined truth."""
    ok = ~np.isnan(Y) & ~np.isnan(pred)
    p, y = pred[ok] == 1, Y[ok] == 1
    tp, fn, fp = (p & y).sum(), (~p & y).sum(), (p & ~y).sum()
    T, A = ok.sum(), y.sum()
    if A == 0:
        return np.nan
    w = (T - A) / (2 * A)
    smax = A * w
    s = tp * w - fn * w - fp * 1.0
    return float(smax / (2 * smax - s))


def clinical(pred, Y):
    """Positive-class macro F1 / precision / recall over labels with >= 1 positive, and CRG."""
    f1, pr, rc = [], [], []
    for j in range(Y.shape[1]):
        ok = ~np.isnan(Y[:, j])
        y, p = Y[ok, j] == 1, pred[ok, j] == 1
        if y.sum() == 0:
            continue
        tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
        f1.append(2 * tp / max(1, 2 * tp + fp + fn)); pr.append(tp / max(1, tp + fp)); rc.append(tp / max(1, tp + fn))
    return dict(f1=float(np.mean(f1)), prec=float(np.mean(pr)), rec=float(np.mean(rc)), crg=crg(pred, Y))


def boot(pred, Y, pats, B, seed=0):
    rng = np.random.default_rng(seed)
    up = np.unique(pats)
    rows = {p: np.where(pats == p)[0] for p in up}
    bs = []
    for _ in range(B):
        i = np.concatenate([rows[p] for p in rng.choice(up, len(up))])
        bs.append(clinical(pred[i], Y[i]))
    return {k: np.nanquantile([b[k] for b in bs], [0.025, 0.975]) for k in bs[0]}


def merlin_findings(report):
    t = str(report or '')
    m = re.search(r'\bFINDINGS?\s*:', t, re.I)
    t = t[m.end():] if m else t
    e = re.search(r'\bIMPRESSIONS?\s*:', t, re.I)
    t = t[:e.start()] if e else t
    return re.sub(r'\s+', ' ', t).strip()


def references(d, vols):
    if d == 'ctrate':
        import ctrate_report_classifier as ctc
        r = pd.read_csv(ctc.REPORTS).set_index('VolumeName').loc[list(vols)]
        return r.Findings_EN.fillna('').astype(str).tolist(), ctc.released_labels(vols), CTRATE_FINDINGS
    m = pd.read_parquet(f'{W}/manifests/merlin.parquet').set_index('volume').loc[list(vols)]
    Y = m[[f'label_{f}' for f in MERLIN_FINDINGS]].to_numpy(float)
    return [merlin_findings(x) for x in m.report], Y, MERLIN_FINDINGS


def extract(texts, d):
    if d == 'ctrate':
        import ctrate_report_classifier as ctc
        return (ctc.classify(texts) >= 0.5).astype(float)
    from organic.ladder import merlin_report_labels
    return merlin_report_labels(texts)[:, [FINDING_NAMES.index(f) for f in MERLIN_FINDINGS]]


def coco(hyps, refs):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    tok = PTBTokenizer()
    g = tok.tokenize({i: [{'caption': r or ' '}] for i, r in enumerate(refs)})
    h = tok.tokenize({i: [{'caption': x or ' '}] for i, x in enumerate(hyps)})
    out = {'bleu4': Bleu(4).compute_score(g, h)[0][3], 'rougeL': Rouge().compute_score(g, h)[0]}
    out['meteor'] = Meteor().compute_score(g, h)[0]
    return out


def radeval(hyps, refs, green):
    from radeval import RadEval
    names = ['bertscore', 'radgraph'] + (['green'] if green else [])
    r = RadEval(metrics=names, show_progress=False)(refs=refs, hyps=[h or ' ' for h in hyps])
    return {k: float(v) for k, v in r.items() if isinstance(v, (int, float, np.floating))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', required=True)
    ap.add_argument('--datasets', default='ctrate,merlin')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--green', action='store_true')
    ap.add_argument('--no_text', action='store_true', help='clinical metrics only')
    a = ap.parse_args()
    rows = []
    for d in a.datasets.split(','):
        st = pd.read_parquet(f'{W}/extaudit/{d}_set.parquet')
        if d == 'ctrate':                                       # CT-RATE v2: drop the 3 non-chest series
            from organic.ctrate_v2 import no_chest_volumes
            st = st[~st.volume.isin(no_chest_volumes(W))]
        for name in a.names.split(','):
            rep = load(name, d)
            have = st[st['index'].isin(rep)]
            if len(have) < len(st):
                print(f'{name} {d}: {len(have)} / {len(st)} reports, skipped (incomplete)'); continue
            hyps = [rep[int(i)] for i in st['index']]
            refs, Y, fs = references(d, st.volume)
            pred = extract(hyps, d)
            r = dict(model=name, dataset=d, n=len(st), **clinical(pred, Y))
            if a.boot:
                for k, (lo, hi) in boot(pred, Y, st.patient_id.values, a.boot).items():
                    r[f'{k}_lo'], r[f'{k}_hi'] = lo, hi
            if not a.no_text:
                r.update(coco(hyps, refs)); r.update(radeval(hyps, refs, a.green))
            r['words'] = float(np.mean([len(h.split()) for h in hyps]))
            r['distinct'] = len(set(hyps)) / len(hyps)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
            rows.append(r)
    out = pd.DataFrame(rows)
    dst = f'{W}/eval/report_std.csv'
    if os.path.exists(dst):                                     # keep rows of models / datasets not re-run
        old = pd.read_csv(dst)
        old = old[~(old.model + '|' + old.dataset).isin(set(out.model + '|' + out.dataset))]
        out = pd.concat([old, out], ignore_index=True)
    out.to_csv(dst, index=False)
    print('wrote', dst)


if __name__ == '__main__':
    main()
