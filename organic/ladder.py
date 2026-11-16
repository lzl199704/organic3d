"""Shared ladder metrics: matched-pair AUROC (step 5 control) and report labellers for clinical F1 / Ret."""
import os
import re
import sys

import numpy as np

from .vocab import FINDING_NAMES, CTRATE_FINDINGS, MERLIN_FINDINGS


def matched_pair_auroc(L, s, j):
    """AUROC over positive/negative pairs identical on every other labelled finding of the same row family.
    L (N, F) labels with NaN, s (N,) scores, j finding column. Returns (auroc, n_pairs)."""
    y = L[:, j]
    ok = ~np.isnan(y) & ~np.isnan(s)
    if ok.sum() < 2:
        return np.nan, 0
    others = np.delete(np.nan_to_num(L, nan=-1), j, axis=1)[ok]
    yy, ss = y[ok], s[ok]
    keys = {}
    for r, k in enumerate(map(bytes, others.astype(np.int8))):
        keys.setdefault(k, []).append(r)
    num, den = 0.0, 0
    for rows in keys.values():
        rows = np.asarray(rows)
        p, n = ss[rows[yy[rows] == 1]], ss[rows[yy[rows] == 0]]
        if len(p) and len(n):
            cmp = p[:, None] - n[None]
            num += (cmp > 0).sum() + 0.5 * (cmp == 0).sum(); den += cmp.size
    return (num / den if den else np.nan), int(den)


# ------------------------------------------------------------------ report labellers (generated text -> labels)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'scripts'))


def merlin_report_labels(texts):
    """Rule labeller of scripts/label_rules.py (the one used for Merlin training labels) on generated reports."""
    import pandas as pd
    import parse_reports as pr
    from .data import split_sentences
    rows = [dict(volume=i, idx=k, text=t, finding='other', polarity='none', organ='other')
            for i, r in enumerate(texts) for k, t in enumerate(split_sentences(r or '.', max_n=80))]
    lab = pr.merlin_labels(pd.DataFrame(rows)).reindex(range(len(texts)))
    out = np.full((len(texts), len(FINDING_NAMES)), np.nan)
    for f in MERLIN_FINDINGS:
        out[:, FINDING_NAMES.index(f)] = lab[f'label_{f}'].fillna(0).values
    return out


def ctrate_report_labels(texts):
    """Keyword + NegEx labeller for the 18 CT-RATE findings (same gates as the claim parser); closed world."""
    import label_rules as lr
    import parse_reports as pr
    from .data import split_sentences
    out = np.zeros((len(texts), len(FINDING_NAMES)))
    out[:, [FINDING_NAMES.index(f) for f in MERLIN_FINDINGS]] = np.nan
    for i, r in enumerate(texts):
        for t in split_sentences(r or '.', max_n=80):
            for f in CTRATE_FINDINGS:
                m = pr.GATE_RE[f].search(t)
                if not m:
                    continue
                neg = pr.NEG_CUE.search(lr.clause_before(t, m.start())) or pr.CT_POST_NEG.search(t[m.end():])
                if not neg and not lr.HEDGE.search(t):
                    out[i, FINDING_NAMES.index(f)] = 1
    return out


def clinical_f1(pred, true, cols):
    """Macro F1 over findings with at least one positive in `true`."""
    f1s = []
    for j in cols:
        y, p = true[:, j], pred[:, j]
        ok = ~np.isnan(y) & ~np.isnan(p)
        y, p = y[ok], p[ok]
        if y.sum() == 0:
            continue
        tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else np.nan
