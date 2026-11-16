"""Classification on the lesion datasets (2026-10-23): the columns that join CT-RATE, RAD-ChestCT and Merlin in the one
classification table. Clean scans only; no edits.

ReXGroundingCT (464 radiologist-annotated scans: CT-RATE test scans and ReXGroundingCT validation patients; never in
any of our training sets). Label of finding f on a scan = the radiologists drew a lesion of a category that maps to f
(CAT_FINDING below, 10 CT-RATE findings); 0 = no lesion of that category was drawn on the scan. Score =
the model's CT-RATE finding score on the clean scan: encoders Delta_f (d_clean of extaudit/core.py --task grounding,
the model's own prompts, as in the CT-RATE column), VLMs the yes/no logit (extaudit/vlm_core.py --set_file
rex_heldout_set.parquet --clean_only). Macro AUROC over findings with >= 10 positives and >= 10 negatives.
95% CIs: 2,000 resamples of patients.

    python scripts/metrics/lesion_cls_metrics.py
Writes {W}/eval/lesion_cls_summary.csv (model, kind, dataset, metric, n, n_findings, value, lo,
hi).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from organic.vocab import FINDING_NAMES

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from organic.workroot import work_root                              # noqa: E402

W = work_root()
# ReXGroundingCT lesion category -> CT-RATE finding
CAT_FINDING = {'1a': 'ct_peribronchial_thickening', '1b': 'ct_bronchiectasis', '1c': 'ct_emphysema',
               '1d': 'ct_septal_thickening', '2a': 'ct_atelectasis', '2b': 'ct_consolidation', '2c': 'ct_lung_opacity',
               '2d': 'ct_lung_nodule', '2e': 'ct_pleural_effusion', '2f': 'ct_fibrotic_sequela'}
R = f'{W}/extaudit'
B = 2000
ENCODERS = ['enc4_organic3d', 'COLIPRI', 'Jolia', 'Jolia-long', 'SPECTRE', 'Merlin', 'RADAR-plus', 'RADAR-plus-ft', 'CT-CLIP',
            'CT-VocabFine', 'HLIP', 'ItemizedCLIP', 'fVLM', 'ViSD-Boost']
VLMS = ['vlm4_organic3d', 'vlm1_plain', 'NV-Reason-CT', 'MedGemma-1.5-4B', 'Hulu-Med-7B', 'OmniCT-7B', 'M3D-LaMed', 'CT-CHAT',
        'RadFM']


def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot(fn, groups, rng):
    """fn(row index array) -> value; resample groups (patients) with replacement."""
    ug = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in ug}
    v = [fn(np.concatenate([rows[g] for g in rng.choice(ug, len(ug))])) for _ in range(B)]
    return np.nanpercentile(v, [2.5, 97.5])


# ------------------------------------------------------------------ ReXGroundingCT
def rex_labels():
    gt = pd.read_parquet(f'{W}/rexground/findings.parquet')
    st = pd.read_parquet(f'{R}/rex_heldout_set.parquet')
    vols = st.volume.to_numpy()
    cols = sorted(set(CAT_FINDING.values()), key=FINDING_NAMES.index)
    Y = np.zeros((len(vols), len(cols)))
    pos = gt[gt.category.isin(CAT_FINDING)].assign(f=lambda t: t.category.map(CAT_FINDING))
    have = set(zip(pos.volume, pos.f))
    for i, v in enumerate(vols):
        for j, f in enumerate(cols):
            Y[i, j] = (v, f) in have
    pats = st.patient_id.astype(str).to_numpy()
    return st, vols, cols, Y, pats


def rex_encoder(name, vols, cols):
    d = f'{R}/out/{name}'
    old = sorted(glob.glob(f'{d}/grounding_shard*.parquet'))
    fs = old + sorted(glob.glob(f'{d}/grounding_v2new_shard*.parquet')) if old else sorted(glob.glob(f'{d}/grounding_v2_shard*.parquet'))
    if not fs:
        return None
    t = pd.concat([pd.read_parquet(f, columns=['volume', 'd_clean']) for f in fs]).drop_duplicates('volume').set_index('volume')
    if not set(vols) <= set(t.index):
        print(f'{name}: {len(set(vols) - set(t.index))} held-out ReX scans missing'); return None
    D = np.stack(t.loc[vols, 'd_clean'].to_numpy())
    return D[:, [FINDING_NAMES.index(f) for f in cols]]


def rex_vlm(name, st, cols):
    fs = sorted(glob.glob(f'{R}/out_vlm/{name}/rex_heldout_shard*.npz'))
    if not fs:
        return None
    Z = [dict(np.load(f)) for f in fs]
    key = 'D__clean' if 'D__clean' in Z[0] else 'P__clean'
    idx = np.concatenate([z['index'] for z in Z]); S = np.concatenate([z[key] for z in Z])
    fs_ = list(Z[0]['findings'])
    pos = {int(i): r for r, i in enumerate(idx)}
    if not all(int(i) in pos for i in st['index']):
        print(f'{name}: VLM ReX shards incomplete'); return None
    S = S[[pos[int(i)] for i in st['index']]]
    return S[:, [fs_.index(f) for f in cols]]


def macro_auroc(S, Y, rows):
    v = []
    for j in range(Y.shape[1]):
        y = Y[rows, j]
        if y.sum() >= 10 and (1 - y).sum() >= 10:
            v.append(auroc(y, S[rows, j]))
    return np.nanmean(v) if v else np.nan


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoders', default=','.join(ENCODERS), help='harness names of the encoders')
    ap.add_argument('--vlms', default=','.join(VLMS), help='harness names of the VLMs')
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = []
    st, vols, cols, Y, pats = rex_labels()
    print(f'ReXGroundingCT held-out: {len(vols)} scans, {len(np.unique(pats))} patients; positives per finding',
          dict(zip(cols, Y.sum(0).astype(int))))
    for kind, names in (('encoder', [n for n in a.encoders.split(',') if n]), ('vlm', [n for n in a.vlms.split(',') if n])):
        for n in names:
            S = rex_encoder(n, vols, cols) if kind == 'encoder' else rex_vlm(n, st, cols)
            if S is not None:
                allr = np.arange(len(vols))
                v = macro_auroc(S, Y, allr)
                lo, hi = boot(lambda r: macro_auroc(S, Y, r), pats, rng)
                nf = int(sum(Y[:, j].sum() >= 10 and (1 - Y[:, j]).sum() >= 10 for j in range(len(cols))))
                out.append(dict(model=n, kind=kind, dataset='rexgroundingct_heldout', metric='macro_auroc', n=len(vols),
                                n_findings=nf, value=v, lo=lo, hi=hi))
    o = pd.DataFrame(out)
    dst = f'{W}/eval/lesion_cls_summary.csv'
    if os.path.exists(dst) and len(o):                  # keep other models' rows
        old = pd.read_csv(dst)
        o = pd.concat([old[~old.model.isin(set(o.model))], o], ignore_index=True)
    o.to_csv(dst, index=False)
    pd.set_option('display.width', 200)
    print(o.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
