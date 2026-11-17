"""ReXGroundingCT radiologist-annotated scans (464; never in any of our training sets) for the Table 2 VLMs: the same
answer metrics as the CT-RATE / Merlin columns.

Labels: finding f is positive on a scan when the radiologists drew a lesion of a category that maps to f
(lesion_cls_metrics.rex_labels: 10 CT-RATE findings), else 0; the other CT-RATE findings have no label here (NaN).
  --mode clean   existing shards out_vlm/<name>/rex_heldout_shard*.npz (vlm_core --clean_only): answer AUROC, BACC, F1
                 (Yes when P(Yes) >= 0.5 / logit difference >= 0; main_tables.macro, findings with >= 10 positives and
                 >= 10 negatives)  -> W/eval/rex_vlm_clean_cls.csv
  --mode ladder  shards out_vlm/<name>/rex_heldout_ladder_shard*.npz (vlm_core --dataset ctrate --set_file
                 rex_heldout_set.parquet --out_tag rex_heldout_ladder: clean, mirror and the CT-RATE organ erasures): the
                 above plus MP (matched-pair AUROC: positive/negative pairs identical on the other 9 lesion labels,
                 organic.metrics.matched_pair_auroc via organic.vlm.evaluate.metrics) and OER (answer test,
                 finding_reports.evaluate retraction: cross-fitted thresholds, non-size findings, target organ vs control
                 organ)  -> W/eval/rex_vlm_ladder.csv
Point estimates only (the paper reports no intervals).

    python scripts/metrics/rex_vlm_metrics.py --mode clean|ladder
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from organic.vlm.evaluate import metrics  # noqa: E402
from organic.vocab import FINDING_NAMES  # noqa: E402
from lesion_cls_metrics import rex_labels  # noqa: E402
from main_tables import answer, macro  # noqa: E402
import finding_reports as RT  # noqa: E402

from organic.workroot import work_root                              # noqa: E402

W = work_root()
R = f'{W}/extaudit'
VLMS = ['Hulu-Med-7B', 'OmniCT-7B', 'MedGemma-1.5-4B', 'M3D-LaMed', 'RadFM', 'CT-CHAT', 'NV-Reason-CT', 'vlm1_plain',
        'vlm4_organic3d']


def load(name, tag, st):
    fs = sorted(glob.glob(f'{R}/out_vlm/{name}/{tag}_shard*.npz'))
    if not fs:
        return None
    Z = [dict(np.load(f, allow_pickle=True)) for f in fs]
    cat = {k: np.concatenate([z[k] for z in Z]) for k in Z[0] if k.startswith(('P__', 'D__', 'nvox__', 'index', 'feat'))}
    pos = {int(i): r for r, i in enumerate(cat['index'])}
    if not all(int(i) in pos for i in st['index']):
        print(f'{name} {tag}: incomplete'); return None
    o = np.array([pos[int(i)] for i in st['index']])          # rows in rex_heldout_set order
    cat = {k: v[o] for k, v in cat.items()}
    return cat, [str(f) for f in Z[0]['findings']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['clean', 'ladder'], required=True)
    ap.add_argument('--names', default=','.join(VLMS))
    a = ap.parse_args()
    st, vols, cols, Y10, pats = rex_labels()
    n = len(st)
    L = np.full((n, len(FINDING_NAMES)), np.nan, np.float32)    # labels in FINDING_NAMES order, row = rex set row
    for j, f in enumerate(cols):
        L[:, FINDING_NAMES.index(f)] = Y10[:, j]
    tag = 'rex_heldout' if a.mode == 'clean' else 'rex_heldout_ladder'
    rows = []
    for name in a.names.split(','):
        r = load(name, tag, st)
        if r is None:
            print(f'{name}: no {tag} shards'); continue
        cat, finds = r
        key = 'D__' if any(k.startswith('D__') for k in cat) else 'P__'
        S = cat[f'{key}clean']
        jcols = [FINDING_NAMES.index(f) for f in finds]
        Yq = L[:, jcols]
        c = macro(answer(S), Yq, S, 10, 10)
        row = dict(model=name, n=n, n_findings=int(sum((np.nansum(Yq[:, j]) >= 10) and (np.nansum(1 - Yq[:, j]) >= 10)
                                                      for j in range(Yq.shape[1]))),
                   auroc=c.get('auroc'), bacc=c.get('bacc'), f1=c.get('f1'), yes_rate=float(np.nanmean(answer(S))))
        if a.mode == 'ladder':
            res = {k[3:].split('@')[0]: torch.tensor(v) for k, v in cat.items() if k.startswith(key)}
            res['wrong'] = res['clean'][np.roll(np.arange(n), 1)]
            res.update({'nvox_' + k[6:].split('@')[0]: torch.tensor(v) for k, v in cat.items() if k.startswith('nvox__')})
            res['feat'] = torch.tensor(cat['feat']); res['index'] = torch.arange(n)
            res['gen_index'] = []; res['texts'] = {'clean': [], 'wrong': []}
            res['js'] = jcols
            _, summ = metrics(res, L, 'ctrate')
            P = {k[3:].split('@')[0]: v.astype(float) for k, v in cat.items() if k.startswith(key)}
            nv = {k[6:].split('@')[0]: v for k, v in cat.items() if k.startswith('nvox__')}
            ev = RT.evaluate((P, nv, finds, L, st.volume.to_numpy()), 'ctrate', 1, np.random.default_rng(0))   # CT-RATE naming: patient rule, no non-chest scans here; 1 resample (CIs unused)
            row.update(mc=summ.get('matched_pair_auroc'), oer=ev['retraction'], composed_f1=ev['f1'],
                       composed_ia_f1=ev['attributable'])
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    out = f'{W}/eval/rex_vlm_clean_cls.csv' if a.mode == 'clean' else f'{W}/eval/rex_vlm_ladder.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out)


if __name__ == '__main__':
    main()
