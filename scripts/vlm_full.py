"""Full-test-set VLM answers, keyed by the FULL-set index (<dataset>_set.parquet: CT-RATE v2 3,005 rows, Merlin 5,109).
Simplest: one harness run on the whole set, extaudit/vlm_core.py --set_file <dataset>_set.parquet --out_tag <dataset>_full
(out_vlm/<name>/<dataset>_full_shard*.npz). The paper's runs instead merged two parts:

  audit part  harness shards out_vlm/<name>/<dataset>_shard*.npz (index = <dataset>_audit1000_set.parquet row). The two
              models whose current Table 2 audit came from another pipeline (MedGemma-1.5-4B: organic.audit_medgemma;
              CT-CHAT on CT-RATE: $CTCHAT_SCORES) are re-run in the harness on the audit scans as well (queue
              vlm_rest_1025.txt, vaud_* jobs), so every full-set number comes from one harness. _audit_pipeline() reads
              the old pipeline files in harness keys; since 2026-10-26 it supplies the audit part of MedGemma (both datasets) and CT-CHAT (CT-RATE).
              Merlin: 7 of the 1,000 audit scans are not in merlin_set.parquet and are dropped (993 kept).
  rest part   out_vlm/<name>/<dataset>_rest_shard*.npz from extaudit/vlm_core.py --set_file <dataset>_rest_set.parquet
              --out_tag <dataset>_rest (index = full-set index; CT-RATE 2,005 scans, Merlin 4,116).
Same conditions in both parts (clean, mirror, every organ of meta['eval_organs'] erased with fat -100 HU + N(0, 15),
mask dilated 2 mm). The fill noise comes from a generator seeded with the shard number and drawn in scan order
(extaudit/vlm_core.py: torch.Generator().manual_seed(k)), so the noise realisation of a scan depends on its run, shard
and position, not on the scan; audit and rest scans get the same rule with different random draws (SD 15 HU).
Scores: the logit difference D__ when every part saved it, else P(Yes) (same ranking; MedGemma's audit saved P only).

    load(name, dataset) -> dict(index, feat, findings, P__<cond>|D__<cond>, nvox__<cond>) sorted by full-set index,
                           or None if a part is missing.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workroot import work_root                              # noqa: E402

W = work_root()
CTCHAT_SCORES = os.environ.get('CTCHAT_SCORES', f'{W}/evalB/CT-CHAT/scores_shard*.npz')
R = f'{W}/extaudit'
# 2026-10-26 (user): the audit-scan results of these runs were saved by their original pipelines with every condition
# (clean, mirror, each eval organ erased with fat + N(0, 15)); they are used as the audit part instead of a harness re-run.
# Checked against the harness on 8 scans: clean P identical for MedGemma (max |dP| 0.0000) and within 0.004 for CT-CHAT.
PIPELINE = {('MedGemma-1.5-4B', 'ctrate'): 'medgemma', ('MedGemma-1.5-4B', 'merlin'): 'medgemma',
            ('CT-CHAT', 'ctrate'): 'ctchat'}


def _cat(files):
    Z = [dict(np.load(f, allow_pickle=True)) for f in files]
    keep = [k for k in Z[0] if k.startswith(('P__', 'D__', 'nvox__', 'index', 'feat'))]
    return {k: np.concatenate([z[k] for z in Z]) for k in keep}, [str(x) for x in Z[0]['findings']]


def _audit_pipeline(kind, d):
    """Audit scores of another pipeline -> harness keys (P__<cond>, nvox__<cond>, index = audit index, feat)."""
    if kind == 'medgemma':
        z = dict(np.load(f'{W}/evalB/MedGemma-1.5-4B_scores/{d}_scores.npz', allow_pickle=True))
        st = pd.read_parquet(f'{R}/{d}_audit1000_set.parquet').set_index('volume')
        out = {'index': st.loc[z['volume'].astype(str), 'index'].to_numpy(), 'feat': z['feat']}
        for k, v in z.items():
            if k in ('clean', 'mirror'):
                out[f'P__{k}'] = v; out[f'nvox__{k}'] = np.zeros(len(v), int)
            elif k.startswith('rem_'):
                out[f'P__{k}@fat'] = v
            elif k.startswith('nvox_rem_'):
                out[f'nvox__{k[5:]}@fat'] = v
        from organic.vocab import CTRATE_FINDINGS, MERLIN_FINDINGS
        return out, list(CTRATE_FINDINGS if d == 'ctrate' else MERLIN_FINDINGS)
    Z = [dict(np.load(f)) for f in sorted(glob.glob(CTCHAT_SCORES))]
    cat = {k: np.concatenate([z[k] for z in Z]) for k in Z[0]}
    out = {'index': cat['index'], 'feat': cat['feat']}
    for k, v in cat.items():
        if k.startswith(('P__', 'nvox__')):
            c = k.split('__', 1)[1]
            out[f'{k.split("__")[0]}__{c}@fat' if c.startswith('rem_') else k] = v
    from organic.vocab import CTRATE_FINDINGS
    return out, list(CTRATE_FINDINGS)


def load(name, d):
    full = pd.read_parquet(f'{R}/{d}_set.parquet')
    vol2full = dict(zip(full.volume, full['index']))
    one = sorted(glob.glob(f'{R}/out_vlm/{name}/{d}_full_shard*.npz'))
    if one:                       # one run on the whole test set: vlm_core.py --set_file <d>_set.parquet --out_tag <d>_full
        out, fa = _cat(one)
        o = np.argsort(out['index'])
        out = {k: v[o] for k, v in out.items()}
        if len(np.unique(out['index'])) != len(full):
            print(f'{name} {d}: {len(full) - len(np.unique(out["index"]))} of {len(full)} full-set scans missing')
            return None
        out['findings'] = np.asarray(fa)
        out['n_audit'] = 0; out['n_rest'] = len(out['index'])
        return out
    if (name, d) in PIPELINE:
        a, fa = _audit_pipeline(PIPELINE[(name, d)], d)
    else:
        fs = sorted(glob.glob(f'{R}/out_vlm/{name}/{d}_shard*.npz'))
        if not fs:
            return None
        a, fa = _cat(fs)
    aud = pd.read_parquet(f'{R}/{d}_audit1000_set.parquet').set_index('index').volume
    av = aud.loc[a['index']].to_numpy()
    ka = np.array([v in vol2full for v in av])
    a = {k: v[ka] for k, v in a.items()}
    a['index'] = np.array([vol2full[v] for v in av[ka]])
    rf = sorted(glob.glob(f'{R}/out_vlm/{name}/{d}_rest_shard*.npz'))
    if not rf:
        return None
    b, fb = _cat(rf)
    assert fa == fb, (name, d, 'finding lists differ between audit and rest parts')
    key = 'D__' if any(k.startswith('D__') for k in a) and any(k.startswith('D__') for k in b) else 'P__'
    conds = sorted(k for k in a if k.startswith(key)); assert conds == sorted(k for k in b if k.startswith(key)), (name, d)
    nvk = sorted(set(k for k in a if k.startswith('nvox__')) & set(k for k in b if k.startswith('nvox__')))
    out = {k: np.concatenate([a[k], b[k]]) for k in ['index'] + conds + nvk}
    out['feat'] = np.concatenate([a['feat'], b['feat']]) if a['feat'].shape[1:] == b['feat'].shape[1:] else \
        np.full((len(out['index']), 1), np.nan, np.float32)            # pipelines with different pooled features
    o = np.argsort(out['index'])
    out = {k: v[o] for k, v in out.items()}
    assert len(np.unique(out['index'])) == len(out['index']), (name, d, 'duplicate scans')
    missing = len(full) - len(out['index'])
    if missing:
        print(f'{name} {d}: {missing} of {len(full)} full-set scans missing')
        return None
    out['findings'] = np.asarray(fa)
    out['n_audit'] = int(ka.sum()); out['n_rest'] = len(b['index'])
    return out
