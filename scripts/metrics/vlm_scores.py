"""VLM answer scores on the full test set, keyed by the set index (<dataset>_set.parquet: CT-RATE v2 3,005 rows, Merlin
5,109): the harness shards of one extaudit/vlm_core.py run on the whole set (out_vlm/<name>/<dataset>_shard*.npz).
Conditions: clean, mirror, every organ of meta['eval_organs'] erased with fat -100 HU + N(0, 15), mask dilated 2 mm.
Scores: the logit difference D__ when the adapter saved it, else P(Yes) (same ranking).

    load(name, dataset) -> dict(index, feat, findings, P__<cond>|D__<cond>, nvox__<cond>) sorted by set index,
                           or None if shards are missing or the run does not cover the whole set (partial=True
                           accepts a run on the first N scans, extaudit/vlm_core.py --first N).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from organic.workroot import work_root                              # noqa: E402

W = work_root()
R = f'{W}/extaudit'


def load(name, d, partial=False):
    fs = sorted(glob.glob(f'{R}/out_vlm/{name}/{d}_shard*.npz'))
    if not fs:
        return None
    Z = [dict(np.load(f, allow_pickle=True)) for f in fs]
    keep = [k for k in Z[0] if k.startswith(('P__', 'D__', 'nvox__', 'index', 'feat'))]
    out = {k: np.concatenate([z[k] for z in Z]) for k in keep}
    o = np.argsort(out['index'])
    out = {k: v[o] for k, v in out.items()}
    assert len(np.unique(out['index'])) == len(out['index']), (name, d, 'duplicate scans')
    n = len(pd.read_parquet(f'{R}/{d}_set.parquet'))
    if len(out['index']) != n and not partial:
        print(f'{name} {d}: {n - len(out["index"])} of {n} test scans missing')
        return None
    out['findings'] = np.asarray([str(x) for x in Z[0]['findings']])
    return out
