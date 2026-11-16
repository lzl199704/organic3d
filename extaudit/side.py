"""Side accuracy (Side) on original scans: a flipped CT is anatomically situs inversus, so this test does not flip the scan.
Per sentence that names one side, the score of the sentence vs the same sentence with
the side swapped, on the original scan only; scripts/metrics/side_metrics.py turns them into balanced side accuracy (mean of
the accuracy on left-side and right-side sentences: 50% for chance and for any model that ignores side).

  SIDE_SET=ctrate | merlin   report sentences of the harness test set ({dataset}_set.parquet lat_pairs: [true, swapped])
                             SIDE_EVERY=k keeps every k-th scan with side sentences (11/03: ctrate 3, merlin 6; the
                             machine was CPU-bound at ~30 s/scan for RADAR+, so the full sets would take > 1 day)
  SIDE_SET=rex | aa3         expert-mask sides (scripts/data/prepare_side.py): ReXGroundingCT lung findings, AbdomenAtlas 3.0
                             kidney tumours; scored on the lesion unit's crop as in extaudit/lesion_erasure.py
Writes {root}/out/<name>/side_<set>_shard{k}of{n}.parquet (one row per sentence).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SIDE_SET = os.environ.get('SIDE_SET')
ZWIN_MM = 480


def run_side(a, meta, mdl, dev, k, n):
    from extaudit.core import VolSet
    from extaudit.lesion_erasure import ORGAN_ID
    W = os.path.join(a.root, '..')
    if SIDE_SET in ('ctrate', 'merlin'):
        st = pd.read_parquet(os.path.join(a.root, f'{SIDE_SET}_set.parquet'))
        st = st[st.lat_pairs.map(lambda v: bool(v) and v != '[]')]
        st = st.iloc[::int(os.environ.get('SIDE_EVERY', '1'))]   # 11/03: fixed subset, the same scans for every model
        sents = None
    else:
        sents = pd.read_parquet(os.path.join(W, 'side', f'{SIDE_SET}.parquet'))
        st = pd.read_parquet(os.path.join(a.root, f'heal_{SIDE_SET}_set.parquet'))
        st = st[st.volume.isin(set(sents.volume))]
        by_vol = {v: g for v, g in sents.groupby('volume')}
    st = st.iloc[k::n]
    if a.limit:
        st = st.head(a.limit)
    if sents is not None:                                    # Jolia: pre-encoded sentences (missing ones are encoded)
        cache = os.path.join(W, 'side', f'jolia_text_cache_{SIDE_SET}.pt')
        if isinstance(getattr(mdl.ad, 'cache', None), dict) and os.path.exists(cache):
            mdl.ad.cache.update(torch.load(cache))
    loader = DataLoader(VolSet(st), batch_size=None, num_workers=a.workers, prefetch_factor=2)
    rows, errors, t0 = [], [], time.time()
    for it, b in enumerate(loader):
        if 'error' in b:
            errors.append(b['error']); print('ERROR', b['error'], flush=True)
            continue
        hu, seg = b['hu'].to(dev), b['seg'].to(dev)
        if sents is None:
            pairs = json.loads(b['lat'])
            true, swap = [p for p, _ in pairs], [q for _, q in pairs]
            meta_rows = [dict(volume=b['volume'], j=j) for j in range(len(pairs))]
        else:
            g = by_vol[b['volume']]
            organ = g.organ.iloc[0]
            zs = torch.nonzero(torch.isin(seg, torch.as_tensor(ORGAN_ID[organ], device=dev, dtype=seg.dtype)).any(0).any(0))[:, 0]
            z0 = 0
            if hu.shape[2] > ZWIN_MM and len(zs):                # head-foot window centred on the organ (as the heal test)
                z0 = min(max(int(round((int(zs.min()) + int(zs.max())) / 2 - ZWIN_MM / 2)), 0), hu.shape[2] - ZWIN_MM)
            hu, seg = hu[:, :, z0:z0 + ZWIN_MM].contiguous(), seg[:, :, z0:z0 + ZWIN_MM].contiguous()
            true, swap = list(g.text_true), list(g.text_swap)
            meta_rows = [dict(volume=b['volume'], j=int(r.k), organ=organ, patient=r.patient) for r in g.itertuples()]
        mdl.scan(seg, b['volume'])
        state = mdl.image(hu)[0]
        s = mdl.sentences(state, true + swap).cpu().numpy()
        m = len(true)
        for j, r in enumerate(meta_rows):
            rows.append(dict(r, text_true=true[j], s_true=float(s[j]), s_swap=float(s[m + j])))
        if it % 50 == 0:
            el = time.time() - t0
            print(f'{it + 1}/{len(st)}  {el / 60:.1f} min, {el / (it + 1):.1f} s/scan', flush=True)
    out = os.path.join(a.root, 'out', a.name)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(out, f'side_{SIDE_SET}_shard{k}of{n}.parquet'), index=False)
    return len(rows), len(errors), t0
