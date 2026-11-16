"""--task lesion_erasure: the lesion-erasure test (LE, paper Sec. 3.4) on the sets of scripts/prepare_heal.py.
LE_SET=rex (ReXGroundingCT lung findings, chest crop) or LE_SET=aa3 (AbdomenAtlas 3.0 liver, pancreas and kidney
tumours, abdominal crop); run through extaudit/core.py.

Per lesion (>= 10 mm, sentence from the report or "There is a {d} cm mass in the {organ}."):
  clean   the sentence's score on the clean scan;
  les     the lesion erased: its voxels set to the median HU of the organ's normal parenchyma (organ minus every
          lesion dilated 3 mm) + N(0, noise_sd); LE_FILL=sample: values drawn from that parenchyma instead;
  ctrl    control: the lesion's own shape translated inside the same organ (scripts/prepare_heal.py:
          >= 70% inside the parenchyma, < 5% overlap with any lesion dilated 10 mm), erased the same way.
Hit (scripts/heal_metrics.py) = s_clean - s_les > s_clean - s_ctrl.
Field of view: scans longer than ZWIN_MM are cut head-foot to a 480 mm window centred on the organ, for every model.
Writes {root}/out/<name>/heal_<set>_shard{k}of{n}.parquet.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ZWIN_MM = 480
FILL_MODE = os.environ.get('LE_FILL', 'median')
ORGAN_ID = {'liver': [5], 'pancreas': [7], 'right_kidney': [2], 'left_kidney': [3],          # TotalSegmentator v2 ids
            'left_lung': [10, 11], 'right_lung': [12, 13, 14], 'lungs': [10, 11, 12, 13, 14]}
LE_SET = os.environ.get('LE_SET')               # 'rex' | 'aa3': the sets of scripts/prepare_heal.py


def flat_mask(idx, shape, dev):
    m = torch.zeros(int(np.prod(shape)), dtype=torch.bool, device=dev)
    if len(idx):
        m[torch.as_tensor(np.asarray(idx), device=dev).long()] = True
    return m.view(shape)


def run_lesion_erasure(a, meta, mdl, dev, k, n):
    from extaudit.core import VolSet
    if LE_SET not in ('rex', 'aa3'):
        raise SystemExit('set LE_SET=rex or LE_SET=aa3')
    W = os.path.join(a.root, '..')
    base = os.path.join(W, 'heal', LE_SET)
    gt = pd.read_parquet(os.path.join(base, 'findings.parquet'))
    st = pd.read_parquet(os.path.join(a.root, f'heal_{LE_SET}_set.parquet'))
    st = st[st.volume.isin(set(gt.volume))].iloc[k::n]
    if a.limit:
        st = st.head(a.limit)
    cache = os.path.join(base, 'jolia_text_cache.pt')       # Jolia: sentences pre-encoded (Qwen3-8B); missing texts are encoded
    if isinstance(getattr(mdl.ad, 'cache', None), dict) and os.path.exists(cache):
        mdl.ad.cache.update(torch.load(cache))
    by_vol = {v: g for v, g in gt.groupby('volume')}
    loader = DataLoader(VolSet(st, os.path.join(base, 'masks')), batch_size=None, num_workers=a.workers,
                        prefetch_factor=2)
    gen = torch.Generator(device=dev).manual_seed(k)
    sd = meta['noise_sd']
    rows, errors, t0 = [], [], time.time()
    for it, b in enumerate(loader):
        if 'error' in b:
            errors.append(b['error']); print('ERROR', b['error'], flush=True)
            continue
        hu, seg, L = b['hu'].to(dev), b['seg'].to(dev), b['les']
        full = hu.shape
        g = by_vol[b['volume']]
        organ = g.organ.iloc[0]
        zs = torch.nonzero(torch.isin(seg, torch.as_tensor(ORGAN_ID[organ], device=seg.device, dtype=seg.dtype)).any(0).any(0))[:, 0]
        z0 = 0
        if full[2] > ZWIN_MM and len(zs):                    # head-foot window centred on the organ
            z0 = min(max(int(round((int(zs.min()) + int(zs.max())) / 2 - ZWIN_MM / 2)), 0), full[2] - ZWIN_MM)
        z1 = min(z0 + ZWIN_MM, full[2])
        hu, seg = hu[:, :, z0:z1].contiguous(), seg[:, :, z0:z1].contiguous()

        def M(idx):
            return flat_mask(idx, full, dev)[:, :, z0:z1].contiguous()

        mdl.scan(seg, b['volume'])
        par = M(L['par'])
        pool = hu[par]
        s_clean = mdl.image(hu)[0]
        texts = list(g.text)
        sc = mdl.sentences(s_clean, texts).cpu().numpy()
        for j, r in enumerate(g.itertuples()):
            key = f'k{r.k}'
            if key not in L:
                continue
            les = M(L[key])
            ctrl = M(L[f'c{r.k}']) if f'c{r.k}' in L else None
            row = dict(volume=b['volume'], k=r.k, organ=organ, source=r.source, spectre_seen=r.spectre_seen,
                       patient=r.patient, size_cm=r.size_cm, n_vox=int(les.sum()), n_vox_full=int(r.n_vox),
                       ctrl_valid=bool(r.ctrl_valid) and ctrl is not None, z0=z0, z1=z1, z_full=int(full[2]),
                       fill=r.par_hu, s_clean=float(sc[j]), n_ctrl=int(ctrl.sum()) if ctrl is not None else 0)
            for name, m in (('les', les), ('ctrl', ctrl)):
                if m is None or not m.any():
                    row[f's_{name}'] = np.nan
                    continue
                v = hu.clone()
                cnt = int(m.sum())
                if FILL_MODE == 'sample' and len(pool):
                    v[m] = pool[torch.randint(len(pool), (cnt,), device=dev, generator=gen)]
                else:
                    v[m] = float(r.par_hu) + torch.randn(cnt, device=dev, generator=gen) * sd
                row[f's_{name}'] = float(mdl.sentences(mdl.image(v)[0], [r.text])[0])
            rows.append(row)
        if it % 20 == 0:
            el = time.time() - t0
            print(f'{it + 1}/{len(st)}  {el / 60:.1f} min, {el / (it + 1):.1f} s/scan', flush=True)
    out = os.path.join(a.root, 'out', a.name)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(out, f'heal_{LE_SET}_shard{k}of{n}.parquet'), index=False)
    return len(rows), len(errors), t0
