"""--task chest: the chest rungs of organic/encoder/evaluate_chest.py (lobe localisation and density-edit specificity) for any
harness encoder (2026-10-24), on the CT-RATE v2 test scans of extaudit/ctrate_chest_set.parquet (scripts/
prepare_chest_set.py). Same tests and metric code as the stage-A chest evaluator; the edits are made here on the 1 mm
RAS volume before the model's own preprocessing (as every harness perturbation), not on our 2 mm crop.

WHERE (scans with lobe sentences). Each sentence template is filled with each of the five lobe names. Conditions: clean
and each lobe erased in turn (TotalSegmentator v2 lobes 12, 13, 14, 10, 11 = LOBES order, 2 mm dilation, fat fill
-100 HU + N(0, 15) as the organ rung). Scores: the five lobe versions on the clean scan (text test) and the true-lobe
version under each condition (image test).
WHAT (every scan). Edits (organic.encoder.evaluate_chest.edit_masks / calcium_masks, here at 1 mm):
  emph    lung voxels < -950 HU not connected to the trachea's air (trachea dilated 2 mm) -> -860 HU + N(0, 15)
  ca_ao   aortic calcium (>= 130 HU after 3 x 3 x 3 mean filter, < 2000 HU raw, components >= 3 voxels, plus a 1 mm
          rim inside > 0 HU), in the aorta dilated 1 mm (unlabelled or aorta voxels) -> 40 HU + N(0, 15)
  ca_cor  the same in a 3 mm shell around the heart, outside heart and aorta (dilated 3 mm) and every labelled structure
Calcium edits are skipped on contrast-enhanced scans (median of the eroded aorta > 120 HU). Score: the finding deltas
(the model's own prompts) on the clean and the three edited volumes.
Writes {root}/out/<name>/chest_shard{k}of{n}.npz; metrics: scripts/metrics/chest_harness_metrics.py.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

LOBES = ['right upper lobe', 'right middle lobe', 'right lower lobe', 'left upper lobe', 'left lower lobe']
LOBE_TS = [12, 13, 14, 10, 11]
EDITS = {'emph': -860.0, 'ca_ao': 40.0, 'ca_cor': 40.0}
TRACHEA, HEART, AORTA = 16, 51, 52


def edit_masks(hu, seg):
    """1 mm numpy HU and TotalSegmentator ids -> {edit: bool mask}, contrast flag (organic.encoder.evaluate_chest rules at 1 mm)."""
    from scipy import ndimage
    lung = np.isin(seg, LOBE_TS)
    trachea = ndimage.binary_dilation(seg == TRACHEA, iterations=2)
    low = (hu < -950) & (lung | trachea)
    lab, n = ndimage.label(low)
    airway = np.unique(lab[trachea & low]) if n else np.zeros(0, int)
    emph = (hu < -950) & lung & ~np.isin(lab, airway[airway > 0])
    aorta, heart = seg == AORTA, seg == HEART
    empty = np.zeros_like(emph)
    if aorta.sum() > 2000 and np.median(hu[ndimage.binary_erosion(aorta, iterations=2)]) > 120:
        return {'emph': emph, 'ca_ao': empty, 'ca_cor': empty.copy()}, True
    hs = ndimage.uniform_filter(hu, size=3)

    def ca(region):
        m = region & (hs >= 130) & (hu < 2000)
        lab_, n_ = ndimage.label(m)
        if n_:
            size = ndimage.sum(m, lab_, np.arange(1, n_ + 1))
            m = np.isin(lab_, np.where(size >= 3)[0] + 1)
        return ndimage.binary_dilation(m, iterations=1) & (hu > 0)
    shell = ndimage.binary_dilation(heart, iterations=3) & ~heart & ~ndimage.binary_dilation(aorta, iterations=3) & (seg == 0)
    ao_region = ndimage.binary_dilation(aorta, iterations=1) & ((seg == 0) | (seg == AORTA))
    return {'emph': emph, 'ca_ao': ca(ao_region), 'ca_cor': ca(shell)}, False


def chest_set(rows):
    """VolSet whose items also carry the edit masks (computed in the loader workers, CPU)."""
    from extaudit.core import VolSet

    class ChestSet(VolSet):
        def __getitem__(self, i):
            b = super().__getitem__(i)
            if 'error' not in b:
                masks, b['contrast'] = edit_masks(b['hu'].numpy(), b['seg'].numpy())
                b['masks'] = {e: torch.from_numpy(m) for e, m in masks.items()}
            return b
    return ChestSet(rows)


def run_chest(a, meta, mdl, dev, k, n):
    rows = pd.read_parquet(os.path.join(a.root, 'ctrate_chest_set.parquet')).iloc[k::n]
    if a.limit:
        rows = rows.head(a.limit)
    items = dict(zip(rows['index'].astype(int), rows.lobe_items))
    gen = torch.Generator(device=dev).manual_seed(k)
    fill, sd = meta['fills_hu']['fat'], meta['noise_sd']
    lobe_rec, edit_rec, errors = [], [], []
    loader = DataLoader(chest_set(rows), batch_size=None, num_workers=a.workers, prefetch_factor=2)
    t0 = time.time()
    for it, b in enumerate(loader):
        if 'error' in b:
            errors.append((b['index'], b['error'])); print('ERROR', b['index'], b['error'], flush=True)
            continue
        hu, seg = b['hu'].to(dev), b['seg'].to(dev)
        mdl.scan(seg, b['volume'])
        st0, _ = mdl.image(hu)
        masks, contrast = {e: m.numpy() for e, m in b['masks'].items()}, bool(b['contrast'])
        d = [mdl.deltas(st0)]
        for e, hu_fill in EDITS.items():
            mk = torch.from_numpy(masks[e]).to(dev)
            v = hu.clone()
            if mk.any():
                v[mk] = hu_fill + torch.randn(int(mk.sum()), device=dev, generator=gen) * sd
                d.append(mdl.deltas(mdl.image(v)[0]))
            else:
                d.append(d[0].copy())
        edit_rec.append(dict(index=b['index'], nvox=[int(masks[e].sum()) for e in EDITS], contrast=contrast,
                             delta=np.stack(d)))
        its = json.loads(items[b['index']])
        if its:
            sents = [t.replace('{LOBE}', L) for t, _, _ in its for L in LOBES]
            s_clean = mdl.sentences(st0, sents).cpu().numpy().reshape(len(its), 5)
            true_sents = [t.replace('{LOBE}', LOBES[l]) for t, l, _ in its]
            img = [mdl.sentences(st0, true_sents).cpu().numpy()]
            nvox = []
            for lid in LOBE_TS:
                m = (seg == lid).float()[None, None]
                m = F.max_pool3d(m, 5, stride=1, padding=2)[0, 0] > 0
                nvox.append(int((seg == lid).sum()))
                v = hu.clone()
                if m.any():
                    v[m] = fill + torch.randn(int(m.sum()), device=dev, generator=gen) * sd
                img.append(mdl.sentences(mdl.image(v)[0], true_sents).cpu().numpy())
            img = np.stack(img, 1)                                            # (sentences, 6)
            for s_i, (t, l, w) in enumerate(its):
                lobe_rec.append(dict(index=b['index'], true=int(l), word=w, nvox=nvox, text=s_clean[s_i].tolist(),
                                     img=img[s_i].tolist()))
        if it % 50 == 0:
            el = time.time() - t0
            print(f'{it + 1}/{len(rows)}  {el / 60:.1f} min, {el / (it + 1):.1f} s/scan', flush=True)
    out = os.path.join(a.root, 'out', a.name)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f'chest_shard{k}of{n}.npz'),
                        edit_index=np.asarray([r['index'] for r in edit_rec]),
                        edit_nvox=np.asarray([r['nvox'] for r in edit_rec]).reshape(-1, 3),
                        edit_contrast=np.asarray([r['contrast'] for r in edit_rec]),
                        edit_delta=np.stack([r['delta'] for r in edit_rec]) if edit_rec else np.zeros((0, 4, 0)),
                        lobes=np.asarray(json.dumps(lobe_rec)), errors=np.asarray([json.dumps(e) for e in errors]))
    return len(edit_rec), len(errors), t0
