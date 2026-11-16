"""--task aa3_det (2026-10-27): zero-shot tumour detection on the AbdomenAtlas 3.0 IID test split (scripts/prepare_aa3.py).
Every scan has radiologist-revised tumour masks for liver, pancreas and kidney, so a scan without a lesion in an organ is a
checked negative from the same cohort. Detection score = positive minus negative prompt, two prompt sets per organ:
  native  each encoder's own published template and score rule, exactly as for the MLC columns (extaudit/native_prompts.py;
          findings liver / pancreatic / renal mass), e.g. CT-CLIP 'Liver mass is present.' / 'Liver mass is not
          present.', ours 'There is liver mass.' / 'No liver mass.'
  free    one sentence pair for every encoder: "There is a mass in the {organ}." / "There is no mass in the {organ}."
scripts/lesion_det_metrics.py: AUROC per organ (Table 1a: native; supplement: both).
Field of view: as extaudit/lesion_erasure.py, scans longer than ZWIN_MM are cut head-foot to a 480 mm window centred on
liver + pancreas (the same rule for every scan).
Writes {root}/out/<name>/aa3_det_shard{k}of{n}.parquet (one row per scan; labels are joined in the metrics script).
"""
import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from extaudit.native_prompts import native_prompts
from extaudit.lesion_erasure import ORGAN_ID, ZWIN_MM

ORGANS = ['liver', 'pancreas', 'kidney']
TEXTS = [f'There is a mass in the {o}.' for o in ORGANS] + [f'There is no mass in the {o}.' for o in ORGANS]
KEYS = {'liver': 'mr_liver_mass', 'pancreas': 'mr_pancreatic_mass', 'kidney': 'mr_renal_mass'}


def run_aa3_det(a, meta, mdl, dev, k, n):
    from extaudit.core import VolSet
    W = os.path.join(a.root, '..')
    st = pd.read_parquet(os.path.join(a.root, os.environ.get('AA3_SET', 'aa3_set.parquet'))).iloc[k::n]   # AA3_SET: smoke tests
    if a.limit:
        st = st.head(a.limit)
    cache = os.path.join(W, 'aa3', 'jolia_text_cache.pt')             # Jolia: sentences pre-encoded (Qwen3-8B), optional
    if isinstance(getattr(mdl.ad, 'cache', None), dict) and os.path.exists(cache):
        mdl.ad.cache.update(torch.load(cache))
    tp, tn = native_prompts(mdl.ad, meta, [KEYS[o] for o in ORGANS])
    loader = DataLoader(VolSet(st), batch_size=None, num_workers=a.workers, prefetch_factor=2)
    rows, errors, t0 = [], [], time.time()
    for it, b in enumerate(loader):
        if 'error' in b:
            errors.append(b['error']); print('ERROR', b['error'], flush=True)
            continue
        hu, seg = b['hu'].to(dev), b['seg'].to(dev)
        full = hu.shape
        both = torch.isin(seg, torch.as_tensor(ORGAN_ID['liver'] + ORGAN_ID['pancreas'], device=seg.device, dtype=seg.dtype))   # 11/03: ids are lists
        zs = torch.nonzero(both.any(0).any(0))[:, 0]
        z0 = 0
        if full[2] > ZWIN_MM and len(zs):
            z0 = min(max(int(round((int(zs.min()) + int(zs.max())) / 2 - ZWIN_MM / 2)), 0), full[2] - ZWIN_MM)
        z1 = min(z0 + ZWIN_MM, full[2])
        hu, seg = hu[:, :, z0:z1].contiguous(), seg[:, :, z0:z1].contiguous()
        mdl.scan(seg, b['volume'])
        state = mdl.image(hu)[0]
        sc = mdl.sentences(state, TEXTS).cpu().numpy()
        with torch.no_grad():
            sp, sn = mdl.sim(state, tp).float().cpu().numpy(), mdl.sim(state, tn).float().cpu().numpy()
        r = dict(case=b['volume'].replace('.nii.gz', ''), z0=z0, z1=z1, z_full=int(full[2]))
        for i, o in enumerate(ORGANS):
            r[f's_{o}'], r[f's_{o}_no'] = float(sc[i]), float(sc[len(ORGANS) + i])          # free sentences
            r[f'n_{o}'], r[f'n_{o}_no'] = float(sp[i]), float(sn[i])                        # native template
        rows.append(r)
        if it % 50 == 0:
            el = time.time() - t0
            print(f'{it + 1}/{len(st)}  {el / 60:.1f} min, {el / (it + 1):.1f} s/scan', flush=True)
    out = os.path.join(a.root, 'out', a.name)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(out, f'aa3_det_shard{k}of{n}.parquet'), index=False)
    return len(rows), len(errors), t0
