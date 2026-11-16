"""--task occlusion (2026-10-25): occlusion-sensitivity maps of one positive sentence per case, the same procedure for
every encoder (model-agnostic; used for the cross-encoder qualitative figure, not as a paper metric).

Cases: a case table ($FIG4_CASES; fixed selection rule, public data only). --dataset ctrate runs the ReXGroundingCT
cases (chest crop / prompt family), --dataset merlin the AbdomenAtlas 3.0 cases (abdominal; the same 480 mm head-foot
window as --task lesion_erasure, centred on the tumour's organ, for every model).

Per case: the display plane is the axial slice through the lesion centroid. A block of 24 x 24 mm in-plane and 32 mm
head-foot, centred on that plane, is moved over the plane with a 16 mm stride; positions whose 24 x 24 mm footprint is
< 25% body (TotalSegmentator label > 0 or HU > -500 on the plane) are skipped. Occluder fill (model-independent, no ground truth used):
lung voxels (TotalSegmentator lobes 10-14, and unlabelled body voxels in a block whose most frequent label is a lobe)
take the scan's median lung HU, so a nodule or consolidation becomes normal lung and no artificial dense blob appears in
the lung; every other body voxel (labelled, or unlabelled with HU > -500) takes the benchmark's fat fill (-100 HU); both
plus N(0, noise_sd); air (unlabelled, HU <= -500) is kept. (A first version gave each voxel its label's median HU; it
left kidney tumours that TotalSegmentator labels as kidney almost unchanged.) The noise generator is
seeded by (case, position), so every model sees the same occluded volumes.
Heat = s(clean) - s(occluded) for the case's sentence (Model.sentences: each model's own sentence score, the path
--task grounding / lesion_erasure use). Writes {root}/out/<name>/occlusion/<case>.parquet (one row per position: cx, cy, drop)
and <case>.json (s_clean, z, z0, block, n positions, seconds). Resumable (finished cases are skipped).
"""
import json
import os
import time
import zlib

import numpy as np
import pandas as pd
import torch

CASES = os.environ.get('FIG4_CASES', '')   # case table (volume, finding key, sentence, plane); built from your own dataset copies, not distributed
BXY, BZ, STRIDE, MIN_BODY = 24, 32, 16, 0.25
ZWIN_MM = 480
LUNG = [10, 11, 12, 13, 14]


def flat(idx, shape, dev):
    m = torch.zeros(int(np.prod(shape)), dtype=torch.bool, device=dev)
    if len(idx):
        m[torch.as_tensor(np.asarray(idx), device=dev).long()] = True
    return m.view(shape)


def zcut(n, zs):
    if n <= ZWIN_MM or not len(zs):
        return 0, n
    z0 = min(max(int(round((int(zs.min()) + int(zs.max())) / 2 - ZWIN_MM / 2)), 0), n - ZWIN_MM)
    return z0, z0 + ZWIN_MM


def run_occlusion(a, meta, mdl, dev, k, n):
    from extaudit.core import load_vol
    W = os.path.join(a.root, '..')
    cases = pd.read_parquet(CASES)
    cases = cases[cases.dataset.isin(['rex'] if a.dataset == 'ctrate' else ['aa3'])].iloc[k::n]
    if a.limit:
        cases = cases.head(a.limit)
    sets = {'rex': 'ctrate_rex_set_v2.parquet', 'aa3': 'heal_aa3_set.parquet'}   # AbdomenAtlas 3.0: the lesion-erasure set
    mdirs = {'rex': 'rexground', 'aa3': 'heal/aa3'}
    sets = {d: pd.read_parquet(os.path.join(a.root, f)).set_index('volume') for d, f in sets.items()}
    if isinstance(getattr(mdl.ad, 'cache', None), dict):                 # Jolia: pre-encoded sentences
        for d in mdirs.values():
            p = os.path.join(W, d, 'jolia_text_cache.pt')
            if os.path.exists(p):
                mdl.ad.cache.update(torch.load(p))
    out = os.path.join(a.root, 'out', a.name, 'occlusion')
    os.makedirs(out, exist_ok=True)
    sd = float(meta['noise_sd'])
    done, errors, t0 = 0, 0, time.time()
    for c in cases.itertuples():
        if os.path.exists(os.path.join(out, f'{c.case}.json')):
            done += 1; continue
        tc = time.time()
        try:
            r = sets[c.dataset].loc[c.volume]
            hu = torch.from_numpy(load_vol(r.img)).to(dev)
            seg = torch.from_numpy(load_vol(r.seg, mask=True)).to(dev)
            L = np.load(os.path.join(W, mdirs[c.dataset], 'masks', c.volume.replace('.nii.gz', '.npz')))
            full = hu.shape
            les = flat(L[c.key], full, dev)
            z0, z1 = 0, full[2]
            if c.dataset == 'aa3':                              # head-foot window on the tumour's organ (kidneys: both)
                lab = {'liver': [5], 'pancreas': [7]}.get(c.row, [2, 3])
                z0, z1 = zcut(full[2], torch.nonzero(torch.isin(seg, torch.as_tensor(lab, device=dev, dtype=seg.dtype))
                                                     .any(0).any(0))[:, 0])
            hu, seg, les = hu[:, :, z0:z1].contiguous(), seg[:, :, z0:z1].contiguous(), les[:, :, z0:z1].contiguous()
            mdl.scan(seg, c.volume)
            # model-independent fill table: lung lobes -> median lung HU, every other label (and 0) -> fat
            table = torch.full((256,), float(meta['fills_hu']['fat']), device=dev)
            lungm = torch.isin(seg, torch.as_tensor(LUNG, device=dev, dtype=seg.dtype))
            if lungm.any():
                table[LUNG] = hu[lungm].float().median()
            z = int(torch.nonzero(les)[:, 2].float().mean().round())
            plane = (seg[:, :, z] > 0) | (hu[:, :, z] > -500)
            nz = torch.nonzero(plane)
            (xlo, ylo), (xhi, yhi) = nz.min(0).values.tolist(), nz.max(0).values.tolist()
            zs0, zs1 = max(z - BZ // 2, 0), min(z + BZ // 2, hu.shape[2])
            s0 = float(mdl.sentences(mdl.image(hu)[0], [c.text])[0])
            rows = []
            h = BXY // 2
            for cx in range(xlo + h, xhi - h + STRIDE, STRIDE):
                for cy in range(ylo + h, yhi - h + STRIDE, STRIDE):
                    xs = slice(max(cx - h, 0), min(cx + h, hu.shape[0]))
                    ys = slice(max(cy - h, 0), min(cy + h, hu.shape[1]))
                    if float(plane[xs, ys].float().mean()) < MIN_BODY:
                        continue
                    sub_h, sub_s = hu[xs, ys, zs0:zs1], seg[xs, ys, zs0:zs1]
                    lab = sub_s.long().clone()
                    nzl = lab[lab > 0]
                    maj = int(torch.mode(nzl).values) if nzl.numel() else 0
                    unl = (lab == 0) & (sub_h > -500)
                    lab[unl] = maj
                    rep = (sub_s > 0) | unl
                    g = torch.Generator(device=dev).manual_seed(zlib.crc32(f'{c.case}:{cx}:{cy}'.encode()))
                    v = hu.clone()
                    vals = table[lab] + torch.randn(lab.shape, device=dev, generator=g) * sd
                    blk = v[xs, ys, zs0:zs1]
                    blk[rep] = vals[rep].to(blk.dtype)
                    v[xs, ys, zs0:zs1] = blk
                    s = float(mdl.sentences(mdl.image(v)[0], [c.text])[0])
                    rows.append(dict(cx=cx, cy=cy, drop=s0 - s))
            pd.DataFrame(rows).to_parquet(os.path.join(out, f'{c.case}.parquet'), index=False)
            json.dump(dict(case=c.case, volume=c.volume, text=c.text, s_clean=s0, z=z, z0=z0, bxy=BXY, bz=BZ,
                           stride=STRIDE, n_pos=len(rows), seconds=time.time() - tc),
                      open(os.path.join(out, f'{c.case}.json'), 'w'))
            done += 1
            print(f'{c.case}: {len(rows)} positions, {time.time() - tc:.0f} s', flush=True)
        except Exception as ex:                                            # noqa: BLE001
            errors += 1
            print('ERROR', c.case, repr(ex)[:300], flush=True)
    return done, errors, t0
