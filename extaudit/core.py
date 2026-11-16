"""External-encoder audit: the stage-A ladder for released 3D CT image-text encoders, each in its own python env.

--task ladder (default): the same perturbations as our models (clean, left-right mirror, each organ group erased with
fat or soft-tissue fill + N(0, 15) HU, mask dilated by 2 mm) are applied to the 1 mm RAS volume BEFORE the model's own
preprocessing, so every model sees its native input geometry. Scores are the model's own zero-shot rule: similarity to
the positive prompt minus the negative prompt (monotone in its two-prompt softmax). Laterality: a report sentence naming
one side vs the same sentence with the side swapped, on the clean and the mirrored scan.

--task grounding: ReXGroundingCT lesions (scripts/data/prepare_rexgrounding.py). Per report finding, the drawn lesion is
"healed" (voxels set to the scan's own median lung HU + N(0, 15)), and, as control, the same-shaped region mirrored to
the other side of the lungs (only where it overlaps no drawn lesion). Saved: the finding sentence's score and all
finding scores (Delta_f) on clean / healed lesion / healed control. Grounded = the lesion's healing lowers its own
sentence more than the control's.

Inputs come from scripts/data/export_audit_set.py (rows = evaluate_stageA's test order). Only numpy / torch / nibabel /
pandas are imported here (adapters may import more), so any model env can run it. Metrics: scripts/metrics/extaudit_metrics.py.

    CUDA_VISIBLE_DEVICES=0 <model env>/bin/python -m extaudit.core --model ctclip --name CT-CLIP --dataset ctrate \
        --root <work_root>/extaudit --shard 0/2
Writes {root}/out/{name}/{dataset}_shard{k}of{n}.npz (grounding: {root}/out/{name}/grounding_shard{k}of{n}.parquet).
"""
import argparse
import importlib
import json
import os
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LUNG_IDS = [10, 11, 12, 13, 14]


def load_vol(path, mask=False):
    """NIfTI or the simple .bin format (b"ZST0"/b"RAW0", 3 x uint32 shape, int16 C order; 1 mm RAS, no header)."""
    if path.endswith('.bin'):
        import struct
        with open(path, 'rb') as f:
            magic, shape, data = f.read(4), struct.unpack('3I', f.read(12)), f.read()
        if magic == b'ZST0':
            import zstandard
            data = zstandard.ZstdDecompressor().decompress(data)
        a = np.frombuffer(data, dtype=np.int16).reshape(shape)
    else:
        a = np.asarray(nib.load(path).dataobj)
    return a.astype(np.uint8) if mask else a.astype(np.float32)


class VolSet(Dataset):
    def __init__(self, rows, mask_dir=None):
        self.rows, self.mask_dir = rows.reset_index(drop=True), mask_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        try:
            hu, seg = load_vol(r.img), load_vol(r.seg, mask=True)
            les = dict(np.load(os.path.join(self.mask_dir, r.volume.replace('.nii.gz', '.npz')))) if self.mask_dir else {}
        except Exception as ex:                                       # noqa: BLE001
            return {'index': int(r['index']), 'error': str(ex)[:200]}
        return {'index': int(r['index']), 'hu': torch.from_numpy(hu), 'seg': torch.from_numpy(seg),
                'lat': r.get('lat_pairs', '[]'), 'les': les, 'volume': r.volume}


def conditions(meta, dataset, fills):
    out = [('clean', None, None), ('mirror', None, None)]
    for name, ids in meta['eval_organs'][dataset]:
        for f in fills:
            out.append((f'{name}@{f}', ids, f))
    return out


def perturb(hu, seg, cond, meta, gen):
    """hu (X, Y, Z) float cuda HU, seg uint8 cuda -> perturbed HU, number of erased voxels."""
    name, ids, fill = cond
    if name == 'clean':
        return hu, 0
    if name == 'mirror':
        return torch.flip(hu, dims=[0]), 0                           # axis 0 = left-right (RAS)
    m = torch.isin(seg, torch.as_tensor(ids, device=seg.device, dtype=seg.dtype)).float()[None, None]
    m = F.max_pool3d(m, 5, stride=1, padding=2)[0, 0] > 0           # 2 mm dilation (= 1 voxel at 2 mm in our models)
    n = int(m.sum())
    out = hu.clone()
    if n:
        noise = torch.randn(n, device=hu.device, generator=gen) * meta['noise_sd']
        if fill in meta['fills_hu']:                                  # trained fills: fat -100 HU, soft tissue 40 HU
            out[m] = meta['fills_hu'][fill] + noise
        elif fill in UNSEEN_FILLS:                                   # fills no training step uses (supplement S8)
            out[m] = UNSEEN_FILLS[fill](hu, m) + noise
        else:
            raise ValueError(f'unknown fill {fill}')
    return out, n


def _box(m, pad):
    idx = torch.nonzero(m)
    lo = (idx.min(0).values - pad).clamp(min=0)
    hi = torch.minimum(idx.max(0).values + pad + 1, torch.as_tensor(m.shape, device=m.device))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _blur_fill(hu, m, sigma=6.0):
    """Organ voxels replaced by the scan smoothed with a 6 mm Gaussian (1 mm grid): texture and small lesions vanish,
    the organ's own density and outline stay. Computed on the organ's bounding box only."""
    r = int(3 * sigma)
    sl = _box(m, r)
    x = hu[sl][None, None].float()
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=hu.device, dtype=torch.float32) / sigma) ** 2)
    k = k / k.sum()
    for d in range(3):                                               # separable 3D Gaussian, replicate padding
        shape = [1, 1, 1, 1, 1]; shape[2 + d] = 2 * r + 1
        pad = [0, 0, 0, 0, 0, 0]; pad[2 * (2 - d)] = pad[2 * (2 - d) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode='replicate'), k.view(shape))
    sm = torch.zeros_like(hu)
    sm[sl] = x[0, 0].to(hu.dtype)
    return sm[m]


UNSEEN_FILLS = {
    'water': lambda hu, m: torch.zeros(int(m.sum()), device=hu.device, dtype=hu.dtype),   # 0 HU, never trained
    'median': lambda hu, m: torch.full((int(m.sum()),), float(hu[m].median()), device=hu.device, dtype=hu.dtype),
    'blur': _blur_fill,
}


class Model:
    """Adapter + its text prompts; one place for the score rule."""
    def __init__(self, ad, meta, dataset, prompts=True):
        self.ad = ad
        fam = 'ct_' if dataset in ('ctrate', 'radchest', 'grounding') else 'mr_'
        self.fidx = [j for j, f in enumerate(meta['findings']) if f.startswith(fam)]
        self.nf = len(meta['findings'])
        if prompts:                                     # retrieval needs no finding prompts (chest models have none for mr_)
            pos, neg = zip(*[ad.prompts(meta['findings'][j], meta) for j in self.fidx])
            with torch.no_grad():
                self.tp, self.tn = ad.encode_text(list(pos)), ad.encode_text(list(neg))
        self.sim = getattr(ad, 'similarity', None) or (lambda z, t: t @ z)

    def scan(self, seg, volume=None):
        if hasattr(self.ad, 'set_scan'):
            self.ad.set_scan(seg)
        if volume is not None and hasattr(self.ad, 'set_volume'):    # e.g. Merlin: restore the scan's original FOV
            self.ad.set_volume(volume)

    @torch.no_grad()
    def image(self, hu):
        return self.ad.encode_image(hu)

    @torch.no_grad()
    def deltas(self, state):
        d = np.full(self.nf, np.nan, np.float32)
        d[self.fidx] = (self.sim(state, self.tp) - self.sim(state, self.tn)).float().cpu().numpy()
        return d

    @torch.no_grad()
    def sentences(self, state, sents):
        return self.sim(state, self.ad.encode_text(list(sents))).float()


def run_ladder(a, meta, mdl, rows, dev, k, n):
    conds = conditions(meta, a.dataset, a.fills.split(','))
    res = {c[0]: {'delta': [], 'feat': [], 'nvox': []} for c in conds}
    idx, lat, errors = [], [], []
    gen = torch.Generator(device=dev).manual_seed(k)
    loader = DataLoader(VolSet(rows), batch_size=None, num_workers=a.workers, prefetch_factor=2)
    t0 = time.time()
    for it, b in enumerate(loader):
        if 'error' in b:
            errors.append((b['index'], b['error'])); print('ERROR', b['index'], b['error'], flush=True)
            continue
        hu, seg = b['hu'].to(dev), b['seg'].to(dev)
        mdl.scan(seg, b['volume'])
        states = {}
        for c in conds:
            v, nv = perturb(hu, seg, c, meta, gen)
            if c[0] == 'mirror':              # mask-guided adapters (crop centre, organ windows) see the mirrored mask
                mdl.scan(torch.flip(seg, dims=[0]))
            st, feat = mdl.image(v)
            if c[0] == 'mirror':
                mdl.scan(seg, b['volume'])
            res[c[0]]['delta'].append(mdl.deltas(st))
            res[c[0]]['feat'].append(feat.float().cpu().numpy().astype(np.float16))
            res[c[0]]['nvox'].append(nv)
            if c[0] in ('clean', 'mirror'):
                states[c[0]] = st
        pairs = json.loads(b['lat'])
        if pairs:
            sents = [p for p, _ in pairs] + [q for _, q in pairs]
            m_ = len(pairs)
            s0, s1 = mdl.sentences(states['clean'], sents), mdl.sentences(states['mirror'], sents)
            ok0, ok1 = (s0[:m_] > s0[m_:]).float(), (s1[m_:] > s1[:m_]).float()
            lat.append([b['index'], m_, float(ok0.sum()), float(ok1.sum()), float((ok0 * ok1).sum())])
        idx.append(b['index'])
        if it % 50 == 0:
            el = time.time() - t0
            print(f'{it + 1}/{len(rows)}  {el / 60:.1f} min, {el / (it + 1):.1f} s/volume', flush=True)
    out = os.path.join(a.root, 'out', a.name)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f'{a.dataset}_shard{k}of{n}.npz'), index=np.asarray(idx),
                        lat=np.asarray(lat, np.float32).reshape(-1, 5), conds=np.asarray([c[0] for c in conds]),
                        errors=np.asarray([json.dumps(e) for e in errors]),
                        **{f'delta__{c}': np.stack(r['delta']) for c, r in res.items() if r['delta']},
                        **{f'feat__{c}': np.stack(r['feat']) for c, r in res.items() if r['feat']},
                        **{f'nvox__{c}': np.asarray(r['nvox']) for c, r in res.items() if r['delta']})
    return len(idx), len(errors), t0


def mirror_mask(les, lung):
    """Lesion mask reflected left-right about the lungs' midline (same shape and size, other side)."""
    xs = torch.where(lung.any(dim=2).any(dim=1))[0]
    x0, x1 = int(xs.min()), int(xs.max())
    out = torch.zeros_like(les)
    src = torch.arange(les.shape[0], device=les.device)
    dst = (x0 + x1) - src
    ok = (dst >= 0) & (dst < les.shape[0])
    out[dst[ok]] = les[src[ok]]
    return out


def run_grounding(a, meta, mdl, dev, k, n):
    gt = pd.read_parquet(os.path.join(a.root, '..', 'rexground', 'findings.parquet'))
    st = pd.read_parquet(os.path.join(a.root, a.rex_set))
    if a.rex_exclude:                                   # e.g. v2 run on the scans the old set lacked only
        st = st[~st.volume.isin(set(pd.read_parquet(os.path.join(a.root, a.rex_exclude), columns=['volume']).volume))]
    st = st[st.volume.isin(set(gt.volume))].iloc[k::n]
    if a.limit:
        st = st.head(a.limit)
    by_vol = {v: g for v, g in gt.groupby('volume')}
    cache = os.path.join(a.root, '..', 'rexground', 'jolia_text_cache.pt')   # scripts/data/rex_jolia_text_cache.py
    if a.model == 'jolia' and isinstance(getattr(mdl.ad, 'cache', None), dict) and os.path.exists(cache):
        mdl.ad.cache.update(torch.load(cache))
    loader = DataLoader(VolSet(st, os.path.join(a.root, '..', 'rexground', 'masks')), batch_size=None,
                        num_workers=a.workers, prefetch_factor=2)
    gen = torch.Generator(device=dev).manual_seed(k)
    rows, errors, t0 = [], [], time.time()
    for it, b in enumerate(loader):
        if 'error' in b:
            errors.append(b['error']); print('ERROR', b['error'], flush=True)
            continue
        hu, seg = b['hu'].to(dev), b['seg'].to(dev)
        mdl.scan(seg, b['volume'])
        lung = torch.isin(seg, torch.as_tensor(LUNG_IDS, device=dev, dtype=seg.dtype))
        g = by_vol[b['volume']]
        masks = {}
        for r in g.itertuples():
            key = f'k{r.k}'
            if key in b['les']:
                m = torch.zeros(hu.numel(), dtype=torch.bool, device=dev)
                m[torch.as_tensor(b['les'][key], device=dev)] = True
                masks[r.k] = m.view(hu.shape)
        if not masks or not lung.any():
            continue
        allles = torch.stack(list(masks.values())).any(0)
        allles_d = F.max_pool3d(allles.float()[None, None], 7, stride=1, padding=3)[0, 0] > 0
        fill = float(hu[lung & ~allles_d].median()) if (lung & ~allles_d).any() else -850.0
        s_clean = mdl.image(hu)[0]
        d_clean = mdl.deltas(s_clean)
        texts = list(g.text)
        sc_clean = mdl.sentences(s_clean, texts).cpu().numpy()
        for j, r in enumerate(g.itertuples()):
            if r.k not in masks:
                continue
            les = masks[r.k]
            ctrl = mirror_mask(les, lung)
            nl = int(les.sum())
            overlap = float((ctrl & allles_d).sum()) / max(nl, 1)
            ctrl_lung = float((ctrl & lung).sum()) / max(int(ctrl.sum()), 1)
            row = dict(volume=b['volume'], k=r.k, category=r.category, n_vox=nl, ctrl_overlap=overlap,
                       ctrl_in_lung=ctrl_lung, fill=fill, s_clean=float(sc_clean[j]), d_clean=d_clean.tolist())
            for name, m in (('les', les), ('ctrl', ctrl)):
                v = hu.clone()
                cnt = int(m.sum())
                if cnt:
                    v[m] = fill + torch.randn(cnt, device=dev, generator=gen) * meta['noise_sd']
                s = mdl.image(v)[0]
                row[f's_{name}'] = float(mdl.sentences(s, [r.text])[0])
                row[f'd_{name}'] = mdl.deltas(s).tolist()
            rows.append(row)
        if it % 50 == 0:
            el = time.time() - t0
            print(f'{it + 1}/{len(st)}  {el / 60:.1f} min, {el / (it + 1):.1f} s/scan', flush=True)
    out = os.path.join(a.root, 'out', a.name)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(out, f'{a.out_tag}_shard{k}of{n}.parquet'), index=False)
    return len(rows), len(errors), t0


@torch.no_grad()
def encode_texts(mdl, texts, bs):
    """All texts through the adapter's text encoder, kept on the CPU (Merlin: ~140k sentences)."""
    return torch.cat([mdl.ad.encode_text(list(texts[i:i + bs])).float().cpu() for i in range(0, len(texts), bs)])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='adapter module in extaudit/models')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--name', required=True)
    ap.add_argument('--task', default='ladder', choices=['ladder', 'grounding', 'lesion_erasure', 'aa3_det', 'chest',
                                                     'occlusion', 'side'])
    ap.add_argument('--dataset', default='ctrate')
    ap.add_argument('--root', required=True, help='{work_root}/extaudit')
    ap.add_argument('--fills', default='fat,soft')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--first', type=int, default=0, help='ladder: only the first N scans of the set (before sharding)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--rex_set', default='ctrate_rex_set.parquet', help='grounding: scan set under --root')
    ap.add_argument('--rex_exclude', default='', help='grounding: skip scans listed in this set (under --root)')
    ap.add_argument('--out_tag', default='grounding', help='grounding: shard file prefix in out/<name>/')
    a = ap.parse_args()
    k, n = map(int, a.shard.split('/'))
    if a.task == 'aa3_det' or (a.task == 'lesion_erasure' and os.environ.get('LE_SET') == 'aa3'):   # abdominal crop + prompts
        a.dataset = 'merlin'
    if a.task == 'lesion_erasure' and os.environ.get('LE_SET') == 'rex':     # ReXGroundingCT: chest crop + prompts
        a.dataset = 'ctrate'
    if a.task == 'side':                                 # 11/03: side accuracy on unflipped scans (extaudit/side.py)
        a.dataset = {'ctrate': 'ctrate', 'merlin': 'merlin', 'rex': 'ctrate', 'aa3': 'merlin'}[os.environ['SIDE_SET']]
    meta = json.load(open(os.path.join(a.root, 'meta.json')))
    dev = torch.device('cuda')
    ad = importlib.import_module(f'extaudit.models.{a.model}').Adapter(a.ckpt, dev)
    if hasattr(ad, 'set_dataset'):                      # e.g. our crop: chest (CT-RATE, RAD-ChestCT) or abdomen (Merlin)
        ad.set_dataset('ctrate' if a.task == 'grounding' else a.dataset)
    mdl = Model(ad, meta, 'grounding' if a.task == 'grounding' else a.dataset, prompts=True)
    if a.task == 'grounding':
        done, nerr, t0 = run_grounding(a, meta, mdl, dev, k, n)
    elif a.task == 'aa3_det':                           # AbdomenAtlas 3.0 tumour detection (2026-10-27)
        from extaudit.aa3_det import run_aa3_det
        done, nerr, t0 = run_aa3_det(a, meta, mdl, dev, k, n)
    elif a.task == 'chest':                             # lobe localisation and density edits (CT-RATE v2)
        from extaudit.chest import run_chest
        done, nerr, t0 = run_chest(a, meta, mdl, dev, k, n)
    elif a.task == 'occlusion':                         # cross-encoder occlusion maps (qualitative figure)
        from extaudit.occlusion import run_occlusion
        done, nerr, t0 = run_occlusion(a, meta, mdl, dev, k, n)
    elif a.task == 'side':                              # side accuracy on unflipped scans (2026-11-03)
        from extaudit.side import run_side
        done, nerr, t0 = run_side(a, meta, mdl, dev, k, n)
    elif a.task == 'lesion_erasure':                    # LE on ReXGroundingCT / AbdomenAtlas 3.0 (LE_SET)
        from extaudit.lesion_erasure import run_lesion_erasure
        done, nerr, t0 = run_lesion_erasure(a, meta, mdl, dev, k, n)
    else:
        rows = pd.read_parquet(os.path.join(a.root, f'{a.dataset}_set.parquet'))
        rows = (rows.head(a.first) if a.first else rows).iloc[k::n]
        if a.limit:
            rows = rows.head(a.limit)
        done, nerr, t0 = run_ladder(a, meta, mdl, rows, dev, k, n)
    print(f'DONE {a.name} {a.task} {a.dataset} shard {k}/{n}: {done} items, {nerr} errors, '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
