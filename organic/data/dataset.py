"""Volumes, organ maps, labels and report sentences for stage A.

Per item (CPU, in DataLoader workers):
  1 mm RAS NIfTI + TotalSegmentator mask -> crop centred on the region's anatomy (chest: lungs + heart;
  abdomen: liver, spleen, kidneys, pancreas; random jitter in training) -> 2 mm grid (COLIPRI's spacing)
  -> 192^3 volume in HU and an organ-group map (vocab.ORGANS ids, 255 = background).
Erasure, intensity normalisation and mirroring happen on the GPU in the training step (erasure needs HU).
"""
import os
import re
import zlib

import nibabel as nib
from .volio import read_image, read_mask
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from ..config import abspath, read_manifest
from ..vocab import ORGANS, ORGAN_ID, BACKGROUND, FINDING_NAMES

SIZE, SPACING = 192, 2.0
CENTER_ORGANS = {'ctrate': ['lung_left', 'lung_right', 'heart'],
                 'merlin': ['liver', 'spleen', 'kidney_left', 'kidney_right', 'pancreas']}
CENTER_ORGANS['radchest'] = CENTER_ORGANS['ctrate']      # RAD-ChestCT (external chest test set)

_LUT = np.full(256, BACKGROUND, np.uint8)
for _o, _ids in ORGANS.items():
    _LUT[_ids] = ORGAN_ID[_o]

# Merlin covers abdomen and pelvis (median 467 mm), longer than the 384 mm window. Head to foot, the window is
# centred on the span from the top of the liver to the bottom of the pelvic organs (median 390 mm), so both the upper
# abdomen and the bladder/prostate stay in view; left-right and front-back keep the organ centroid.
SPAN_ORGANS = {'merlin': (['liver'], ['pelvic_organs'])}


def crop_center(organ1, dataset, step=4):
    """Crop centre in 1 mm voxel coordinates from the organ-group map (computed on a `step` mm subsample)."""
    sub = organ1[::step, ::step, ::step]
    pts = np.argwhere(np.isin(sub, [ORGAN_ID[o] for o in CENTER_ORGANS[dataset]]))
    center = (pts.mean(0) * step) if len(pts) else np.array(organ1.shape) / 2
    if dataset in SPAN_ORGANS:
        z = []
        for group in SPAN_ORGANS[dataset]:
            zz = np.where(np.isin(sub, [ORGAN_ID[o] for o in group]).any(axis=(0, 1)))[0]
            z.append(zz)
        if all(len(zz) for zz in z):
            lo, hi = min(zz.min() for zz in z), max(zz.max() for zz in z)
            center[2] = (lo + hi) / 2 * step + step / 2   # axis 2 = superior-inferior (RAS)
    return center


_SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def split_sentences(report, max_n=40):
    """Fallback sentence splitter (used until parsed claims are available)."""
    report = re.sub(r'\s+', ' ', str(report or '')).strip()
    report = re.sub(r'^(FINDINGS|Findings)\s*:\s*', '', report)
    out = [s.strip() for s in _SENT.split(report) if len(s.strip()) > 8]
    return out[:max_n]


def _crop_bounds(center, shape, size):
    lo = [int(round(c - size / 2)) for c in center]
    return [(l, l + size) for l in lo]


def _crop_pad(arr, bounds, fill):
    out = np.full([b - a for a, b in bounds], fill, dtype=arr.dtype)
    src = tuple(slice(max(a, 0), min(b, n)) for (a, b), n in zip(bounds, arr.shape))
    dst = tuple(slice(s.start - a, s.stop - a) for s, (a, _) in zip(src, bounds))
    out[dst] = arr[src]
    return out


class CTDataset(Dataset):
    def __init__(self, cfg, frames, train=True, sentences=None):
        """frames: list of manifest DataFrames (already filtered to one split). sentences: optional
        {volume -> list of claim dicts} from parse_reports.py."""
        self.cfg, self.train = cfg, train
        self.df = pd.concat(frames, ignore_index=True)
        self.labels = self.df[[f'label_{f}' for f in FINDING_NAMES]].to_numpy(np.float32)
        self.sentences = sentences or {}
        self.jitter_mm = cfg.data.crop_jitter_mm if train else 0.0

    def __len__(self):
        return len(self.df)

    def _load(self, r):
        img = read_image(abspath(self.cfg, r.dataset, r.img_rel))
        seg = read_mask(abspath(self.cfg, r.dataset, r.seg_rel))
        return img, seg

    def _from_cache(self, r):
        p = os.path.join(self.cfg.paths.work_root, 'cache', r.dataset, f'{r.volume}.npz')
        if not (self.cfg.data.get('use_cache', False) and os.path.exists(p)):
            return None
        z = np.load(p)
        img, organ = z['img'], z['organ']
        m = (img.shape[0] - SIZE) // 2                              # jitter room in voxels
        if self.train and m:
            o = np.random.default_rng().integers(0, 2 * m + 1, 3)
        else:
            o = np.array([m, m, m])
        sl = tuple(slice(int(a), int(a) + SIZE) for a in o)
        vol = torch.from_numpy(img[sl].astype(np.float32))[None]
        return vol, torch.from_numpy(np.ascontiguousarray(organ[sl]))

    def __getitem__(self, i):
        r = self.df.iloc[i]
        cached = self._from_cache(r)
        if cached is not None:
            vol, organ = cached
            claims = self.sentences.get(r.volume)
            if claims is None:
                claims = [{'text': s} for s in split_sentences(r.report)]
            return {'img': vol.to(torch.float16), 'organ': organ, 'labels': torch.from_numpy(self.labels[i]),
                    'claims': claims, 'dataset': r.dataset, 'volume': r.volume, 'index': i}
        img, seg = self._load(r)
        organ1 = _LUT[seg]                                        # 1 mm organ map
        # centre on the region's anatomy (computed on a 4 mm subsample for speed)
        center = crop_center(organ1, r.dataset)
        if self.jitter_mm:
            rng = np.random.default_rng()
            center = center + rng.uniform(-self.jitter_mm, self.jitter_mm, 3)
        # 1 mm crop of 2*SIZE, then 2 mm by trilinear (image) / strided nearest (organs)
        b1 = _crop_bounds(center, img.shape, int(SIZE * SPACING))
        img_c = _crop_pad(img, b1, -1000.0)
        org_c = _crop_pad(organ1, b1, BACKGROUND)
        t = torch.from_numpy(img_c)[None, None]
        vol = F.interpolate(t, size=(SIZE, SIZE, SIZE), mode='trilinear', align_corners=False)[0]
        organ = torch.from_numpy(np.ascontiguousarray(org_c[1::2, 1::2, 1::2]))
        claims = self.sentences.get(r.volume)
        if claims is None:
            claims = [{'text': s} for s in split_sentences(r.report)]
        return {'img': vol.to(torch.float16), 'organ': organ, 'labels': torch.from_numpy(self.labels[i]),
                'claims': claims, 'dataset': r.dataset, 'volume': r.volume, 'index': i}


def val_indices(ds, n, rank, world, seed=0):
    """Indices of a fixed validation slice with an equal share of every dataset (the frames are concatenated per
    dataset, so the first n rows would all be CT-RATE), split across ranks."""
    groups = [np.where(ds.df.dataset.values == d)[0] for d in pd.unique(ds.df.dataset.values)]
    per = max(1, n // max(1, len(groups)))
    rng = np.random.default_rng(seed)
    pick = np.sort(np.concatenate([rng.choice(g, size=min(per, len(g)), replace=False) for g in groups if len(g)]))
    return [int(i) for i in pick[rank::world]]


def collate(items):
    out = {k: torch.stack([it[k] for it in items]) for k in ('img', 'organ', 'labels')}
    for k in ('claims', 'dataset', 'volume', 'index', 'report'):
        out[k] = [it.get(k, '') for it in items]
    return out


class MixedBatchSampler(Sampler):
    """Per-rank batches mixing datasets by weight; a fraction of each batch is label-matched pairs
    (positive and negative for one finding, identical on every other labelled finding of that dataset)."""

    def __init__(self, ds, batch_size, dataset_weights, pair_frac, rank, world, seed=0, steps_per_epoch=None):
        self.ds, self.bs, self.pair_frac = ds, batch_size, pair_frac
        self.rank, self.world, self.seed, self.epoch = rank, world, seed, 0
        self.groups = {d: np.where(ds.df.dataset.values == d)[0] for d in dataset_weights if (ds.df.dataset.values == d).any()}
        w = np.array([dataset_weights[d] for d in self.groups], float)
        self.names, self.w = list(self.groups), w / w.sum()
        n = sum(len(v) for v in self.groups.values())
        self.steps = steps_per_epoch or max(1, n // (batch_size * world))
        self.pairs = self._build_pairs()

    def _build_pairs(self):
        """finding id -> list of (pos_index, [matched neg indices])."""
        L = self.ds.labels
        pairs = {}
        for d, idx in self.groups.items():
            cols = np.where(~np.isnan(L[idx]).all(0))[0]
            if len(cols) < 2:
                continue
            sub = np.nan_to_num(L[idx][:, cols], nan=-1).astype(np.int8)
            for j, f in enumerate(cols):
                others = np.delete(sub, j, axis=1)
                keys = [zlib.crc32(row.tobytes()) for row in others]
                key_arr = np.array(keys)
                pos = np.where(sub[:, j] == 1)[0]
                neg = np.where(sub[:, j] == 0)[0]
                by_key = pd.Series(neg).groupby(key_arr[neg]).apply(list).to_dict()
                lst = [(idx[p], [idx[q] for q in by_key[key_arr[p]]]) for p in pos if key_arr[p] in by_key]
                if len(lst) >= 5:
                    pairs.setdefault(int(f), []).extend(lst)
        return pairs

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1000 * self.epoch + self.rank)
        exp_pairs = self.bs * self.pair_frac / 2 if self.pairs else 0.0   # expected pairs per batch
        fs = list(self.pairs)
        for _ in range(self.steps):
            batch = []
            n_pairs = int(exp_pairs) + int(rng.random() < exp_pairs - int(exp_pairs))
            n_pairs = min(n_pairs, self.bs // 2)
            for _ in range(n_pairs):
                f = fs[rng.integers(len(fs))]
                p, negs = self.pairs[f][rng.integers(len(self.pairs[f]))]
                batch += [int(p), int(negs[rng.integers(len(negs))])]
            while len(batch) < self.bs:
                d = self.names[rng.choice(len(self.names), p=self.w)]
                batch.append(int(self.groups[d][rng.integers(len(self.groups[d]))]))
            yield batch


def load_manifests(cfg, split):
    frames = []
    ds = cfg.data.datasets
    if split == 'val' and cfg.data.get('val_datasets'):
        ds = [d for d in ds if d in cfg.data.val_datasets]
    for d in ds:
        m = read_manifest(cfg, d)                                # minus <dataset>_exclude.csv, if present
        m = m[m.split == split]
        if split == 'train' and cfg.data.get('one_recon_per_scan', True) and d == 'ctrate':
            m = m.groupby('scan_id', group_keys=False).head(1)   # sampler still sees every scan once
        if cfg.data.get('limit'):
            m = m.head(int(cfg.data.limit))
        frames.append(m)
    return frames


def load_sentences(cfg):
    p = os.path.join(cfg.paths.work_root, 'parsed', 'claims.parquet')
    if not os.path.exists(p):
        return {}
    df = pd.read_parquet(p)
    return {v: g.drop(columns='volume').to_dict('records') for v, g in df.groupby('volume')}
