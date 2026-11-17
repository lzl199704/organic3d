"""Alignment-ladder audit of NV-Reason-CT (released 3D CT VLM, Qwen3.5-4B + Primus/COLIPRI ViT, 13,824 tokens).

Scored with the same metric code as our stage-B model (organic.vlm.evaluate.metrics): answer AUROC, LLM-input probe,
Use, organ-erasure effect, mirror, matched-pair AUROC, wrong-patient AUROC, and report clinical F1 / Ret.

The model's own preprocessing is reproduced (LPS, 2 mm, its anatomy-aware 192^3 crop), applied to the image and the
TotalSegmentator mask together so organ erasure happens in exactly the geometry the model sees. Per condition the
13,824-token image prefix is encoded once and its cache reused for every question.

    torchrun --nproc_per_node 4 -m organic.baselines.nv_reason_ct --env configs/env/local.yaml --datasets merlin,ctrate --limit 500 --gen 100
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from ..data.interventions import erase, organ_mask
from ..config import abspath, load_config, read_manifest
from ..data.dataset import _LUT
from ..encoder.evaluate import EVAL_ORGANS, cond_name, organ_ids
from ..vlm.evaluate import FAM, metrics
from ..utils import gather_padded, setup_dist
from ..data.volio import image_space
from ..vocab import FINDINGS, FINDING_NAMES

MODEL_ID = 'nvidia/NV-Reason-CT'
REGION = {'merlin': 'abdomen', 'ctrate': 'chest'}
REPORT = {'merlin': 'Write a structured abdominal CT report.', 'ctrate': 'Write a structured chest CT report.'}


def nv_question(f):
    return f'Is {FINDINGS[f][2]} present in this CT? Answer only Yes or No.'


class NVPrep:
    """NV-Reason-CT preprocessing only (CPU): its LPS / 2 mm / anatomy crop, applied to image and mask together.
    With cache_root set, load() reads precomputed {cache_root}/{dataset}/{volume}.npz inputs."""

    def __init__(self, cache_root=None):
        from transformers import AutoProcessor
        self.cache_root = cache_root
        self.proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.il = self.proc.image_processor_3d
        self.Crop = sys.modules[type(self.il).__module__].AnatomySpatialCropd
        from monai.transforms import Compose, EnsureTyped, LoadImaged, Orientationd, SpatialPadd, Spacingd
        k = ['image', 'seg']
        px = self.il.pixdim
        rest = [
            EnsureTyped(keys=k, data_type='tensor', dtype=torch.float),
            Orientationd(keys=k, axcodes='LPS'),
            Spacingd(keys=k, pixdim=px, mode=('bilinear', 'nearest'), dtype=torch.float,
                     min_pixdim=np.array(px) * 0.9, max_pixdim=np.array(px) * 1.1),
            SpatialPadd(keys=['image'], spatial_size=self.il.spatial_size, value=-1000),
            SpatialPadd(keys=['seg'], spatial_size=self.il.spatial_size, value=0),
        ]
        self.tf = Compose([LoadImaged(keys=k, ensure_channel_first=True, image_only=True)] + rest)
        self.tf_arrays = Compose(rest)         # .bin volumes, already loaded as MetaTensors

    def cached(self, dataset, volume):
        p = os.path.join(self.cache_root, dataset, f'{volume}.npz') if self.cache_root else None
        if p and os.path.exists(p):
            z = np.load(p)
            return torch.from_numpy(z['hu'].astype(np.float32))[None], torch.from_numpy(z['organ'])
        return None

    def load(self, img_path, seg_path, region, raw_space=False):
        """-> HU volume (1, Z, Y, X) and organ-group map (Z, Y, X), both in the model's input geometry.
        raw_space: img_path is the original scan; it is first resampled onto the 1 mm mask grid (e.g. when the env layout reads Merlin's original scans)."""
        if raw_space:
            import nibabel as nib
            from monai.data import MetaTensor
            from ..data.volio import sample_raw_on_mask_grid
            snii = nib.load(seg_path); sarr = np.asarray(snii.dataobj).astype(np.float32)
            img = sample_raw_on_mask_grid(img_path, snii.affine, sarr.shape, [0, 0, 0], 1.0, 0.0, sarr.shape)
            aff = torch.as_tensor(snii.affine)
            d = self.tf_arrays({'image': MetaTensor(torch.from_numpy(img)[None], affine=aff),
                                'seg': MetaTensor(torch.from_numpy(sarr)[None], affine=aff)})
        elif img_path.endswith('.bin'):
            from monai.data import MetaTensor
            from ..data.volio import read_image, read_mask, ras_affine_1mm
            aff = torch.as_tensor(ras_affine_1mm())
            d = self.tf_arrays({'image': MetaTensor(torch.from_numpy(read_image(img_path))[None], affine=aff),
                                'seg': MetaTensor(torch.from_numpy(read_mask(seg_path).astype(np.float32))[None], affine=aff)})
        else:
            d = self.tf({'image': img_path, 'seg': seg_path})
        img, seg = d['image'].as_subclass(torch.Tensor), d['seg'].as_subclass(torch.Tensor)
        crop = self.Crop(keys=['image'], roi_size=self.il.spatial_size, pixdim=self.il.pixdim)
        sl = crop_slices(crop, img, region)
        img, seg = img[sl], seg[sl]
        img = img.permute(0, 3, 2, 1).contiguous()
        seg = seg.permute(0, 3, 2, 1)[0].round().long().clamp(0, len(_LUT) - 1)
        organ = torch.from_numpy(_LUT[seg.numpy()])
        return img, organ

    def normalise(self, hu):
        return self.il._apply_normalization(hu, self.il._resolve_normalize_mode(self.il.normalize_mode))


class NVReason(NVPrep):
    def __init__(self, device, cache_root=None):
        super().__init__(cache_root)
        from transformers import AutoModelForImageTextToText
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16,
                                                                 attn_implementation='sdpa').eval().to(device)
        self.tok = self.proc.tokenizer
        self.img_tok = '<|image_pad|>'
        self.img_id = self.tok.convert_tokens_to_ids(self.img_tok)
        self.n_img = int(np.prod(self.il.final_grid_size))
        self.grid = torch.tensor([self.il.final_grid_size], device=device)
        self.yes = self.tok('Yes', add_special_tokens=False).input_ids[0]
        self.no = self.tok('No', add_special_tokens=False).input_ids[0]

    def prompt(self, text):
        msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}]
        return self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _ids(self, s):
        return torch.tensor([self.tok(s.replace(self.img_tok, self.img_tok * self.n_img), add_special_tokens=False).input_ids],
                            device=self.device)

    @torch.no_grad()
    def prefix(self, pv):
        """Encode the image prefix once; returns (cache, pooled visual features)."""
        full = self.prompt('Q')
        pre = full[:full.index('<|vision_end|>') + len('<|vision_end|>')]
        ids = self._ids(pre)
        mm = (ids == self.img_id).int()
        out = self.model(input_ids=ids, pixel_values=pv[None].to(self.device, torch.bfloat16), image_grid_thw=self.grid,
                         mm_token_type_ids=mm, use_cache=True)
        feats = self.model.model.get_image_features(pv[None].to(self.device, torch.bfloat16), self.grid).pooler_output[0]
        return out.past_key_values, feats.float().mean(0)

    @torch.no_grad()
    def p_yes(self, cache, questions):
        out = []
        for q in questions:
            full = self.prompt(q)
            suf = full[full.index('<|vision_end|>') + len('<|vision_end|>'):]
            ids = self._ids(suf)
            o = self.model(input_ids=ids, past_key_values=copy.deepcopy(cache), use_cache=True)
            lg = o.logits[0, -1].float()
            out.append(torch.softmax(torch.stack([lg[self.yes], lg[self.no]]), 0)[0])
        return torch.stack(out)

    @torch.no_grad()
    def generate(self, pv, text, max_new_tokens=512):
        ids = self._ids(self.prompt(text))
        mm = (ids == self.img_id).int()
        g = self.model.generate(input_ids=ids, pixel_values=pv[None].to(self.device, torch.bfloat16), image_grid_thw=self.grid,
                                mm_token_type_ids=mm, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        return self.tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)


def audit_sample(cfg, dset, limit):
    """A fixed-seed sample of the test volumes (limit = 0: all)."""
    m = read_manifest(cfg, dset)
    m = m[m.split == 'test']
    if limit:
        m = m.sample(n=min(limit, len(m)), random_state=0)
    return m.reset_index(drop=True)


def crop_slices(crop, image, region):
    """Slices of AnatomySpatialCropd._crop_one for this image (so the mask can be cropped identically)."""
    shp = tuple(int(x) for x in image.shape[-3:])
    roi = crop.roi_size
    b = crop._region_bounds(image, region)
    xs = crop._axis_start(shp[0], roi[0], (shp[0] - 1) / 2.0)
    ys = crop._axis_start(shp[1], roi[1], (shp[1] - 1) / 2.0)
    zs = shp[2] - roi[2] if b is None else crop._axis_start(shp[2], roi[2], (float(b[0]) + float(b[1])) / 2.0)
    return (slice(None), slice(xs, xs + roi[0]), slice(ys, ys + roi[1]), slice(zs, zs + roi[2]))


def run(nv, cfg, m, dset, a, rank, world):
    fs = FAM[dset]
    qs = [nv_question(f) for f in fs]
    conds = ['clean', 'mirror'] + [cond_name(o) for o in EVAL_ORGANS[dset]]
    rows = list(range(rank, len(m), world))
    P = {c: [] for c in conds + ['wrong']}
    nvox = {c: [] for c in conds}
    feats, order, pvs = [], [], []
    gen = torch.Generator(); gen.manual_seed(0)
    region = REGION[dset]
    for n, i in enumerate(rows):
        r = m.iloc[i]
        got = nv.cached(dset, r.volume)
        hu, organ = got if got is not None else nv.load(abspath(cfg, dset, r.img_rel), abspath(cfg, dset, r.seg_rel), region,
                                                        raw_space=image_space(cfg, dset) == 'raw')
        vols = {'clean': hu, 'mirror': torch.flip(hu, dims=[-1])}
        for o in EVAL_ORGANS[dset]:
            mk = organ_mask(organ, organ_ids(o), 1)
            vols[cond_name(o)] = erase(hu[0], mk, a.fill, 15.0, gen)[None]
            nvox[cond_name(o)].append(int(mk.sum()))
        for c in conds:
            cache, feat = nv.prefix(nv.normalise(vols[c]))
            P[c].append(nv.p_yes(cache, qs).cpu())
            if c == 'clean':
                feats.append(feat.cpu())
            del cache
        pvs.append(nv.normalise(hu).to(torch.bfloat16))
        order.append(i)
        if rank == 0 and n % 10 == 0:
            print(f'{dset} {n + 1}/{len(rows)} on rank 0', flush=True)
    k = len(order)
    perm = np.roll(np.arange(k), 1)
    for s in range(k):
        cache, _ = nv.prefix(pvs[perm[s]])
        P['wrong'].append(nv.p_yes(cache, qs).cpu()); del cache
    n_gen = min(k, -(-a.gen // world)) if a.gen else 0
    texts = {'clean': [nv.generate(pvs[s], REPORT[dset]) for s in range(n_gen)],
             'wrong': [nv.generate(pvs[perm[s]], REPORT[dset]) for s in range(n_gen)]}
    dev = nv.device
    res = {c: torch.stack(v) for c, v in P.items() if v}
    res.update({f'nvox_{c}': torch.tensor(v) for c, v in nvox.items() if v})
    res['feat'] = torch.stack(feats); res['index'] = torch.tensor(order)
    if dist.is_initialized():
        for key in list(res):
            t, _ = gather_padded(res[key].to(dev), grad=False); res[key] = t.cpu()
        allt = [None] * world
        dist.all_gather_object(allt, (texts, order[:n_gen]))
        texts = {'clean': sum((t[0]['clean'] for t in allt), []), 'wrong': sum((t[0]['wrong'] for t in allt), [])}
        res['gen_index'] = sum((t[1] for t in allt), [])
    else:
        res['gen_index'] = order[:n_gen]
    res['texts'] = texts
    res['js'] = [FINDING_NAMES.index(f) for f in fs]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--datasets', default='merlin,ctrate')
    ap.add_argument('--limit', type=int, default=500)
    ap.add_argument('--gen', type=int, default=100)
    ap.add_argument('--fill', default='fat')
    a = ap.parse_args()
    cfg = load_config(os.path.join(os.path.dirname(__file__), '../../configs/stageA.yaml'), env=a.env)
    os.environ.setdefault('HF_HOME', cfg.paths.hf_home)
    rank, world, device = setup_dist(timeout_min=240)
    nv = NVReason(device, cache_root=os.path.join(cfg.paths.work_root, 'cache_nv'))
    out_dir = os.path.join(cfg.paths.work_root, 'evalB', 'NV-Reason-CT')
    summary = {}
    for dset in a.datasets.split(','):
        m = audit_sample(cfg, dset, a.limit)
        res = run(nv, cfg, m, dset, a, rank, world)
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            L = m[[f'label_{f}' for f in FINDING_NAMES]].to_numpy(np.float32)
            tab, summ = metrics(res, L, dset)
            tab.to_csv(os.path.join(out_dir, f'{dset}_ladder.csv'))
            with open(os.path.join(out_dir, f'{dset}_reports.json'), 'w') as fh:
                json.dump({'volume_index': [int(i) for i in res['gen_index']], **res['texts']}, fh, indent=1)
            summary[dset] = summ
            print('SUMMARY', json.dumps({'run': 'NV-Reason-CT', 'dataset': dset, **summ}), flush=True)
            # per-scan scores (as organic.vlm.evaluate) for paired bootstrap CIs against our models on the same scans
            np.savez_compressed(os.path.join(out_dir, f'{dset}_scores.npz'), labels=L[res['index'].numpy()],
                                volume=m.volume.to_numpy()[res['index'].numpy()].astype(str),
                                **{k: (v.half() if k == 'feat' else v).numpy() for k, v in res.items() if torch.is_tensor(v)})
    if rank == 0:
        with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
            json.dump(summary, fh, indent=1)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
