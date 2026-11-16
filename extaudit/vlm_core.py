"""External 3D CT VLM audit: P(Yes) per finding question on clean, mirrored and organ-erased scans, for released
generative VLMs, each in its own python env (adapters in extaudit/vlm_models/). Every model gets the same scans (the
full test set, <dataset>_set.parquet: CT-RATE v2 3,005 rows, 3,002 scored after the non-chest drop; Merlin 5,109), the
same perturbations (on the 1 mm RAS volume before the model's own preprocessing; fat fill -100 HU + N(0, 15), mask
dilated 2 mm) and the same metric code (evaluate_stageB.metrics via scripts/metrics/vlm_metrics.py).

Adapter API (class Adapter(ckpt, device)):
    question(phrase) -> str                 the model's own yes/no question for a finding phrase (default below)
    encode(hu) -> state, feat               hu: (X, Y, Z) 1 mm RAS HU cuda tensor; state = whatever p_yes needs
                                            (e.g. visual tokens); feat = 1-D pooled visual feature (probe rung)
    p_yes(state, questions) -> (n,) tensor  P(Yes) = sigmoid(logit Yes - logit No) at the first answer token
    set_scan(seg), set_dataset(name)        optional: TotalSegmentator mask of the scan (mirrored for 'mirror'), dataset
    yes_no_logit(state, questions)          optional: the logit difference itself; saved as D__ and used by the
                                            metrics instead of P (same ranking, but no ties where P saturates at 1.0)

    CUDA_VISIBLE_DEVICES=0 <model env>/bin/python -m extaudit.vlm_core --model m3d_lamed --name M3D-LaMed --dataset ctrate \
        --root <work_root>/extaudit --shard 0/2
Writes {root}/out_vlm/{name}/{dataset}_shard{k}of{n}.npz.
"""
import argparse
import importlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from extaudit.core import VolSet, perturb


def default_question(phrase):
    return f'Is there {phrase} in this CT? Answer Yes or No.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='adapter module in extaudit/vlm_models')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--name', required=True)
    ap.add_argument('--dataset', default='ctrate')
    ap.add_argument('--root', required=True, help='{work_root}/extaudit')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--set_file', default='', help='scan set under --root instead of <dataset>_set.parquet, '
                    'e.g. rex_heldout_set.parquet (ReXGroundingCT held-out scans, classification only)')
    ap.add_argument('--clean_only', action='store_true', help='clean scan only (no mirror, no erasure)')
    ap.add_argument('--out_tag', default='', help='shard file prefix in out_vlm/<name>/ (default: --dataset)')
    ap.add_argument('--fills', default='fat', help='erasure fills, comma-separated (2026-11-13: also water, median, '
                    'blur; use one fill per --name, since the metrics key conditions by organ only)')
    ap.add_argument('--no_mirror', action='store_true', help='skip the mirrored condition')
    ap.add_argument('--first', type=int, default=0, help='use only the first N scans of the set (before sharding)')
    a = ap.parse_args()
    k, n = map(int, a.shard.split('/'))
    meta = json.load(open(os.path.join(a.root, 'meta.json')))
    rows = pd.read_parquet(os.path.join(a.root, a.set_file or f'{a.dataset}_set.parquet'))
    rows = (rows.head(a.first) if a.first else rows).iloc[k::n]
    if a.limit:
        rows = rows.head(a.limit)
    dev = torch.device('cuda')
    ad = importlib.import_module(f'extaudit.vlm_models.{a.model}').Adapter(a.ckpt, dev)
    if hasattr(ad, 'set_dataset'):
        ad.set_dataset(a.dataset)
    scan = getattr(ad, 'set_scan', None)                 # optional: mask-guided input (crop centre), e.g. our VLMs
    fam = 'ct_' if a.dataset in ('ctrate', 'radchest') else 'mr_'
    fs = [f for f in meta['findings'] if f.startswith(fam)]
    qfn = getattr(ad, 'question', None) or default_question
    qs = [qfn(meta['phrase'][f]) for f in fs]
    conds = [('clean', None, None)] + ([] if a.no_mirror else [('mirror', None, None)]) + \
            [(f'{name}@{fl}', ids, fl) for name, ids in meta['eval_organs'][a.dataset] for fl in a.fills.split(',')]
    if a.clean_only:
        conds = conds[:1]
    P = {c[0]: [] for c in conds}
    D = {c[0]: [] for c in conds}
    has_d = hasattr(ad, 'yes_no_logit')
    nvox = {c[0]: [] for c in conds}
    feats, idx, errors = [], [], []
    gen = torch.Generator(device=dev).manual_seed(k)
    loader = DataLoader(VolSet(rows), batch_size=None, num_workers=a.workers, prefetch_factor=2)
    t0 = time.time()
    for it, b in enumerate(loader):
        if 'error' in b:
            errors.append(b['error']); print('ERROR', b['index'], b['error'], flush=True)
            continue
        hu, seg = b['hu'].to(dev), b['seg'].to(dev)
        if scan:
            scan(seg)
        for c in conds:
            v, nv = perturb(hu, seg, c, meta, gen)
            if scan and c[0] == 'mirror':
                scan(torch.flip(seg, dims=[0]))           # the mask follows the mirrored image
            with torch.no_grad():
                st, feat = ad.encode(v)
                if has_d:
                    d = ad.yes_no_logit(st, qs).float()
                    p = torch.sigmoid(d)
                    D[c[0]].append(d.cpu().numpy())
                else:
                    p = ad.p_yes(st, qs)
            P[c[0]].append(p.float().cpu().numpy()); nvox[c[0]].append(nv)
            if c[0] == 'clean':
                feats.append(feat.float().cpu().numpy())
            if scan and c[0] == 'mirror':
                scan(seg)
        idx.append(b['index'])
        if it % 10 == 0:
            el = time.time() - t0
            print(f'{it + 1}/{len(rows)}  {el / 60:.1f} min, {el / (it + 1):.1f} s/scan', flush=True)
    out = os.path.join(a.root, 'out_vlm', a.name)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f'{a.out_tag or a.dataset}_shard{k}of{n}.npz'), index=np.asarray(idx),
                        feat=np.stack(feats), findings=np.asarray(fs), questions=np.asarray(qs),
                        **{f'P__{c}': np.stack(v) for c, v in P.items()},
                        **{f'D__{c}': np.stack(v) for c, v in D.items() if v},
                        **{f'nvox__{c}': np.asarray(v) for c, v in nvox.items()})
    print(f'DONE {a.name} {a.dataset} shard {k}/{n}: {len(idx)} scans, {len(errors)} errors, '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
