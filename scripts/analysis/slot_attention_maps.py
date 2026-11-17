"""Anatomy-slot attention of our stage-A encoders: numbers and qualitative maps for the slot analysis.

The model and its input are exactly the extaudit harness path (extaudit/models/organic3d_enc.Adapter): crop centred by
TotalSegmentator (chest: lungs + heart; abdomen: Merlin's rule), 384 mm at 1 mm -> 192^3 at 2 mm, HU clipped to
[-1000, 1000] / 1000. encode_image returns (T, A, pbar); A (1, 144, 24^3) is the last cross-attention layer's
head-averaged attention, tokens in (x y z) order; slot k < 128 belongs to organ group k // 8 (vocab.ORGAN_NAMES), slots
128-143 are global.

--mode stats: per scan and organ group o (organ map of the crop at 2 mm, as in training; a token is "inside" o when its
8^3 patch holds any voxel of o, the definition of losses.anatomy_loss):
  own    mean over o's 8 slots of the attention mass on o's tokens;
  other  mean over the 120 slots of the other organ groups of their mass on o's tokens;
  glob   mean over the 16 global slots of their mass on o's tokens;
  share  o's share of the 13,824 tokens (= the mass of a uniform slot).
Also each slot's effective number of tokens exp(entropy), and the sentence pooling weights beta_k(s) of each finding's
positive canonical prompt: mass on the finding's organ slots vs the uniform share.
--mode figs: saves the crop, organ map, attention and betas of the figure scans to --npz_dir.

    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/slot_attention_maps.py --mode stats --out <dir> [--ckpt_dir <work_root>/runs]
Inputs: the harness sets in {work_root}/extaudit (organic/workroot.py: $ORGANIC3D_WORK_ROOT, else
paths.work_root of $ORGANIC3D_ENV); checkpoints <ckpt_dir>/<model>/last.pt.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from extaudit.core import VolSet                                     # noqa: E402
from extaudit.models.organic3d_enc import Adapter                     # noqa: E402
from organic.data.dataset import _LUT, _crop_pad                            # noqa: E402
from organic.vocab import ORGAN_NAMES, BACKGROUND, FINDING_NAMES, finding_organ_ids   # noqa: E402
sys.path.insert(0, HERE)
from organic.workroot import work_root                                       # noqa: E402

W = work_root()
ROOT = os.path.join(W, 'extaudit')
NO, G, P = len(ORGAN_NAMES), 24, 8
SLOT_ORGAN = np.r_[np.arange(NO * 8) // 8, np.full(16, -1)]


def organ_crop(seg, bounds):
    """2 mm organ-group map of the model's crop (training: CTDataset org_c[1::2, 1::2, 1::2])."""
    return _crop_pad(_LUT[seg], bounds, BACKGROUND)[1::2, 1::2, 1::2]


def inside_tokens(org, dev):
    """(NO, N) bool: token's 8^3 patch holds any voxel of the organ group (losses.anatomy_loss)."""
    b = torch.as_tensor(org, device=dev).reshape(G, P, G, P, G, P)
    return torch.stack([(b == o).any(5).any(3).any(1).reshape(-1) for o in range(NO)])


@torch.no_grad()
def forward(ad, hu, seg):
    ad.set_scan(seg)
    x = ad.prep(hu)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        T, A, _ = ad.m.encode_image(x)
    return x, T, A[0].float()


@torch.no_grad()
def betas(ad, T, t):
    """Sentence pooling weights beta (S, K), as Organic3D.score (fixed routing: route columns split off and the
    slots outside the sentence's route masked, 2026-10-24)."""
    m = ad.m
    R = None
    if getattr(m, 'pool_mode', None) == 'fixed' and t.shape[-1] > T.shape[-1]:
        t, R = t[:, :-T.shape[1]], t[:, -T.shape[1]:]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        lg = torch.einsum('kd,sd->sk', m.W_b(T[0]), m.W_a(t)) / math.sqrt(T.shape[-1])
    lg = lg.float()
    if R is not None:
        lg = lg.masked_fill(R.float() <= 0, float('-inf'))
    return lg.softmax(-1)



def rows_for(ds, n_scans):
    st = pd.read_parquet(os.path.join(ROOT, f'{ds}_set.parquet'), columns=['index', 'volume', 'img', 'seg'])
    return st.iloc[::max(len(st) // n_scans, 1)].head(n_scans), None


def run_stats(a, models, dev):
    meta = json.load(open(os.path.join(ROOT, 'meta.json')))
    organ_rows, slot_rows, beta_rows = [], [], []
    lift_sum = {nm: np.zeros((NO, NO)) for nm in models}
    lift_n = {nm: np.zeros((NO, NO)) for nm in models}
    for ds, n_scans in [('ctrate', a.n), ('merlin', a.n)]:
        rows, _ = rows_for(ds, n_scans)
        fam = 'ct_' if ds == 'ctrate' else 'mr_'
        fins = [f for f in FINDING_NAMES if f.startswith(fam) and finding_organ_ids(f)]
        prompts = [f"There is {meta['phrase'][f]}." for f in fins]
        tp = {}
        for nm, ad in models.items():
            ad.set_dataset(ds)
            tp[nm] = ad.encode_text(prompts)
        loader = DataLoader(VolSet(rows), batch_size=None, num_workers=a.workers, prefetch_factor=2)
        for it, b in enumerate(loader):
            if 'error' in b:
                print('ERROR', b['error'], flush=True)
                continue
            hu, seg = b['hu'].to(dev), b['seg'].to(dev)
            for nm, ad in models.items():
                _, T, A = forward(ad, hu, seg)
                org = organ_crop(seg.cpu().numpy(), ad.bounds)
                ins = inside_tokens(org, dev)                                     # (NO, N)
                ntok = ins.sum(1).cpu().numpy()
                mass = (A @ ins.float().T).cpu().numpy()                          # (K, NO)
                ent = -(A.clamp_min(1e-12) * A.clamp_min(1e-12).log()).sum(1)
                eff = ent.exp().cpu().numpy()
                for o in range(NO):
                    if ntok[o] == 0:
                        continue
                    own = mass[SLOT_ORGAN == o, o].mean()
                    oth = mass[(SLOT_ORGAN >= 0) & (SLOT_ORGAN != o), o].mean()
                    glo = mass[SLOT_ORGAN < 0, o].mean()
                    organ_rows.append(dict(model=nm, dataset=ds, volume=b['volume'], organ=ORGAN_NAMES[o],
                                           n_tok=int(ntok[o]), share=ntok[o] / G ** 3, own=own, other=oth, glob=glo))
                pres = np.where(ntok > 0)[0]
                for q in range(NO):
                    if ntok[q] == 0:
                        continue
                    lift_sum[nm][q, pres] += mass[SLOT_ORGAN == q][:, pres].mean(0) / (ntok[pres] / G ** 3)
                    lift_n[nm][q, pres] += 1
                slot_rows.append(dict(model=nm, dataset=ds, volume=b['volume'],
                                      eff_organ=float(eff[SLOT_ORGAN >= 0].mean()),
                                      eff_global=float(eff[SLOT_ORGAN < 0].mean()),
                                      eff_organ_median=float(np.median(eff[SLOT_ORGAN >= 0])),
                                      eff_global_median=float(np.median(eff[SLOT_ORGAN < 0]))))
                be = betas(ad, T, tp[nm]).cpu().numpy()                           # (S, K)
                for f, bb in zip(fins, be):
                    ids = finding_organ_ids(f)
                    on = np.isin(SLOT_ORGAN, ids)
                    beta_rows.append(dict(model=nm, dataset=ds, volume=b['volume'], finding=f,
                                          organs='+'.join(ORGAN_NAMES[i] for i in ids), beta_organ=bb[on].sum(),
                                          beta_global=bb[SLOT_ORGAN < 0].sum(), uniform=on.sum() / len(SLOT_ORGAN)))
            if it % 20 == 0:
                print(f'{ds} {it}/{len(rows)}', flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(organ_rows).to_csv(os.path.join(a.out, 'per_scan_organ.csv'), index=False)
    pd.DataFrame(slot_rows).to_csv(os.path.join(a.out, 'per_scan_entropy.csv'), index=False)
    pd.DataFrame(beta_rows).to_csv(os.path.join(a.out, 'per_scan_beta.csv'), index=False)
    for nm in models:
        lift = lift_sum[nm] / np.maximum(lift_n[nm], 1)
        pd.DataFrame(lift, index=ORGAN_NAMES, columns=ORGAN_NAMES).to_csv(os.path.join(a.out, f'lift_{nm}.csv'))


def run_figs(a, models, dev):
    """Figure scans: crop HU (2 mm), organ map, attention (144 x 24^3), betas of the given sentences."""
    os.makedirs(a.npz_dir, exist_ok=True)
    meta = json.load(open(os.path.join(ROOT, 'meta.json')))
    for spec in a.fig:
        ds, vol = spec.split(':')
        st = pd.read_parquet(os.path.join(ROOT, f'{ds}_set.parquet'))
        rows = st[st.volume == vol]
        b = VolSet(rows)[0]
        hu, seg = b['hu'].to(dev), b['seg'].to(dev)
        les, z0 = None, 0
        sents = []
        fam = 'ct_' if ds == 'ctrate' else 'mr_'
        sents = [f"There is {meta['phrase'][f]}." for f in FINDING_NAMES if f.startswith(fam)]
        for nm, ad in models.items():
            ad.set_dataset(ds)
            x, T, A = forward(ad, hu, seg)
            org = organ_crop(seg.cpu().numpy(), ad.bounds)
            lesc = None
            if les is not None:
                lesc = _crop_pad(les.cpu().numpy().astype(np.uint8), ad.bounds, 0)[1::2, 1::2, 1::2]
            be = betas(ad, T, ad.encode_text(sents)).cpu().numpy()
            np.savez_compressed(os.path.join(a.npz_dir, f'{nm}_{ds}_{vol.replace(".nii.gz", "")}.npz'),
                                hu=(x[0, 0].float().cpu().numpy() * 1000).astype(np.int16), organ=org,
                                lesion=lesc if lesc is not None else np.zeros(1, np.uint8),
                                A=A.cpu().numpy().astype(np.float32), beta=be, sentences=np.array(sents),
                                bounds=np.array(ad.bounds), z0=z0, slot_organ=SLOT_ORGAN)
            print('saved', nm, ds, vol, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['stats', 'figs'], required=True)
    ap.add_argument('--models', default='enc4_organic3d,enc3_routing')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(W, 'slot_attention'))
    ap.add_argument('--npz_dir', default=os.path.join(W, 'slot_attention', 'npz'))
    ap.add_argument('--ckpt_dir', default=os.path.join(W, 'runs'), help='stage-A runs: <ckpt_dir>/<model>/last.pt')
    ap.add_argument('--fig', nargs='*', default=[])
    a = ap.parse_args()
    dev = torch.device('cuda')
    models = {nm: Adapter(os.path.join(a.ckpt_dir, nm, 'last.pt'), dev) for nm in a.models.split(',')}
    (run_stats if a.mode == 'stats' else run_figs)(a, models, dev)


if __name__ == '__main__':
    main()
