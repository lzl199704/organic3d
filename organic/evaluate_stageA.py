"""Ladder evaluation of a stage-A encoder on Merlin test / CT-RATE test (steps 1-3 plus 3D perturbations).

Conditions per volume: clean, each organ group erased (fat fill by default), left-right mirror.
Metrics per finding: zero-shot AUROC of Delta_f; organ-removal effect (target minus mean control dAUROC, the
audit's C3); probe AUROC on mean-pooled tokens (Dec) and its organ-removal effect (Evi); mirror dAUROC.

    torchrun --nproc_per_node 4 -m organic.evaluate_stageA --env configs/env/local.yaml \
        --ckpt <work_root>/runs/enc4_organic3d/last.pt --datasets merlin --limit 1000
    # released COLIPRI with its own heads (A0):   --ckpt colipri
"""
import argparse
import re
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from .augment import erase, lateral_swap, normalise, one_side_only, organ_mask
from .config import load_config, read_manifest
from .ctrate_v2 import no_chest_volumes
from .data import CTDataset, collate, split_sentences
from .train_stageA import auroc, gather_padded, setup_dist
from .ladder import matched_pair_auroc
from .vocab import FINDING_NAMES, FINDINGS, ORGAN_ID, CTRATE_FINDINGS, MERLIN_FINDINGS, SIZE_FINDINGS, canonical_prompts, finding_organ_ids, control_organ_ids

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_ORGANS = {   # organ groups erased at evaluation (each is one condition)
    'merlin': ['liver', 'gallbladder', 'spleen', 'pancreas', ['kidney_left', 'kidney_right'], ['esophagus', 'stomach'],
               'small_bowel', 'colon', 'aorta', 'pelvic_organs'],
    'ctrate': [['lung_left', 'lung_right'], 'heart', 'aorta', ['esophagus', 'stomach'], 'trachea'],
}
EVAL_ORGANS['radchest'] = EVAL_ORGANS['ctrate']           # RAD-ChestCT: external chest test set, CT-RATE findings


# Size-neutral erasure (ladder v2): a fixed box, larger than the organ in ~95% of scans, centred on the organ, so the
# erased region no longer shows the organ's size. Control: the same box moved sideways by its own width.
# Sizes (x, y, z mm) from the 95th-percentile extents of 300 Merlin crops (+~10%); heart from chest anatomy. Liver,
# bowel and hiatal hernia are left out: a box hiding them covers most of the abdomen (stated limitation).
BLOCK_MM = {'heart': (170, 150, 150), 'spleen': (130, 160, 160), 'gallbladder': (80, 90, 90),
            'pancreas': (170, 110, 120), 'pelvic_organs': (135, 130, 135), 'aorta': (80, 120, 260)}


def block_masks(organ, ids, mm, step=2.0):
    """organ (D,H,W) map at 2 mm -> (target box, control box) bool masks; empty if the organ is absent."""
    m = torch.isin(organ, torch.as_tensor(ids, device=organ.device, dtype=organ.dtype))
    tgt, ctl = torch.zeros_like(m), torch.zeros_like(m)
    if not m.any():
        return tgt, ctl
    c = torch.nonzero(m).float().mean(0)
    half = [int(round(v / step / 2)) for v in mm]
    shape = organ.shape
    def box(cx):
        lo = [max(0, int(round(float(v))) - h) for v, h in zip(cx, half)]
        hi = [min(n, int(round(float(v))) + h) for v, h, n in zip(cx, half, shape)]
        out = torch.zeros_like(m)
        out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
        return out
    tgt = box(c)
    d = 2 * half[0] * (1 if c[0] < shape[0] / 2 else -1)                 # sideways, towards the crop centre
    ctl = box(c + torch.tensor([d, 0, 0], device=c.device))
    return tgt, ctl


def cond_name(o):
    return 'rem_' + ('+'.join(o) if isinstance(o, list) else o)


def organ_ids(o):
    return [ORGAN_ID[x] for x in (o if isinstance(o, list) else [o])]


class Scorer:
    """Wraps either our model (Delta_f from sentence-conditioned scores) or released COLIPRI (A0)."""
    def __init__(self, cfg, ckpt, device):
        from colipri import get_model, get_processor
        self.tok = get_processor()._text_tokenizer
        self.device, self.native = device, ckpt == 'colipri'
        if cfg.eval.get('prompt_style', 'ours') == 'ctclip':     # CT-CLIP / CT-RATE published protocol phrasing
            canon = [s for f in FINDING_NAMES for s in (f'{FINDINGS[f][0]} is present.', f'{FINDINGS[f][0]} is not present.')]
        else:
            canon = [s for f in FINDING_NAMES for s in canonical_prompts(f)]
        enc = self.tok(canon, max_length=96, padding='longest', truncation=True, return_tensors='pt')
        if self.native:
            self.m = get_model().to(device).eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                self.c = self.m.encode_text(enc['input_ids'].to(device), enc['attention_mask'].to(device)).float()
        else:
            from .model import build_model
            cfg.model.pretrained = False
            self.m = build_model(cfg).to(device).eval()
            state = torch.load(ckpt, map_location='cpu', weights_only=False)
            self.m.load_state_dict(state['model'])
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                self.c = self.m.encode_text(enc['input_ids'].to(device), enc['attention_mask'].to(device))
                self.c = self.m.attach_route(self.c, canon)

    @torch.no_grad()
    def sentence_scores(self, x, sents):
        """x (1,1,D,H,W) normalised, sents list of str -> similarity per sentence (len(sents),)."""
        enc = self.tok(sents, max_length=96, padding='longest', truncation=True, return_tensors='pt')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            t = self.m.encode_text(enc['input_ids'].to(self.device), enc['attention_mask'].to(self.device))
            if self.native:
                e = self.m.encode_image(x, pool=True, project=True, normalize=True).float()
                return (e @ t.float().T)[0]
            T, _, _ = self.m.encode_image(x)
            return self.m.score(T, self.m.attach_route(t, sents))[0].float()

    @torch.no_grad()
    def tokens(self, x):
        """Our models only: query tokens T (N, K, d) of normalised volumes."""
        with torch.autocast('cuda', dtype=torch.bfloat16):
            T, _, _ = self.m.encode_image(x)
        return T

    @torch.no_grad()
    def delta_from_tokens(self, T):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            S = self.m.score(T, self.c).float()
        return (S[:, 0::2] - S[:, 1::2]).float()

    @torch.no_grad()
    def __call__(self, x):
        """x normalised (N,1,192,192,192) -> Delta (N, F), pooled features (N, D)."""
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if self.native:
                e = self.m.encode_image(x, pool=True, project=True, normalize=True).float()
                S = e @ self.c.T
                feat = e
            else:
                T, _, _ = self.m.encode_image(x)
                S = self.m.score(T, self.c).float()
                feat = T.float().mean(1)
        return (S[:, 0::2] - S[:, 1::2]).float(), feat.float()


_NEG = re.compile(r'\b(no|not|without|negative for|resolved|absent)\b', re.I)


def lateral_sentences(report, max_n=6):
    """Report sentences that name exactly one side and state something present (laterality rung, ladder v2)."""
    out = []
    for s in split_sentences(report):
        if one_side_only(s) and not _NEG.search(s):
            sw = lateral_swap(s)
            if sw and sw != s:
                out.append((s, sw))
    return out[:max_n]


def run(cfg, scorer, dataset, limit, fills, rank, world, device, bs):
    m = read_manifest(cfg, dataset)
    m = m[m.split == 'test']
    if dataset == 'merlin':
        m = m[m[[f'label_{f}' for f in MERLIN_FINDINGS]].notna().any(axis=1)]
    if limit:
        m = m.sample(n=min(limit, len(m)), random_state=0)
    ds = CTDataset(cfg, [m], train=False)
    ids = list(range(rank, len(ds), world))
    loader = DataLoader(Subset(ds, ids), batch_size=bs, num_workers=cfg.data.num_workers, collate_fn=collate)
    blocks = [o for o in EVAL_ORGANS[dataset] if isinstance(o, str) and o in BLOCK_MM]
    conds = ['clean', 'mirror'] + [cond_name(o) + f'@{f}' for o in EVAL_ORGANS[dataset] for f in fills] + \
            [f'{k}_{o}@{fills[0]}' for o in blocks for k in ('blk', 'blkc')]
    out = {c: {'delta': [], 'feat': [], 'nvox': []} for c in conds}
    idx_all, labels, lat, toks = [], [], [], []
    gen = torch.Generator(device=device).manual_seed(rank)
    for b in loader:
        hu = b['img'].to(device).float()
        organ = b['organ'].to(device)
        for c in conds:
            vols, nv = [], []
            for i in range(hu.shape[0]):
                v = hu[i, 0]
                if c == 'mirror':
                    v = torch.flip(v, dims=[0]); k = 0
                elif c.startswith(('blk_', 'blkc_')):
                    kind, rest = c.split('_', 1); name, fill = rest.split('@')
                    t_box, c_box = block_masks(organ[i], organ_ids(name), BLOCK_MM[name])
                    mk = t_box if kind == 'blk' else c_box
                    v = erase(v, mk, fill, 15.0, gen); k = int(t_box.sum())        # nvox > 0 iff the organ is present
                elif c.startswith('rem_'):
                    name, fill = c[4:].split('@')
                    mk = organ_mask(organ[i], organ_ids(name.split('+') if '+' in name else name), 1)
                    v = erase(v, mk, fill, 15.0, gen); k = int(mk.sum())
                else:
                    k = 0
                vols.append(v); nv.append(k)
            d, f = scorer(normalise(torch.stack(vols)[:, None]))
            out[c]['delta'].append(d.cpu()); out[c]['feat'].append(f.cpu()); out[c]['nvox'].append(torch.tensor(nv))
            if c == 'clean' and not scorer.native:
                toks.append(scorer.tokens(normalise(torch.stack(vols)[:, None])).to(torch.float16).cpu())
        for i, vi in enumerate(b['index']):                          # laterality rung: clean and mirrored volume
            pairs = lateral_sentences(ds.df.report.iloc[vi]) if 'report' in ds.df else []
            if pairs:
                sents = [p for p, _ in pairs] + [q for _, q in pairs]; n = len(pairs)
                s0 = scorer.sentence_scores(normalise(hu[i:i + 1]), sents)
                s1 = scorer.sentence_scores(normalise(torch.flip(hu[i:i + 1], dims=[2])), sents)
                ok0, ok1 = (s0[:n] > s0[n:]).float(), (s1[n:] > s1[:n]).float()
                lat.append(torch.tensor([float(vi), n, ok0.sum(), ok1.sum(), (ok0 * ok1).sum()]))
        idx_all += b['index']; labels.append(b['labels'])
    if toks:   # organ-slot swap test (ladder v2, our models): one organ group's slots taken from the next volume on this rank
        Tc = torch.cat(toks); n = len(Tc); perm = np.roll(np.arange(n), 1)
        oq = scorer.m.queries.organ_of_query()
        for o in EVAL_ORGANS[dataset]:
            c = 'swap_' + cond_name(o)[4:]
            slots = torch.isin(oq, torch.as_tensor(organ_ids(o)))
            ds_ = []
            for s in range(0, n, 8):
                T2 = Tc[s:s + 8].clone(); T2[:, slots] = Tc[perm[s:s + 8]][:, slots]
                ds_.append(scorer.delta_from_tokens(T2.to(device)).cpu())
            out[c] = {'delta': ds_, 'feat': [torch.zeros(n, 1)], 'nvox': [torch.ones(n, dtype=torch.long) * int(slots.sum())]}
            conds.append(c)
    res = {c: {k: torch.cat(v) for k, v in out[c].items()} for c in conds}
    lab = torch.cat(labels)
    if world > 1:
        for c in conds:
            for k in res[c]:
                res[c][k], _ = gather_padded(res[c][k].to(device), grad=False)
                res[c][k] = res[c][k].cpu()
        lab, _ = gather_padded(lab.to(device), grad=False); lab = lab.cpu()
    idx = torch.tensor(idx_all, dtype=torch.long)
    if world > 1:
        idx, _ = gather_padded(idx.to(device), grad=False); idx = idx.cpu()
    res['index'] = idx.numpy()
    L = torch.stack(lat) if lat else torch.zeros(0, 5)
    if world > 1:
        L, _ = gather_padded(L.to(device), grad=False); L = L.cpu()
    res['lat'] = L.numpy()                                             # (volumes with side sentences, 5)
    res['patient'] = ds.df.patient_id.values[res['index']].astype(str)   # for patient-level bootstrap
    res['volume'] = ds.df.volume.values[res['index']].astype(str)
    return res, lab.numpy(), conds


def drop_no_chest(res, lab, conds, work_root):
    """CT-RATE v2 test set: drop the brain series (organic/ctrate_v2.py), as scripts/metrics/extaudit_metrics.py does, so
    the training evaluator and the harness score the same 3,002 volumes. Scan indices are kept."""
    keep = ~np.isin(res['volume'], list(no_chest_volumes(work_root)))
    if keep.all():
        return res, lab
    kt = torch.from_numpy(keep)
    for c in conds:
        res[c] = {k: v[kt] for k, v in res[c].items()}
    if len(res['lat']):
        res['lat'] = res['lat'][np.isin(res['lat'][:, 0], res['index'][keep])]
    for k in ('index', 'patient', 'volume'):
        res[k] = res[k][keep]
    return res, lab[keep]


def probe_scores(X, y, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ok = ~np.isnan(y)
    models, oof = [], np.full(len(y), np.nan)
    idx = np.where(ok)[0]
    yy = y[idx].astype(int)
    if yy.sum() < folds or (1 - yy).sum() < folds:
        return None, None
    splits = list(StratifiedKFold(folds, shuffle=True, random_state=0).split(idx, yy))
    for tr, te in splits:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight='balanced'))
        clf.fit(X[idx[tr]], yy[tr]); models.append((clf, idx[te]))
        oof[idx[te]] = clf.decision_function(X[idx[te]])
    return models, oof


def metrics(res, lab, dataset, fills, min_pos=10):
    fs = CTRATE_FINDINGS if dataset in ('ctrate', 'radchest') else MERLIN_FINDINGS
    rows = []
    for f in fs:
        j = FINDING_NAMES.index(f)
        y = lab[:, j]
        if np.nansum(y) < min_pos:
            continue
        d0 = res['clean']['delta'][:, j].numpy()
        a0 = auroc(y, d0)
        r = dict(finding=f, n_pos=int(np.nansum(y)), zero_shot=a0,
                 mirror_d=auroc(y, res['mirror']['delta'][:, j].numpy()) - a0)
        tgt = set(finding_organ_ids(f)); ctrl = set(control_organ_ids(f))
        models, oof = probe_scores(res['clean']['feat'].numpy(), y)
        r['probe'] = auroc(y, oof) if oof is not None else np.nan
        # step 3 Addressable: how much of what a probe reads the text side reaches (chance-corrected ratio)
        r['adr'] = (a0 - 0.5) / (r['probe'] - 0.5) if r['probe'] == r['probe'] and r['probe'] > 0.5 else np.nan
        # step 5 control: AUROC between patients identical on every other labelled finding
        mp, npairs = matched_pair_auroc(lab, d0, j)
        r['matched_pair_auroc'], r['matched_gap'], r['n_pairs'] = mp, (a0 - mp) if mp == mp else np.nan, npairs
        for fill in fills:
            t_eff, c_eff, pt_eff, pc_eff = [], [], [], []
            for o in EVAL_ORGANS[dataset]:
                ids = set(organ_ids(o)); c = cond_name(o) + f'@{fill}'
                keep = res[c]['nvox'].numpy() > 0
                dz = auroc(np.where(keep, y, np.nan), res[c]['delta'][:, j].numpy()) - auroc(np.where(keep, y, np.nan), d0)
                if models is not None:
                    pe = np.full(len(y), np.nan)
                    for clf, te in models:
                        pe[te] = clf.decision_function(res[c]['feat'].numpy()[te])
                    dp = auroc(np.where(keep, y, np.nan), pe) - auroc(np.where(keep, y, np.nan), oof)
                else:
                    dp = np.nan
                if tgt and ids & tgt:
                    t_eff.append(dz); pt_eff.append(dp)
                elif ids <= ctrl:
                    c_eff.append(dz); pc_eff.append(dp)
            r[f'effect_{fill}'] = (np.nanmean(t_eff) - np.nanmean(c_eff)) if t_eff else np.nan
            r[f'probe_effect_{fill}'] = (np.nanmean(pt_eff) - np.nanmean(pc_eff)) if pt_eff else np.nan
        # organ-slot swap test: AUROC change when the finding's organ slots come from another scan, minus control organs
        if any(c.startswith('swap_') for c in res if isinstance(res[c], dict)):
            t_eff, c_eff = [], []
            for o in EVAL_ORGANS[dataset]:
                c = 'swap_' + cond_name(o)[4:]
                if c not in res:
                    continue
                dz = auroc(y, res[c]['delta'][:, j].numpy()) - a0
                ids = set(organ_ids(o))
                if tgt and ids & tgt:
                    t_eff.append(dz)
                elif ids <= ctrl:
                    c_eff.append(dz)
            r['swap_effect'] = (np.nanmean(t_eff) - np.nanmean(c_eff)) if t_eff else np.nan
        # size-neutral block test: target box minus the same box moved sideways (compact organs only)
        bl = [o for o in FINDINGS[f][1] if f'blk_{o}@{fills[0]}' in res]
        if bl:
            dd = []
            for o in bl:
                keep = res[f'blk_{o}@{fills[0]}']['nvox'].numpy() > 0
                yk = np.where(keep, y, np.nan)
                dd.append(auroc(yk, res[f'blk_{o}@{fills[0]}']['delta'][:, j].numpy()) -
                          auroc(yk, res[f'blkc_{o}@{fills[0]}']['delta'][:, j].numpy()))
            r['block_effect'] = float(np.nanmean(dd))
        rows.append(r)
    return pd.DataFrame(rows).set_index('finding') if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(HERE, '../configs/stageA.yaml'))
    ap.add_argument('--env', required=True)
    ap.add_argument('--ckpt', required=True, help="checkpoint path, or 'colipri' for the released model")
    ap.add_argument('--name', default=None)
    ap.add_argument('--datasets', default='merlin,ctrate')
    ap.add_argument('--fills', default=None, help='comma list; default: eval.fills of the run config')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--min_pos', type=int, default=10)
    ap.add_argument('overrides', nargs='*')
    a = ap.parse_args()
    run_cfg = os.path.join(os.path.dirname(a.ckpt), 'config.yaml') if a.ckpt != 'colipri' else None
    cfg = load_config(run_cfg or a.config, env=a.env, overrides=a.overrides)
    os.environ.setdefault('HF_HOME', cfg.paths.hf_home)
    rank, world, device = setup_dist(timeout_min=240)
    scorer = Scorer(cfg, a.ckpt, device)
    name = a.name or (os.path.basename(os.path.dirname(a.ckpt)) if a.ckpt != 'colipri' else 'COLIPRI')
    out_dir = os.path.join(cfg.paths.work_root, 'eval', name)
    fills = a.fills.split(',') if a.fills else list(cfg.eval.get('fills', ['fat']))
    for dset in a.datasets.split(','):
        res, lab, conds = run(cfg, scorer, dset, a.limit, fills, rank, world, device, a.bs)
        if rank == 0:
            if dset == 'ctrate':
                res, lab = drop_no_chest(res, lab, conds, cfg.paths.work_root)
            write_outputs(out_dir, name, dset, res, lab, conds, fills, a.min_pos)
    if dist.is_initialized():
        dist.destroy_process_group()


def write_outputs(out_dir, name, dset, res, lab, conds, fills, min_pos=10):
    """Ladder table, per-scan scores and summary (shared with scripts/metrics/extaudit_metrics.py for released encoders)."""
    os.makedirs(out_dir, exist_ok=True)
    df = metrics(res, lab, dset, fills, min_pos)
    df.to_csv(os.path.join(out_dir, f'{dset}_ladder.csv'))
    np.savez_compressed(os.path.join(out_dir, f'{dset}_scores.npz'), labels=lab, index=res['index'],
                        patient=np.asarray(res['patient']), feat=res['clean']['feat'].numpy().astype(np.float16),
                        **{f'delta__{c}': res[c]['delta'].numpy() for c in conds},
                        **{f'nvox__{c}': res[c]['nvox'].numpy() for c in conds}, lat=res['lat'])
    summ = {'model': name, 'dataset': dset, 'n': int(len(lab)),
            **{k: float(df[k].mean()) for k in df.columns if k != 'n_pos'}}
    tex = df[~df.index.isin(SIZE_FINDINGS)]
    for k in [c for c in df.columns if c.startswith(('effect_', 'probe_effect_'))]:
        summ[k + '_texture'] = float(tex[k].mean())     # main Evidential number: size findings excluded
    Lr = res['lat']
    if len(Lr):
        summ['lat_n_sent'] = int(Lr[:, 1].sum())
        summ['lat_acc'] = float(Lr[:, 2].sum() / Lr[:, 1].sum())            # true side beats swapped side
        summ['lat_acc_mirror'] = float(Lr[:, 3].sum() / Lr[:, 1].sum())     # after flipping, swapped side wins
        summ['lat_both'] = float(Lr[:, 4].sum() / Lr[:, 1].sum())           # both: the side follows the image
    if 'swap_effect' in df:
        summ['swap_effect_all'] = float(df.swap_effect.mean())                 # valid for size findings too
    if 'block_effect' in df:
        summ['block_effect_size'] = float(df.block_effect[df.index.isin(SIZE_FINDINGS)].mean())
    if 'probe' in df and df['probe'].mean() > 0.5:
        summ['adr_of_means'] = float((df['zero_shot'].mean() - 0.5) / (df['probe'].mean() - 0.5))
    with open(os.path.join(out_dir, f'{dset}_summary.json'), 'w') as fh:
        json.dump(summ, fh, indent=1)
    print('SUMMARY', json.dumps(summ), flush=True)
    print(df.round(3).to_string(), flush=True)
    return df, summ


if __name__ == '__main__':
    main()
