"""Stage-A training (encoder, core 1). Launch with torchrun; works single-node (local GPUs) or multi-node.

    torchrun --nproc_per_node 4 -m organic.encoder.train --env configs/env/local.yaml \
        --ablation configs/encoder/enc4_organic3d.yaml train.max_steps=100
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.interventions import one_side_only, plan_counterfactuals, build_image_batch, lateral_swap, severity_swap
from ..config import load_config
from ..utils import allreduce_grads, auroc, gather_padded, offset, setup_dist  # noqa: F401
from ..data.dataset import CTDataset, MixedBatchSampler, collate, load_manifests, load_sentences, val_indices
from .losses import (i2t_loss, t2i_loss, txt_loss, cls_loss, evidence_loss, pair_loss, mirror_loss, anatomy_loss)
from .model import build_model
from ..vocab import FAMILY, FINDING_NAMES, FINDING_ID, ORGAN_ID, ORGAN_NAMES, canonical_prompts, control_organ_ids, finding_organ_ids
from ..data.interventions import present_organs

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ text batch
def build_text(batch, cfg, rng):
    """Positives (own-report sentences), hard negatives, mirror pairs. Returns a dict of string lists + indices."""
    c = cfg.text
    pos, owner, claim_f, claim_pol, claim_org = [], [], [], [], []
    organ_normal = []                  # per volume: organs mentioned in its report with no 'present' claim (R2)
    intact_only = cfg.text.get('intact_only', False)
    neg_rows = []                      # (positive index, [edited strings])
    mirror = []                        # (volume index, sentence, side-swapped sentence)
    for i, claims in enumerate(batch['claims']):
        abn = {cl.get('organ') for cl in (claims or []) if cl.get('polarity') == 'present'}
        organ_normal.append({ORGAN_ID[cl['organ']] for cl in (claims or []) if cl.get('organ') in ORGAN_ID and cl['organ'] not in abn})
        if not claims:
            continue
        ok = intact_organs(batch['organ'][i], cfg.aug.min_organ_vox) if intact_only else None
        if intact_only:           # R3 (from RADAR): drop sentences about an organ that is missing or cut by the crop edge
            claims = [cl for cl in claims if cl.get('organ') not in ORGAN_ID or ORGAN_ID[cl['organ']] in ok]
            if not claims:
                continue
        sel = rng.choice(len(claims), size=min(c.max_sentences, len(claims)), replace=False)
        for k in sel:
            cl = claims[k]
            s = cl['text']
            claim_org.append(ORGAN_ID.get(cl.get('organ'), -1))
            j = len(pos)
            pos.append(s); owner.append(i)
            f = cl.get('finding'); p = cl.get('polarity')
            claim_f.append(FINDING_ID.get(f, -1) if f else -1)
            claim_pol.append(1 if p == 'present' else (0 if p == 'absent' else -1))
            edits = []
            for key in ('neg_text', 'swap_text'):
                e = cl.get(key)
                if isinstance(e, str) and e and e != s:
                    edits.append(e)
            # rule edits are contradictions only for stated abnormalities (negated/normal sentences can stay true)
            if p == 'present':
                for fn in ((lambda x: lateral_swap(x) if one_side_only(x) else None), severity_swap):
                    e = fn(s)
                    if e and e != s:
                        edits.append(e)
            if edits:
                neg_rows.append((j, edits[:c.max_edits]))
            ls = lateral_swap(s)
            if ls:
                mirror.append((i, s, ls))
    return {'pos': pos, 'owner': owner, 'claim_f': claim_f, 'claim_pol': claim_pol, 'claim_org': claim_org,
            'organ_normal': organ_normal, 'neg_rows': neg_rows, 'mirror': mirror}


def intact_organs(organ, min_vox):
    """Organ groups present in the crop (>= min_vox voxels) and not touching its border (R3, after RADAR)."""
    present = present_organs(organ, min_vox)
    faces = torch.cat([organ[0].flatten(), organ[-1].flatten(), organ[:, 0].flatten(), organ[:, -1].flatten(),
                       organ[:, :, 0].flatten(), organ[:, :, -1].flatten()])
    return present - set(torch.unique(faces).tolist())


def encode_strings(model, tok, strings, device, max_len):
    uniq = list(dict.fromkeys(strings))
    if not uniq:
        return {}, None
    enc = tok(uniq, max_length=max_len, padding='longest', truncation=True, return_tensors='pt')
    emb = model.encode_text(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    emb = model.attach_route(emb, uniq)                                  # no-op unless model.pool = fixed
    return {s: i for i, s in enumerate(uniq)}, emb


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(HERE, '../../configs/stageA.yaml'))
    ap.add_argument('--env', required=True)
    ap.add_argument('--ablation', default=None)
    ap.add_argument('overrides', nargs='*')
    a = ap.parse_args()
    cfg = load_config(a.config, env=a.env, ablation=a.ablation, overrides=a.overrides)
    os.environ.setdefault('HF_HOME', cfg.paths.hf_home)
    rank, world, device = setup_dist()
    torch.manual_seed(cfg.seed + rank); rng = np.random.default_rng(cfg.seed + rank)
    run_dir = os.path.join(cfg.paths.work_root, 'runs', cfg.run_name)
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
        from omegaconf import OmegaConf
        OmegaConf.save(cfg, os.path.join(run_dir, 'config.yaml'))

    # ---- data
    sentences = load_sentences(cfg)
    train_ds = CTDataset(cfg, load_manifests(cfg, 'train'), train=True, sentences=sentences)
    val_ds = CTDataset(cfg, load_manifests(cfg, 'val'), train=False, sentences=sentences)
    sampler = MixedBatchSampler(train_ds, cfg.train.batch_size, dict(cfg.data.dataset_weights), cfg.data.pair_frac,
                                rank, world, seed=cfg.seed, steps_per_epoch=cfg.train.get('steps_per_epoch'))
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=cfg.data.num_workers, collate_fn=collate,
                        pin_memory=True, persistent_workers=cfg.data.num_workers > 0, prefetch_factor=2 if cfg.data.num_workers else None)
    prev = np.nanmean(train_ds.labels, 0)

    # ---- model
    model = build_model(cfg).to(device)
    from colipri import get_processor
    tok = get_processor()._text_tokenizer
    with torch.no_grad():
        p = np.clip(np.nan_to_num(prev, nan=0.5), 1e-3, 1 - 1e-3)
        model.ev_bias.copy_(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))
        model.cls_bias.copy_(model.ev_bias)
    organ_of_query = model.queries.organ_of_query()
    groups = {'backbone': [], 'text': [], 'new': []}
    for n, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        groups['backbone' if n.startswith('backbone.') else 'text' if n.startswith('text.') else 'new'].append(prm)
    opt = torch.optim.AdamW([{'params': groups['backbone'], 'lr': cfg.optim.lr_backbone},
                             {'params': groups['text'], 'lr': cfg.optim.lr_text},
                             {'params': groups['new'], 'lr': cfg.optim.lr_new}],
                            weight_decay=cfg.optim.weight_decay, betas=(0.9, 0.98))
    base_lrs = [g['lr'] for g in opt.param_groups]
    core = ddp = model     # gradients are averaged explicitly (allreduce_grads): several forwards per step
    params = [p for p in model.parameters() if p.requires_grad]
    if world > 1:
        for p in params:
            dist.broadcast(p.data, 0)

    step, start = 0, time.time()
    ck = os.path.join(run_dir, 'last.pt')
    if cfg.train.resume and os.path.exists(ck):
        state = torch.load(ck, map_location='cpu', weights_only=False)
        core.load_state_dict(state['model']); opt.load_state_dict(state['opt']); step = state['step']
        if rank == 0:
            print(f'resumed from {ck} at step {step}', flush=True)

    canon = [s for f in FINDING_NAMES for s in canonical_prompts(f)]           # 2F strings: pos, neg interleaved
    W = cfg.loss
    log_f = open(os.path.join(run_dir, 'train_log.jsonl'), 'a') if rank == 0 else None
    epoch = 0
    if cfg.train.max_steps == 0:                         # evaluation-only run (e.g. A0: released COLIPRI)
        ev = evaluate(core, tok, val_ds, cfg, device, rank, world, canon)
        if rank == 0:
            print('EVAL', json.dumps(ev), flush=True)
            with open(os.path.join(run_dir, 'eval_log.jsonl'), 'a') as fh:
                fh.write(json.dumps(ev) + '\n')
    while step < cfg.train.max_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            if step >= cfg.train.max_steps:
                break
            lr_scale = min(1.0, (step + 1) / cfg.optim.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / cfg.train.max_steps)))
            for g, b in zip(opt.param_groups, base_lrs):
                g['lr'] = b * lr_scale
            t0 = time.time()
            logs = step_fn(core, tok, batch, cfg, W, rng, device, rank, world, canon, organ_of_query, step, opt, params)
            logs.update(step=step, lr=opt.param_groups[0]['lr'], sec=time.time() - t0)
            if rank == 0 and step % cfg.train.log_every == 0:
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in logs.items()}), flush=True)
                log_f.write(json.dumps(logs) + '\n'); log_f.flush()
            step += 1
            if step % cfg.train.eval_every == 0 or step == cfg.train.max_steps:
                ev = evaluate(core, tok, val_ds, cfg, device, rank, world, canon)
                if rank == 0:
                    ev['step'] = step
                    print('EVAL', json.dumps(ev), flush=True)
                    with open(os.path.join(run_dir, 'eval_log.jsonl'), 'a') as fh:
                        fh.write(json.dumps(ev) + '\n')
            if rank == 0 and (step % cfg.train.save_every == 0 or step == cfg.train.max_steps):
                torch.save({'model': core.state_dict(), 'opt': opt.state_dict(), 'step': step}, ck + '.tmp')
                os.replace(ck + '.tmp', ck)
        epoch += 1
    if rank == 0:
        print(f'done {step} steps in {(time.time() - start) / 3600:.2f} h', flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


def slot_swap_loss(core, T, c_emb, batch, organ_of_query, cfg, rng, device):
    """U4. T (B, K, d) clean tokens; c_emb (2F, d) canonical prompts (detached). For each scan i, a partner j of the
    same dataset and an organ group (the organs of one finding) present in both: swap that group's slots j -> i.
    Findings whose organs meet the group follow j; findings whose organs are all controls of the group stay with i."""
    B = T.shape[0]
    oq = organ_of_query.to(device)
    pres = [present_organs(batch['organ'][b], cfg.aug.min_organ_vox) for b in range(B)]
    terms, n = [], 0
    with torch.no_grad():
        S0 = core.score(T, c_emb).float(); D0 = S0[:, 0::2] - S0[:, 1::2]           # (B, F) unswapped scores
    for i in range(B):
        js = [j for j in range(B) if j != i and batch['dataset'][j] == batch['dataset'][i]]
        if not js:
            continue
        j = js[int(rng.integers(len(js)))]
        fam = [f for f in FINDING_NAMES if f.startswith(FAMILY[batch['dataset'][i]])]
        groups = [tuple(finding_organ_ids(f)) for f in fam if finding_organ_ids(f)]
        groups = [g for g in dict.fromkeys(groups) if all(o in pres[i] and o in pres[j] for o in g)]
        if not groups:
            continue
        g = groups[int(rng.integers(len(groups)))]
        slots = torch.isin(oq, torch.as_tensor(g, device=device))
        if not slots.any():
            continue
        Tsw = T[i].clone(); Tsw[slots] = T[j][slots]
        S = core.score(Tsw[None], c_emb).float(); D = (S[:, 0::2] - S[:, 1::2])[0]
        follow = [FINDING_ID[f] for f in fam if set(finding_organ_ids(f)) & set(g)]
        gctrl = set.intersection(*[set(control_organ_ids(f)) for f in fam if tuple(finding_organ_ids(f)) == g])
        stay = [FINDING_ID[f] for f in fam if finding_organ_ids(f) and set(finding_organ_ids(f)) <= gctrl]
        if follow:
            terms.append(F.huber_loss(D[follow], D0[j, follow], delta=cfg.loss.huber_delta))
        if stay:
            terms.append(cfg.loss.get('swap_eta', 1.0) * F.huber_loss(D[stay], D0[i, stay], delta=cfg.loss.huber_delta))
        n += 1
    return (sum(terms) if terms else T.sum() * 0), n


def step_fn(core, tok, batch, cfg, W, rng, device, rank, world, canon, organ_of_query, step, opt, params):
    logs = {}
    B = len(batch['volume'])
    plans = plan_counterfactuals(batch, cfg, rng) if W.ev > 0 else [None] * B
    txt = build_text(batch, cfg, rng)
    mirror = [m for m in txt['mirror'] if rng.random() < cfg.aug.mirror_prob] if W.mir > 0 else []
    mirror_vols = sorted({m[0] for m in mirror})
    x, idx = build_image_batch(batch, plans, mirror_vols, cfg, device, seed=cfg.seed * 100003 + step * 131 + rank)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        T, A, _ = core.encode_image(x)
        strings = txt['pos'] + [e for _, es in txt['neg_rows'] for e in es] + canon + [m[2] for m in mirror]
        s_idx, E = encode_strings(core, tok, strings, device, cfg.text.max_len)

        loss = torch.zeros((), device=device)
        labels = batch['labels'].to(device)
        # ---- base contrastive (gathered across ranks)
        if txt['pos']:
            t_pos = E[[s_idx[s] for s in txt['pos']]]
            T_clean = T[:B]
            T_all, vc = gather_padded(T_clean)
            t_all, sc = gather_padded(t_pos)
            lab_all, _ = gather_padded(labels, grad=False)
            meta = torch.tensor([[o + offset(vc, rank), f, p, g] for o, f, p, g in
                                 zip(txt['owner'], txt['claim_f'], txt['claim_pol'], txt['claim_org'])], device=device, dtype=torch.long)
            meta_all, _ = gather_padded(meta, grad=False)
            S = core.score(T_all, t_all).float()                                  # (Vg, Sg)
            own = torch.zeros_like(S, dtype=torch.bool)
            own[meta_all[:, 0], torch.arange(len(meta_all), device=device)] = True
            f_all, p_all = meta_all[:, 1], meta_all[:, 2]
            valid = (f_all >= 0) & (p_all >= 0)
            lab_f = lab_all[:, f_all.clamp(min=0)]                                # (Vg, Sg) label of each sentence's finding
            true_mask = valid[None] & ~torch.isnan(lab_f) & (lab_f == p_all[None].float())
            g_all = meta_all[:, 3]
            if cfg.loss.get('normal_pos', False):
                # R2 (from RADAR): a sentence stating an organ is normal / a finding absent is also true for every scan
                # whose report mentions that organ with nothing abnormal, so it is not that scan's negative
                on = torch.zeros(B, len(ORGAN_NAMES), dtype=torch.bool, device=device)
                for b, s_ in enumerate(txt['organ_normal']):
                    if s_:
                        on[b, list(s_)] = True
                on_all, _ = gather_padded(on.float(), grad=False)
                stmt = (p_all == 0) & (g_all >= 0)
                true_mask = true_mask | (stmt[None] & (on_all[:, g_all.clamp(min=0)] > 0))
            unknown = None
            ign = list(cfg.loss.get('ignore_unknown', []) or [])
            if ign:
                # partial labels (datasets in loss.ignore_unknown ask only some findings per scan): a parsed sentence
                # about a finding of the scan's own region whose label was never asked is neither true nor false for
                # that scan, so it is left out of the contrast instead of counting as a negative. Rows of other
                # datasets (complete labels) are unchanged. Not used by the released models (public data only).
                fam_id = {'ct_': 1.0, 'mr_': 2.0}
                rinfo = torch.tensor([[float(d in ign), fam_id.get(FAMILY.get(d, ''), 0.0)] for d in batch['dataset']],
                                     device=device)
                rinfo_all, _ = gather_padded(rinfo, grad=False)                     # padded rows are (0, 0)
                f_fam = torch.tensor([fam_id.get(n[:3], -1.0) for n in FINDING_NAMES], device=device)[f_all.clamp(min=0)]
                unknown = (valid[None] & torch.isnan(lab_f) & (rinfo_all[:, :1] > 0)
                           & (rinfo_all[:, 1:2] == f_fam[None]) & ~true_mask)
                logs['unk_frac'] = unknown.float().mean().item()
            r0, c0 = offset(vc, rank), offset(sc, rank)
            rows = slice(r0, r0 + B); cols = slice(c0, c0 + len(t_pos))
            l_i2t = i2t_loss(S[rows], own[rows], true_mask[rows], None if unknown is None else unknown[rows])
            l_t2i = t2i_loss(S[:, cols], own[:, cols], true_mask[:, cols], cfg.loss.lambda_q,
                             None if unknown is None else unknown[:, cols])
            loss = loss + W.i2t * l_i2t + W.t2i * l_t2i
            logs.update(i2t=l_i2t.item(), t2i=l_t2i.item(), n_sent=len(t_pos))
            if W.get('report', 0) > 0:
                # A8: report-level contrast - a scan's score for a report is the mean of its scores for that report's
                # sampled sentences (the retrieval rule); each local scan finds its own report among the global batch's
                # and each local report its own scan
                o = own.float()                                                   # (Vg reports, Sg sentences)
                n_o = o.sum(1)
                R = (S @ o.T) / n_o.clamp(min=1)[None]                            # (Vg scans, Vg reports)
                R = R.masked_fill((n_o == 0)[None], float('-inf'))
                tgt = torch.arange(r0, r0 + B, device=device)
                has = n_o[tgt] > 0
                if has.any():
                    l_rep = 0.5 * (F.cross_entropy(R[rows][has], tgt[has]) +
                                   F.cross_entropy(R[:, rows].T[has], tgt[has]))
                    loss = loss + W.report * l_rep; logs['report'] = l_rep.item()
            if W.get('organ_con', 0) > 0:
                # R1 (from RADAR): same-organ contrast - each scan's sentence about organ o against the o-sentences of
                # the other scans in the global batch (hard negatives), false negatives removed as above
                terms = []
                for o in torch.unique(g_all[g_all >= 0]).tolist():
                    cols_o = torch.where(g_all == o)[0]
                    sub_own, sub_true = own[rows][:, cols_o], true_mask[rows][:, cols_o]
                    if sub_own.any() and len(cols_o) > 1:
                        terms.append(i2t_loss(S[rows][:, cols_o], sub_own, sub_true))
                if terms:
                    l_oc = torch.stack(terms).mean(); loss = loss + W.organ_con * l_oc; logs['organ_con'] = l_oc.item()
            # ---- hard-negative sentences
            if W.txt > 0 and txt['neg_rows']:
                j_idx = [j for j, _ in txt['neg_rows']]
                Emax = max(len(es) for _, es in txt['neg_rows'])
                owners = [txt['owner'][j] for j in j_idx]
                s_true = S[r0 + torch.tensor(owners, device=device), c0 + torch.tensor(j_idx, device=device)]
                neg_emb = [E[[s_idx[e] for e in es]] for _, es in txt['neg_rows']]
                s_neg = torch.full((len(j_idx), Emax), float('-inf'), device=device)
                for r, (o, ne) in enumerate(zip(owners, neg_emb)):
                    s_neg[r, :len(ne)] = core.score(T[o:o + 1], ne)[0].float()
                l_txt = txt_loss(s_true, s_neg)
                loss = loss + W.txt * l_txt
                logs['txt'] = l_txt.item()

        # ---- finding logits with canonical prompts for every image pass
        c_emb = E[[s_idx[s] for s in canon]]
        Sc = core.score(T, c_emb).float()
        delta = Sc[:, 0::2] - Sc[:, 1::2]                                        # (N_pass, F)
        if W.cls > 0:
            l_cls = cls_loss(delta[:B], labels, core.cls_bias)
            loss = loss + W.cls * l_cls; logs['cls'] = l_cls.item()
        # ---- core: evidence term
        if W.ev > 0 and (idx['sens'] or idx['inv']):
            sens_i = list(idx['sens']); inv_i = list(idx['inv'])
            f_s = torch.tensor([plans[i]['finding'] for i in sens_i], device=device, dtype=torch.long)
            f_v = torch.tensor([plans[i]['finding'] for i in inv_i], device=device, dtype=torch.long)
            # the evidence term constrains the image side only: with gradients into the text tower, the cheapest way to
            # meet it is to pull the canonical positive/negative prompts together (constant Delta), which kills zero-shot
            dl = delta
            if cfg.loss.get('ev_detach_text', True):
                Sd = core.score(T, c_emb.detach()).float()
                dl = Sd[:, 0::2] - Sd[:, 1::2]
            d_s = dl[[idx['sens'][i] for i in sens_i], f_s] if sens_i else delta.new_zeros(0)
            d_v = dl[[idx['inv'][i] for i in inv_i], f_v] if inv_i else delta.new_zeros(0)
            d_c = dl[inv_i, f_v] if inv_i else delta.new_zeros(0)
            mode = cfg.get('ev_target_mode', 'prior')
            if mode == 'inv_only':
                d_s, f_s = d_s[:0], f_s[:0]
            # targets on the calibrated scale used by cls_loss: logit_f = Delta_f + c_f. Erasure should give the prior,
            # logit = b_f (ev_bias, initialised at the prior log-odds), i.e. Delta_f -> b_f - c_f (0 at init);
            # ablation B3 ('absent') asks for logit = -absent_logit instead.
            calib = core.cls_bias.detach()
            if mode != 'absent':
                bias = core.ev_bias - calib
            else:
                bias = torch.full_like(core.ev_bias, -cfg.loss.get('absent_logit', 5.0)) - calib
            l_ev = evidence_loss(d_c, d_s, d_v, f_s, f_v, bias, cfg.loss.eta, cfg.loss.huber_delta)
            # schedule: off while contrastive alignment forms, then linear ramp to full weight
            n = cfg.train.max_steps
            s0, r = cfg.loss.get('ev_start_frac', 0.2) * n, max(1.0, cfg.loss.get('ev_ramp_frac', 0.2) * n)
            w_ev = W.ev * min(1.0, max(0.0, (step - s0) / r))
            loss = loss + w_ev * l_ev
            logs['w_ev'] = w_ev
            logs.update(ev=l_ev.item(), n_sens=len(sens_i), n_inv=len(inv_i))
        # ---- D1 mirror equivariance
        if W.mir > 0 and mirror:
            vi = torch.tensor([idx['mirror'][m[0]] for m in mirror], device=device)
            e_orig = E[[s_idx[m[1]] for m in mirror]]; e_swap = E[[s_idx[m[2]] for m in mirror]]
            s_sw = torch.stack([core.score(T[v:v + 1], e_swap[k:k + 1])[0, 0] for k, v in enumerate(vi.tolist())]).float()
            s_or = torch.stack([core.score(T[v:v + 1], e_orig[k:k + 1])[0, 0] for k, v in enumerate(vi.tolist())]).float()
            l_mir = mirror_loss(s_sw, s_or)
            loss = loss + W.mir * l_mir; logs['mir'] = l_mir.item()
        # ---- U4 organ-slot swap consistency: one organ group's slots move from scan j into scan i; that organ's finding
        # scores must follow scan j, findings of unrelated organs must stay with scan i. A feature-level counterfactual:
        # no image editing (no fill artefacts) and valid for size findings (the slots carry the organ's size).
        if W.get('swap', 0) > 0:
            l_sw, n_sw = slot_swap_loss(core, T[:B], c_emb.detach(), batch, organ_of_query, cfg, rng, device)
            if n_sw:
                n = cfg.train.max_steps
                s0, r = cfg.loss.get('ev_start_frac', 0.2) * n, max(1.0, cfg.loss.get('ev_ramp_frac', 0.2) * n)
                w_sw = W.swap * min(1.0, max(0.0, (step - s0) / r))
                loss = loss + w_sw * l_sw; logs.update(swap=l_sw.item(), n_swap=n_sw, w_swap=w_sw)
        # ---- D2 label-matched pairs (inside the local batch)
        if W.pair > 0:
            l_pair = pair_loss(delta[:B], labels, cfg.loss.pair_tau)
            loss = loss + W.pair * l_pair; logs['pair'] = l_pair.item()
        # ---- D3 anatomy prior
        if W.anat > 0:
            l_anat = anatomy_loss(A[:B], batch['organ'].to(device), organ_of_query, len(ORGAN_NAMES))
            loss = loss + W.anat * l_anat; logs['anat'] = l_anat.item()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    allreduce_grads(params, world)
    if os.environ.get('DEBUG_GN') and rank == 0:
        top = sorted(((float(p.grad.norm()), n) for n, p in core.named_parameters() if p.grad is not None), reverse=True)[:8]
        print('TOPGN', [(n, round(v, 1)) for v, n in top], flush=True)
    if step % cfg.train.log_every == 0:
        for gname, gp in core.param_groups_for_logging().items():
            gs = [p.grad.norm() for p in gp if p.grad is not None]
            logs[f'gn_{gname}'] = float(torch.stack(gs).norm()) if gs else 0.0
    gn = sum(float(torch.nn.utils.clip_grad_norm_(g['params'], cfg.optim.grad_clip)) ** 2 for g in opt.param_groups) ** 0.5
    opt.step()
    logs.update(loss=float(loss), grad_norm=float(gn), n_pass=int(x.shape[0]), mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    return logs


@torch.no_grad()
def evaluate(core, tok, val_ds, cfg, device, rank, world, canon):
    """Zero-shot AUROC of Delta_f on (a slice of) the validation split, per labelled finding."""
    core.eval()
    ids = val_indices(val_ds, cfg.eval.n_val, rank, world)   # equal share of CT-RATE and Merlin
    enc = tok(canon, max_length=cfg.text.max_len, padding='longest', truncation=True, return_tensors='pt')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        c_emb = core.encode_text(enc['input_ids'].to(device), enc['attention_mask'].to(device))
        c_emb = core.attach_route(c_emb, canon)
    out_d, out_y = [], []
    # 2 workers: the training workers stay alive during validation, and each worker holds a copy of the report data
    loader = DataLoader(torch.utils.data.Subset(val_ds, ids), batch_size=cfg.eval.batch_size,
                        num_workers=min(2, cfg.data.num_workers), collate_fn=collate)
    for b in loader:
        x = (b['img'].to(device).float().clamp(-1000, 1000) / 1000)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            T, _, _ = core.encode_image(x)
            Sc = core.score(T, c_emb).float()
        out_d.append((Sc[:, 0::2] - Sc[:, 1::2]).cpu()); out_y.append(b['labels'])
    d = torch.cat(out_d) if out_d else torch.zeros(0, len(FINDING_NAMES))
    y = torch.cat(out_y) if out_y else torch.zeros(0, len(FINDING_NAMES))
    if dist.is_initialized():
        d, _ = gather_padded(d.to(device), grad=False); y, _ = gather_padded(y.to(device), grad=False)
    core.train()
    d, y = d.cpu().numpy(), y.cpu().numpy()
    res = {}
    for j, f in enumerate(FINDING_NAMES):
        if np.nansum(y[:, j]) >= cfg.eval.min_pos:
            res[f] = auroc(y[:, j], d[:, j])
    vals = [v for v in res.values() if v == v]
    res['macro_ct'] = float(np.mean([v for k, v in res.items() if k.startswith('ct_') and v == v])) if any(k.startswith('ct_') for k in res) else None
    res['macro_mr'] = float(np.mean([v for k, v in res.items() if k.startswith('mr_') and v == v])) if any(k.startswith('mr_') for k in res) else None
    res['n_val'] = int(len(y))
    return res


if __name__ == '__main__':
    main()
