"""Stage B (core 2): 3D evidence tokens + Qwen3.5-4B (LoRA), trained on per-finding questions and reports.

Terms (weights in cfg.lossB; 0 disables):
  lm    teacher-forced CE on answers ("Yes"/"No") and on report findings text
  dep   dependence margin (core 2):
          pair   [m - (log p(a_i|V_i) - log p(a_i|V_j))]_+   for label-matched partners (i, j) on finding f
          erase  [m - (log p(Yes|V) - log p(Yes|E_o V))]_+   for positive answers whose organ is in the crop
          inv    |log p(a|V) - log p(a|E_U V)|               unrelated organs erased
  dpo   D5, reference-free image-side preference: -log sigmoid(beta * (log p(a|V) - log p(a|V_cf)))
  read  D4, LLM hidden state at the visual positions predicts the encoder's Delta_f (stage-A margin)
  rep_dep  report-level dependence: [m_r - (mean token log p(report_i|V_i) - mean token log p(report_i|V_j))]_+ with
           j another scan of the batch (same dataset when possible)
Reports (vlm.report_mode): 'free' = the findings text alone (rounds 1-10); 'list' = answer-then-write, the target is
"Abnormal findings: <phrases of the labelled positives, or none>.\nReport: <findings text>", so the report is written
conditioned on an explicit finding list. At inference the list is either generated or prefilled from the model's own
yes/no answers (evaluate_stageB --gen, extaudit/vlm_models/organic3d_vlm.py); only the text after 'Report:' is scored.

    torchrun --nproc_per_node 4 -m organic.train_stageB --env configs/env/local.yaml --ablation configs/vlm/vlm4_organic3d.yaml
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

from .augment import erase, normalise, organ_mask, present_organs
from .config import load_config
from .data import CTDataset, MixedBatchSampler, collate, load_manifests, val_indices
from .train_stageA import allreduce_grads, auroc, gather_padded, setup_dist
from .vlm import Organic3DVLM, build_encoder, sync_encoder_cfg
from .vocab import (FAMILY, FINDINGS, FINDING_NAMES, REGION, SIZE_FINDINGS, canonical_prompts, control_organ_ids,
                    finding_organ_ids)


def organ_load(f):
    """Number of findings of f's dataset whose organ group overlaps f's (lungs: 10 CT-RATE findings share 16 tokens)."""
    org, fam = set(finding_organ_ids(f)), f.split('_')[0]
    return sum(1 for g in FINDING_NAMES if g.split('_')[0] == fam and org & set(finding_organ_ids(g)))


HERE = os.path.dirname(__file__)
LIST_HEAD, REPORT_HEAD = 'Abnormal findings:', 'Report:'


def question(f):
    return f'Is there {FINDINGS[f][2]} in this CT scan? Answer Yes or No.'


def report_prompt(dataset, mode='free'):
    """Report request; in 'list' mode (answer-then-write) the model first lists the abnormal findings."""
    p = f'Write the findings section of the radiology report for this {REGION[dataset]} CT.'
    if mode == 'list':
        p += ' First list the abnormal findings, then write the report.'
    return p


def finding_list(findings):
    """Answer-then-write prefix for a list of finding names (vocabulary order)."""
    ph = [FINDINGS[f][2] for f in findings]
    return f"{LIST_HEAD} {', '.join(ph) if ph else 'none'}.\n{REPORT_HEAD} "


def split_report(text):
    """(finding list, report body) of an answer-then-write output; ('', text) for a free report."""
    t = str(text or '')
    k = t.find(REPORT_HEAD)
    if t.lstrip().startswith(LIST_HEAD) and k >= 0:
        return t[:k].strip(), t[k + len(REPORT_HEAD):].strip()
    return '', t.strip()


def report_target(batch, i, labels, c):
    """Training target of report row i: the findings text, in 'list' mode preceded by its labelled positives."""
    rep = str(batch_report(batch, i, c))[:c.report_chars]
    if c.get('report_mode', 'free') != 'list':
        return rep
    fam = FAMILY[batch['dataset'][i]]
    pos = [f for j, f in enumerate(FINDING_NAMES) if f.startswith(fam) and labels[i, j] == 1]
    return finding_list(pos) + rep


class PairSampler(MixedBatchSampler):
    """Pairs for the dependence margin. 'matched': partner identical on every other labelled finding (default);
    'unmatched': any opposite-label patient of the same dataset (Pair-DPO-style ablation)."""

    def __init__(self, *a, pair_mode='matched', **k):
        self.pair_mode = pair_mode
        super().__init__(*a, **k)

    def _build_pairs(self):
        if self.pair_mode == 'matched':
            return super()._build_pairs()
        L = self.ds.labels
        pairs = {}
        for d, idx in self.groups.items():
            for f in np.where(~np.isnan(L[idx]).all(0))[0]:
                pos = idx[L[idx, f] == 1]; neg = list(idx[L[idx, f] == 0])
                if len(pos) >= 5 and neg:
                    pairs[int(f)] = [(int(p), neg) for p in pos]
        return pairs


def find_pairs(labels, datasets, mode='matched'):
    """Consecutive (i, i+1) that the sampler emitted as a pair (positive first): same dataset and opposite on exactly
    one defined finding (matched), or on at least one (unmatched: the first opposite finding is used)."""
    out, i, used = [], 0, set()
    L = labels.numpy()
    while i + 1 < len(L):
        a, b = L[i], L[i + 1]
        both = ~np.isnan(a) & ~np.isnan(b)
        diff = np.where(both & (a != b))[0]
        ok = len(diff) == 1 if mode == 'matched' else len(diff) >= 1
        if datasets[i] == datasets[i + 1] and ok and a[diff[0]] == 1:
            out.append((i, i + 1, int(diff[0]))); used |= {i, i + 1}; i += 2
        else:
            i += 1
    return out, used


def pick_finding(lab, rng):
    """A labelled finding for a single sample; positives are oversampled to half the questions when present."""
    known = np.where(~np.isnan(lab))[0]
    if not len(known):
        return None
    pos = known[lab[known] == 1]
    if len(pos) and rng.random() < 0.5:
        return int(pos[rng.integers(len(pos))])
    return int(known[rng.integers(len(known))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(HERE, '../configs/stageB.yaml'))
    ap.add_argument('--env', required=True)
    ap.add_argument('--ablation', default=None)
    ap.add_argument('overrides', nargs='*')
    a = ap.parse_args()
    base = load_config(os.path.join(HERE, '../configs/stageA.yaml'))
    cfg = load_config(a.config, env=a.env, ablation=a.ablation, overrides=a.overrides)
    from omegaconf import OmegaConf
    cfg = OmegaConf.merge(base, cfg)
    sync_encoder_cfg(cfg)
    os.environ.setdefault('HF_HOME', cfg.paths.hf_home)
    rank, world, device = setup_dist()
    torch.manual_seed(cfg.seed + rank); rng = np.random.default_rng(cfg.seed + rank)
    run_dir = os.path.join(cfg.paths.work_root, 'runsB', cfg.run_name)
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(run_dir, 'config.yaml'))

    c, W = cfg.vlm, cfg.lossB
    train_ds = CTDataset(cfg, load_manifests(cfg, 'train'), train=True)
    val_ds = CTDataset(cfg, load_manifests(cfg, 'val'), train=False)
    pair_frac = c.pair_frac if (W.dep > 0 or W.dpo > 0) else 0.0
    sampler = PairSampler(train_ds, c.batch_size, dict(cfg.data.dataset_weights), pair_frac, rank, world,
                          seed=cfg.seed, pair_mode=c.pair_mode)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=cfg.data.num_workers, collate_fn=collate,
                        pin_memory=True, persistent_workers=cfg.data.num_workers > 0,
                        prefetch_factor=2 if cfg.data.num_workers else None)

    encoder = build_encoder(cfg, device)
    model = Organic3DVLM(cfg, encoder).to(device)
    for n, p in model.named_parameters():                     # fp32 master weights for everything trained
        if p.requires_grad:
            p.data = p.data.float()
    canon_emb = None
    if W.read > 0:
        from colipri import get_processor
        tok_e = get_processor()._text_tokenizer
        canon = [s for f in FINDING_NAMES for s in canonical_prompts(f)]
        enc = tok_e(canon, max_length=cfg.text.max_len, padding='longest', truncation=True, return_tensors='pt')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            canon_emb = encoder.encode_text(enc['input_ids'].to(device), enc['attention_mask'].to(device)).float()
            canon_emb = encoder.attach_route(canon_emb, canon)                 # no-op unless encoder pool = fixed
    params = [p for p in model.parameters() if p.requires_grad]
    if world > 1:
        for p in params:
            dist.broadcast(p.data, 0)
    opt = torch.optim.AdamW(params, lr=c.lr, weight_decay=c.weight_decay, betas=(0.9, 0.98))
    if rank == 0:
        print('trainable params (M):', round(sum(p.numel() for p in params) / 1e6, 2), flush=True)

    step, ck = 0, os.path.join(run_dir, 'last.pt')
    if c.resume and os.path.exists(ck):
        st = torch.load(ck, map_location='cpu', weights_only=False)
        model.load_state_dict(st['model'], strict=False); opt.load_state_dict(st['opt']); step = st['step']
    log_f = open(os.path.join(run_dir, 'train_log.jsonl'), 'a') if rank == 0 else None
    t_start, epoch = time.time(), 0
    while step < c.max_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            if step >= c.max_steps:
                break
            lr = c.lr * min(1.0, (step + 1) / c.warmup_steps) * 0.5 * (1 + math.cos(math.pi * step / c.max_steps))
            for g in opt.param_groups:
                g['lr'] = lr
            t0 = time.time()
            logs = step_fn(model, batch, cfg, rng, device, step, canon_emb)
            opt.zero_grad(set_to_none=True)
            logs['loss_t'].backward()
            allreduce_grads(params, world)
            gn = torch.nn.utils.clip_grad_norm_(params, c.grad_clip)
            opt.step()
            logs = {k: (float(v) if torch.is_tensor(v) else v) for k, v in logs.items() if k != 'loss_t'}
            logs.update(step=step, lr=lr, sec=time.time() - t0, grad_norm=float(gn),
                        mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            if rank == 0 and step % c.log_every == 0:
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in logs.items()}), flush=True)
                log_f.write(json.dumps(logs) + '\n'); log_f.flush()
            step += 1
            if step % c.eval_every == 0 or step == c.max_steps:
                ev = evaluate_qa(model, val_ds, cfg, device, rank, world)
                if rank == 0:
                    ev['step'] = step
                    print('EVAL', json.dumps(ev), flush=True)
                    with open(os.path.join(run_dir, 'eval_log.jsonl'), 'a') as fh:
                        fh.write(json.dumps(ev) + '\n')
            if rank == 0 and (step % c.save_every == 0 or step == c.max_steps):
                torch.save({'model': model.trainable_state(), 'opt': opt.state_dict(), 'step': step}, ck + '.tmp')
                os.replace(ck + '.tmp', ck)
        epoch += 1
    if rank == 0:
        print(f'done {step} steps in {(time.time() - t_start) / 3600:.2f} h', flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


def step_fn(model, batch, cfg, rng, device, step, canon_emb):
    c, W, A = cfg.vlm, cfg.lossB, cfg.aug
    B = len(batch['volume'])
    labels = batch['labels']
    hu = batch['img'].to(device, non_blocking=True).float()          # (B,1,D,H,W) HU
    organ = batch['organ'].to(device, non_blocking=True)
    gen = torch.Generator(device=device); gen.manual_seed(cfg.seed * 7919 + step)
    pairs, in_pair = find_pairs(labels, batch['dataset'], c.pair_mode)

    # ---- what each sample is asked
    qa = {}                                                          # sample -> finding
    for i, j, f in pairs:
        qa[i] = f; qa[j] = f
    report_rows = []
    for i in range(B):
        if i in qa:
            continue
        if rng.random() < c.report_frac and batch_report(batch, i, c):
            report_rows.append(i)
        else:
            f = pick_finding(labels[i].numpy(), rng)
            if f is not None:
                qa[i] = f

    # ---- counterfactual volumes for the organ-erased and unrelated-erasure terms
    need_cf = W.dep > 0 or W.dpo > 0
    vols, erased, unrel = [hu[:, 0]], {}, {}
    if need_cf:
        extra = []
        for i, f in qa.items():
            fname = FINDING_NAMES[f]
            present = present_organs(organ[i], A.min_organ_vox)
            fill = A.fills[int(rng.integers(len(A.fills)))]
            tgt = [o for o in finding_organ_ids(fname) if o in present]
            skip = fname in A.get('holdout_findings', []) or (A.get('ev_skip_size', False) and fname in SIZE_FINDINGS)
            if labels[i, f] == 1 and tgt and not skip:
                erased[i] = B + len(extra)
                extra.append(erase(hu[i, 0], organ_mask(organ[i], tgt, A.dilate), fill, A.noise_sd, gen))
            ctrl = [o for o in control_organ_ids(fname) if o in present]
            if ctrl:
                U = list(rng.choice(ctrl, size=min(len(ctrl), int(rng.integers(1, A.max_controls + 1))), replace=False))
                unrel[i] = B + len(extra)
                extra.append(erase(hu[i, 0], organ_mask(organ[i], [int(u) for u in U], A.dilate), fill, A.noise_sd, gen))
        if extra:
            vols.append(torch.stack(extra))
    x = normalise(torch.cat(vols)[:, None])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        Vp, Traw = model.visual_tokens(x)                              # (N, K, d_llm)

    # ---- QA rows: clean, partner-crossed, erased, unrelated-erased
    rows_v, rows_q, rows_a, tags = [], [], [], []
    def add(v_idx, i, tag, ans=None):
        f = qa[i]
        rows_v.append(v_idx); rows_q.append(question(FINDING_NAMES[f]))
        rows_a.append(ans or ('Yes' if labels[i, f] == 1 else 'No')); tags.append((tag, i))
    for i in qa:
        add(i, i, 'clean')
    for i, j, f in pairs:
        if W.dep > 0 or W.dpo > 0:
            add(j, i, 'cross'); add(i, j, 'cross')                    # a_i scored on the partner's scan and vice versa
    for i, k in erased.items():
        add(k, i, 'erased')
    if W.dep > 0 and W.inv > 0:
        for i, k in unrel.items():
            add(k, i, 'unrel')
    # U6 organ-token swap: in a matched pair (i, j differ only in finding f), scan i with f's organ tokens taken from j.
    # The organ now comes from a patient with the opposite label, so i's answer must become clearly less likely.
    n_ts_skip = 0
    if W.get('tswap', 0) > 0 and pairs:
        tok_org = model.token_organ.to(Vp.device)
        extra_v = []
        cap = W.get('tswap_max_findings', 0)
        for i, j, f in pairs:
            org = finding_organ_ids(FINDING_NAMES[f])
            if not org or (cap and organ_load(FINDING_NAMES[f]) > cap):
                n_ts_skip += 1
                continue
            on = torch.isin(tok_org, torch.as_tensor(org, device=Vp.device))
            for a_, b_ in ((i, j), (j, i)):
                v = Vp[a_].clone(); v[on] = Vp[b_][on]
                extra_v.append((a_, v))
        if extra_v:
            base = Vp.shape[0]
            Vp = torch.cat([Vp, torch.stack([v for _, v in extra_v])])
            for k, (a_, _) in enumerate(extra_v):
                add(base + k, a_, 'tswap')
    logs, loss = {'n_qa': len(qa), 'n_pairs': len(pairs), 'n_erased': len(erased), 'n_report': len(report_rows),
                  'n_tswap_skip': n_ts_skip}, 0.0
    read_layer = c.get('readout_layer') if W.read > 0 else None
    with torch.autocast('cuda', dtype=torch.bfloat16):
        if rows_v:
            want_att = W.get('att', 0) > 0 or c.get('log_att', True)
            got = model.answer_logprob(Vp[rows_v], rows_q, rows_a, device, hidden_layer=read_layer, attention=want_att)
            lp, ce_qa, hs = got[:3]
            if want_att:
                # B (organ-routed attention prior): for clean rows whose finding has an organ, the share of the visual
                # attention (first answer prediction, full-attention layers att_layers) that falls on that organ's tokens
                p_vis = got[3]                                                       # (R, layers, heads, K)
                tok_org = model.token_organ.to(device)
                share = []
                for n, (tag, i) in enumerate(tags):
                    org = finding_organ_ids(FINDING_NAMES[qa[i]]) if tag == 'clean' else []
                    if org:
                        on = torch.isin(tok_org, torch.as_tensor(org, device=device))
                        share.append(p_vis[n][..., on].sum(-1) / p_vis[n].sum(-1).clamp_min(1e-8))   # (layers, heads)
                if share:
                    s = torch.stack(share)
                    logs['att_share'] = s.mean().detach()
                    logs['att_vis_mass'] = p_vis.sum(-1).mean().detach()
                    if W.get('att', 0) > 0:
                        t = -(s.mean((1, 2)) + 1e-6).log().mean(); loss = loss + W.att * t; logs['att'] = t.detach()
            clean_ids = [n for n, (t, _) in enumerate(tags) if t == 'clean']
            ce_clean = -(lp[clean_ids].mean())
            loss = loss + W.lm * ce_clean
            logs['ce_qa'] = ce_clean.detach()
            lp_of = {(t, i): lp[n] for n, (t, i) in enumerate(tags)}
            m = W.margin
            d_pair = [lp_of[('clean', i)] - lp_of[('cross', i)] for i in qa if ('cross', i) in lp_of]
            d_er = [lp_of[('clean', i)] - lp_of[('erased', i)] for i in erased]
            d_inv = [lp_of[('clean', i)] - lp_of[('unrel', i)] for i in unrel if ('unrel', i) in lp_of]
            d_ts = any(t == 'tswap' for t, _ in tags)
            if d_ts and W.get('tswap', 0) > 0:
                dd = torch.stack([lp_of[('clean', i)] - lp[n0] for n0, (t, i) in enumerate(tags) if t == 'tswap'])
                t_ = F.relu(m - dd).mean(); loss = loss + W.tswap * t_
                logs['tswap'] = t_.detach(); logs['D_tswap'] = dd.mean().detach()
            if W.dep > 0:
                l_dep = 0.0
                if d_pair and W.pair > 0:
                    t = F.relu(m - torch.stack(d_pair)).mean(); l_dep = l_dep + W.pair * t; logs['dep_pair'] = t.detach()
                    logs['D_pair'] = torch.stack(d_pair).mean().detach()
                if d_er and W.erase > 0:
                    t = F.relu(m - torch.stack(d_er)).mean(); l_dep = l_dep + W.erase * t; logs['dep_erase'] = t.detach()
                    logs['D_erase'] = torch.stack(d_er).mean().detach()
                if d_inv and W.inv > 0:
                    t = torch.stack(d_inv).abs().mean(); l_dep = l_dep + W.inv * t; logs['dep_inv'] = t.detach()
                if torch.is_tensor(l_dep):
                    loss = loss + W.dep * l_dep
            if W.dpo > 0 and (d_pair or d_er):
                d = torch.stack(d_pair + d_er)
                t = -F.logsigmoid(W.dpo_beta * d).mean(); loss = loss + W.dpo * t; logs['dpo'] = t.detach()
            if read_layer is not None and canon_emb is not None:
                K = Vp.shape[1]; s0 = len(model.pre_ids)
                pooled = hs[clean_ids, s0:s0 + K].float().mean(1)
                with torch.no_grad():
                    sc = model.encoder.score(Traw[[rows_v[n] for n in clean_ids]].float(), canon_emb).float()
                    delta = sc[:, 0::2] - sc[:, 1::2]
                t = F.mse_loss(model.readout(pooled), delta); loss = loss + W.read * t; logs['read'] = t.detach()
        if report_rows:
            mode = c.get('report_mode', 'free')
            reps = [report_target(batch, i, labels, c) for i in report_rows]
            prompts = [report_prompt(batch['dataset'][i], mode) for i in report_rows]
            lp_r, ce_rep, _ = model.answer_logprob(Vp[report_rows], prompts, reps, device)
            loss = loss + W.lm * W.report * ce_rep
            logs['ce_report'] = ce_rep.detach()
            if W.get('rep_dep', 0) > 0 and B > 1:
                # the same report scored on another scan of the batch (same dataset when there is one)
                part = []
                for i in report_rows:
                    same = [j for j in range(B) if j != i and batch['dataset'][j] == batch['dataset'][i]]
                    other = same or [j for j in range(B) if j != i]
                    part.append(other[int(rng.integers(len(other)))])
                lp_x, _, _ = model.answer_logprob(Vp[part], prompts, reps, device)
                n_tok = torch.tensor([len(model.tok(r + model.eos, add_special_tokens=False).input_ids) for r in reps],
                                     device=device, dtype=lp_r.dtype)
                d = (lp_r - lp_x) / n_tok
                t = F.relu(W.get('rep_margin', 0.05) - d).mean()
                loss = loss + W.rep_dep * t
                logs['rep_dep'] = t.detach(); logs['D_rep'] = d.mean().detach()
    if not torch.is_tensor(loss):
        loss = Vp.sum() * 0.0
    logs['loss'] = loss.detach(); logs['loss_t'] = loss
    return logs


def batch_report(batch, i, c=None):
    """Report text of sample i: its claims joined (the parsed report sentences of CT-RATE / Merlin)."""
    claims = batch['claims'][i]
    return ' '.join(cl['text'] for cl in claims) if claims else ''


@torch.no_grad()
def evaluate_qa(model, ds, cfg, device, rank, world):
    """Per-finding AUROC of P(Yes) on a validation slice (clean scans)."""
    model.eval()
    ids = val_indices(ds, cfg.vlm.n_val, rank, world)        # equal share of CT-RATE and Merlin
    P, Y = [], []
    loader = DataLoader(torch.utils.data.Subset(ds, ids), batch_size=2, num_workers=min(2, cfg.data.num_workers), collate_fn=collate)   # training workers stay alive
    fam = {d: [j for j, f in enumerate(FINDING_NAMES) if f.startswith(p)] for d, p in FAMILY.items()}
    for b in loader:
        x = normalise(b['img'].to(device).float())
        with torch.autocast('cuda', dtype=torch.bfloat16):
            Vp, _ = model.visual_tokens(x)
            for i in range(len(b['volume'])):
                js = fam[b['dataset'][i]]
                p = torch.full((len(FINDING_NAMES),), float('nan'), device=device)
                for s in range(0, len(js), 24):
                    jj = js[s:s + 24]
                    p[jj] = model.p_yes(Vp[i:i + 1].expand(len(jj), -1, -1), [question(FINDING_NAMES[j]) for j in jj], device)
                P.append(p[None]); Y.append(b['labels'][i:i + 1].to(device))
    P = torch.cat(P) if P else torch.zeros(0, len(FINDING_NAMES), device=device)
    Y = torch.cat(Y) if Y else torch.zeros(0, len(FINDING_NAMES), device=device)
    if dist.is_initialized():
        P, _ = gather_padded(P, grad=False); Y, _ = gather_padded(Y, grad=False)
    model.train()
    P, Y = P.cpu().numpy(), Y.cpu().numpy()
    res = {}
    for j, f in enumerate(FINDING_NAMES):
        ok = ~np.isnan(P[:, j]) & ~np.isnan(Y[:, j])
        if Y[ok, j].sum() >= cfg.eval.min_pos:
            res[f] = auroc(Y[ok, j], P[ok, j])
    for pre, key in (('ct_', 'macro_ct'), ('mr_', 'macro_mr')):
        v = [x for k, x in res.items() if k.startswith(pre) and x == x]
        res[key] = float(np.mean(v)) if v else None
    res['n_val'] = int(len(Y))
    return res


if __name__ == '__main__':
    main()
