"""Ladder evaluation of a stage-B VLM (steps 4-5, plus LLM-level steps 1-2) on Merlin test / CT-RATE test.

Per volume: P(Yes) for every finding of its dataset under clean, mirrored, each-organ-erased and wrong-patient
visual tokens; mean-pooled projector output for the LLM-input probe; optionally generated reports (clean and
wrong patient) for clinical F1 and Ret.

Per finding: answer AUROC; Dec_LLM-in (5-fold probe on projector output); Use = AUROC / Dec_LLM-in; organ-erasure
effect (target minus control dAUROC of P(Yes)); mirror dAUROC; matched-pair AUROC and its gap; wrong-patient AUROC.
Summary: Ret = clinical F1(report | wrong patient) / clinical F1(report | own scan).

    torchrun --nproc_per_node 4 -m organic.evaluate_stageB --env configs/env/local.yaml \
        --run <work_root>/runsB/vlm4_organic3d --datasets merlin,ctrate --limit 1000 --gen 200
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from .augment import erase, normalise, organ_mask
from .config import load_config, read_manifest
from .data import CTDataset, collate
from .evaluate_stageA import EVAL_ORGANS, cond_name, organ_ids, probe_scores
from .ladder import clinical_f1, ctrate_report_labels, matched_pair_auroc, merlin_report_labels
from .train_stageA import auroc, gather_padded, setup_dist
from .train_stageB import finding_list, question, report_prompt
from .vlm import Organic3DVLM, build_encoder
from .vocab import CTRATE_FINDINGS, FINDINGS, FINDING_NAMES, MERLIN_FINDINGS, SIZE_FINDINGS, control_organ_ids, finding_organ_ids

FAM = {'ctrate': CTRATE_FINDINGS, 'merlin': MERLIN_FINDINGS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--run', required=True, help='stage-B run directory (config.yaml + last.pt)')
    ap.add_argument('--datasets', default='merlin,ctrate')
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--gen', type=int, default=0, help='volumes per dataset with generated reports (clean + wrong)')
    ap.add_argument('--max_new', type=int, default=256, help='new-token cap of generated reports (768 for answer-then-write)')
    ap.add_argument('--fill', default='fat')
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--qchunk', type=int, default=64)
    ap.add_argument('--name', default=None, help='output folder under evalB/ (default: the run name)')
    ap.add_argument('--val_n', type=int, default=600, help='validation scans used to pick each finding\'s Yes threshold (reports from answers)')
    ap.add_argument('--also_n', type=int, default=1000, help='also report metrics on the first N scans of the sample '
                    '(= the external-model audit sample); 0 = off')
    ap.add_argument('overrides', nargs='*')
    a = ap.parse_args()
    cfg = load_config(os.path.join(a.run, 'config.yaml'), env=a.env, overrides=a.overrides)
    os.environ.setdefault('HF_HOME', cfg.paths.hf_home)
    rank, world, device = setup_dist(timeout_min=240)
    cfg.model.pretrained = cfg.vlm.encoder_ckpt == 'colipri'
    model = Organic3DVLM(cfg, build_encoder(cfg, device)).to(device)
    ck = os.path.join(a.run, 'last.pt')
    if os.path.exists(ck):
        st = torch.load(ck, map_location='cpu', weights_only=False)['model']
        missing, unexpected = model.load_state_dict(st, strict=False)
        if rank == 0:
            print('loaded', ck, 'tensors', len(st), 'unexpected', len(unexpected), flush=True)
    elif rank == 0:
        print('WARNING: no checkpoint, evaluating the untrained stage-B model', flush=True)
    model.eval()
    out_dir = os.path.join(cfg.paths.work_root, 'evalB', a.name or os.path.basename(a.run.rstrip('/')))
    summary, summary_sub = {}, {}
    for dset in a.datasets.split(','):
        m = read_manifest(cfg, dset)
        m = m[m.split == 'test']
        if a.limit:
            m = m.sample(n=min(a.limit, len(m)), random_state=0)
        ds = CTDataset(cfg, [m], train=False)
        thr = val_thresholds(model, cfg, dset, a, rank, world, device) if a.val_n else None
        res = run(model, ds, dset, a, rank, world, device)
        res['thr'] = thr
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            tab, summ = metrics(res, ds.labels_all, dset)
            tab.to_csv(os.path.join(out_dir, f'{dset}_ladder.csv'))
            with open(os.path.join(out_dir, f'{dset}_reports.json'), 'w') as fh:
                json.dump({'volume_index': [int(i) for i in res.get('gen_index') or []], **res['texts'],
                           'composed_examples': res.get('composed_examples', []),
                           'thresholds': [float(t) for t in (res['thr'] if res.get('thr') is not None else [])]}, fh, indent=1)
            summary[dset] = summ
            print('SUMMARY', json.dumps({'run': os.path.basename(a.run.rstrip('/')), 'dataset': dset, **summ}), flush=True)
            np.savez_compressed(os.path.join(out_dir, f'{dset}_scores.npz'), labels=ds.labels_all[res['index'].numpy()],
                                **{k: (v.half() if k == 'feat' else v).numpy() for k, v in res.items() if torch.is_tensor(v)})
            if a.also_n and a.limit and a.also_n < len(m):
                # the external-model audits score the first also_n scans of this same fixed-seed sample
                # (audit_nvreason.audit_sample draws with the same seed, so its 1,000 = our first 1,000 of 1,500)
                tab_s, summ_s = metrics(subset_res(res, res['index'].numpy() < a.also_n), ds.labels_all, dset)
                tab_s.to_csv(os.path.join(out_dir, f'{dset}_ladder_n{a.also_n}.csv'))
                summary_sub[dset] = summ_s
                print(f'SUMMARY_N{a.also_n}', json.dumps({'dataset': dset, **summ_s}), flush=True)
    if rank == 0:
        with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
            json.dump(summary, fh, indent=1)
        if summary_sub:
            with open(os.path.join(out_dir, f'summary_n{a.also_n}.json'), 'w') as fh:
                json.dump(summary_sub, fh, indent=1)
    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def answers(model, ds, fs, a, rank, world, device):
    """P(Yes) of every finding question on clean scans (all ranks gathered): (n, len(fs)) and the labels."""
    qs = [question(f) for f in fs]; js = [FINDING_NAMES.index(f) for f in fs]
    ids = list(range(rank, len(ds), world))
    loader = DataLoader(Subset(ds, ids), batch_size=a.bs, num_workers=4, collate_fn=collate)
    P, Y = [], []
    for b in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            Vp, _ = model.visual_tokens(normalise(b['img'].to(device).float()))
            out = torch.empty(Vp.shape[0], len(fs), device=device)
            for i in range(Vp.shape[0]):
                for s in range(0, len(fs), a.qchunk):
                    sl = slice(s, s + a.qchunk)
                    out[i, sl] = model.p_yes(Vp[i:i + 1].expand(len(qs[sl]), -1, -1), qs[sl], device)
        P.append(out.float()); Y.append(b['labels'][:, js].to(device))
    P = torch.cat(P) if P else torch.zeros(0, len(fs), device=device)
    Y = torch.cat(Y) if Y else torch.zeros(0, len(fs), device=device)
    if dist.is_initialized():
        P, _ = gather_padded(P, grad=False); Y, _ = gather_padded(Y, grad=False)
    return P.cpu().numpy(), Y.cpu().numpy()


def val_thresholds(model, cfg, dset, a, rank, world, device):
    """Per finding, the P(Yes) threshold with the best F1 on a validation slice (never the test set)."""
    mv = read_manifest(cfg, dset); mv = mv[mv.split == 'val']
    mv = mv.sample(n=min(a.val_n, len(mv)), random_state=0)
    P, Y = answers(model, CTDataset(cfg, [mv], train=False), FAM[dset], a, rank, world, device)
    thr = np.full(P.shape[1], 0.5)
    for k in range(P.shape[1]):
        y, p = Y[:, k], P[:, k]; ok = ~np.isnan(y); y, p = y[ok], p[ok]
        if y.sum() < 3:
            continue
        cands = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 49)))
        f1 = [2 * ((p >= t) & (y == 1)).sum() / max(1, 2 * ((p >= t) & (y == 1)).sum() + ((p >= t) & (y == 0)).sum() + ((p < t) & (y == 1)).sum()) for t in cands]
        thr[k] = cands[int(np.argmax(f1))]
    return thr


def compose_report(p_row, thr, fs):
    """A report assembled from the answers: every finding above its threshold, grouped by organ section."""
    sec = {}
    for k, f in enumerate(fs):
        if p_row[k] >= thr[k]:
            org = FINDINGS[f][1][0].replace('_left', 's').replace('_right', 's').replace('_', ' ').capitalize() if FINDINGS[f][1] else 'Other'
            sec.setdefault(org, []).append(FINDINGS[f][2])
    return ' '.join(f'{o}: {", ".join(v)}.' for o, v in sec.items()) or 'No abnormality among the assessed findings.'


@torch.no_grad()
def run(model, ds, dset, a, rank, world, device):
    fs = FAM[dset]
    js = [FINDING_NAMES.index(f) for f in fs]
    qs = [question(f) for f in fs]
    conds = ['clean', 'mirror'] + [cond_name(o) for o in EVAL_ORGANS[dset]]
    ids = list(range(rank, len(ds), world))
    ds.labels_all = ds.labels
    loader = DataLoader(Subset(ds, ids), batch_size=a.bs, num_workers=4, collate_fn=collate)
    tconds = ['tok_' + cond_name(o) for o in EVAL_ORGANS[dset]]
    P = {c: [] for c in conds + ['wrong'] + tconds}
    nvox = {c: [] for c in conds}
    feats, toks, order = [], [], []
    gen = torch.Generator(device=device); gen.manual_seed(0)

    def pyes(Vp):
        out = torch.empty(Vp.shape[0], len(fs), device=device)
        for i in range(Vp.shape[0]):
            for s in range(0, len(fs), a.qchunk):
                sl = slice(s, s + a.qchunk)
                out[i, sl] = model.p_yes(Vp[i:i + 1].expand(len(qs[sl]), -1, -1), qs[sl], device)
        return out

    for b in loader:
        hu = b['img'].to(device).float()
        organ = b['organ'].to(device)
        B = hu.shape[0]
        vols = {'clean': hu, 'mirror': torch.flip(hu, dims=[2])}
        for o in EVAL_ORGANS[dset]:
            c = cond_name(o)
            ms = [organ_mask(organ[i], organ_ids(o), 1) for i in range(B)]
            vols[c] = torch.stack([erase(hu[i, 0], ms[i], a.fill, 15.0, gen) for i in range(B)])[:, None]
            nvox[c].append(torch.tensor([int(mm.sum()) for mm in ms]))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for c in conds:
                Vp, _ = model.visual_tokens(normalise(vols[c]))
                P[c].append(pyes(Vp).cpu())
                if c == 'clean':
                    feats.append(Vp.float().mean(1).cpu()); toks.append(Vp.cpu())
        order += [int(i) for i in b['index']]
    Vall = torch.cat(toks) if toks else None
    n = len(order)
    perm = np.roll(np.arange(n), 1)                                  # wrong patient: the next volume on this rank
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for s in range(0, n, a.bs):
            sl = perm[s:s + a.bs]
            P['wrong'].append(pyes(Vall[sl].to(device)).cpu())
    # organ-token swap ("used" per organ): one organ's visual tokens replaced by the wrong patient's, all else kept
    tok_org = model.token_organ.cpu()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for o in EVAL_ORGANS[dset]:
            on = torch.isin(tok_org, torch.as_tensor(organ_ids(o)))
            for s in range(0, n, a.bs):
                e = min(n, s + a.bs)
                V2 = Vall[s:e].clone()
                V2[:, on] = Vall[perm[s:e]][:, on]
                P['tok_' + cond_name(o)].append(pyes(V2.to(device)).cpu())
    texts = {'clean': [], 'wrong': []}
    n_gen = min(n, -(-a.gen // world)) if a.gen else 0
    mode = model.cfg.vlm.get('report_mode', 'free')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        if mode == 'list':
            # answer-then-write: the finding list is the model's own yes/no answers (P(Yes) >= 0.5) on that scan, forced
            # as the start of the reply; the scored report is the continuation after 'Report:'
            Pc = torch.cat(P['clean']).numpy()
            for i in range(n_gen):
                for key, j in (('clean', i), ('wrong', int(perm[i]))):
                    pre = finding_list([fs[k] for k in range(len(fs)) if Pc[j, k] >= 0.5])
                    texts[key] += model.generate(Vall[j:j + 1].to(device), report_prompt(dset, mode), device,
                                                 max_new_tokens=a.max_new, prefix=pre)
        else:
            for s in range(0, n_gen, a.bs):
                e = min(n_gen, s + a.bs)
                texts['clean'] += model.generate(Vall[s:e].to(device), report_prompt(dset), device, max_new_tokens=a.max_new)
                texts['wrong'] += model.generate(Vall[perm[s:e]].to(device), report_prompt(dset), device, max_new_tokens=a.max_new)
    res = {c: torch.cat(v) for c, v in P.items() if v}
    res.update({f'nvox_{c}': torch.cat(v) for c, v in nvox.items() if v})
    res['feat'] = torch.cat(feats); res['index'] = torch.tensor(order)
    if dist.is_initialized():
        for k in list(res):
            t, _ = gather_padded(res[k].to(device), grad=False); res[k] = t.cpu()
        allt = [None] * world
        dist.all_gather_object(allt, (texts, order[:n_gen], [order[i] for i in perm[:n_gen]]))
        texts = {'clean': sum((t[0]['clean'] for t in allt), []), 'wrong': sum((t[0]['wrong'] for t in allt), [])}
        res['gen_index'] = sum((t[1] for t in allt), [])
    else:
        res['gen_index'] = order[:n_gen]
    res['texts'] = texts
    res['js'] = js
    return res


def subset_res(res, keep):
    """res restricted to the scans where keep is True (per-scan tensors only; generated reports are dropped)."""
    out = {k: (v[torch.as_tensor(keep)] if torch.is_tensor(v) and v.shape[0] == len(keep) else v) for k, v in res.items()}
    out['gen_index'] = []
    return out


def metrics(res, L_all, dset):
    js, fs = res['js'], FAM[dset]
    idx = res['index'].numpy()
    L = L_all[idx]
    rows = []
    X = res['feat'].numpy()
    for k, (f, j) in enumerate(zip(fs, js)):
        y = L[:, j]
        if np.nansum(y) < 10 or np.nansum(1 - y) < 10:
            continue
        p0 = res['clean'][:, k].numpy()
        a0 = auroc(y, p0)
        r = dict(finding=f, n_pos=int(np.nansum(y)), answer_auroc=a0,
                 wrong_auroc=auroc(y, res['wrong'][:, k].numpy()),
                 mirror_d=auroc(y, res['mirror'][:, k].numpy()) - a0)
        _, oof = probe_scores(X, y)
        r['dec_llm_in'] = auroc(y, oof) if oof is not None else np.nan
        r['use'] = (a0 - 0.5) / (r['dec_llm_in'] - 0.5) if r['dec_llm_in'] == r['dec_llm_in'] and r['dec_llm_in'] > 0.5 else np.nan
        mp, npairs = matched_pair_auroc(L, p0, j)
        r['matched_pair_auroc'] = mp; r['matched_gap'] = a0 - mp if mp == mp else np.nan; r['n_pairs'] = npairs
        tgt, ctrl = set(finding_organ_ids(f)), set(control_organ_ids(f))
        t_eff, c_eff = [], []
        for o in EVAL_ORGANS[dset]:
            c = cond_name(o)
            keep = res[f'nvox_{c}'].numpy() > 0
            yk = np.where(keep, y, np.nan)
            d = auroc(yk, res[c][:, k].numpy()) - auroc(yk, p0)
            ids = set(organ_ids(o))
            if tgt and ids & tgt:
                t_eff.append(d)
            elif ids <= ctrl:
                c_eff.append(d)
        r['erase_effect'] = (np.nanmean(t_eff) - np.nanmean(c_eff)) if t_eff else np.nan
        # organ-token swap: answer AUROC change when the finding's organ tokens come from another patient, minus the
        # change for control organs (valid for size findings too: the swapped tokens carry the other organ's size)
        t_eff, c_eff = [], []
        for o in EVAL_ORGANS[dset]:
            c = 'tok_' + cond_name(o)
            if c not in res:
                continue
            d = auroc(y, res[c][:, k].numpy()) - a0
            ids = set(organ_ids(o))
            if tgt and ids & tgt:
                t_eff.append(d)
            elif ids <= ctrl:
                c_eff.append(d)
        r['tok_effect'] = (np.nanmean(t_eff) - np.nanmean(c_eff)) if t_eff else np.nan
        rows.append(r)
    tab = pd.DataFrame(rows).set_index('finding') if rows else pd.DataFrame()
    summ = {k: float(np.nanmean(tab[k])) for k in ['answer_auroc', 'dec_llm_in', 'use', 'erase_effect', 'tok_effect', 'mirror_d',
                                                   'matched_pair_auroc', 'matched_gap', 'wrong_auroc'] if k in tab}
    if 'erase_effect' in tab:                                       # organ erasure is not valid for size findings
        summ['erase_effect_texture'] = float(np.nanmean(tab.erase_effect[~tab.index.isin(SIZE_FINDINGS)]))
    summ['n'] = int(len(idx))
    thr = res.get('thr')
    if thr is not None:                       # reports assembled from the answers (thresholds from validation)
        cols = [FINDING_NAMES.index(f) for f in fs]
        P0, Pw = res['clean'].numpy(), res['wrong'].numpy()
        pred = np.full((len(idx), len(FINDING_NAMES)), np.nan); pred[:, cols] = (P0 >= thr).astype(float)
        predw = np.full_like(pred, np.nan); predw[:, cols] = (Pw >= thr).astype(float)
        f_c, f_w = clinical_f1(pred, L, cols), clinical_f1(predw, L, cols)
        summ.update(composed_f1=f_c, composed_f1_wrong=f_w, composed_ret=(f_w / f_c) if f_c else np.nan)
        res['composed_examples'] = [compose_report(P0[i], thr, fs) for i in range(min(5, len(P0)))]
    gi = res.get('gen_index') or []
    if gi:
        labeller = merlin_report_labels if dset == 'merlin' else ctrate_report_labels
        true = L_all[np.asarray(gi)]
        cols = [FINDING_NAMES.index(f) for f in fs]
        f_clean = clinical_f1(labeller(res['texts']['clean']), true, cols)
        f_wrong = clinical_f1(labeller(res['texts']['wrong']), true, cols)
        summ.update(report_f1=f_clean, report_f1_wrong=f_wrong, ret=(f_wrong / f_clean) if f_clean else np.nan, n_gen=len(gi))
    return tab, summ


if __name__ == '__main__':
    main()
