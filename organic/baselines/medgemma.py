"""Alignment-ladder audit of MedGemma 1.5 4B (google/medgemma-1.5-4b-it): a 2D-slice VLM with 3D CT support.

Same test volumes (audit_sample, fixed seed), conditions and metric code as the NV-Reason-CT audit and our stage-B
evaluation (organic.vlm.evaluate.metrics): answer AUROC, LLM-input probe, use, organ-erasure effect, mirror, matched pairs,
wrong-patient AUROC. Reports are not generated here (QA rungs only); free-text report generation uses MedGemma.generate
through extaudit/vlm_models/medgemma.py and extaudit/vlm_report.py.

MedGemma's own CT recipe (google-health/medgemma notebooks/high_dimensional_ct_hugging_face.ipynb): 85 axial slices
sampled uniformly through the volume; each slice an RGB image whose channels are three windows (-1024..1024,
-135..215, 0..80 HU); each image followed by "SLICE n"; then the question. Erasure and mirroring are applied to the
1 mm HU volume and its TotalSegmentator mask before slicing. Per condition the image prefix is encoded once and its
cache reused for every question.

    torchrun --nproc_per_node 4 -m organic.baselines.medgemma --env configs/env/local.yaml --datasets merlin,ctrate --limit 1000
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.distributed as dist

from ..data.interventions import erase, organ_mask
from .nv_reason_ct import audit_sample
from ..config import abspath, load_config
from ..data.dataset import _LUT
from ..encoder.evaluate import EVAL_ORGANS, cond_name, organ_ids
from ..vlm.evaluate import FAM, metrics
from ..utils import gather_padded, setup_dist
from ..data.volio import read_image, read_mask
from ..vocab import FINDINGS, FINDING_NAMES

MODEL_ID = 'google/medgemma-1.5-4b-it'
N_SLICES = 85
WINDOWS = [(-1024, 1024), (-135, 215), (0, 80)]
REGION = {'merlin': 'abdomen and pelvis', 'ctrate': 'chest'}


def question(f):
    return f'Is there {FINDINGS[f][2]} in this CT? Answer Yes or No.'


def slices_rgb(hu, n=N_SLICES):
    """(X, Y, Z) HU volume (RAS, axis 2 = superior) -> list of n uint8 RGB axial slices, MedGemma windowing."""
    z = hu.shape[2]
    idx = np.unique(np.round(np.linspace(0, z - 1, min(n, z))).astype(int))
    out = []
    for k in idx:
        s = np.rot90(hu[:, :, k])                                  # radiological display: anterior up
        ch = [np.clip((np.clip(s, lo, hi) - lo) / (hi - lo) * 255.0, 0, 255) for lo, hi in WINDOWS]
        out.append(np.round(np.stack(ch, -1)).astype(np.uint8))
    return out


class MedGemma:
    def __init__(self, device):
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        self.device = device
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        self.m = Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
        tok = self.proc.tokenizer
        self.yes = [tok(w, add_special_tokens=False).input_ids[0] for w in ('Yes', ' Yes')]
        self.no = [tok(w, add_special_tokens=False).input_ids[0] for w in ('No', ' No')]

    def _messages(self, imgs, dataset, q):
        from PIL import Image
        content = [{'type': 'text', 'text': f'You are reviewing a CT of the {REGION[dataset]}. The axial slices are '
                                            f'given below in order from inferior to superior.'}]
        for k, im in enumerate(imgs, 1):
            content += [{'type': 'image', 'image': Image.fromarray(im)}, {'type': 'text', 'text': f'SLICE {k}'}]
        content.append({'type': 'text', 'text': '\n\n' + q})
        return [{'role': 'user', 'content': content}]

    @torch.no_grad()
    def prefix(self, imgs, dataset, q0):
        """Encode everything up to the question once. Returns (cache, prefix_ids, pooled image feature)."""
        inputs = self.proc.apply_chat_template(self._messages(imgs, dataset, q0), add_generation_prompt=True,
                                               tokenize=True, return_dict=True, return_tensors='pt').to(self.device)
        ids = inputs['input_ids'][0]
        q_ids = self.proc.tokenizer(q0, add_special_tokens=False).input_ids
        # the prefix ends right before the question text: locate the question tokens from the end
        tail = self._tail_len(ids, q_ids)
        pre = {k: (v[:, :len(ids) - tail] if k in ('input_ids', 'attention_mask', 'token_type_ids') else v) for k, v in inputs.items()}
        pre['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        out = self.m(**pre, use_cache=True)
        feat = self.m.get_image_features(pre['pixel_values'])
        feat = (feat if torch.is_tensor(feat) else feat.pooler_output if hasattr(feat, 'pooler_output') else feat[0]).float()
        return out.past_key_values, ids[:len(ids) - tail], feat.reshape(-1, feat.shape[-1]).mean(0)

    def _tail_len(self, ids, q_ids):
        """Number of tokens from the start of the question text to the end of the prompt."""
        ids = ids.tolist()
        for s in range(len(ids) - len(q_ids), -1, -1):
            if ids[s:s + len(q_ids)] == q_ids:
                return len(ids) - s
        raise ValueError('question tokens not found in the prompt')

    @torch.no_grad()
    def p_yes(self, cache, pre_ids, dataset, questions, imgs_q0=None):
        """P(Yes) / (P(Yes) + P(No)) at the first answer token for all questions in ONE batched pass over a
        batch-repeated copy of the prefix cache (question tails right-padded)."""
        tails = []
        for q in questions:
            full = self.proc.apply_chat_template(self._messages([], dataset, q), add_generation_prompt=True, tokenize=True,
                                                 return_dict=True, return_tensors='pt')['input_ids'][0]
            q_ids = self.proc.tokenizer(q, add_special_tokens=False).input_ids
            tails.append(full[len(full) - self._tail_len(full, q_ids):])
        n, T = len(tails), max(len(t) for t in tails)
        pad = self.proc.tokenizer.pad_token_id or 0
        ids = torch.full((n, T), pad, dtype=torch.long)
        am = torch.zeros((n, len(pre_ids) + T), dtype=torch.long)
        am[:, :len(pre_ids)] = 1
        for k, t in enumerate(tails):
            ids[k, :len(t)] = t; am[k, len(pre_ids):len(pre_ids) + len(t)] = 1
        c = copy.deepcopy(cache); c.batch_repeat_interleave(n)
        pos = torch.arange(len(pre_ids), len(pre_ids) + T, device=self.device)
        o = self.m(input_ids=ids.to(self.device), past_key_values=c, use_cache=True, cache_position=pos,
                   attention_mask=am.to(self.device))
        last = torch.tensor([len(t) - 1 for t in tails], device=self.device)
        lg = o.logits[torch.arange(n, device=self.device), last].float()
        y, no = torch.logsumexp(lg[:, self.yes], 1), torch.logsumexp(lg[:, self.no], 1)
        del c
        return torch.sigmoid(y - no)


    @torch.no_grad()
    def generate(self, imgs, dataset, prompt, max_new_tokens=768):
        """Greedy free-text answer to `prompt` after the slices (the notebook's generate call, do_sample=False); only
        the new tokens are decoded. A thinking block (<unused94>thought ... <unused95>), if any, is dropped."""
        inputs = self.proc.apply_chat_template(self._messages(imgs, dataset, prompt), add_generation_prompt=True,
                                               tokenize=True, return_dict=True, return_tensors='pt').to(self.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        out = self.m.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        text = self.proc.tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        if '<unused95>' in text:
            text = text.split('<unused95>', 1)[1]
        for t in ('<end_of_turn>', '<eos>', '<pad>'):
            text = text.replace(t, '')
        return text.strip()


def load_volume(cfg, dset, r):
    hu = read_image(abspath(cfg, dset, r.img_rel)).astype(np.float32)
    seg = read_mask(abspath(cfg, dset, r.seg_rel))
    return hu, _LUT[seg]


def run(mg, cfg, m, dset, a, rank, world, device):
    fs = FAM[dset]
    qs = [question(f) for f in fs]
    conds = ['clean', 'mirror'] + [cond_name(o) for o in EVAL_ORGANS[dset]]
    rows = list(range(rank, len(m), world))
    P = {c: [] for c in conds + ['wrong']}
    nvox = {c: [] for c in conds}
    feats, order, clean_imgs = [], [], []
    gen = torch.Generator(); gen.manual_seed(0)
    for n, i in enumerate(rows):
        r = m.iloc[i]
        hu, organ = load_volume(cfg, dset, r)
        hu_t, org_t = torch.from_numpy(hu), torch.from_numpy(organ)
        vols = {'clean': hu, 'mirror': hu[::-1].copy()}
        for o in EVAL_ORGANS[dset]:
            mk = organ_mask(org_t, organ_ids(o), 1)
            vols[cond_name(o)] = erase(hu_t, mk, a.fill, 15.0, gen).numpy()
            nvox[cond_name(o)].append(int(mk.sum()))
        for c in conds:
            imgs = slices_rgb(vols[c])
            cache, pre_ids, feat = mg.prefix(imgs, dset, qs[0])
            P[c].append(mg.p_yes(cache, pre_ids, dset, qs, imgs).cpu())
            if c == 'clean':
                feats.append(feat.cpu()); clean_imgs.append(imgs)
            del cache
        order.append(i)
        if rank == 0 and n % 10 == 0:
            print(f'{dset} {n + 1}/{len(rows)} on rank 0', flush=True)
    k = len(order)
    perm = np.roll(np.arange(k), 1)
    for s in range(k):
        cache, pre_ids, _ = mg.prefix(clean_imgs[perm[s]], dset, qs[0])
        P['wrong'].append(mg.p_yes(cache, pre_ids, dset, qs, None).cpu()); del cache
    res = {c: torch.stack(v) for c, v in P.items() if v}
    res.update({f'nvox_{c}': torch.tensor(v) for c, v in nvox.items() if v})
    res['feat'] = torch.stack(feats); res['index'] = torch.tensor(order)
    if dist.is_initialized():
        for key in list(res):
            t, _ = gather_padded(res[key].to(device), grad=False); res[key] = t.cpu()
    res['gen_index'] = []; res['texts'] = {'clean': [], 'wrong': []}
    res['js'] = [FINDING_NAMES.index(f) for f in fs]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--datasets', default='merlin,ctrate')
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--fill', default='fat')
    ap.add_argument('--name', default='MedGemma-1.5-4B')
    a = ap.parse_args()
    cfg = load_config(os.path.join(os.path.dirname(__file__), '../../configs/stageA.yaml'), env=a.env)
    os.environ.setdefault('HF_HOME', cfg.paths.hf_home)
    rank, world, device = setup_dist(timeout_min=240)
    mg = MedGemma(device)
    out_dir = os.path.join(cfg.paths.work_root, 'evalB', a.name)
    summary = {}
    for dset in a.datasets.split(','):
        m = audit_sample(cfg, dset, a.limit)
        res = run(mg, cfg, m, dset, a, rank, world, device)
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            L = m[[f'label_{f}' for f in FINDING_NAMES]].to_numpy(np.float32)
            tab, summ = metrics(res, L, dset)
            tab.to_csv(os.path.join(out_dir, f'{dset}_ladder.csv'))
            summary[dset] = summ
            print('SUMMARY', json.dumps({'run': a.name, 'dataset': dset, **summ}), flush=True)
            # per-scan scores (as organic.vlm.evaluate) for the finding-based reports and paired bootstrap CIs
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
