"""MedGemma 1.5 4B (google/medgemma-1.5-4b-it) as an extaudit adapter for report generation
(extaudit/vlm_report.py). Run in the organic env (transformers 5.x); weights google/medgemma-1.5-4b-it from the
Hugging Face cache ($HF_HOME if set, else the default ~/.cache/huggingface).

Input = the audit's own CT recipe (organic/audit_medgemma.py, from google-health/medgemma
notebooks/high_dimensional_ct_hugging_face.ipynb): 85 axial slices sampled uniformly through the 1 mm RAS volume,
each an RGB image of three windows (-1024..1024, -135..215, 0..80 HU), each followed by "SLICE n"; then the request.
Prompt: Google publishes no CT report-generation instruction (the CT notebook asks a liver yes/no question with
reasoning), so the audit's framing ("You are reviewing a CT of the <region>. The axial slices are given below in order
from inferior to superior.") is followed by the notebook's query lead-in plus a findings-section request (REPORT_Q).
Greedy decoding (the notebook's do_sample=False); new tokens only; a thinking block, if emitted, is dropped.

    CUDA_VISIBLE_DEVICES=0 <organic env>/bin/python -W ignore -m extaudit.vlm_report --model medgemma \
        --name MedGemma-1.5-4B --dataset ctrate --root <work_root>/extaudit
"""
import copy

import torch

from organic.audit_medgemma import REGION, MedGemma, slices_rgb

REPORT_Q = ('Based on the visual evidence in the slices provided above, write the findings section of the radiology '
            'report for this {region} CT.')


class Adapter:
    def __init__(self, ckpt, device):
        self.mg = MedGemma(device)

    @staticmethod
    def report_prompt(dataset):
        return REPORT_Q.format(region=REGION[dataset])

    def report(self, hu, dataset, max_new_tokens=768):
        imgs = slices_rgb(hu.float().cpu().numpy())
        return self.mg.generate(imgs, dataset, self.report_prompt(dataset), max_new_tokens)

    # ---- yes/no scoring (extaudit/vlm_core.py API), same recipe and batched scoring as the audit
    def set_dataset(self, dataset):
        self.dataset = dataset

    @staticmethod
    def question(phrase):
        return f'Is there {phrase} in this CT? Answer Yes or No.'          # = organic.audit_medgemma.question

    @torch.no_grad()
    def encode(self, hu):
        imgs = slices_rgb(hu.float().cpu().numpy())
        cache, pre_ids, feat = self.mg.prefix(imgs, self.dataset, self.question('a mass'))
        return (cache, pre_ids), feat

    @torch.no_grad()
    def yes_no_logit(self, state, questions):
        """log P(Yes) - log P(No) at the first answer token (audit_medgemma.MedGemma.p_yes without the sigmoid)."""
        mg, (cache, pre_ids) = self.mg, state
        tails = []
        for q in questions:
            full = mg.proc.apply_chat_template(mg._messages([], self.dataset, q), add_generation_prompt=True,
                                               tokenize=True, return_dict=True, return_tensors='pt')['input_ids'][0]
            q_ids = mg.proc.tokenizer(q, add_special_tokens=False).input_ids
            tails.append(full[len(full) - mg._tail_len(full, q_ids):])
        n, T = len(tails), max(len(t) for t in tails)
        pad = mg.proc.tokenizer.pad_token_id or 0
        ids = torch.full((n, T), pad, dtype=torch.long)
        am = torch.zeros((n, len(pre_ids) + T), dtype=torch.long)
        am[:, :len(pre_ids)] = 1
        for k, t in enumerate(tails):
            ids[k, :len(t)] = t; am[k, len(pre_ids):len(pre_ids) + len(t)] = 1
        c = copy.deepcopy(cache); c.batch_repeat_interleave(n)
        pos = torch.arange(len(pre_ids), len(pre_ids) + T, device=mg.device)
        o = mg.m(input_ids=ids.to(mg.device), past_key_values=c, use_cache=True, cache_position=pos,
                 attention_mask=am.to(mg.device))
        last = torch.tensor([len(t) - 1 for t in tails], device=mg.device)
        lg = o.logits[torch.arange(n, device=mg.device), last].float()
        del c
        return torch.logsumexp(lg[:, mg.yes], 1) - torch.logsumexp(lg[:, mg.no], 1)
