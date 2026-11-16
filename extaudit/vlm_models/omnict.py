"""OmniCT-7B (Lin et al., ICLR 2026; HF Alibaba-DAMO-Academy/OmniCT-7B, Apache-2.0; code github.com/alibaba-damo-academy/
OmniCT @ d9ddc8b), run in $ORGANIC3D_BASELINES/omnict/venv. SigLIP-so400m-384 (frozen, google/siglip-so400m-patch14-384)
+ TPE + 3x3 token shuffle + MLP adapter -> Qwen2.5-7B-Instruct.

Preprocessing = MedEval-CT's CT-RATE recipe (datasets/dataset_info.json; paper App. E: "windowed to [-1000, 1000] and
resampled to 32x384x384", D x H x W along z, y, x) followed by the repo's own 3D inference path
(OmniCTQwen2ForCausalLM.generate + utils/build_datasets.sample_slices_rgb_from_3d_image): our 1 mm RAS volume -> LPS
(CT-RATE file order: flip axes 0 and 1) -> (z, y, x) array, HU clipped to [-1000, 1000], whole volume resized (no crop,
trilinear) to 32 x 384 x 384; volume min-max scaled to [0, 1]; 10 slices at np.linspace(0, 32, 12, int)[1:-1], each an
RGB image (previous, current, next slice) quantised to uint8; SigLIP normalisation (x / 255 - 0.5) / 0.5.
Orientation check: the organ boxes ("bbx", z/y/x on the 32x384x384 grid) in the released MedEval-CT SFT json put the
liver / gallbladder / right lobes at small x, kidneys posterior (large y), thyroid at large z = LPS, z up, as above
(TotalSegmentator boxes of our CT-RATE valid scans through this route match them in x and z). Variant test on 150
CT-RATE audit scans (clean, macro AUROC over 18 findings): LPS 0.807, LPS z-flipped 0.801, LPS l-r flipped 0.801,
RAS 0.777; OmniCT-style question "Are there any indications of X in this CT?" 0.775 vs harness default 0.807.

Organ token block (<|organ_start|> 90 tokens <|organ_end|>): the model's OSE pathway pools a TotalSegmentator organ box
when a training question targets an organ; the official inference script passes bbox_token=[] (whole-volume pooling to
10 x 3 x 3), which is used here, so no mask is needed.
Prompt = evaluation/utils.build_multimodal_prompt (Qwen2.5 chat template, default system prompt):
"<|image3d_start|>[810 x <|image_pad|>]<|image3d_end|>\\n<|organ_start|>[90 x <|image_pad|>]<|organ_end|>\\n" + question,
with the harness default yes/no question (OmniCT has no published binary-finding prompt; its CT-RATE VQA answers
closed questions with "Yes." / "No."). yes_no_logit = logit 'Yes' - logit 'No' at the first answer token;
P(Yes) = its sigmoid. Prefix KV cache + batched question tails (bf16; max |dP| 0.03 vs the uncached full-prompt batch,
which matches the official generate() route exactly).
Probe feature = mean of the 810 projected 3D visual tokens (LLM input space, 3584-d).
Deviation: SDPA attention instead of flash_attention_2 (not installed); bf16 as in the official script.

Report generation (extaudit/vlm_report.py): same image + organ block and chat prompt as above, question =
"Write a radiology report for the following CT scan.", the most frequent CT-RATE report-generation request in the
released MedEval-CT SFT json (medeval-ct-dataset-sft.json, CT-RATE entries answered "Findings: ... Impression: ...";
the CT-RATE-CAP templates of CT-CHAT's VQA set with the tags stripped). It names no body region, so it is used for
Merlin too (OmniCT has no abdominal report set; its only other 3D captioning data, M3D-Cap, asks for one-line captions,
"Please caption this medical scan with findings."). Decoding: the model's own generate() route (inputs_embeds with the
visual tokens merged, as OmniCTQwen2ForCausalLM.generate) with the released generation_config except greedy
(do_sample=False instead of T 0.7 / top-k 20 / top-p 0.8; repetition_penalty 1.05 kept); new tokens decoded only.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from extaudit.paths import baseline

BASE = baseline('omnict')                 # repo/ (code clone) and weights/ (HF snapshots) below it
sys.path.insert(0, f'{BASE}/repo/src')
CKPT = f'{BASE}/weights/OmniCT-7B'
SIGLIP = f'{BASE}/weights/siglip-so400m-patch14-384'
IMAGE_PAD = 151655
SLICES = np.linspace(0, 32, num=12, dtype=int)[1:-1]                       # [2, 5, ..., 29]


class Adapter:
    def __init__(self, ckpt, device):
        from transformers import AutoTokenizer
        from omnict import OmniCTQwen2ForCausalLM
        self.dev = device
        self.m = OmniCTQwen2ForCausalLM.from_pretrained(
            ckpt or CKPT, vision_tower_path=SIGLIP, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            training_stage='eval', device_map={'': device}, local_files_only=True)
        self.m.init_vision_tower()
        self.m.vision_tower.to(dtype=torch.bfloat16, device=device)
        self.m.eval()
        self.tok = AutoTokenizer.from_pretrained(ckpt or CKPT, local_files_only=True, use_fast=True)
        self.yes = self.tok('Yes', add_special_tokens=False).input_ids[0]
        self.no = self.tok('No', add_special_tokens=False).input_ids[0]
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self._ids = {}

    def prompt_ids(self, q):
        if q not in self._ids:
            msg = [{'role': 'user', 'content': '<|image3d_start|><|image3d|><|image3d_end|>\n'
                                               '<|organ_start|><|mask3d|><|organ_end|>\n' + q}]
            p = self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            p = p.replace('<|image3d|>', '<|image_pad|>' * 810).replace('<|mask3d|>', '<|image_pad|>' * 90)
            ids = torch.tensor(self.tok(p).input_ids)
            assert int((ids == IMAGE_PAD).sum()) == 900
            self._ids[q] = ids
        return self._ids[q]

    @staticmethod
    def prep(hu):
        """(X, Y, Z) 1 mm RAS HU (cuda) -> (10, 3, 384, 384) SigLIP pixel values."""
        a = torch.flip(hu, dims=[0, 1]).clamp(-1000, 1000).permute(2, 1, 0)   # LPS, (z, y, x)
        v = F.interpolate(a[None, None].float(), size=(32, 384, 384), mode='trilinear', align_corners=False)[0, 0]
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)                       # sample_slices_rgb_from_3d_image
        idx = torch.as_tensor(SLICES, device=v.device)
        prev, nxt = (idx - 1).clamp(min=0), (idx + 1).clamp(max=31)
        rgb = torch.stack([v[prev], v[idx], v[nxt]], dim=1)                  # (10, 3, H, W): channels prev/curr/next
        rgb = torch.floor(rgb * 255).clamp(0, 255) / 255.0                   # uint8 cast, then rescale 1/255
        return (rgb - 0.5) / 0.5

    def encode(self, hu):
        m = self.m
        px = self.prep(hu).to(torch.bfloat16)
        e = m.vision_tower(px)                                               # (10, 729, 1152), last hidden state
        e = m.TPE_embedding(e.view(-1, 10, 27, 27, e.shape[-1])).view(-1, 729, e.shape[-1] + m.TPE_lens)
        e = m.compress_vision_token_shuffle_siglip(e).view(1, -1, e.shape[-1] * 9)
        e = m.vision_adapter_3d(e.to(m.vision_adapter_3d_head[0].weight.dtype)).squeeze(0)   # (810, 3584)
        organ = m.aggregator_3d(e.reshape(10, 9, 9, e.shape[-1])).reshape(-1, e.shape[-1])  # bbox_token = []
        img = torch.cat((e, organ), dim=0)                                   # (900, 3584)
        return img, e.float().mean(0)

    def p_yes(self, img, questions):
        return torch.sigmoid(self.yes_no_logit(img, questions))

    def yes_no_logit(self, img, questions):
        """logit(Yes) - logit(No) at the first answer token. Prefix (system prompt + image block, identical for all
        questions) encoded once; the question tails (from <|organ_end|> on) run as one right-padded batch over a
        batch-repeated copy of its KV cache."""
        ids = [self.prompt_ids(q) for q in questions]
        P = int(torch.where(ids[0] == IMAGE_PAD)[0][-1]) + 1
        assert all(torch.equal(t[:P], ids[0][:P]) for t in ids)
        pre = ids[0][:P].to(self.dev)[None]
        emb = self.m.model.get_input_embeddings()(pre)
        emb[pre == IMAGE_PAD] = img.to(emb.dtype)
        cache = self.m.model(inputs_embeds=emb, use_cache=True).past_key_values
        tails = [t[P:] for t in ids]
        n, T = len(tails), max(len(t) for t in tails)
        cache.batch_repeat_interleave(n)
        x = torch.full((n, T), self.pad, dtype=torch.long)
        am = torch.zeros((n, P + T), dtype=torch.long)
        am[:, :P] = 1
        for k, t in enumerate(tails):
            x[k, :len(t)] = t; am[k, P:P + len(t)] = 1                     # right padding
        h = self.m.model(input_ids=x.to(self.dev), attention_mask=am.to(self.dev), past_key_values=cache,
                         use_cache=True, cache_position=torch.arange(P, P + T, device=self.dev)).last_hidden_state
        last = torch.tensor([len(t) - 1 for t in tails], device=self.dev)
        lg = self.m.lm_head(h[torch.arange(n, device=self.dev), last]).float()
        return lg[:, self.yes] - lg[:, self.no]

    def p_yes_full(self, img, questions):
        """Reference route (no cache): every full prompt in one right-padded batch; used to check p_yes."""
        ids = [self.prompt_ids(q) for q in questions]
        n, T = len(ids), max(len(t) for t in ids)
        x = torch.full((n, T), self.pad, dtype=torch.long)
        am = torch.zeros((n, T), dtype=torch.long)
        for k, t in enumerate(ids):
            x[k, :len(t)] = t; am[k, :len(t)] = 1
        x, am = x.to(self.dev), am.to(self.dev)
        emb = self.m.model.get_input_embeddings()(x)
        emb[x == IMAGE_PAD] = img.to(emb.dtype).repeat(n, 1)
        h = self.m.model(inputs_embeds=emb, attention_mask=am, use_cache=False).last_hidden_state
        last = torch.tensor([len(t) - 1 for t in ids], device=self.dev)
        lg = self.m.lm_head(h[torch.arange(n, device=self.dev), last]).float()
        return torch.sigmoid(lg[:, self.yes] - lg[:, self.no])

    REPORT_Q = 'Write a radiology report for the following CT scan.'

    def report_prompt(self, dataset):
        return self.REPORT_Q

    @torch.no_grad()
    def report(self, hu, dataset, max_new_tokens=768):
        """Greedy report through the model's own generate() route (OmniCTQwen2ForCausalLM.generate minus the file
        loading: the same merged inputs_embeds handed to the parent generate)."""
        img, _ = self.encode(hu)
        ids = self.prompt_ids(self.report_prompt(dataset)).to(self.dev)[None]
        emb = self.m.model.get_input_embeddings()(ids)
        emb = self.m.merge_image_embeds_with_text_embeds(img, emb, ids)
        out = super(type(self.m), self.m).generate(
            inputs_embeds=emb, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_k=None, top_p=None, repetition_penalty=1.05, pad_token_id=self.pad)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()     # inputs_embeds only: new tokens only
