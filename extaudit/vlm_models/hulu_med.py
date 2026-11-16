"""Hulu-Med-7B (ZJU-AI4H/Hulu-Med-7B @258594714a, Apache-2.0; Qwen2.5-7B + SigLIP-NaViT encoder, HF remote code), run
in $ORGANIC3D_BASELINES/hulu_med/venv (torch 2.4 cu121, transformers 4.51.2, flash-attn 2.7.3).

Hulu-Med reads a CT volume as a 'video' of 2D slices: processing_hulumed.HulumedProcessor.load_nii samples slices with
np.linspace(0, n - 1, k, dtype=int), min-max normalises EACH slice to uint8 (no HU window), grey -> RGB; video merge
size 2; use_token_compression is off in the released config. The authors' released 3D evaluation frames
(ModelScope Med-Team/Hulu-Med, Eval/*, read by MedUniEval) show two slice geometries, both reproduced here:
  'axial' (DEFAULT, all datasets): Eval/3D-AMOS-MM 'amos_test_crop_32' = 32 axial slices, inferior -> superior,
    radiological display (rows anterior -> posterior, columns patient's right -> left), per-slice min-max, square
    512 x 512 frames. Ours: RAS -> LPS (flip axes 0, 1), x-y padded symmetrically to a square with -1024 HU (our 1 mm
    volumes are cropped in y; scanner FOVs are square), 32 slices by load_nii's linspace, frame = lps[:, :, k].T.
  'rad3d': Eval/3DRad_eval 'valid_crop' (3D-RAD = CT-RATE validation) = the CT-RATE NIfTI array (LPS file order)
    resized trilinearly (align_corners=False) to 32 x 256 x 256 along the ARRAY axes, i.e. 32 SAGITTAL slices
    (right -> left), rows anterior -> posterior, columns inferior -> superior, per-slice min-max, floor to uint8
    (matches the released JPEGs computed from the raw CT-RATE files to MAE 2.1-2.7 grey levels = JPEG noise).
  'nii': load_nii on a CT-RATE file as is (axial, rows = LPS x, columns = LPS y), for comparison.
Choice (150 CT-RATE audit scans, clean, macro AUROC over 18 findings, A/B question): axial-32 0.794, axial-64 0.802,
axial-180 (README's nii_num_slices) 0.775, nii-32 0.781, rad3d 0.725, the released 3D-RAD JPEGs themselves 0.706.
The 3D-RAD format is reproduced faithfully but is clearly worse than axial slicing, so CT-Rate-Cap / CT-Rate-VQA
training most likely used axial slices; axial-32 = the released AMOS-MM recipe (same slice count as both released
3D sets) is the default. Merlin (60 scans, 17 findings): axial-32 0.702, nii-32 0.723, rad3d 0.621. --ckpt selects others: 'rad3d', 'nii', 'axial:s64', and 'yesno' (e.g. 'axial:yesno').
Frames go through the model's own processor exactly as MedUniEval/models/HuluMed_Qwen2 (PIL RGB frames,
{"type": "video", "num_frames": 32} + text, add_system_prompt=False, add_generation_prompt=True; bicubic resize to
multiples of 28 under the 16384-token cap, mean/std 0.5).
Question = 3D-RAD Existence Detection (CT-RATE yes/no on the 18 findings, in Hulu-Med's 3D eval): one of its 18
templates, "Is {X} present on the CT scan?\\nA.Yes\\nB.No", + MedUniEval Rad3D's closed-question suffix
"\\nAnswer with the option's letter from the given choices directly." yes_no_logit = logit 'A' - logit 'B' at the
first answer token (lm_head rows applied in float32), P(Yes) = its sigmoid. ('yesno': the harness default question,
Yes/No tokens; AUROC within 0.01 of A/B.)
Per condition the visual prefix (chat header + slice tokens) is encoded once; question tails are batched on a
repeated copy of its KV cache. feat = mean of the projected visual tokens (LLM input space, 3584-d).
Track B reports (report / report_prompt, extaudit/vlm_report.py): the same frames and processor call, prompt =
MedUniEval's 3D report-generation prompt (utils/question_formats.py:82 get_report_generation_prompt, used by
utils/Amos/Amos.py:76 for 3D-AMOS-MM, the only 3D report benchmark in their eval; no CT-RATE report task there, so the
same prompt for both datasets), the model's own generate() as models/HuluMed_Qwen2 generate_output (greedy:
do_sample=False, repetition_penalty 1.0 = eval.sh defaults; max_new_tokens from the caller instead of 16384).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

from extaudit.paths import default_hf_home

os.environ.setdefault('HF_HOME', default_hf_home())
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')               # DataLoader workers fork after tokenizer use
MODEL_ID = 'ZJU-AI4H/Hulu-Med-7B'
REVISION = '258594714a0d3835eb2c9e4cc165a4242e606d71'
N_SLICES = 32
MC_SUFFIX = "\nA.Yes\nB.No\nAnswer with the option's letter from the given choices directly."
REPORT_PROMPT = ('You are a helpful assistant. Please generate a report for the given images, including both findings and '
                 'impressions. Return the report in the following format: Findings: {} Impression: {}.')


def _minmax_u8(t):
    """(n, H, W) float -> per-slice min-max to [0, 255], floored (astype uint8); constant slice -> 0."""
    lo = t.amin(dim=(1, 2), keepdim=True)
    hi = t.amax(dim=(1, 2), keepdim=True)
    out = (t - lo) / (hi - lo).clamp_min(1e-6) * 255.0
    return torch.floor(out.clamp(0, 255)).to(torch.uint8)


def _square_xy(v, fill=-1024.0):
    """(X, Y, Z) -> symmetric pad of x / y to a square in-plane field of view."""
    X, Y, _ = v.shape
    s = max(X, Y)
    if X == Y:
        return v
    px, py = s - X, s - Y
    v = F.pad(v.permute(2, 0, 1)[None], (py // 2, py - py // 2, px // 2, px - px // 2), value=fill)[0]
    return v.permute(1, 2, 0)


def frames_rad3d(hu):
    """(X, Y, Z) 1 mm RAS HU -> (32, 256, 256) uint8: 3D-RAD/CT-RATE frames (sagittal, right -> left)."""
    lps = _square_xy(torch.flip(hu, dims=[0, 1]))
    t = F.interpolate(lps[None, None].float(), size=(N_SLICES, 256, 256), mode='trilinear', align_corners=False)[0, 0]
    return _minmax_u8(t)


def frames_axial(hu, n=N_SLICES, radiological=True):
    """(X, Y, Z) 1 mm RAS HU -> (n, S, S) uint8 axial slices, inferior -> superior. radiological: rows anterior ->
    posterior, columns patient's right -> left (AMOS-MM frames); else load_nii's raw array order on a CT-RATE file
    (rows = LPS x, columns = LPS y)."""
    lps = _square_xy(torch.flip(hu, dims=[0, 1]))
    idx = np.linspace(0, lps.shape[2] - 1, n, dtype=int)
    t = lps[:, :, torch.as_tensor(idx, device=hu.device)].permute(2, 1, 0) if radiological else \
        lps[:, :, torch.as_tensor(idx, device=hu.device)].permute(2, 0, 1)
    return _minmax_u8(t.float())


class Adapter:
    def __init__(self, ckpt, device):
        from transformers import AutoModelForCausalLM, AutoProcessor
        opts = (ckpt or '').split(':')
        modes = [o for o in opts if o in ('rad3d', 'axial', 'nii')]
        self.mode = modes[0] if modes else 'axial'
        ns = [int(o[1:]) for o in opts if o[:1] == 's' and o[1:].isdigit()]
        self.n_slices = ns[0] if ns else N_SLICES
        self.yesno = 'yesno' in opts
        self.dev = device
        try:
            import flash_attn  # noqa: F401
            attn = 'flash_attention_2'
        except ImportError:
            attn = 'sdpa'
        self.m = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True,
                                                      torch_dtype=torch.bfloat16, attn_implementation=attn).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
        self.tok = self.proc.tokenizer
        words = ('Yes', 'No') if self.yesno else ('A', 'B')
        self.pos, self.neg = [self.tok(w, add_special_tokens=False).input_ids[0] for w in words]
        self.img_tok = self.m.config.image_token_index
        print(f'Hulu-Med-7B: mode={self.mode} slices={self.n_slices} question={"yes/no" if self.yesno else "3D-RAD A/B"} attn={attn}',
              flush=True)

    def question(self, phrase):
        if self.yesno:
            return f'Is there {phrase} in this CT? Answer Yes or No.'
        return f'Is {phrase} present on the CT scan?' + MC_SUFFIX

    def frames(self, hu):
        if self.mode == 'rad3d':
            return frames_rad3d(hu)
        return frames_axial(hu, self.n_slices, radiological=self.mode == 'axial')

    def _inputs(self, fr, text):
        from PIL import Image
        imgs = [Image.fromarray(f).convert('RGB') for f in fr.cpu().numpy()]
        conv = [{'role': 'user', 'content': [{'type': 'video', 'num_frames': len(imgs)}, {'type': 'text', 'text': text}]}]
        x = self.proc(images=[imgs], conversation=conv, add_system_prompt=False, add_generation_prompt=True,
                      return_tensors='pt')
        return {k: (v.to(self.dev) if torch.is_tensor(v) else v) for k, v in x.items()}

    @torch.no_grad()
    def encode(self, hu):
        """Encode the frames and the chat prefix up to the question text; state = (KV cache, prefix length)."""
        x = self._inputs(self.frames(hu), 'Q')
        ids = x['input_ids']
        tail = self._tail('Q')
        P = ids.shape[1] - len(tail)
        assert ids[0, P:].tolist() == tail, 'prompt tail mismatch'
        _, _, _, _, emb, _ = self.m.prepare_inputs_labels_for_multimodal(
            input_ids=ids[:, :P], attention_mask=None, position_ids=None, past_key_values=None, labels=None,
            pixel_values=x['pixel_values'].to(torch.bfloat16), grid_sizes=x['grid_sizes'],
            merge_sizes=x['merge_sizes'], modals=x['modals'])
        out = self.m.model(inputs_embeds=emb, use_cache=True)
        feat = emb[0][ids[0, :P] == self.img_tok].float().mean(0)
        return (out.past_key_values, P), feat

    def _tail(self, q):
        return self.tok(q + '<|im_end|>\n<|im_start|>assistant\n', add_special_tokens=False).input_ids

    @torch.no_grad()
    def p_yes(self, state, questions):
        return torch.sigmoid(self.yes_no_logit(state, questions))

    @torch.no_grad()
    def yes_no_logit(self, state, questions, chunk=6):
        """logit('A') - logit('B') (or 'Yes' - 'No') at the first answer token. Question tails run in batches of
        `chunk` on a repeated copy of the prefix KV cache (bounds memory with ~8k visual tokens); checked equal to the
        first-token scores of the model's own generate() on the full prompt."""
        from transformers import DynamicCache
        cache, P = state
        legacy = cache.to_legacy_cache() if hasattr(cache, 'to_legacy_cache') else cache
        w = self.m.lm_head.weight
        wd = w[self.pos].float() - w[self.neg].float()            # the two answer rows of lm_head, in float32
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        out = []
        for s0 in range(0, len(questions), chunk):
            tails = [self._tail(q) for q in questions[s0:s0 + chunk]]
            n, T = len(tails), max(len(t) for t in tails)
            ids = torch.full((n, T), pad, dtype=torch.long)
            for k, t in enumerate(tails):
                ids[k, :len(t)] = torch.tensor(t)                 # right padding: causal, pads never attended to
            c = DynamicCache.from_legacy_cache(tuple((kk.expand(n, -1, -1, -1).contiguous(),
                                                      vv.expand(n, -1, -1, -1).contiguous()) for kk, vv in legacy))
            h = self.m.model(input_ids=ids.to(self.dev), past_key_values=c, use_cache=True).last_hidden_state
            last = torch.tensor([len(t) - 1 for t in tails], device=self.dev)
            out.append(h[torch.arange(n, device=self.dev), last].float() @ wd)   # final norm is inside self.m.model
            del c, h
        return torch.cat(out)

    def report_prompt(self, dataset):
        return REPORT_PROMPT

    @torch.no_grad()
    def report(self, hu, dataset, max_new_tokens=768):
        """Greedy report with the model's own generate() (MedUniEval HuluMed_Qwen2.generate_output)."""
        x = self._inputs(self.frames(hu), self.report_prompt(dataset))
        x['pixel_values'] = x['pixel_values'].to(torch.bfloat16)
        out = self.m.generate(**x, do_sample=False, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                              use_cache=True, pad_token_id=self.tok.eos_token_id)
        n_in = x['input_ids'].shape[1]
        if out.shape[1] > n_in and torch.equal(out[0, :n_in], x['input_ids'][0]):
            out = out[:, n_in:]                                  # keep only the new tokens if the prompt is echoed
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()
