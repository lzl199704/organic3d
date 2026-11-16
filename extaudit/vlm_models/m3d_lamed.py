"""M3D-LaMed-Phi-3-4B (Bai et al. 2024, arXiv 2404.00578; weights HF GoodBaiBai88/M3D-LaMed-Phi-3-4B, Apache-2.0;
code github.com/BAAI-DCAI/M3D, MIT), run in $ORGANIC3D_BASELINES/m3d_lamed/venv (transformers 4.42.4, monai 1.3.2,
the HF repo's own modeling_m3d_lamed.py via trust_remote_code; bf16 as in the README quickstart / online demo;
M3D_LAMED_FP32=1 loads fp32 instead: bf16 moves near-threshold logits by up to ~1.4 vs another bf16 path).

Preprocessing = the repo's Data/process scripts (m3d_cap_data_prepare.py, m3d_seg_data_prepare.py): volume as
(D, H, W) = (axial slice, row, column), min-max normalised to [0, 1], monai CropForeground (voxels > 0, i.e. drop the
bounding margin at the volume minimum), monai Resize to 32 x 256 x 256 (mode 'bilinear' = trilinear,
align_corners=False, no anti-aliasing; checked identical to F.interpolate), input 1 x 1 x 32 x 256 x 256.
HU handling: the repo has no HU window for 3D NIfTI input (M3D-Cap/VQA are Radiopaedia display-windowed JPEG slices;
M3D-Seg clips to foreground percentiles). We clip to WINDOW = [-1000, 1000] HU before min-max (also removes Merlin's
-3024 FOV padding). On the first 150 CT-RATE audit scans (clean, 18 findings, macro AUROC): [-1000, 1000] 0.579,
M3D-Seg foreground-percentile normalise 0.558, lung window [-1350, 150] 0.566, soft tissue [-160, 240] 0.572.
Orientation (ORIENT, default 'rad'): radiological display order, as for Radiopaedia slices and for M3D-Seg's LPS-like
NIfTIs after its swapaxes(-1, -3): D = slices inferior -> superior, H = anterior -> posterior, W = patient right ->
left. Training used RandRotate90 in-plane p=0.5 and random flips p=0.1 per axis; orientation barely matters (same 150
scans: 'rad' 0.579, slices reversed 0.587, plain RAS 0.579, in-plane transposed 0.582; P(Yes) r = 0.94 rad vs reversed).

Prompt (no chat template; LaMed's VQA training / eval format, LaMed/src/dataset/multi_dataset.py VQADataset and
Bench/eval/eval_vqa.py): tokenizer("<im_patch>" * 256 + " " + question) with BOS; the 256 projected visual tokens
replace the <im_patch> embeddings (positions 1..256). Question = the M3D open-ended yes/no style ("Is ... in this
image?", online_demo.py usage note; M3D-VQA contains Yes/No open answers), i.e. "Is there <phrase> in this image?".
P(Yes) = sigmoid(logit('▁Yes') - logit('▁No')) at the first answer token (training text = question + " " + answer).
Probe feature = mean over the 256 projected visual tokens (the LLM input, 3072-d).
Report (extaudit/vlm_report.py): M3D-Cap caption format of the authors' caption eval (Bench/eval/eval_caption.py
+ Bench/dataset/multi_dataset.CapDataset: question = "<im_patch>" * 256 + one of Bench/dataset/prompt_templates
Caption_templates, no space in between; the eval draws one at random, we fix the first: "Can you provide a caption
consists of findings for this medical image?"), same image preprocessing as above, the model's own
generate(image, input_ids) with do_sample=False; eval_caption's max_new_tokens default 256 caps our request.
"""
import os

import torch
import torch.nn.functional as F

from extaudit.paths import baseline

WEIGHTS = baseline('m3d_lamed', 'weights')      # local snapshot of HF GoodBaiBai88/M3D-LaMed-Phi-3-4B
N_IMG = 256
CAPTION_PROMPT = 'Can you provide a caption consists of findings for this medical image?'
MAX_NEW = 256                  # eval_caption.py --max_new_tokens default


class Adapter:
    WINDOW = (-1000.0, 1000.0)     # HU clip before min-max; None = M3D-Seg foreground-percentile normalise
    ORIENT = 'rad'                 # 'rad' | 'rad_zflip' | 'ras' | 'rad_T' (orientation tests)

    def __init__(self, ckpt, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = device
        path = ckpt or WEIGHTS
        self.tok = AutoTokenizer.from_pretrained(path, model_max_length=512, padding_side='right', use_fast=False,
                                                 trust_remote_code=True)
        dtype = torch.float32 if os.environ.get('M3D_LAMED_FP32') == '1' else torch.bfloat16
        self.m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
        self.yes = self.tok('Yes', add_special_tokens=False).input_ids[-1]      # '▁Yes'
        self.no = self.tok('No', add_special_tokens=False).input_ids[-1]        # '▁No'
        self._qcache = {}

    @staticmethod
    def question(phrase):
        return f'Is there {phrase} in this image?'

    def prep(self, hu):
        """(X, Y, Z) 1 mm RAS HU (cuda) -> (1, 1, 32, 256, 256) in [0, 1]."""
        if self.ORIENT == 'ras':
            v = hu.permute(2, 1, 0)                                   # (z, y: P->A, x: L->R)
        else:
            v = torch.flip(hu, dims=[0, 1]).permute(2, 1, 0)          # (z: I->S, A->P, R->L)
            if self.ORIENT == 'rad_zflip':
                v = torch.flip(v, dims=[0])                           # slices S->I
            elif self.ORIENT == 'rad_T':
                v = v.transpose(1, 2)
        v = v.float()
        if self.WINDOW is None:                                      # m3d_seg_data_preprocess.normalize
            flat = v.flatten()
            fg = flat[flat > flat.mean()]
            q = torch.quantile(fg[torch.randperm(fg.numel(), device=fg.device)[:1_000_000]].float(),
                               torch.tensor([0.0005, 0.9995], device=fg.device))
            v = v.clamp(float(q[0]), float(q[1]))
        else:
            v = v.clamp(*self.WINDOW)
        v = v - v.min()
        v = v / v.max().clamp_min(1e-8)
        nz = v > 0                                                    # monai CropForeground (select_fn x > 0)
        sl = []
        for d in range(3):
            ax = nz.any(dim=tuple(i for i in range(3) if i != d)).nonzero()
            sl.append(slice(int(ax[0]), int(ax[-1]) + 1) if len(ax) else slice(None))
        v = v[sl[0], sl[1], sl[2]]
        v = F.interpolate(v[None, None], size=(32, 256, 256), mode='trilinear', align_corners=False)
        return v

    def encode(self, hu):
        x = self.prep(hu).to(self.m.dtype)
        tokens = self.m.encode_images(x)                              # (1, 256, 3072) projected visual tokens
        return tokens, tokens[0].float().mean(0)

    def _ids(self, questions):
        key = tuple(questions)
        if key not in self._qcache:
            seqs = [self.tok('<im_patch>' * N_IMG + ' ' + q).input_ids for q in questions]
            assert all(s[0] == self.tok.bos_token_id and len(set(s[1:N_IMG + 1])) == 1 for s in seqs)
            T = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), T), self.tok.pad_token_id, dtype=torch.long)
            am = torch.zeros((len(seqs), T), dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, :len(s)] = torch.tensor(s); am[i, :len(s)] = 1
            last = torch.tensor([len(s) - 1 for s in seqs])
            self._qcache[key] = (ids.to(self.dev), am.to(self.dev), last.to(self.dev))
        return self._qcache[key]

    def p_yes(self, state, questions):
        return torch.sigmoid(self.yes_no_logit(state, questions))

    def yes_no_logit(self, state, questions):
        """logit(Yes) - logit(No) at the first answer token (rank-equivalent to P(Yes), no float32 saturation).
        Shared prefix (BOS + 256 visual tokens) encoded once; question tails batched on its KV cache (same causal
        computation as the full sequence; checked against the model's own forward(images, input_ids))."""
        from transformers import DynamicCache
        ids, am, last = self._ids(questions)
        n, P = ids.shape[0], N_IMG + 1
        emb = self.m.get_model().embed_tokens(ids[:1, :1])
        pre = torch.cat([emb, state.to(emb.dtype)], dim=1)                           # (1, 257, d)
        kv = self.m(inputs_embeds=pre, use_cache=True).past_key_values
        kv = kv.to_legacy_cache() if hasattr(kv, 'to_legacy_cache') else kv
        kv = DynamicCache.from_legacy_cache(tuple((k.expand(n, -1, -1, -1).contiguous(),
                                                   v.expand(n, -1, -1, -1).contiguous()) for k, v in kv))
        tail = ids[:, P:]
        mask = torch.cat([torch.ones((n, P), dtype=am.dtype, device=self.dev), am[:, P:]], dim=1)
        pos = torch.arange(P, ids.shape[1], device=self.dev)[None].expand(n, -1)
        lg = self.m(input_ids=tail, attention_mask=mask, position_ids=pos, past_key_values=kv, use_cache=True).logits
        lg = lg[torch.arange(n, device=self.dev), last - P].float()
        return lg[:, self.yes] - lg[:, self.no]

    @staticmethod
    def report_prompt(dataset):
        return CAPTION_PROMPT

    @torch.no_grad()
    def report(self, hu, dataset, max_new_tokens=MAX_NEW):
        """Greedy caption (findings) as eval_caption.py: model.generate(image, input_ids); returns the new text only."""
        x = self.prep(hu).to(self.m.dtype)
        ids = self.tok('<im_patch>' * N_IMG + self.report_prompt(dataset), return_tensors='pt').input_ids.to(self.dev)
        out = self.m.generate(x, ids, max_new_tokens=min(max_new_tokens, MAX_NEW), do_sample=False)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()
