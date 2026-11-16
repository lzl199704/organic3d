"""RadFM (Wu et al. 2023, arXiv 2308.02463; weights HF chaoyi-wu/RadFM pytorch_model.zip, code
github.com/chaoyi-wu/RadFM, MIT), run in $ORGANIC3D_BASELINES/radfm/venv (torch 2.5.1 cu124, transformers 4.44.2).
The repo's own model code (repo/Quick_demo/Model/RadFM: 3D ViT + perceiver + LLaMA-13B) with the fp32 checkpoint
converted once to bf16 (with a local conversion helper, not distributed with this repository; drops the unused Med-KEBERT keyword-loss weights and rotary buffers).

Preprocessing = RadFM's 3D data path (src/Dataset/dataset/jpg2nii_data_convert.py + nii2npy_for_radiopaedio.py /
radiopaedia.py resize_image, multi_dataset_test.stack_images): a 3D scan is a stack of display slices, array
(C=3, H=row, W=column, D=slice) with the grey image repeated over 3 channels, in-plane resized to 512 x 512 and depth
to at most 64 slices (ndimage.zoom order 0 / F.interpolate default 'nearest'; stack_images then picks D from
{4, 8, ..., 64}), min-max normalised to [0, 1] per volume; input vision_x = (B=1, S=1, 3, 512, 512, D).
HU handling: RadFM has no HU window (its 3D CT data are Radiopaedia display-windowed JPEG stacks). We clip to WINDOW
(default [-1000, 1000] HU, the CT-RATE convention; chosen with the orientation on CT-RATE scans outside the audit
sample, see the report) before the min-max; this also removes Merlin's -3024 FOV padding.
Orientation (ORIENT, default 'rad'): radiological display slices as in Radiopaedia JPEGs, H = anterior -> posterior,
W = patient right -> left, D = slice order inferior -> superior.
SIZE: 512 (official inference path; multi_dataset_test / Quick_demo keep 512 x 512); the training collator used
256 x 256 for single 3D samples (src/train.py DataCollator), kept as an option.

Prompt (Quick_demo/test.py and the training format question + ' ' + answer, answers 'yes' / 'no'): the image is
inserted at position 0 as "<image>" + "<image0>...<image31>" + "</image>" followed by the question, tokenizer with BOS
(id 1, as set in get_tokenizer). Question = RadFM's own binary-diagnosis prompt (Quick_demo and
src/Dataset/dataset/yes_no_prompt.json): "Can you identify any visible signs of <phrase> in the image?".
The 32 perceiver tokens (projected to 5120-d) replace the <image0..31> embeddings exactly as MyEmbedding's one-hot
matmul does (gathered instead of matmul, same values). P(Yes) = sigmoid(logsumexp logits{'▁yes','▁Yes'} -
logsumexp logits{'▁no','▁No'}) at the first answer token
(yes_no_logit returns the logit difference itself). Probe feature = mean of the 32 projected visual tokens.
Report (extaudit/vlm_report.py): RadFM's 3D report task (multi_dataset_test 'caption' split,
radiopaedia.RadioCaption_Dataset: question = age/sex + presentation + a caption_prompt.json template, answer
"Finding: ... Impression: ..."); no clinical history here, so the question is the first template alone, "Can you provide
a caption consists of finding and impression for this medical image?", image at position 0 as above. Decoding as
MultiLLaMAForCausalLM.generate (embedding layer -> lang_model.generate(inputs_embeds), max_new_tokens=200, which caps
our request), but greedy (do_sample=False; their top_k=50 is inert without sampling) and stopping at the tokenizer's
EOS (id 2): the checkpoint's LLaMA config carries the old eos_token_id=1, with which generation never stops early.
Smoke test (3 + 3 scans): this 3D caption template returns the literal "Finding: nanImpression: nan" for every scan (the
Radiopaedia training answers with missing finding / impression); RADFM_REPORT_PROMPT=report selects the first
report_prompt.json template (the MIMIC-CXR report task of the same test split), "Can you provide a radiology report
for this medical image?", which gives one-sentence Radiopaedia-style captions.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from extaudit.paths import baseline

BASE = baseline('radfm')                  # repo/ (code clone) and weights/ below it
WEIGHTS = f'{BASE}/weights/radfm_bf16.safetensors'
LANG = f'{BASE}/repo/Quick_demo/Language_files'
N_IMG = 32
CAPTION_PROMPT = 'Can you provide a caption consists of finding and impression for this medical image?'
REPORT_PROMPT = 'Can you provide a radiology report for this medical image?'
MAX_NEW = 200                  # MultiLLaMAForCausalLM.generate


class _RadFM(nn.Module):
    """Same attribute names as MultiLLaMAForCausalLM, so the checkpoint keys load unchanged."""
    def __init__(self):
        super().__init__()
        sys.path.insert(0, f'{BASE}/repo/Quick_demo')
        import Model.RadFM.my_embedding_layer as mel
        from transformers import LlamaConfig, LlamaForCausalLM

        class _NoBert:                                           # Med-KEBERT only feeds the disabled keyword loss
            @staticmethod
            def from_pretrained(*a, **k):
                return None
        mel.AutoTokenizer = mel.AutoModel = _NoBert
        self.lang_model = LlamaForCausalLM(LlamaConfig.from_pretrained(LANG))
        self.embedding_layer = mel.MyEmbedding()
        self.embedding_layer.weight = self.lang_model.get_input_embeddings().weight


class Adapter:
    WINDOW = (-1000.0, 1000.0)     # HU clip before min-max
    ORIENT = 'rad'                 # 'rad' (slices I->S) | 'rad_zflip' (S->I) | 'rad_lr' (W = left -> right)
    SIZE = 512                     # in-plane size (512 official inference; 256 training collator for 3D)
    MAX_D = 64

    def __init__(self, ckpt, device):
        from safetensors.torch import load_file
        from transformers import LlamaTokenizer
        self.dev = device
        tok = LlamaTokenizer.from_pretrained(LANG, legacy=True)
        tok.add_special_tokens({'additional_special_tokens': ['<image>', '</image>'] +
                                [f'<image{j}>' for j in range(100 * N_IMG)]})          # = get_tokenizer(max_img_size=100)
        tok.pad_token_id, tok.bos_token_id, tok.eos_token_id = 0, 1, 2
        self.tok = tok
        self.img_text = '<image>' + ''.join(f'<image{j}>' for j in range(N_IMG)) + '</image>'
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(device):
                m = _RadFM()
        finally:
            torch.set_default_dtype(torch.float32)
        missing, unexpected = m.load_state_dict(load_file(ckpt or WEIGHTS), strict=False)
        assert not unexpected and not [k for k in missing if 'bert_model' not in k], (missing, unexpected)
        self.m = m.eval()
        self.emb = m.embedding_layer
        self.yes = [tok.convert_tokens_to_ids(w) for w in ('▁yes', '▁Yes')]
        self.no = [tok.convert_tokens_to_ids(w) for w in ('▁no', '▁No')]
        self._qcache = {}

    @staticmethod
    def question(phrase):
        return f'Can you identify any visible signs of {phrase} in the image?'

    def prep(self, hu):
        """(X, Y, Z) 1 mm RAS HU (cuda) -> vision_x (1, 1, 3, SIZE, SIZE, D) in [0, 1]."""
        v = torch.flip(hu, dims=[0, 1]).permute(1, 0, 2)             # (H: A->P, W: R->L, D: I->S)
        if self.ORIENT == 'rad_zflip':
            v = torch.flip(v, dims=[2])
        elif self.ORIENT == 'rad_lr':
            v = torch.flip(v, dims=[1])
        v = v.float().clamp(*self.WINDOW)
        d = min(v.shape[2], self.MAX_D)                               # radiopaedia.resize_image: depth > 64 -> 64
        d = min(range(4, 65, 4), key=lambda t: (abs(t - d), t))       # stack_images: nearest of {4, ..., 64}
        v = F.interpolate(v[None, None], size=(self.SIZE, self.SIZE, d), mode='nearest')[0, 0]
        v = (v - v.min()) / (v.max() - v.min()).clamp_min(1e-8)
        return v[None, None, None].expand(1, 1, 3, -1, -1, -1)

    def encode(self, hu):
        x = self.prep(hu).to(torch.bfloat16)                         # (B, S, C, H, W, D)
        feats, _ = self.emb.vision_encoder(x[:, 0])                  # (1, N, 768)
        feats = self.emb.perceiver(feats[:, None, None])             # (b, s, F, v, d) -> (1, 1, 32, 768)
        tokens = self.emb.fc(feats[0, 0])                            # (32, 5120)
        return tokens, tokens.float().mean(0)

    def _ids(self, questions):
        key = tuple(questions)
        if key not in self._qcache:
            seqs = [self.tok(self.img_text + q).input_ids for q in questions]
            img_ids = list(range(32002, 32002 + N_IMG))
            assert all(s[0] == 1 and s[1] == 32000 and s[2:2 + N_IMG] == img_ids and s[2 + N_IMG] == 32001
                       for s in seqs)
            T = max(len(s) for s in seqs)
            ids = torch.zeros((len(seqs), T), dtype=torch.long)
            am = torch.zeros((len(seqs), T), dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, :len(s)] = torch.tensor(s); am[i, :len(s)] = 1
            last = torch.tensor([len(s) - 1 for s in seqs])
            self._qcache[key] = (ids.to(self.dev), am.to(self.dev), last.to(self.dev))
        return self._qcache[key]

    def p_yes(self, state, questions):
        return torch.sigmoid(self.yes_no_logit(state, questions))

    def yes_no_logit(self, state, questions):
        """logsumexp logit{'▁yes','▁Yes'} - logsumexp logit{'▁no','▁No'} at the first answer token."""
        ids, am, last = self._ids(questions)
        n = ids.shape[0]
        table = torch.cat([self.emb.weight, self.emb.figure_token_weight, state.to(self.emb.weight.dtype)], 0)
        emb = table[ids]                                             # = one_hot(ids) @ [weight; figure; vision]
        lg = self.m.lang_model(inputs_embeds=emb, attention_mask=am, use_cache=False).logits
        lg = lg[torch.arange(n, device=self.dev), last].float()
        return torch.logsumexp(lg[:, self.yes], 1) - torch.logsumexp(lg[:, self.no], 1)

    @staticmethod
    def report_prompt(dataset):
        return REPORT_PROMPT if os.environ.get('RADFM_REPORT_PROMPT') == 'report' else CAPTION_PROMPT

    @torch.no_grad()
    def report(self, hu, dataset, max_new_tokens=MAX_NEW):
        """Greedy report: image tokens at position 0 + prompt, inputs_embeds -> lang_model.generate; new text only."""
        tokens, _ = self.encode(hu)
        ids = torch.tensor([self.tok(self.img_text + self.report_prompt(dataset)).input_ids], device=self.dev)
        table = torch.cat([self.emb.weight, self.emb.figure_token_weight, tokens.to(self.emb.weight.dtype)], 0)
        out = self.m.lang_model.generate(inputs_embeds=table[ids], attention_mask=torch.ones_like(ids),
                                         max_new_tokens=min(max_new_tokens, MAX_NEW), do_sample=False,
                                         eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()
