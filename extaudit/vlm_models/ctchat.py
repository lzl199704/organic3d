"""CT-CHAT (CT-RATE authors; Llama 3.1 8B LoRA + CT-CLIP v2 CTViT, CC-BY-NC-SA) as an extaudit adapter for report
generation (Track B, extaudit/vlm_report.py) and yes/no answers (extaudit/vlm_core.py: the CT-CHAT audit's question
route, "<image>\\n <provided>" + 'Is there X in this CT? Answer Yes or No.' + " <short_answer>", P(Yes) = sigmoid of the
logsumexp of the 'Yes' / ' Yes' logits minus that of 'No' / ' No' at the first answer token; used for Merlin and
ReXGroundingCT). CT-CHAT is a chest model: on Merlin it is run as is, and its report question drops the word "chest".
Run in $ORGANIC3D_BASELINES/ctchat/venv (transformers 4.44, the
repo's llava code in $ORGANIC3D_BASELINES/ctchat/code). Loading and input route = $ORGANIC3D_BASELINES/ctchat/ctchat_score.py
(the CT-CHAT yes/no audit script; NOT distributed with this repository, see extaudit/README.md): CTViT encoded
tokens of preprocess() (RAS -> LPS, HU [-1000, 1000] / 1000, 0.75 x 0.75 x 1.5 mm, crop / pad to 480 x 480 x 240).
The audit set's CT-RATE input is the same 1 mm NIfTI array ctchat_score.py reads (nibabel dataobj, RAS).

Prompt = the CT-CHAT demo route (llava/serve/gradio_web_server.py): conv_templates['llama3'] with
"<image>\\n <provided>" + q + " <report_generation>", q = the most frequent report-generation question of CT-RATE's
valid_vqa.json ("Could you write the radiology report for this chest CT scan?", 148 of its report questions).
Decoding as llava/serve/ctchat_validation_llama.py at its default temperature 0 (greedy, use_cache); its default
max_new_tokens is 512, the driver's --max_new_tokens is used here.

    CUDA_VISIBLE_DEVICES=0 $ORGANIC3D_BASELINES/ctchat/venv/bin/python -W ignore \
        -m extaudit.vlm_report --model ctchat --name CT-CHAT --dataset ctrate --root <work_root>/extaudit
"""
import sys

import torch

from extaudit.paths import baseline

ROOT = baseline('ctchat')
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/code')
sys.path.insert(0, baseline('CT-CLIP', 'transformer_maskgit'))
REPORT_Q = 'Could you write the radiology report for this chest CT scan?'
REPORT_Q_OTHER = 'Could you write the radiology report for this CT scan?'


class Adapter:
    def __init__(self, ckpt, device):
        from transformer_maskgit import CTViT
        from llava.model.builder import load_pretrained_model
        from ctchat_score import BASE, MODEL
        self.dev = device
        enc = CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20, temporal_patch_size=10, spatial_depth=4,
                    temporal_depth=4, dim_head=32, heads=8)
        sd = torch.load(baseline('CT-CLIP', 'CT-CLIP_v2.pt'), map_location='cpu')
        sd = {k[len('visual_transformer.'):]: v for k, v in sd.items() if k.startswith('visual_transformer.')}
        print('encoder', enc.load_state_dict(sd, strict=False), flush=True)
        self.enc = enc.to(device).eval()
        self.tok, self.m, _, _ = load_pretrained_model(MODEL, BASE, 'llava-lora-llama3.1-8b', device_map={'': device})
        self.m.eval()
        t = self.tok
        self.yes = [t('Yes', add_special_tokens=False).input_ids[0], t(' Yes', add_special_tokens=False).input_ids[0]]
        self.no = [t('No', add_special_tokens=False).input_ids[0], t(' No', add_special_tokens=False).input_ids[0]]
        self._ids = {}

    @staticmethod
    def question(phrase):
        return f'Is there {phrase} in this CT? Answer Yes or No.'

    @torch.no_grad()
    def encode(self, hu):
        from ctchat_score import preprocess
        emb = self.enc(preprocess(hu.float().cpu().numpy()).to(self.dev).unsqueeze(0), return_encoded_tokens=True)
        return emb, emb.float().mean(dim=tuple(range(1, emb.dim() - 1)))[0]

    def _prompt(self, q):
        if q not in self._ids:
            from llava.constants import IMAGE_TOKEN_INDEX
            from llava.conversation import conv_templates
            from llava.mm_utils import tokenizer_image_token
            conv = conv_templates['llama3'].copy()
            conv.append_message(conv.roles[0], '<image>\n <provided>' + q + ' <short_answer>')
            conv.append_message(conv.roles[1], None)
            self._ids[q] = tokenizer_image_token(conv.get_prompt(), self.tok, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return self._ids[q]

    @torch.no_grad()
    def yes_no_logit(self, state, questions):
        out = []
        for q in questions:
            lg = self.m(input_ids=self._prompt(q)[None].to(self.dev), images=state.to(torch.float16), use_cache=False).logits[0, -1].float()
            out.append(torch.logsumexp(lg[self.yes], 0) - torch.logsumexp(lg[self.no], 0))
        return torch.stack(out)

    def p_yes(self, state, questions):
        return torch.sigmoid(self.yes_no_logit(state, questions))

    @staticmethod
    def report_prompt(dataset):
        return REPORT_Q if dataset == 'ctrate' else REPORT_Q_OTHER

    def report(self, hu, dataset, max_new_tokens=768):
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from ctchat_score import preprocess
        conv = conv_templates['llama3'].copy()
        conv.append_message(conv.roles[0], '<image>\n <provided>' + self.report_prompt(dataset) + ' <report_generation>')
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), self.tok, IMAGE_TOKEN_INDEX, return_tensors='pt')[None].to(self.dev)
        emb = self.enc(preprocess(hu.float().cpu().numpy()).to(self.dev).unsqueeze(0), return_encoded_tokens=True)
        out = self.m.generate(ids, images=emb.to(torch.float16), do_sample=False, max_new_tokens=max_new_tokens,
                              use_cache=True)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()
