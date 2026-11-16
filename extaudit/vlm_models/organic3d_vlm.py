"""Our stage-B VLMs (Organic3DVLM: stage-A encoder -> projector -> Qwen3.5-4B + LoRA) as an extaudit VLM adapter, so they
answer the external-VLM audit (same 1,000 full 1 mm scans per dataset, same perturbations and metric code) next to
NV-Reason-CT, CT-CHAT, MedGemma, M3D-LaMed, RadFM, OmniCT and Hulu-Med. Run in the organic env.

--ckpt: the released weights (a Hugging Face repo id such as zllhhhh/organic3d-vlm, or a downloaded folder; optionally
"<vlm>::<encoder>" to choose the encoder repo or folder), or "<stage-B run dir>::<stage-A encoder last.pt>" for a
training run: the stage-B checkpoint keeps only the trained parts (LoRA,
projector, query head), so the frozen encoder is loaded from its own checkpoint (the encoder path stored in the run config
is replaced by the one given here).
Input = CTDataset's evaluation crop (as organic3d_enc.py): centred on lungs + heart (CT-RATE) or the abdominal organs
(Merlin) from the TotalSegmentator mask, 384 mm at 1 mm, trilinear to 192^3 at 2 mm, HU clipped to [-1000, 1000] / 1000.
Question = our training question; score = logit(Yes) - logit(No) at the first answer position (as Organic3DVLM.p_yes).
Track B report (extaudit/vlm_report.py): our training report prompt (train_stageB.report_prompt) through Organic3DVLM.generate,
greedy, as evaluate_stageB's --gen path (which capped at 256 new tokens; here the driver's cap).
"""
import os

import torch

from organic.config import load_config
from organic.train_stageB import finding_list, question as our_question, report_prompt as our_report_prompt
from organic.vocab import FAMILY, FINDING_NAMES, FINDINGS
from extaudit.models.organic3d_enc import ENV, Adapter as EncAdapter, inference_module

PHRASE_TO_FINDING = {v[2]: k for k, v in FINDINGS.items()}


class Adapter:
    def __init__(self, ckpt, device):
        from organic.vlm import Organic3DVLM, build_encoder
        run, enc_ckpt = ckpt.split('::') if '::' in ckpt else (ckpt, None)
        if not os.path.exists(os.path.join(run, 'last.pt')):            # released weights (inference script's loader)
            O = inference_module()
            self.dev = device
            self.m = O.VLM(run, enc_ckpt or O.ENCODER_REPO, device).m.eval()
            self.crop = EncAdapter.__new__(EncAdapter)
            self.crop.dataset = 'ctrate'
            return
        cfg = load_config(os.path.join(run, 'config.yaml'), env=ENV, overrides=[f'vlm.encoder_ckpt={enc_ckpt}'])
        os.environ.setdefault('HF_HOME', cfg.paths.hf_home)
        cfg.model.pretrained = cfg.vlm.encoder_ckpt == 'colipri'
        self.dev = device
        self.m = Organic3DVLM(cfg, build_encoder(cfg, device)).to(device)
        st = torch.load(os.path.join(run, 'last.pt'), map_location='cpu', weights_only=False)['model']
        missing, unexpected = self.m.load_state_dict(st, strict=False)
        assert not unexpected, unexpected[:5]
        self.m.eval()
        self.crop = EncAdapter.__new__(EncAdapter)            # reuse the crop code only
        self.crop.dataset = 'ctrate'

    def set_dataset(self, dataset):
        self.crop.dataset = 'merlin' if dataset == 'merlin' else 'ctrate'

    @staticmethod
    def question(phrase):
        if phrase in PHRASE_TO_FINDING:
            return our_question(PHRASE_TO_FINDING[phrase])
        return f'Is there {phrase} in this CT scan? Answer Yes or No.'     # free phrase (lesion tests): same template

    def set_scan(self, seg):
        self.crop.set_scan(seg)

    def encode(self, hu):
        x = self.crop.prep(hu)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            V, _ = self.m.visual_tokens(x)
        return V, V.float().mean(1)[0]

    def yes_no_logit(self, V, questions):
        n = len(questions)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb, mask, _, _ = self.m.build(V.expand(n, -1, -1), questions, [''] * n, self.dev)
            h, _ = self.m.hidden(emb, mask)
        last = mask.sum(1) - 1
        w = self.m.head().weight[[self.m.yes_id, self.m.no_id]].float()
        two = h[torch.arange(n, device=self.dev), last].float() @ w.T
        return two[:, 0] - two[:, 1]

    def p_yes(self, V, questions):
        return torch.sigmoid(self.yes_no_logit(V, questions))

    def report_prompt(self, dataset):
        return our_report_prompt('merlin' if dataset == 'merlin' else 'ctrate', self.m.cfg.vlm.get('report_mode', 'free'))

    def report(self, hu, dataset, max_new_tokens=768):
        """Free mode: greedy report. Answer-then-write ('list' runs): the reply starts with the model's own yes/no
        answers on this scan (P(Yes) >= 0.5 over the dataset's findings), the report is the continuation after 'Report:'
        (only that is returned and scored; the list is kept in self.last_meta)."""
        V, _ = self.encode(hu)
        pre = None
        if self.m.cfg.vlm.get('report_mode', 'free') == 'list':
            d = 'merlin' if dataset == 'merlin' else 'ctrate'
            fs = [f for f in FINDING_NAMES if f.startswith(FAMILY[d])]
            p = self.p_yes(V, [our_question(f) for f in fs]).float().cpu().numpy()
            pre = finding_list([f for f, x in zip(fs, p) if x >= 0.5])
            self.last_meta = {'list': pre.strip()}
        # REPORT_HEAD_MERLIN / REPORT_HEAD_CTRATE (2026-10-26, default off): start the report with the first item of the
        # dataset's findings order (Merlin 'Lower thorax:', CT-RATE 'Trachea'), so answer-then-write cannot skip to a
        # numbered impression; the head is part of the returned report.
        head = os.environ.get('REPORT_HEAD_MERLIN' if dataset == 'merlin' else 'REPORT_HEAD_CTRATE', '')
        if head:
            pre = f'{pre} {head}' if pre else head
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = self.m.generate(V, self.report_prompt(dataset), self.dev, max_new_tokens=max_new_tokens, prefix=pre)
        body = out[0].strip()
        if not head:
            return body
        return head + ('' if body[:1] in ',.;:' else ' ') + body
