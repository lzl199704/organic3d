"""ORGANIC-3D inference: the encoder (finding scores, sentence scores) and the VLM (yes/no answers, reports) for one CT.

Needs the ORGANIC-3D code (the `organic` package, https://github.com/lzl199704/organic3d; `pip install -e .` in the repository) and its
requirements (torch, colipri, transformers >= 5, peft, nibabel, scipy, safetensors). Optional: TotalSegmentator for the
crop centre.

Preprocessing (as in the paper's evaluation):
  1. reorient to RAS (nibabel as_closest_canonical) and resample to 1 mm (HU trilinear, align_corners=False, rounded);
  2. crop centre: from a TotalSegmentator v2 'total' mask on the 1 mm grid (chest: lungs + heart; abdomen: liver,
     spleen, kidneys, pancreas, head-foot from the top of the liver to the pelvic organs; recommended), or, without a
     mask, an estimate (chest: the air inside the body; abdomen: the body), off by ~3 mm (chest) and ~11 mm (abdomen)
     per axis on average, so scores differ from the paper's;
  3. 384 mm cube at 1 mm (padded with -1000 HU), trilinear to 192^3 (2 mm), HU clipped to [-1000, 1000] / 1000.

Encoder: S(V, s) is the sentence-conditioned similarity; a finding f is scored by Delta_f = S(V, "There is f.") -
S(V, "No f."), the score used for every result in the paper. Any sentence can be scored with S(V, s).
VLM: P(Yes) against No at the first answer token of "Is there f in this CT scan? Answer Yes or No."; reports are
written answer-first (the model's own Yes answers, P(Yes) >= 0.5, are listed, then the report), and the report is
started with the first item of the findings order ("Trachea" chest, "Lower thorax:" abdomen), as in the paper.

    python organic3d.py encoder --ct scan.nii.gz --region chest [--seg ts_1mm.nii.gz | --totalseg] \
        [--sentence "There is a nodule in the right upper lobe."] [--weights <encoder repo or dir>]
    python organic3d.py vlm --ct scan.nii.gz --region chest [--seg ... | --totalseg] [--question "..."] [--report] \
        [--weights <vlm repo or dir>] [--encoder <encoder repo or dir>]

Research use only; not for clinical use.
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

QWEN_ID, QWEN_REV = 'Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
ENCODER_REPO = 'zllhhhh/organic3d-encoder'
VLM_REPO = 'zllhhhh/organic3d-vlm'
DATASET = {'chest': 'ctrate', 'abdomen': 'merlin'}       # region -> the evaluation convention it follows
REPORT_HEAD = {'chest': 'Trachea', 'abdomen': 'Lower thorax:'}


def resolve(repo):
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download
    return snapshot_download(repo)


# ------------------------------------------------------------------ preprocessing
def load_ct(path, resample=True):
    """NIfTI (HU) -> (1 mm RAS float32 array, affine). resample=False: the file is already 1 mm RAS."""
    im = nib.as_closest_canonical(nib.load(path))
    a = np.ascontiguousarray(np.asarray(im.dataobj, dtype=np.float32))
    if not resample:
        return a, im.affine
    zoom = np.asarray(im.header.get_zooms()[:3], float)
    n_old = np.asarray(a.shape)
    size = tuple(int(x) for x in np.maximum(1, np.round(n_old * zoom)).astype(int))
    hu = F.interpolate(torch.from_numpy(a)[None, None], size=size, mode='trilinear', align_corners=False)[0, 0].numpy()
    hu = np.clip(np.rint(hu), -32768, 32767).astype(np.int16).astype(np.float32)
    sc = n_old / np.asarray(size)
    S = np.eye(4)
    S[:3, :3] = np.diag(sc)
    S[:3, 3] = 0.5 * sc - 0.5
    return hu, im.affine @ S


def totalsegmentator(hu, affine, device='gpu'):
    """TotalSegmentator v2 'total' (fast, multilabel) on the 1 mm volume -> uint8 mask on the same grid. Uses the python
    API when TotalSegmentator is installed in this environment, else the `TotalSegmentator` command ($TOTALSEG_BIN or
    PATH), e.g. from a separate environment."""
    import shutil
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, 'ct.nii.gz'), os.path.join(d, 'seg.nii.gz')
        nib.save(nib.Nifti1Image(hu.astype(np.int16), affine), src)
        try:
            from totalsegmentator.python_api import totalsegmentator as ts
            ts(src, dst, ml=True, fast=True, device=device, quiet=True)
        except ImportError:
            exe = os.environ.get('TOTALSEG_BIN') or shutil.which('TotalSegmentator')
            if not exe:
                raise RuntimeError('TotalSegmentator not found: install it, set TOTALSEG_BIN, or pass --seg')
            subprocess.run([exe, '-i', src, '-o', dst, '--ml', '--fast', '--device', device, '--quiet'], check=True)
        return np.asarray(nib.as_closest_canonical(nib.load(dst)).dataobj).astype(np.uint8)


# mean offset (mm) from the mask-free estimate to the TotalSegmentator centre (30 CT-RATE, 20 Merlin scans; on 50 other
# scans the corrected estimate is off by 2.4 / 2.8 / 3.5 mm (chest) and 10 / 11 / 11 mm (abdomen) per axis on average)
FALLBACK_OFFSET = {'chest': np.array([-3.6, 3.1, -4.7]), 'abdomen': np.array([32.8, 15.8, 16.4])}


def crop_centre(hu, region, seg=None):
    """Crop centre (1 mm voxel coordinates). seg: TotalSegmentator v2 multilabel mask on the 1 mm grid, or None.
    Without a mask (approximation): chest = centroid of the air inside the body (the lungs), abdomen = centroid of the
    body (largest region above -500 HU, holes filled per slice), each plus FALLBACK_OFFSET."""
    from organic.data.dataset import _LUT, crop_center
    if seg is not None:
        return crop_center(_LUT[seg], DATASET[region])
    from scipy import ndimage
    sub = hu[::4, ::4, ::4]
    lab, n = ndimage.label(sub > -500)
    if not n:
        return np.asarray(hu.shape) / 2
    body = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    body = np.stack([ndimage.binary_fill_holes(body[:, :, z]) for z in range(body.shape[2])], 2)
    m = body
    if region == 'chest':
        lab, n = ndimage.label((sub < -400) & body)
        if n:
            sz = np.bincount(lab.ravel())
            sz[0] = 0
            m = np.isin(lab, [k for k in np.argsort(sz)[::-1][:2] if sz[k] > 0.1 * sz.max()])
    return np.argwhere(m).mean(0) * 4 + FALLBACK_OFFSET[region]


def model_input(hu, centre, device):
    """1 mm HU volume + crop centre -> (1, 1, 192, 192, 192) input (as the evaluation harness)."""
    from organic.data.interventions import normalise
    from organic.data.dataset import SIZE, SPACING, _crop_bounds
    bounds = _crop_bounds(centre, hu.shape, int(SIZE * SPACING))
    t = torch.as_tensor(hu, device=device)
    out = torch.full([b - a for a, b in bounds], -1000.0, device=device)
    src = tuple(slice(max(a, 0), min(b, n)) for (a, b), n in zip(bounds, t.shape))
    dst = tuple(slice(s.start - a, s.stop - a) for s, (a, _) in zip(src, bounds))
    out[dst] = t[src]
    v = F.interpolate(out[None, None], size=(SIZE, SIZE, SIZE), mode='trilinear', align_corners=False)
    return normalise(v)


# ------------------------------------------------------------------ encoder
class Encoder:
    def __init__(self, weights=ENCODER_REPO, device='cuda', region=None):
        """region ('chest' / 'abdomen'): embed that region's finding prompts at load, before any image, as the evaluation
        harness does (the bf16 image tower's last digits depend on the GPU kernels chosen earlier in the process; in this
        order the scores equal the harness bit for bit)."""
        from colipri import get_model, get_processor
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from organic.encoder.model import Organic3D
        d = resolve(weights)
        self.cfg = json.load(open(os.path.join(d, 'config.json')))
        self.dev = torch.device(device)
        self.m = Organic3D(OmegaConf.create({'model': self.cfg['model']}), get_model(pretrained=False))
        self.m.load_state_dict(load_file(os.path.join(d, 'model.safetensors')), strict=True)
        self.m.to(self.dev).eval()
        self.tok = get_processor()._text_tokenizer
        self.max_len = int(self.cfg.get('max_text_len', 96))
        self.prompt_emb = {}
        if region:
            self.prompts(region)

    @torch.no_grad()
    def encode_image(self, x):
        """(1, 1, 192, 192, 192) input -> slot tokens T (1, K, 768)."""
        with torch.autocast('cuda', dtype=torch.bfloat16):
            T, _, _ = self.m.encode_image(x)
        return T.float()

    @torch.no_grad()
    def encode_text(self, sents):
        enc = self.tok(list(sents), max_length=self.max_len, padding='longest', truncation=True, return_tensors='pt')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            t = self.m.encode_text(enc['input_ids'].to(self.dev), enc['attention_mask'].to(self.dev))
            t = self.m.attach_route(t, list(sents))           # organ route of each sentence (fixed organ routing)
        return t.float()

    def prompts(self, region):
        """(findings, positive and negative prompt embeddings) of the region: "There is {f}." / "No {f}."."""
        if region not in self.prompt_emb:
            from organic.vocab import FAMILY, FINDING_NAMES, canonical_prompts
            fs = [f for f in FINDING_NAMES if f.startswith(FAMILY[DATASET[region]])]
            pos, neg = zip(*[canonical_prompts(f) for f in fs])
            self.prompt_emb[region] = (fs, self.encode_text(pos), self.encode_text(neg))
        return self.prompt_emb[region]

    @torch.no_grad()
    def score(self, T, t):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.m.score(T, t)[0].float().cpu().numpy()

    def similarity(self, T, sents):
        """S(V, s) for each sentence (numpy)."""
        return self.score(T, self.encode_text(sents))

    def finding_scores(self, T, region):
        """{finding: Delta_f} for the region's findings (chest: the 18 CT-RATE findings; abdomen: 21 Merlin findings)."""
        fs, tp, tn = self.prompts(region)
        return dict(zip(fs, (self.score(T, tp) - self.score(T, tn)).tolist()))


# ------------------------------------------------------------------ VLM
class VLM:
    def __init__(self, weights=VLM_REPO, encoder=None, device='cuda'):
        from huggingface_hub import snapshot_download
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from organic.vlm.model import Organic3DVLM
        d = resolve(weights)
        self.cfg = json.load(open(os.path.join(d, 'config.json')))
        self.enc = encoder if isinstance(encoder, Encoder) else Encoder(encoder or ENCODER_REPO, device)
        self.dev = self.enc.dev
        v = dict(self.cfg['vlm'])
        v['llm'] = snapshot_download(self.cfg['base_model'], revision=self.cfg['base_model_revision'])
        cfg = OmegaConf.create({'model': self.enc.cfg['model'], 'vlm': v})
        self.m = Organic3DVLM(cfg, self.enc.m)
        lora = {'llm.' + k.replace('.lora_A.', '.lora_A.default.').replace('.lora_B.', '.lora_B.default.'): x
                for k, x in load_file(os.path.join(d, 'adapter_model.safetensors')).items()}
        proj = {'proj.' + k: x for k, x in load_file(os.path.join(d, 'projector.safetensors')).items()}
        missing, unexpected = self.m.load_state_dict({**lora, **proj}, strict=False)
        assert not unexpected, unexpected[:5]
        expect = {k for k in self.m.state_dict() if '.lora_' in k or k.startswith('proj.')}
        assert not expect & set(missing), sorted(expect & set(missing))[:5]
        self.m.to(self.dev).eval()

    @torch.no_grad()
    def visual_tokens(self, x):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            V, _ = self.m.visual_tokens(x)
        return V

    @staticmethod
    def question(f_or_phrase):
        from organic.vlm.train import question
        from organic.vocab import FINDINGS
        if f_or_phrase in FINDINGS:
            return question(f_or_phrase)
        return f'Is there {f_or_phrase} in this CT scan? Answer Yes or No.'

    @torch.no_grad()
    def p_yes(self, V, questions):
        """P(Yes) against No at the first answer position, one value per question."""
        n = len(questions)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb, mask, _, _ = self.m.build(V.expand(n, -1, -1), list(questions), [''] * n, self.dev)
            h, _ = self.m.hidden(emb, mask)
        last = mask.sum(1) - 1
        w = self.m.head().weight[[self.m.yes_id, self.m.no_id]].float()
        two = h[torch.arange(n, device=self.dev), last].float() @ w.T
        return torch.sigmoid(two[:, 0] - two[:, 1]).cpu().numpy()

    def finding_answers(self, V, region):
        from organic.vocab import FAMILY, FINDING_NAMES
        fs = [f for f in FINDING_NAMES if f.startswith(FAMILY[DATASET[region]])]
        return dict(zip(fs, self.p_yes(V, [self.question(f) for f in fs]).tolist()))

    @torch.no_grad()
    def report(self, V, region, max_new_tokens=768, head=True):
        """Answer-first report: (finding list, report). The report starts with REPORT_HEAD[region] (head=True)."""
        from organic.vlm.train import finding_list, report_prompt
        ans = self.finding_answers(V, region)
        pre = finding_list([f for f, p in ans.items() if p >= 0.5])
        h = REPORT_HEAD[region] if head else ''
        if h:
            pre = f'{pre} {h}'
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = self.m.generate(V, report_prompt(DATASET[region], self.cfg['vlm']['report_mode']), self.dev,
                                  max_new_tokens=max_new_tokens, prefix=pre)
        body = out[0].strip()
        text = h + ('' if body[:1] in ',.;:' else ' ') + body if h else body
        return pre.split('\n')[0].strip(), text


# ------------------------------------------------------------------ CLI
def prepare(a, device):
    hu, aff = load_ct(a.ct, resample=not a.is_1mm)
    seg = None
    if a.seg:
        seg = np.asarray(nib.as_closest_canonical(nib.load(a.seg)).dataobj).astype(np.uint8)
        assert seg.shape == hu.shape, f'mask {seg.shape} is not on the 1 mm grid {hu.shape}'
    elif a.totalseg:
        seg = totalsegmentator(hu, aff)
    return model_input(hu, crop_centre(hu, a.region, seg), device), seg is not None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mode', choices=['encoder', 'vlm'])
    ap.add_argument('--ct', required=True, help='CT NIfTI in HU')
    ap.add_argument('--is_1mm', action='store_true', help='the CT is already 1 mm RAS (skip resampling)')
    ap.add_argument('--region', choices=['chest', 'abdomen'], required=True)
    ap.add_argument('--seg', default=None, help='TotalSegmentator v2 total multilabel mask on the 1 mm grid')
    ap.add_argument('--totalseg', action='store_true', help='run TotalSegmentator for the crop centre')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--encoder', default=ENCODER_REPO, help='vlm mode: the encoder repo or directory')
    ap.add_argument('--sentence', action='append', default=[], help='encoder: extra sentence to score')
    ap.add_argument('--question', action='append', default=[], help='vlm: extra finding phrase or yes/no question')
    ap.add_argument('--report', action='store_true', help='vlm: write a report')
    ap.add_argument('--max_new_tokens', type=int, default=768)
    ap.add_argument('--out', default=None, help='write the results as JSON')
    a = ap.parse_args()
    dev = 'cuda'
    x, masked = prepare(a, dev)
    res = {'ct': os.path.basename(a.ct), 'region': a.region,
           'crop_centre': 'TotalSegmentator' if masked else 'estimate without a mask'}
    if a.mode == 'encoder':
        enc = Encoder(a.weights or ENCODER_REPO, dev, region=a.region)
        T = enc.encode_image(x)
        res['finding_scores'] = enc.finding_scores(T, a.region)
        if a.sentence:
            res['sentence_scores'] = dict(zip(a.sentence, enc.similarity(T, a.sentence).tolist()))
    else:
        vlm = VLM(a.weights or VLM_REPO, a.encoder, dev)
        V = vlm.visual_tokens(x)
        res['p_yes'] = vlm.finding_answers(V, a.region)
        if a.question:
            qs = [q if q.endswith('?') or 'Answer Yes or No' in q else vlm.question(q) for q in a.question]
            res['questions'] = dict(zip(qs, vlm.p_yes(V, qs).tolist()))
        if a.report:
            res['finding_list'], res['report'] = vlm.report(V, a.region, a.max_new_tokens)
    s = json.dumps(res, indent=1)
    if a.out:
        open(a.out, 'w').write(s)
    print(s)


if __name__ == '__main__':
    sys.exit(main())
