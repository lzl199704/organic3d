"""Merlin (Blankemeier et al., 2024; github.com/StanfordMIMI/Merlin, pip merlin-vlm 2.x, MIT; weights
stanfordmimi/Merlin, i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt), the abdominal CT CLIP
model trained on the Merlin training split (our Merlin test scans are held out from it). Run in $ORGANIC3D_BASELINES/merlin/venv.

Preprocessing = merlin/data/monai_transforms.py ImageTransforms on our 1 mm RAS volume (the same RAS array order MONAI's
Orientationd produces). Our Merlin 1 mm volumes are cropped to the body (<MERLIN_ROOT>/1mm/transform/<scan>.json; the
environment variable MERLIN_ROOT points to the Merlin dataset root with these transform files and the original
raw/img/<scan>.nii.gz; if it is unset or the files are missing, the body-cropped 1 mm volume is used as is), so for Merlin scans
the original field of view is restored first (padded with -1000 HU = the official zero pad; the CT table is the only
thing lost) so the centre crop sees what the official pipeline sees; then resample to 1.5 x 1.5 x 3 mm (trilinear),
HU [-1000, 1000] -> [0, 1] clipped, symmetric zero pad / centre crop to 224 x 224 x 160. Input (1, 1, X, Y, Z); the I3D-ResNet152 treats Z as time.
Text = Clinical-Longformer CLS -> linear 512 (TextEncoder lower-cases and word-tokenises; max 1,024 tokens).
Zero-shot rule (the paper's): cosine of the l2-normalised 512-d image (contrastive head) and text embeddings, present
vs absent prompt. The paper's radiologist-written prompts for its 30 findings are not released with the package, so
prompts follow its README example ('<x> is present.' / '<x> is not present.') with our phrases.
Image latent = the l2-normalised 512-d contrastive embedding (also the probe feature).
"""
import json
import os

import nibabel as nib
import torch
import torch.nn.functional as F

SPACING = (1.5, 1.5, 3.0)
SHAPE = (224, 224, 160)
MERLIN = os.environ.get('MERLIN_ROOT')          # optional: see the module docstring


def fov_pad(volume):
    """Merlin scan -> (before, after) 1 mm padding per axis that undoes the body crop, None if not a Merlin scan."""
    if not MERLIN:
        return None
    stem = volume.replace('.nii.gz', '')
    tf, raw = f'{MERLIN}/1mm/transform/{stem}.json', f'{MERLIN}/raw/img/{stem}.nii.gz'
    if not (os.path.exists(tf) and os.path.exists(raw)):
        return None
    t = json.load(open(tf))
    sl = next((x['slices'] for x in t if x['type'] == 'slice'), None)
    if sl is None:                                   # 36 Merlin test scans were only resampled, never cropped
        return None
    im = nib.load(raw)
    full = [round(n * z) for n, z in zip(im.shape, im.header.get_zooms()[:3])]
    return [(a, max(f - b, 0)) for (a, b, _), f in zip(sl, full)]


def prep(hu, pad=None):
    """(X, Y, Z) 1 mm RAS HU (cuda) -> (1, 1, 224, 224, 160) in [0, 1]."""
    if pad:
        hu = F.pad(hu, [p for d in reversed(pad) for p in d], value=-1000.0)
    size = [max(1, round(n / s)) for n, s in zip(hu.shape, SPACING)]
    a = F.interpolate(hu[None, None].float(), size=size, mode='trilinear', align_corners=False)[0, 0]
    a = ((a + 1000.0) / 2000.0).clamp(0.0, 1.0)
    pads = [max(t - n, 0) for t, n in zip(SHAPE, a.shape)]
    a = F.pad(a, [p for d in reversed(pads) for p in (d // 2, d - d // 2)], value=0.0)
    st = [(n - t) // 2 for t, n in zip(SHAPE, a.shape)]
    a = a[st[0]:st[0] + SHAPE[0], st[1]:st[1] + SHAPE[1], st[2]:st[2] + SHAPE[2]]
    return a[None, None]


class Adapter:
    def __init__(self, ckpt, device):
        from merlin import Merlin
        self.dev = device
        self.m = Merlin().to(device).eval()
        self.pad = None

    def set_volume(self, volume):
        self.pad = fov_pad(volume)

    @staticmethod
    def prompts(f, meta):
        p = meta['phrase'][f]
        return f'{p} is present.', f'{p} is not present.'

    def encode_text(self, sents):
        t = self.m.model.encode_text(list(sents))
        return F.normalize(t.float(), dim=-1)

    def encode_image(self, hu):
        z, _ = self.m.model.encode_image(prep(hu, self.pad))
        z = F.normalize(z.float(), dim=-1)[0]
        return z, z
