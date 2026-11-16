"""HLIP (Zhao et al., TMLR 2026; github.com/Zch0414/hlip, tag v1.0, MIT), CT-RATE ViT-Base
(clip_vit_base_singlescan_h2_token2744, patch 8 x 24 x 24, slice + scan attention; checkpoint from the v1.0 README's
Google Drive link), run in $ORGANIC3D_BASELINES/hlip/venv (repo at hlip/hlip_repo, checkpoint at
hlip/weights/hlip_ctrate_vitb.pt). The preprocessing helper `prep` is shared with
itemizedclip.py (ItemizedCLIP is trained on the same HLIP CT-RATE pipeline and backbone).

Preprocessing = HLIP data/ct_rate/process.py + src/hlip_test/zeroshot_ct_rate.py (input_info -1150 350 crop):
our 1 mm RAS volume -> CT-RATE file order (flip axes 0 and 1, as ctclip.py) -> SimpleITK array order (z, y, x),
trilinear resample to 3 x 1 x 1 mm (process.py --spacing 3 1 1; size = int(n * spacing / target)), HU window
[-1150, 350] -> [0, 1] and clip, centre pad (value 0) / centre crop to 112 x 336 x 336, normalise with the mean of
the ImageNet mean / std (0.449, 0.226). Input layout (B, n_scans=1, C=1, D, H, W); fp16 autocast ('amp') as the
official script.
Zero-shot rule (zeroshot_ct_rate.py, --zeroshot-template volume): per finding a 2-class classifier
['The volume shows: Not <x>.', 'The volume shows: <X>.'] on the l2-normalised image / text embeddings, P = softmax
of logit_scale * cos; Delta = cos(pos) - cos(neg) is monotone in it. Prompts from hlip/zeroshot_metadata_ct_rate.py
PROMPTS (with the script's override for lung nodule / lung opacity under the 'volume' template); the two findings
HLIP does not evaluate (medical material, lymphadenopathy) follow the same 'Not <x>' / '<X>' pattern.
Image latent = the model's l2-normalised 512-d projection (also the probe feature).
"""
import os
import sys

import torch
import torch.nn.functional as F

from extaudit.paths import baseline, default_hf_home

os.environ.setdefault('HF_HOME', default_hf_home())
BASE = baseline('hlip')
CKPT = {'hlip': f'{BASE}/weights/hlip_ctrate_vitb.pt'}
MODEL = 'clip_vit_base_singlescan_h2_token2744'
CXR_BERT = 'microsoft/BiomedVLP-CXR-BERT-specialized'
MEAN, STD = 0.449, 0.226                       # mean of timm IMAGENET_DEFAULT_MEAN / _STD (torchvision Normalize)
HU_LO, HU_HI = -1150.0, 350.0
SHAPE = (112, 336, 336)                        # (d, h, w) at 3 x 1 x 1 mm

# HLIP's PROMPTS (negative, positive), keyed by our finding ids; zeroshot_ct_rate.py replaces the lung nodule /
# lung opacity entries with these when the template is not 'organ'.
PROMPTS = {
    'ct_emphysema': ('Not emphysema', 'Emphysema'),
    'ct_atelectasis': ('Not atelectatic', 'Atelectasis'),
    'ct_lung_nodule': ('Not lung nodule', 'Lung nodule'),
    'ct_lung_opacity': ('Not lung opacity', 'Lung opacity'),
    'ct_fibrotic_sequela': ('Not pulmonary fibrotic sequela', 'Pulmonary fibrotic sequela'),
    'ct_pleural_effusion': ('Not pleural effusion', 'Pleural effusion'),
    'ct_mosaic_attenuation': ('Not mosaic attenuation pattern', 'Mosaic attenuation pattern'),
    'ct_peribronchial_thickening': ('Not peribronchial thickening', 'Peribronchial thickening'),
    'ct_consolidation': ('Not consolidation', 'Consolidation'),
    'ct_bronchiectasis': ('Not bronchiectasis', 'Bronchiectasis'),
    'ct_septal_thickening': ('Not interlobular septal thickening', 'Interlobular septal thickening'),
    'ct_cardiomegaly': ('Not cardiomegaly', 'Cardiomegaly'),
    'ct_pericardial_effusion': ('Not pericardial effusion', 'Pericardial effusion'),
    'ct_coronary_calcification': ('Not coronary artery wall calcification', 'Coronary artery wall calcification'),
    'ct_hiatal_hernia': ('Not hiatal hernia', 'Hiatal hernia'),
    'ct_arterial_calcification': ('Not arterial wall calcification', 'Arterial wall calcification'),
    'ct_medical_material': ('Not medical material', 'Medical material'),          # not in HLIP's 16: same pattern
    'ct_lymphadenopathy': ('Not lymphadenopathy', 'Lymphadenopathy'),             # not in HLIP's 16: same pattern
}

# Orientation of the model input relative to our RAS array; 'lps' = CT-RATE file order (chosen, see report).
ORIENT = 'lps'


def prep(hu, orient=None):
    """(X, Y, Z) 1 mm RAS HU (cuda) -> (1, 1, 1, 112, 336, 336) normalised HLIP input."""
    orient = orient or ORIENT
    flips = {'lps': [0, 1], 'ras': [], 'lpi': [0, 1, 2], 'lps_lr': [1]}[orient]
    a = torch.flip(hu, dims=flips) if flips else hu
    a = a.permute(2, 1, 0)                                                    # sitk order (z, y, x)
    a = F.interpolate(a[None, None].float(), size=(int(a.shape[0] / 3.0), a.shape[1], a.shape[2]),
                      mode='trilinear')[0]                                    # 3 x 1 x 1 mm, (1, d, h, w)
    a = ((a - HU_LO) / (HU_HI - HU_LO)).clamp(0.0, 1.0)
    _, d, h, w = a.shape
    pd, ph, pw = (max(t - n, 0) for t, n in zip(SHAPE, (d, h, w)))
    a = F.pad(a[None], (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pd // 2, pd - pd // 2), value=0.0)[0]
    _, d, h, w = a.shape
    sd, sh, sw = (d - SHAPE[0]) // 2, (h - SHAPE[1]) // 2, (w - SHAPE[2]) // 2
    a = a[:, sd:sd + SHAPE[0], sh:sh + SHAPE[1], sw:sw + SHAPE[2]]
    return ((a - MEAN) / STD)[None, None]


def load_text_tower(model, device):
    """Official --use-cxr-bert: replace the text transformer by CXR-BERT's BertModel (weights then come from ckpt)."""
    from transformers import AutoModel
    model.text.transformer = AutoModel.from_pretrained(CXR_BERT, trust_remote_code=True).bert.to(device)


def load_state(model, path):
    sd = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
    model.load_state_dict({k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()})


class Adapter:
    def __init__(self, ckpt, device):
        src = f'{BASE}/hlip_repo/src'
        sys.path.insert(0, src)
        import json
        from open_clip import create_model, get_tokenizer
        from open_clip.factory import _MODEL_CONFIGS
        from hlip import visual_encoder                                      # noqa: F401  (registers the timm ViT)
        cfg = json.load(open(f'{src}/hlip/model_configs/{MODEL}.json'))
        cfg['vision_cfg']['timm_model_pretrained'] = False                 # MAE init is overwritten by the ckpt
        _MODEL_CONFIGS[MODEL] = cfg
        self.dev = device
        self.m = create_model(MODEL, device=device, precision='amp', output_dict=True, pretrained_hf=False)
        load_text_tower(self.m, device)
        load_state(self.m, CKPT.get(ckpt or 'hlip', ckpt))
        self.m = self.m.to(device).eval()
        self.tok = get_tokenizer(MODEL, trust_remote_code=True)

    @staticmethod
    def prompts(f, meta):
        if f in PROMPTS:
            neg, pos = PROMPTS[f]
        else:                               # Merlin findings (2026-10-24): HLIP has no abdominal prompts; same pattern
            p = meta['phrase'][f]
            neg, pos = f'Not {p}', p[:1].upper() + p[1:]
        return f'The volume shows: {pos}.', f'The volume shows: {neg}.'      # TEMPLATES['volume']

    def encode_text(self, sents):
        with torch.autocast('cuda'):
            t = self.m.encode_text(self.tok(list(sents)).to(self.dev), normalize=True)
        return t.float()

    def encode_image(self, hu):
        with torch.autocast('cuda'):
            z = self.m.encode_image(prep(hu), normalize=True).float()[0]
        return z, z
