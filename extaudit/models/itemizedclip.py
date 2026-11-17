"""ItemizedCLIP (Lyu et al., 2026; github.com/MLNeurosurg/ItemizedCLIP, MIT), CT-RATE checkpoint from the
README's Google Drive link (CTRATE_HLIP_BiomedBERT_itemizedclip: HLIP ViT-B singlescan_h2_token2744 + CXR-BERT +
item-conditioned cross-attention pooling), run in $ORGANIC3D_BASELINES/hlip/venv (shared with hlip.py; repo at hlip/itemizedclip_repo, checkpoint at
hlip/weights/itemizedclip_ctrate.pt).

Preprocessing = identical to HLIP (Radiology_Tasks/src/open_ct_rate/zeroshot_ct_rate.py and data_chest_ct.py:
HLIP-processed CT-RATE .pt at 3 x 1 x 1 mm, input_info -1150 350 crop), implemented once in hlip.prep: our 1 mm RAS
volume -> CT-RATE file order -> (z, y, x), 3 x 1 x 1 mm trilinear, HU [-1150, 350] -> [0, 1], centre pad (0) /
crop to 112 x 336 x 336, (x - 0.449) / 0.226. fp16 autocast ('amp') as the official script.
Zero-shot rule (zeroshot_ct_rate.py, get_text / zero_shot): ONE prompt per finding, 'findings consistent with <x>'
(lowercased CT-RATE name; lung nodule -> 'nodules or nodular density', lung opacity -> 'opacity', bronchiectasis ->
'bronchiectasis or bronchiectatic changes'); score = cosine between the l2-normalised text embedding t and the
l2-normalised output of the cross-attention block (query t, keys / values = the l2-normalised 2744 local image tokens).
There is no negative prompt: the adapter's negative is a null sentence encoded as the zero vector, whose score is
exactly 0, so the harness's Delta = score(pos) - score(neg) equals the official score. Laterality sentences use the same
item-conditioned score. Medical material / lymphadenopathy (not in the paper's 16) use the same template.
Image latent for the probe = the l2-normalised global (CLS-pooled, projected) image embedding.
"""
import os
import sys

import torch
import torch.nn.functional as F

from extaudit.models.hlip import BASE, load_text_tower, load_state, prep
from extaudit.paths import default_hf_home

os.environ.setdefault('HF_HOME', default_hf_home())
CKPT = {'itemizedclip': f'{BASE}/weights/itemizedclip_ctrate.pt'}
MODEL = 'CTRATE_HLIP_BiomedBERT_itemizedclip'
NULL = ''                                       # negative "prompt": encoded as the zero vector -> score 0
RENAME = {'lung nodule': 'nodules or nodular density', 'lung opacity': 'opacity',
          'bronchiectasis': 'bronchiectasis or bronchiectatic changes'}


class Adapter:
    def __init__(self, ckpt, device):
        src = f'{BASE}/itemizedclip_repo/Radiology_Tasks/src'
        sys.path.insert(0, src)
        import json
        from open_clip.factory import _MODEL_CONFIGS
        from open_ct_rate import visual_encoder                              # noqa: F401  (registers the timm ViT)
        from open_mri.factory import create_model, get_tokenizer
        cfg = json.load(open(f'{src}/open_mri/model_configs/{MODEL}.json'))
        cfg['vision_cfg']['timm_model_pretrained'] = False                 # MAE init is overwritten by the ckpt
        _MODEL_CONFIGS[MODEL] = cfg
        self.dev, self.dim = device, cfg['embed_dim']
        self.m = create_model(MODEL, device=device, precision='amp', output_dict=True, pretrained_hf=False,
                              my_custom_model=True)
        load_text_tower(self.m, device)
        load_state(self.m, CKPT.get(ckpt or 'itemizedclip', ckpt))
        self.m = self.m.to(device).eval()
        self.tok = get_tokenizer(MODEL, trust_remote_code=True)

    @staticmethod
    def prompts(f, meta):
        d = meta['label_name'][f].lower()
        return f'findings consistent with {RENAME.get(d, d)}', NULL

    def encode_text(self, sents):
        sents = list(sents)
        out = torch.zeros(len(sents), self.dim, device=self.dev)
        keep = [i for i, s in enumerate(sents) if s != NULL]
        if keep:
            with torch.autocast('cuda'):
                t = self.m.encode_text(self.tok([sents[i] for i in keep]).to(self.dev), normalize=False)
            out[keep] = F.normalize(t.float(), dim=-1)
        return out

    def encode_image(self, hu):
        with torch.autocast('cuda'):
            g, loc = self.m.encode_image(prep(hu))
        return F.normalize(loc.float(), dim=-1)[0], F.normalize(g.float(), dim=-1)[0]

    def similarity(self, state, t):
        """state (L, 512) normalised local tokens, t (S, 512) normalised text -> (S,) item-conditioned cosine."""
        with torch.autocast('cuda'):
            a = self.m.visual_attn(t[None], state[None], state[None])
        return (F.normalize(a.float(), dim=-1)[0] * t).sum(-1)
