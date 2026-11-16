"""Our stage-A encoders (and released COLIPRI, ckpt 'colipri') as an extaudit adapter, so the tests that only exist in
extaudit (lesion grounding) run our models through the same code as the released encoders. Run in the organic env.

--ckpt: a training checkpoint (<run dir>/last.pt with its config.yaml), the released weights (a Hugging Face repo id such
as zllhhhh/organic3d-encoder, or a downloaded folder with model.safetensors + config.json), or 'colipri'.
Input = exactly CTDataset's evaluation path: crop centred on lungs + heart (from the TotalSegmentator mask), 384 mm at
1 mm, trilinear to 192^3 at 2 mm, HU clipped to [-1000, 1000] / 1000. Scores: our sentence-conditioned score (Delta_f =
score(present sentence) - score(absent sentence), canonical prompts as in evaluate_stageA), COLIPRI: cosine similarity.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from organic.config import load_config
from organic.data import SIZE, SPACING, _LUT, _crop_bounds, crop_center
from organic.augment import normalise

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.environ.get('ORGANIC3D_ENV', os.path.join(HERE, '../../configs/env/local.yaml'))   # data roots (configs/env/TEMPLATE.yaml)


def released(ckpt):
    """True for released weights (repo id or folder with model.safetensors), False for a training last.pt."""
    return not (ckpt.endswith('.pt') or os.path.exists(os.path.join(ckpt, 'last.pt')))


def inference_module():
    import sys
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'scripts'))
    import organic3d
    return organic3d


class Adapter:
    def __init__(self, ckpt, device, dataset='ctrate'):
        from colipri import get_model, get_processor
        self.dev, self.dataset, self.native = device, dataset, ckpt == 'colipri'
        self.max_len = 96                       # our training sentence limit; the retrieval task's report mode: 512
        self.tok = get_processor()._text_tokenizer
        if self.native:
            self.m = get_model().to(device).eval()
        elif released(ckpt):                                # same construction as a training checkpoint, weights
            import json                                     # from model.safetensors (scores equal bit for bit)
            from omegaconf import OmegaConf
            from safetensors.torch import load_file
            from organic.model import build_model
            d = inference_module().resolve(ckpt)
            cfg = OmegaConf.create({'model': json.load(open(os.path.join(d, 'config.json')))['model']})
            cfg.model.pretrained = False
            self.m = build_model(cfg).to(device).eval()
            self.m.load_state_dict(load_file(os.path.join(d, 'model.safetensors')))
        else:
            from organic.model import build_model
            cfg = load_config(os.path.join(os.path.dirname(ckpt), 'config.yaml'), env=ENV)
            cfg.model.pretrained = False
            self.m = build_model(cfg).to(device).eval()
            self.m.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model'])

    def set_dataset(self, dataset):
        self.dataset = 'merlin' if dataset == 'merlin' else 'ctrate'      # crop centre (RAD-ChestCT = chest)

    @staticmethod
    def prompts(f, meta):
        p = meta['phrase'][f]
        return f'There is {p}.', f'No {p}.'                   # = vocab.canonical_prompts (our evaluation prompts)

    def encode_text(self, sents):
        enc = self.tok(sents, max_length=self.max_len, padding='longest', truncation=True, return_tensors='pt')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            t = self.m.encode_text(enc['input_ids'].to(self.dev), enc['attention_mask'].to(self.dev))
            if not self.native:                                     # released COLIPRI has no route columns
                t = self.m.attach_route(t, sents)                   # no-op unless model.pool = fixed
        return t.float()

    def set_scan(self, seg):
        """Crop centre from this scan's mask (once per scan, before its conditions)."""
        organ1 = _LUT[seg.cpu().numpy()]
        self.bounds = _crop_bounds(crop_center(organ1, self.dataset), seg.shape, int(SIZE * SPACING))

    def prep(self, hu):
        out = torch.full([b - a for a, b in self.bounds], -1000.0, device=hu.device)
        src = tuple(slice(max(a, 0), min(b, n)) for (a, b), n in zip(self.bounds, hu.shape))
        dst = tuple(slice(s.start - a, s.stop - a) for s, (a, _) in zip(src, self.bounds))
        out[dst] = hu[src]
        v = F.interpolate(out[None, None], size=(SIZE, SIZE, SIZE), mode='trilinear', align_corners=False)
        return normalise(v)

    def encode_image(self, hu):
        x = self.prep(hu)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if self.native:
                e = self.m.encode_image(x, pool=True, project=True, normalize=True).float()[0]
                return e, e
            T, _, _ = self.m.encode_image(x)
        return T.float(), T.float().mean(1)[0]

    def similarity(self, state, t):
        """state from encode_image, t (S, d) text -> (S,) scores."""
        if self.native:
            return t @ state
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.m.score(state, t)[0].float()
