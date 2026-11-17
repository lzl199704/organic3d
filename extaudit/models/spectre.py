"""SPECTRE-Large (Claessens et al., arXiv 2511.17209; code github.com/cclaess/SPECTRE @c05af47, MIT; weights
HF cclaess/SPECTRE-Large + cclaess/SPECTRE components, CC-BY-NC-SA 4.0), run in $ORGANIC3D_BASELINES/spectre/venv (the released
components downloaded to spectre/components).

Image = the VLA-trained ViT-L backbone (128 x 128 x 64 crops) + feature combiner (preset 'spectre-large'), then the
released SigLIP image projection head on [combiner CLS ; mean combiner tokens] (ssl/frameworks/vision_language.py).
Preprocessing, RAS (= our array as is), HU clipped to [-1000, 1000] -> [0, 1], pad value 0 (= -1000 HU):
  'vla'  (default) = the VLA pretraining geometry (ssl/transforms/siglip_transform.py): 0.75 x 0.75 x 1.5 mm, centre
         crop / pad to 384 x 384 x 256 (the random crop at eval = centre), 3 x 3 x 4 crops;
  'eval' = the authors' downstream linear-probe pipeline (extract_feat_LP.py in their experiments folder): 0.5 x 0.5 x
         1.0 mm, centre crop / pad to 512 x 512 x 256, 4 x 4 x 4 crops.
Text = Qwen3-Embedding-0.6B + the released LoRA (r 16, alpha 64, q/k/v/o; loralib, merged at eval) + SigLIP text
projection head, last-token pooling, Qwen tokenizer with special tokens (pretrain_siglip.py / utils/collate.py).
Scores: cosine of the L2-normalised projections (SigLIP's learnt t, b are monotone and not released). The paper has no
zero-shot classification protocol (only retrieval), so prompts = harness default "There is X." / "No X."; ckpt suffix
'+tmpl' instead averages the 8 template pairs of the Jolia paper (Table 15), the protocol behind its third-party
SPECTRE numbers (CT-RATE 57.0, Merlin 64.2). Probe feature = the 512-d normalised image projection.
Precision: fp32 as in the authors' inference scripts; ckpt suffix '+amp' = bf16 autocast on the image tower (4x faster;
on 8 CT-RATE scans Delta_f vs fp32: r = 0.99995, max |diff| 0.007 vs between-scan SD 0.15). ckpt examples: vla,
eval, vla+amp, vla+tmpl+amp.
"""
import torch
import torch.nn.functional as F

from extaudit.paths import baseline

BASE = baseline('spectre')
COMP = f'{BASE}/components'
QWEN, QWEN_REV = 'Qwen/Qwen3-Embedding-0.6B', '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'   # tokenizer only (HF snapshot used)
GEOM = {'vla': ((0.75, 0.75, 1.5), (384, 384, 256)), 'eval': ((0.5, 0.5, 1.0), (512, 512, 256))}
JOLIA_TEMPLATES = [
    ('{}', 'no {}'), ('there is evidence of {}', 'there is no evidence of {}'), ('{} present', '{} not present'),
    ('findings consistent with {}', 'no findings consistent with {}'),
    ('The CT scan shows {}', 'The CT scan does not show {}'), ('a CT showing {}', 'a CT without {}'),
    ('Impression: {}', 'Impression: no {}'), ('this is an image of a {}', 'this is an image with no {}'),
]
TAG = '\x00spectre:'
QWEN_CFG = dict(attention_bias=False, attention_dropout=0.0, bos_token_id=151643, eos_token_id=151643, head_dim=128,
                hidden_act='silu', hidden_size=1024, initializer_range=0.02, intermediate_size=3072,
                max_position_embeddings=32768, max_window_layers=28, model_type='qwen3', num_attention_heads=16,
                num_hidden_layers=28, num_key_value_heads=8, rms_norm_eps=1e-06, rope_scaling=None,
                rope_theta=1000000, sliding_window=None, tie_word_embeddings=True, use_cache=False,
                use_sliding_window=False, vocab_size=151669)          # = pretrain_siglip.py


def center_crop_pad(x, size, value):
    """MONAI ResizeWithPadOrCrop on (H, W, D): centre crop (start = n//2 - r//2), then symmetric pad."""
    sl = []
    for n, r in zip(x.shape, size):
        s = max(n // 2 - r // 2, 0) if n > r else 0
        sl.append(slice(s, s + min(n, r)))
    x = x[sl[0], sl[1], sl[2]]
    pad = []
    for n, r in reversed(list(zip(x.shape, size))):
        d = max(r - n, 0); pad += [d // 2, d - d // 2]
    return F.pad(x, pad, value=value)


class Adapter:
    def __init__(self, ckpt, device):
        from transformers import Qwen2TokenizerFast, Qwen3Config, Qwen3Model
        from spectre import SpectreImageFeatureExtractor
        from spectre.presets import BACKBONE_KWARGS, COMBINER_KWARGS
        from spectre.ssl.heads import SigLIPProjectionHead
        from spectre.utils import add_lora_adapters, last_token_pool
        spec = (ckpt or 'vla').split('+')
        self.geom, self.tmpl, self.amp = spec[0], 'tmpl' in spec[1:], 'amp' in spec[1:]
        self.dev, self.pool = device, last_token_pool
        self.m = SpectreImageFeatureExtractor(
            backbone_name='vit_large_patch16_128', backbone_kwargs=dict(BACKBONE_KWARGS),
            backbone_checkpoint_path_or_url=f'{COMP}/spectre_backbone_vit_large_patch16_128.pt',
            feature_combiner_name='feat_vit_large', feature_combiner_kwargs=dict(COMBINER_KWARGS),
            feature_combiner_checkpoint_path_or_url=f'{COMP}/spectre_combiner_feature_vit_large.pt').to(device).eval()
        self.pi = SigLIPProjectionHead(input_dim=2160, output_dim=512)
        self.pt = SigLIPProjectionHead(input_dim=1024, output_dim=512)
        for h, f in ((self.pi, 'image'), (self.pt, 'text')):
            h.load_state_dict(torch.load(f'{COMP}/SigLIP_projection_head_{f}.pt', map_location='cpu'))
            h.to(device).eval()
        self.tok = Qwen2TokenizerFast.from_pretrained(QWEN, revision=QWEN_REV)     # tokenizer only
        self.txt = Qwen3Model(Qwen3Config(**QWEN_CFG))
        add_lora_adapters(self.txt, r=16, lora_alpha=64, lora_dropout=0.05,
                          target_keywords=('q_proj', 'k_proj', 'v_proj', 'o_proj'))
        sd = torch.load(f'{COMP}/spectre_qwen3_embedding_0.6B_lora.pt', map_location='cpu')
        msg = self.txt.load_state_dict(sd, strict=False)
        assert not msg.unexpected_keys and all('rotary' in k for k in msg.missing_keys), msg
        self.txt = self.txt.to(device).eval()                             # loralib merges A, B into W on eval()

    def prompts(self, f, meta):
        if self.tmpl:
            lab = meta['label_name'][f] if f.startswith('ct_') else meta['phrase'][f]
            return TAG + '\x01'.join(p.format(lab) for p, _ in JOLIA_TEMPLATES), \
                TAG + '\x01'.join(n.format(lab) for _, n in JOLIA_TEMPLATES)
        p = meta['phrase'][f]
        return f'There is {p}.', f'No {p}.'

    def _enc(self, sents):
        t = self.tok(sents, add_special_tokens=True, padding=True, truncation=True, max_length=1024,
                     return_tensors='pt').to(self.dev)
        h = self.txt(input_ids=t.input_ids, attention_mask=t.attention_mask).last_hidden_state
        return F.normalize(self.pt(self.pool(h, t.attention_mask)).float(), dim=-1)

    def encode_text(self, sents):
        out = []
        for s in sents:
            if s.startswith(TAG):                                        # mean cosine over templates
                out.append(self._enc(s[len(TAG):].split('\x01')).mean(0))
            else:
                out.append(self._enc([s])[0])
        return torch.stack(out)

    def prep(self, hu):
        """(X, Y, Z) 1 mm RAS HU (cuda) -> crops (N, 1, 128, 128, 64), grid."""
        sp, size = GEOM[self.geom]
        x = ((hu.clamp(-1000, 1000) + 1000) / 2000)
        new = [max(int(round(n / s)), 1) for n, s in zip(x.shape, sp)]
        x = F.interpolate(x[None, None], size=new, mode='trilinear', align_corners=False)[0, 0]
        x = center_crop_pad(x, size, 0.0)
        g = (size[0] // 128, size[1] // 128, size[2] // 64)
        crops = x.view(g[0], 128, g[1], 128, g[2], 64).permute(0, 2, 4, 1, 3, 5).reshape(-1, 1, 128, 128, 64)
        return crops, g                                                  # depth fastest (= spectre grid_patch)

    def encode_image(self, hu):
        crops, g = self.prep(hu)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            f = self.m.extract_backbone_features(crops[None])            # (1, N, T, 1024)
            c = self.m.combine_features(f, g)                            # (1, T', 1080)
            e = torch.cat([c[:, 0], c[:, 1:].mean(1)], dim=-1)
            z = F.normalize(self.pi(e).float(), dim=-1)[0]
        return z, z
