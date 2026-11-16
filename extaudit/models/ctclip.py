"""CT-CLIP and CT-VocabFine (Hamamci et al.; weights *_v2.pt from the CT-RATE repo), run in
$ORGANIC3D_BASELINES/CT-CLIP/ct_clip_venv (code: github.com/ibrahimethemhamamci/CT-CLIP cloned to
$ORGANIC3D_BASELINES/CT-CLIP, which provides transformer_maskgit/ and CT_CLIP/; CT-CLIP_v2.pt and CT_VocabFine_v2.pt
from HF ibrahimhamamci/CT-RATE models/ placed in the same folder).

Preprocessing = CT-CLIP's data_inference / CT-CHAT's encode_script (route checked against CT-CLIP's own predictions,
pooled Spearman 0.80): our 1 mm RAS volume -> LPS (CT-RATE file order), HU clipped to [-1000, 1000] / 1000, resampled to
0.75 x 0.75 x 1.5 mm, centre crop / pad (-1) to 480 x 480 x 240. Prompts: "<label> is present." / "<label> is not
present." with the CT-RATE label names (CT-CLIP zero-shot protocol; VocabFine uses the same inference).
Image latent = the model's l2-normalised projection; the probe feature is the same 512-d latent.
"""
import sys

import torch
import torch.nn.functional as F

from extaudit.paths import baseline

sys.path.insert(0, baseline('CT-CLIP', 'transformer_maskgit'))
sys.path.insert(0, baseline('CT-CLIP', 'CT_CLIP'))
CKPT = {'ctclip': baseline('CT-CLIP', 'CT-CLIP_v2.pt'), 'vocabfine': baseline('CT-CLIP', 'CT_VocabFine_v2.pt')}
TEXT = 'microsoft/BiomedVLP-CXR-BERT-specialized'


class Adapter:
    def __init__(self, ckpt, device):
        from transformer_maskgit import CTViT
        from ct_clip import CTCLIP
        from transformers import BertModel, BertTokenizer
        self.dev = device
        self.tok = BertTokenizer.from_pretrained(TEXT, do_lower_case=True)
        text_encoder = BertModel.from_pretrained(TEXT)
        text_encoder.resize_token_embeddings(len(self.tok))
        image_encoder = CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20, temporal_patch_size=10,
                              spatial_depth=4, temporal_depth=4, dim_head=32, heads=8)
        self.m = CTCLIP(image_encoder=image_encoder, text_encoder=text_encoder, dim_image=294912, dim_text=768,
                        dim_latent=512, extra_latent_projection=False, use_mlm=False, downsample_image_embeds=False,
                        use_all_token_embeds=False)
        self.m.load(CKPT.get(ckpt or 'ctclip', ckpt))
        self.m = self.m.to(device).eval()

    @staticmethod
    def prompts(f, meta):
        lab = meta['label_name'][f]
        return f'{lab} is present.', f'{lab} is not present.'

    def encode_text(self, sents):
        t = self.tok(sents, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(self.dev)
        e = self.m.text_transformer(t.input_ids, attention_mask=t.attention_mask)[0][:, 0, :]
        return F.normalize(self.m.to_text_latent(e), dim=-1)

    def prep(self, hu):
        """(X, Y, Z) 1 mm RAS HU (cuda) -> (1, 1, 240, 480, 480)."""
        a = torch.flip(hu, dims=[0, 1]).clamp(-1000, 1000).permute(2, 0, 1)      # LPS, (z, x, y)
        new = [int(a.shape[0] / 1.5), int(a.shape[1] / 0.75), int(a.shape[2] / 0.75)]
        t = F.interpolate(a[None, None], size=new, mode='trilinear', align_corners=False)[0, 0] / 1000.0
        t = t.permute(1, 2, 0)                                                   # (x, y, z)
        tgt = (480, 480, 240)
        sl = [slice(max((n - d) // 2, 0), max((n - d) // 2, 0) + min(n, d)) for n, d in zip(t.shape, tgt)]
        t = t[sl[0], sl[1], sl[2]]
        pad = []
        for n, d in zip(t.shape, tgt):
            b = (d - n) // 2; pad.append((b, d - n - b))
        t = F.pad(t, (pad[2][0], pad[2][1], pad[1][0], pad[1][1], pad[0][0], pad[0][1]), value=-1)
        return t.permute(2, 0, 1)[None, None]                                   # (1, 1, z, x, y)

    def encode_image(self, hu):
        x = self.prep(hu)
        enc = self.m.visual_transformer(x, return_encoded_tokens=True)
        enc = torch.mean(enc, dim=1).view(enc.shape[0], -1)
        z = F.normalize(self.m.to_visual_latent(enc), dim=-1)[0]
        return z, z
