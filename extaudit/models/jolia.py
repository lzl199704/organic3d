"""Jolia (Raidium 2026, HF raidium/Jolia, Research-Only RAIL-M; "Jolia: Concept-Level Vision-Language Alignment for 3D
CT Contrastive Learning", arXiv 2606.24570), run in $ORGANIC3D_BASELINES/jolia/venv. Chest + abdomen encoder.

Preprocessing = the bundled JoliaPreprocessor (preprocessing_jolia.py, run on GPU with its own transform classes):
volume (H, W, D) -> permute to (D, H, W), flip depth -> trilinear to 1.5 mm isotropic -> centre crop / pad (-1024 HU) to
192^3 -> the 11 CT windows of jolia_windowing ('all': lung, mediastinum, abdomen, liver, bone, brain, subdural, stroke,
temporal bone, soft tissue, per-volume minmax). Orientation (not stated by the model card): our 1 mm RAS array is
handed over as is (ORIENT 'ras'); of ras / lps / las / rps / transposed / no-depth-flip variants on 150 CT-RATE + 150
Merlin scans, ras (+ the preprocessor's depth flip) scored best on both (CT-RATE short .735 vs .70-.72 for lps / las /
rps; Merlin .70 vs .61-.67), consistent with RAS-canonicalised training volumes.
Text = Qwen/Qwen3-Embedding-8B, last-token pooling, max_length 512 (bundled JoliaTextEncoder;
right padding to the longest text instead of 512, identical last-token state), precomputed once into
$ORGANIC3D_BASELINES/jolia/text_cache.pt by a text-cache builder script (not distributed with this repository; missing
texts, including every template prompt, are encoded lazily, loading Qwen in bf16 on this GPU; the 'long' modes need the
prototype reports in that cache, see extaudit/README.md). The HF model repo raidium/Jolia is cloned to jolia/hf.
Zero-shot (paper App. F): short = 8 positive / negative template pairs (Table 15), mean cosine over the templates
(= cosine to the mean of the normalised template embeddings); long (--ckpt long / long_cls) = per finding the mean
embedding of r = 50 positive and 50 negative TRAIN-split reports. Image side ([z_CLS; c_k], the paper's full Jolia):
cosine of the concatenation = (cos(CLS, global text head) + cos(organ query k, organ text head)) / 2, k = the organ slot
of the finding (ORGAN below, our mapping; the paper's finding->concept table is not released). --ckpt cls / long_cls:
CLS only (= the paper's "Jolia - [CLS]" row). Delta_f = score(positive) - score(negative). Free sentences (laterality,
grounding) have no organ -> global head only. Probe feature = the normalised 576-d CLS embedding.
"""
import os
import sys

import torch
import torch.nn.functional as F

from extaudit.paths import baseline

HF = baseline('jolia', 'hf')
CACHE = baseline('jolia', 'text_cache.pt')
QWEN, QWEN_REV = 'Qwen/Qwen3-Embedding-8B', '1d8ad4ca9b3dd8059ad90a75d4983776a23d44af'
R_PROTO = 50
ORIENT = 'ras'          # our RAS array as is = (H, W, D); then the preprocessor's own depth flip
TAG = '\x00jolia:'

TEMPLATES = [                                         # Jolia paper Table 15
    ('{}', 'no {}'),
    ('there is evidence of {}', 'there is no evidence of {}'),
    ('{} present', '{} not present'),
    ('findings consistent with {}', 'no findings consistent with {}'),
    ('The CT scan shows {}', 'The CT scan does not show {}'),
    ('a CT showing {}', 'a CT without {}'),
    ('Impression: {}', 'Impression: no {}'),
    ('this is an image of a {}', 'this is an image with no {}'),
]

ORGAN = {
    'ct_medical_material': 'device', 'ct_arterial_calcification': 'aorta', 'ct_cardiomegaly': 'heart',
    'ct_pericardial_effusion': 'pericardium', 'ct_coronary_calcification': 'heart', 'ct_hiatal_hernia': 'esophagus',
    'ct_lymphadenopathy': 'lymph_nodes', 'ct_emphysema': 'lungs', 'ct_atelectasis': 'lungs', 'ct_lung_nodule': 'lungs',
    'ct_lung_opacity': 'lungs', 'ct_fibrotic_sequela': 'lungs', 'ct_pleural_effusion': 'pleura',
    'ct_mosaic_attenuation': 'lungs', 'ct_peribronchial_thickening': 'lungs', 'ct_consolidation': 'lungs',
    'ct_bronchiectasis': 'lungs', 'ct_septal_thickening': 'lungs',
    'mr_hydronephrosis': 'kidneys', 'mr_renal_calculus': 'kidneys', 'mr_gallstones': 'gallbladder',
    'mr_gb_distension': 'gallbladder', 'mr_biliary_dilatation': 'biliary', 'mr_hepatomegaly': 'liver',
    'mr_fatty_liver': 'liver', 'mr_hiatal_hernia': 'stomach', 'mr_pancreatic_atrophy': 'pancreas',
    'mr_pancreatic_duct': 'pancreas', 'mr_splenomegaly': 'spleen', 'mr_appendicitis': 'appendix',
    'mr_diverticulosis': 'colon', 'mr_bowel_obstruction': 'small_bowel', 'mr_aortic_aneurysm': 'aorta',
    'mr_aortic_stent': 'aorta', 'mr_prostatomegaly': 'prostate', 'mr_ascites': 'peritoneum',
    'mr_lymphadenopathy': 'lymph_nodes', 'mr_uterine_fibroid': 'uterus', 'mr_ovarian_cyst': 'ovaries',
}


def label_of(f, meta):
    """[label]: the CT-RATE label name; Merlin: the plain finding name (Merlin's raw names carry an organ prefix)."""
    return meta['label_name'][f] if f.startswith('ct_') else meta['phrase'][f]


@torch.no_grad()
def encode_pooled(te, texts):
    """JoliaTextEncoder.forward with padding='longest' instead of 'max_length' (512): the Qwen3 tokenizer pads on
    the right and attention is causal, so the last real token's state does not depend on the pad length."""
    dev = next(te.model.parameters()).device
    enc = te.tokenizer(texts, padding='longest', truncation=True, max_length=te.context_length,
                       return_tensors='pt').to(dev)
    assert te.tokenizer.padding_side == 'right'
    h = te.model(**enc).last_hidden_state
    return h[torch.arange(len(texts), device=dev), enc['attention_mask'].sum(1) - 1].float()


def orient(hu, mode):
    """(X, Y, Z) RAS -> the (H, W, D) array JoliaPreprocessor expects."""
    flips = [d for d, c in ((0, 'l'), (1, 'p')) if mode[d] == c]          # 'ras' as is, 'lps' flip 0 and 1, ...
    if flips:
        hu = torch.flip(hu, dims=flips)
    if '_t' in mode:
        hu = hu.transpose(0, 1)
    return hu


class Adapter:
    def __init__(self, ckpt, device, orient_mode=None):
        from transformers import AutoModel
        sys.path.insert(0, HF)
        from jolia_atlas_transform import ApplyWindowing, Crop3D, Pad3D, Resample3D
        self.dev, self.mode = device, ckpt or 'concat'
        assert self.mode in ('concat', 'cls', 'long', 'long_cls'), self.mode
        self.orient = orient_mode or ORIENT
        self.flip_depth = not self.orient.endswith('_nf')
        self.m = AutoModel.from_pretrained(HF, trust_remote_code=True).to(device).eval()
        self.resample, self.crop = Resample3D(target_spacing=(1.5, 1.5, 1.5)), Crop3D((192, 192, 192), training=False)
        self.pad, self.window = Pad3D((192, 192, 192), padding_value=-1024.0), ApplyWindowing('all', 'CT')
        c = torch.load(CACHE) if os.path.exists(CACHE) else {'feat': {}, 'proto': {}}
        self.cache, self.proto, self.te = c['feat'], c['proto'], None
        self.names = self.m.organ_slot_names
        self.meta = None

    def prompts(self, f, meta):
        self.meta = meta
        return f'{TAG}{f}:+', f'{TAG}{f}:-'

    def _pooled(self, texts):
        miss = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if miss:
            if self.te is None:
                from text_encoder_jolia import JoliaTextEncoder
                from huggingface_hub import snapshot_download
                qwen = snapshot_download(QWEN, revision=QWEN_REV)              # the exact snapshot used in the paper
                self.te = JoliaTextEncoder.from_pretrained(qwen, dtype=torch.bfloat16,
                                                           context_length=512).to(self.dev).eval()
            for i in range(0, len(miss), 16):
                e = encode_pooled(self.te, miss[i:i + 16]).cpu()
                self.cache.update(zip(miss[i:i + 16], e))
        return torch.stack([self.cache[t].float() for t in texts]).to(self.dev)

    def _embed(self, texts):
        """Mean of the L2-normalised global and organ-head projections over texts -> (576,), (576,)."""
        q = self._pooled(texts)
        return self.m.encode_text(q).mean(0), self.m.encode_organ_text(q).mean(0)

    def encode_text(self, sents):
        """-> (S, 576 + 576 + 1): global-head text, organ-head text, organ slot (-1 = none)."""
        rows = []
        for s in sents:
            if s.startswith(TAG):
                f, sign = s[len(TAG):].rsplit(':', 1)
                if self.mode.startswith('long'):
                    texts = self.proto[f]['pos' if sign == '+' else 'neg']
                else:
                    lab = label_of(f, self.meta)
                    texts = [(p if sign == '+' else n).format(lab) for p, n in TEMPLATES]
                k = self.names.index(ORGAN[f]) if not self.mode.endswith('cls') else -1
            else:
                texts, k = [s], -1
            g, o = self._embed(texts)
            rows.append(torch.cat([g, o if k >= 0 else torch.zeros_like(o), g.new_tensor([k])]))
        return torch.stack(rows)

    def prep(self, hu):
        """(X, Y, Z) 1 mm RAS HU (cuda) -> (1, 11, 192, 192, 192)."""
        v = orient(hu, self.orient).permute(2, 0, 1)                      # PrepareVolume(depth_last=True)
        if self.flip_depth:
            v = torch.flip(v, dims=[0])
        v = self.resample(v, current_spacing=(1.0, 1.0, 1.0))
        v = self.pad(self.crop(v))
        return self.window(v).float()[None]                              # (1, 11, 192, 192, 192)

    def encode_image(self, hu):
        x = self.prep(hu)
        cls, organ = self.m.forward_with_queries(x)
        z = F.normalize(cls.float(), dim=-1, eps=1e-6)[0]
        q = F.normalize(organ.float(), dim=-1, eps=1e-6)[0]               # (n_organs, 576)
        return torch.cat([z[None], q]), z

    def similarity(self, state, t):
        """state (1 + n_organs, 576), t from encode_text -> (S,) scores."""
        d = state.shape[1]
        g = t[:, :d] @ state[0]
        k = t[:, -1].long()
        has = k >= 0
        if not has.any():
            return g
        o = (t[:, d:2 * d] * state[1 + k.clamp(min=0)]).sum(-1)
        return torch.where(has, (g + o) / 2, g)

