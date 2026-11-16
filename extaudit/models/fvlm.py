"""fVLM (Shui et al., ICLR 2025) and ViSD-Boost (Cao et al., ICCV 2025), Alibaba DAMO organ-level CLIP models trained on
CT-RATE; run in $ORGANIC3D_BASELINES/fvlm/venv. ckpt 'fvlm' (default; model.pth from the README's Google Drive folder)
or 'visd_boost' (checkpoint_vqvae.pth then checkpoint.pth, loaded in that order as in ViSD-Boost eval.py).
Repos: github.com/alibaba-damo-academy/fvlm, /ViSD-Boost (no LICENSE file in either repo).

Preprocessing = their data/resize.py + data/preprocess.py + eval.py, with the TotalSegmentator v2 mask from set_scan in
place of their own TotalSegmentator run (their class_map in resize.py IS the v2 'total' id list, so the merge is the
same ids: lung = 10-14 (5 lobes) -> 1, heart = 51 heart + 61 atrial_appendage_left -> 2, esophagus = 15 -> 3,
aorta = 52 -> 4). Our 1 mm RAS volume -> CT-RATE file order (flip axes 0, 1; = the array MONAI LoadImage reads),
resampled to 1 x 1 x 3 mm (trilinear for HU, nearest for the mask; MONAI Resized, align_corners False), transposed to
(z, y, x), HU window [-1150, 350] -> [0, 1], cropped to the bounding box of the 4 merged organs + 5 slices / 20 px,
padded (symmetric, 0) to at least 112 x 256 x 352. Zero-shot (eval.py): one window per present organ, centred on that
organ, size max(roi, organ extent) with roi = 112 x 288 x 352 (fVLM) / 112 x 256 x 352 (ViSD), window mask = that organ
only, padded at the end to a multiple of 16 x 16 x 32 (fVLM) / 16 x 32 x 32 (ViSD); organ feature = the model's organ
query attention-pooled over the tokens the organ touches -> organ projection -> l2-normalised.
Prompts and organ per finding = eval.py test_items ('<X>.' vs 'Not <X>.', e.g. 'Atelectatic.' / 'Not Atelectatic.';
ViSD says 'Lung nodule.' / 'Lung opacity.' where fVLM says 'Nodule.' / 'Opacity.'): 11 lung findings -> lung;
cardiomegaly, pericardial effusion, coronary calcification -> heart; hiatal hernia -> esophagus; arterial wall
calcification -> aorta. Medical material and lymphadenopathy have no organ in their protocol (16 labels) -> NaN.
Score = organ logit difference (pos - neg) / temp, monotone in their softmax P(pos). Organ absent from the mask: fVLM's
calc_metrics sets P = 0 (here delta = -1e4), ViSD's skips the scan (NaN). Free sentences (laterality / grounding) are
scored against the organ named by keyword (heart / esophagus / aorta, else lung).
Probe feature = the 4 organ embeddings concatenated (1024-d, zeros for an absent organ).

Abdominal data (set_dataset('merlin'): Merlin, AbdomenAtlas 3.0) = FORCED mode (2026-10-25), so that these
chest-only models get a score on every table cell, as CT-CLIP does on Merlin. Their protocol has no organ for any
abdominal finding, so: (i) findings whose name carries a chest-organ keyword use that organ's feature as before
(e.g. hiatal hernia -> esophagus, aortic aneurysm / stent -> aorta); (ii) every other finding or free sentence is
scored against a WHOLE-SCAN feature: one window by their own rule (eval.py center_crop, size max(roi, extent)) on the
bounding box of all TotalSegmentator structures, window mask = all of them; each of the 4 organ queries is
attention-pooled over the tokens that mask touches (their forward_test_win pooling) and the 4 l2-normalised features
are averaged and re-normalised. Prompts for these findings follow their style ('<Phrase>.' / 'Not <Phrase>.', phrase =
meta['phrase']). The crop box becomes the bounding box of all structures (not only the 4 organs), which also covers
abdomen-only scans with no lung or heart. Chest datasets (CT-RATE, RAD-ChestCT, ReXGroundingCT) are unchanged.
Orientation checked on the first 150 full 1 mm CT-RATE scans (fVLM, macro AUROC over 16): flip(0, 1) 0.748, flip0
0.738, flip1 0.741, flip(0, 1, 2) 0.739, none (RAS) 0.722. Organs whose windows coincide share one backbone pass and
token flags are precomputed per scan (features bit-identical to their forward_test_win, checked on full scans).
"""
import contextlib
import io
import os
import re
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from extaudit.paths import baseline

BASE = baseline()                     # repos / weights under fvlm/ and visd_boost/ (extaudit/README.md)
FVLM_REPO, VISD_REPO = f'{BASE}/fvlm/repo', f'{BASE}/visd_boost/repo'
CKPT = {'fvlm': [f'{BASE}/fvlm/weights/checkpoints/model.pth'],
        'visd_boost': [f'{VISD_REPO}/checkpoint_vqvae.pth', f'{VISD_REPO}/checkpoint.pth']}
# their relative 'BiomedVLP-CXR-BERT-specialized' dir: HF snapshot files + its config.json with the BLIP med keys the
# checkpoints imply (no token-type embeddings, no cross-attention layers: add_type_embeddings / add_cross_attention False)
TEXT_DIR = f'{BASE}/fvlm/textenc'
ORGANS = ['lung', 'heart', 'esophagus', 'aorta']
WHOLE = len(ORGANS)                                  # forced mode: state row of the whole-scan feature
TS_IDS = {1: [10, 11, 12, 13, 14], 2: [51, 61], 3: [15], 4: [52]}     # merged id -> TotalSegmentator v2 ids
ROI = {'fvlm': (112, 288, 352), 'visd_boost': (112, 256, 352)}
DIV = {'fvlm': (16, 16, 32), 'visd_boost': (16, 32, 32)}
MIN_SIZE = (112, 256, 352)

# eval.py test_items: (organ, CT-RATE label, negative prompt, positive prompt)
_ITEMS = [
    ['lung', 'Emphysema', 'Not Emphysema.', 'Emphysema.'],
    ['lung', 'Atelectasis', 'Not Atelectatic.', 'Atelectatic.'],
    ['lung', 'Lung nodule', 'Not Nodule.', 'Nodule.'],
    ['lung', 'Lung opacity', 'Not Opacity.', 'Opacity.'],
    ['lung', 'Pulmonary fibrotic sequela', 'Not Pulmonary fibrotic.', 'Pulmonary fibrotic.'],
    ['lung', 'Pleural effusion', 'Not Pleural effusion.', 'Pleural effusion.'],
    ['lung', 'Mosaic attenuation pattern', 'Not Mosaic attenuation pattern.', 'Mosaic attenuation pattern.'],
    ['lung', 'Peribronchial thickening', 'Not Peribronchial thickening.', 'Peribronchial thickening.'],
    ['lung', 'Consolidation', 'Not Consolidation.', 'Consolidation.'],
    ['lung', 'Bronchiectasis', 'Not Bronchiectasis.', 'Bronchiectasis.'],
    ['lung', 'Interlobular septal thickening', 'Not Interlobular septal thickening.', 'Interlobular septal thickening.'],
    ['heart', 'Cardiomegaly', 'Not Cardiomegaly.', 'Cardiomegaly.'],
    ['heart', 'Pericardial effusion', 'Not Pericardial effusion.', 'Pericardial effusion.'],
    ['heart', 'Coronary artery wall calcification', 'Not Coronary artery wall calcification.',
     'Coronary artery wall calcification.'],
    ['esophagus', 'Hiatal hernia', 'Not Hiatal hernia.', 'Hiatal hernia.'],
    ['aorta', 'Arterial wall calcification', 'Not Arterial wall calcification.', 'Arterial wall calcification.'],
]
_VISD_PROMPTS = {'Lung nodule': ('Not Lung nodule.', 'Lung nodule.'), 'Lung opacity': ('Not Lung opacity.', 'Lung opacity.')}


def _stub(pkgs):
    """Register repo sub-packages without running their __init__ (fVLM's lavis/__init__ pulls the whole LAVIS stack)."""
    for name, path in pkgs:
        m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m


@contextlib.contextmanager
def _cwd(d):
    old = os.getcwd(); os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


def _load_med():
    """Their XBertEncoder (lavis med.py; ViSD's copy is fVLM's minus two lavis imports)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('_dam_med', f'{VISD_REPO}/lavis/utils/med.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.XBertEncoder


class _FVLM(nn.Module):
    """The inference part of fVLM's BlipPretrain (blip_pretrain.py) with the same parameter names."""
    def __init__(self):
        super().__init__()
        _stub([('lavis', f'{FVLM_REPO}/lavis'), ('lavis.models', f'{FVLM_REPO}/lavis/models'),
               ('lavis.models.blip_models', f'{FVLM_REPO}/lavis/models/blip_models')])
        from lavis.models.blip_models.vit import ViT
        from transformers import BertTokenizer
        self.visual_encoder = ViT(in_channels=1, img_size=(112, 256, 352), patch_size=(16, 16, 32), num_classes=0,
                                  dropout_rate=0.1, qkv_bias=True)
        with _cwd(TEXT_DIR):
            self.tokenizer = BertTokenizer.from_pretrained('BiomedVLP-CXR-BERT-specialized')
            self.text_encoder = _load_med().from_config(None, from_pretrained=True)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.text_proj = nn.Linear(768, 256)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, dropout=0.1, batch_first=True)
        self.vision_projs = nn.ModuleList([nn.Linear(768, 256) for _ in ORGANS])
        self.query_tokens = nn.Parameter(torch.zeros(len(ORGANS), 768))
        self.max_txt_len = 384                      # pretrain_ct.yaml max_txt_len

    @staticmethod
    def token_index(wmask, organ_id):
        """forward_test_win's token flags (organ mask max-pooled to the 16 x 16 x 32 patch grid) as indices."""
        t = F.max_pool3d(wmask[0, 0][None].eq(organ_id + 1).float(), kernel_size=(16, 16, 32),
                         stride=(16, 16, 32))[0].flatten() > 0
        return torch.nonzero(t)[:, 0]

    def organ_feat(self, window, tok, organ_id, cache):
        """= forward_test_win for the window's own (skip_organ) organ (backbone shared by identical windows)."""
        if id(window) not in cache:
            cache[id(window)] = (window, self.visual_encoder(window)[1])
        hidden = cache[id(window)][1]
        key = torch.cat([lv[0][tok] for lv in hidden]).unsqueeze(0)
        q, _ = self.attention(self.query_tokens[organ_id][None, None], key, key)
        return F.normalize(self.vision_projs[organ_id](q.squeeze(0)), dim=-1)[0]


class _VISD:
    """ViSD-Boost's own VISDBOOST (lavis/utils/model.py) and its forward_test_win."""
    def __init__(self):
        _stub([('lavis', f'{VISD_REPO}/lavis'), ('lavis.utils', f'{VISD_REPO}/lavis/utils')])
        from lavis.utils.vit import ViT
        from lavis.utils.model import VISDBOOST
        image_encoder = ViT(in_channels=1, img_size=(112, 256, 352), patch_size=(16, 16, 32), num_classes=0,
                            dropout_rate=0.1, qkv_bias=True)
        with _cwd(TEXT_DIR):
            self.m = VISDBOOST(image_encoder=image_encoder, text_encoder=_load_med().from_config(None, from_pretrained=True))

    KERNELS = [(2, 4, 4), (4, 8, 8), (8, 16, 16), (16, 32, 32)]          # ViT.forward organ_token_flags1..4

    @classmethod
    def token_index(cls, wmask, organ_id):
        m = wmask[0, 0].eq(organ_id + 1).float()[None, None]
        return [torch.nonzero(F.max_pool3d(m, kernel_size=k, stride=k).flatten(1)[0] > 0)[:, 0] for k in cls.KERNELS]

    @torch.inference_mode()
    def organ_feat(self, window, tok, organ_id, cache):
        """forward_test_win for the window's own (skip_organ) organ, line by line, minus its GPU->CPU syncs (token
        flags precomputed as indices in set_scan; verified identical to organ_feat_official)."""
        m, v = self.m, self.m.visual_encoder
        if id(window) not in cache:
            r = v.res_model(window)
            cache[id(window)] = (window, [p(x).flatten(2).transpose(1, 2)[0]
                                          for p, x in zip((v.proj1, v.proj2, v.proj3, v.proj4), r)])
        e = cache[id(window)][1]
        key1, key2, key3, key4 = [ek[t].unsqueeze(0) for ek, t in zip(e, tok)]
        roi_embeds, roi_embeds_other = key4, torch.cat([key1, key2, key3], dim=1)
        roi_embeds_clone = roi_embeds.clone()
        oid = torch.tensor(organ_id, device=window.device)
        id_embeds = m.id_projs(oid.unsqueeze(0).unsqueeze(0).float()).repeat(roi_embeds.shape[0], 1, 1)
        roi_embeds_min = roi_embeds_clone.min()
        roi_embeds_clone = roi_embeds_clone - roi_embeds_min
        roi_embeds_max = roi_embeds_clone.max()
        roi_embeds_clone = (roi_embeds_clone / roi_embeds_max) - 0.5
        roi_embeds_input = torch.cat([id_embeds, roi_embeds_clone], dim=1)
        z = m.encoder(roi_embeds_input)[:, 1:, :]
        x_recon = m.decoder(m.vq_layer(z))
        recon_feat = (x_recon + 0.5) * roi_embeds_max + roi_embeds_min
        hybrid_feat = torch.cat([torch.cat([roi_embeds_other, roi_embeds], dim=1),
                                 torch.cat([roi_embeds_other, recon_feat], dim=1)], dim=-1)
        hybrid_feat = m.res_projs2[organ_id](m.res_projs1[organ_id](hybrid_feat))
        return F.normalize(m.vision_projs[organ_id](m.get_query_token(hybrid_feat, organ_id)), dim=-1)[0]

    def organ_feat_official(self, window, wmask, organ_id, present, sizes):
        """Their forward_test_win itself (slow under GPU sharing: ~100 host syncs per organ); used to verify."""
        rec = {}
        with contextlib.redirect_stdout(io.StringIO()):          # its 'Incomplete ...' prints
            self.m.forward_test_win(window, wmask, {}, present, {}, rec, sizes, skip_organ=organ_id)
        return torch.tensor(rec[ORGANS[organ_id]][0], device=window.device)


class Adapter:
    def __init__(self, ckpt, device):
        self.kind = ckpt or 'fvlm'
        assert self.kind in CKPT, f'ckpt must be one of {list(CKPT)}'
        self.dev = device
        if self.kind == 'fvlm':
            self.net = _FVLM(); mod = self.net
        else:
            self.net = _VISD(); mod = self.net.m
        for p in CKPT[self.kind]:
            sd = torch.load(p, map_location='cpu', weights_only=False)['model']
            r = mod.load_state_dict(sd, strict=False)
            print(f'[{self.kind}] {os.path.basename(p)}: missing {len(r.missing_keys)} {r.missing_keys[:6]}, '
                  f'unexpected {len(r.unexpected_keys)} {r.unexpected_keys[:6]}', flush=True)
        self.mod = mod.to(device).eval()
        self.tok = mod.tokenizer
        self.temp = float(mod.temp)
        self.items = {}
        for organ, lab, neg, pos in _ITEMS:
            if self.kind == 'visd_boost' and lab in _VISD_PROMPTS:
                neg, pos = _VISD_PROMPTS[lab]
            self.items[lab] = (ORGANS.index(organ), neg, pos)
        self.role = {}                                 # sentence -> (organ id, +1 positive / -1 negative prompt)
        self.forced = False

    def set_dataset(self, ds):
        """Abdominal data -> forced mode (module docstring); chest data -> the authors' protocol."""
        self.forced = ds == 'merlin'

    # ---------------------------------------------------------------- text
    def prompts(self, f, meta):
        lab = meta['label_name'][f]
        if lab in self.items:
            o, neg, pos = self.items[lab]
        elif self.forced:                              # abdominal finding: keyword organ, else the whole-scan feature
            ph = meta['phrase'][f]
            ph = ph[0].upper() + ph[1:]
            o, neg, pos = self._organ_forced(lab), f'Not {ph}.', f'{ph}.'
        else:                                          # medical material, lymphadenopathy: not in their protocol
            o, neg, pos = -1, f'Not {lab}.', f'{lab}.'
        self.role[pos], self.role[neg] = (o, 1.0), (o, -1.0)
        return pos, neg

    @staticmethod
    def _organ_of(s):
        s = s.lower()
        if re.search(r'heart|cardi|pericard|coronary', s):
            return 1
        if re.search(r'esophag|oesophag|hiatal', s):
            return 2
        if re.search(r'aort', s):
            return 3
        return 0

    @classmethod
    def _organ_forced(cls, s):
        """Forced mode: chest-organ keyword -> that organ (lung only if named), otherwise the whole-scan feature."""
        o = cls._organ_of(s)
        if o == 0 and not re.search(r'lung|pulmon|pleura|bronch|lobe', s.lower()):
            return WHOLE
        return o

    def encode_text(self, sents):
        """-> (S, 258): l2-normalised 256-d text feature, organ id, prompt role."""
        out = []
        for i in range(0, len(sents), 32):
            t = self.tok(sents[i:i + 32], padding='max_length', truncation=True, max_length=self.mod.max_txt_len,
                         return_tensors='pt').to(self.dev)
            e = self.mod.text_encoder.forward_text(t).last_hidden_state[:, 0, :]
            out.append(F.normalize(self.mod.text_proj(e), dim=-1))
        org = self._organ_forced if self.forced else self._organ_of
        meta = torch.tensor([self.role.get(s, (org(s), 0.0)) for s in sents], device=self.dev)
        return torch.cat([torch.cat(out).float(), meta.float()], 1)

    # ---------------------------------------------------------------- image
    @staticmethod
    def _to_model_grid(v, mode):
        """(X, Y, Z) 1 mm RAS -> CT-RATE file order, 1 x 1 x 3 mm, (z, y, x)."""
        a = torch.flip(v, dims=[0, 1]).float()
        size = (a.shape[0], a.shape[1], int(a.shape[2] * (1.0 / 3.0)))    # resize.py: int(d * spacing / 3)
        a = F.interpolate(a[None, None], size=size, mode=mode, **({} if mode == 'nearest' else
                                                                   {'align_corners': False}))[0, 0]
        return a.permute(2, 1, 0)

    def set_scan(self, seg):
        merged = torch.zeros_like(seg, dtype=torch.float32)
        for k, ids in TS_IDS.items():
            merged[torch.isin(seg, torch.as_tensor(ids, device=seg.device, dtype=seg.dtype))] = k
        m = self._to_model_grid(merged, 'nearest')
        if self.forced:                                # crop box over all structures; whole-scan window below
            body = self._to_model_grid((seg > 0).float(), 'nearest')
            m = torch.where(m > 0, m, body * (WHOLE + 1))
        nz = torch.nonzero(m)
        lo = (nz.min(0).values - torch.tensor([5, 20, 20], device=m.device)).clamp(min=0)
        hi = torch.minimum(nz.max(0).values + torch.tensor([5, 20, 20], device=m.device),
                           torch.tensor(m.shape, device=m.device))
        self.box = [slice(int(a), int(b)) for a, b in zip(lo, hi)]
        m = m[self.box]
        self.pad = [((max(t - n, 0)) // 2, max(t - n, 0) - (max(t - n, 0)) // 2) for n, t in zip(m.shape, MIN_SIZE)]
        m = self._pad(m, self.pad)
        sizes = {o: int((m == i + 1).sum()) for i, o in enumerate(ORGANS)}       # forced: WHOLE+1 voxels not counted
        self.present = [o for o in ORGANS if sizes[o] > 0]
        self.sizes = sizes
        self.windows = []                # (organ id, crop slices, end padding, window mask, token indices)
        roi, div = ROI[self.kind], DIV[self.kind]
        for o in self.present:
            i = ORGANS.index(o)
            om = m == i + 1
            sl = self._center_crop(om, roi)
            wm = om[sl].float() * (i + 1)
            endpad = [(0, (-n) % k) for n, k in zip(wm.shape, div)]
            wm = self._pad(wm, endpad)[None, None]
            self.windows.append((i, sl, endpad, wm, self.net.token_index(wm, i)))
        self.whole = None
        if self.forced:
            bm = m > 0                                 # every structure (the 4 organs and all others)
            sl = self._center_crop(bm, roi)
            wm = bm[sl].float()
            endpad = [(0, (-n) % k) for n, k in zip(wm.shape, div)]
            wm = self._pad(wm, endpad)[None, None]
            self.whole = (sl, endpad, self.net.token_index(wm, 0))       # tokens where the mask == 1

    @staticmethod
    def _pad(t, pw, value=0.0):
        return F.pad(t, (pw[2][0], pw[2][1], pw[1][0], pw[1][1], pw[0][0], pw[0][1]), value=value)

    @staticmethod
    def _center_crop(mask, crop_size):
        """eval.py center_crop (bounding box from masks_to_boxes_3d: max - min, no +1)."""
        nz = torch.nonzero(mask)
        (z0, y0, x0), (z1, y1, x1) = nz.min(0).values.tolist(), nz.max(0).values.tolist()
        crop = (max(crop_size[0], z1 - z0), max(crop_size[1], y1 - y0), max(crop_size[2], x1 - x0))
        out = []
        for c, lo, hi, n in zip(crop, (z0, y0, x0), (z1, y1, x1), mask.shape):
            ctr = (lo + hi) // 2
            s = max(0, ctr - c // 2); e = min(n, s + c)
            if e - s < c:
                s = max(0, e - c)
            out.append(slice(s, e))
        return tuple(out)

    def encode_image(self, hu):
        x = self._to_model_grid(hu, 'trilinear')
        x = ((x.clamp(-1150, 350) + 1150) / 1500.0)[self.box]
        x = self._pad(x, self.pad)
        feats = torch.full((len(ORGANS) + (1 if self.forced else 0), 256), float('nan'), device=self.dev)
        wins, cache = {}, {}
        for i, sl, endpad, wm, tok in self.windows:
            k = (tuple((s.start, s.stop) for s in sl), tuple(endpad))
            if k not in wins:
                wins[k] = self._pad(x[sl], endpad)[None, None]
            feats[i] = self.net.organ_feat(wins[k], tok, i, cache)
        if self.whole is not None:
            sl, endpad, tok = self.whole
            k = (tuple((s.start, s.stop) for s in sl), tuple(endpad))
            if k not in wins:
                wins[k] = self._pad(x[sl], endpad)[None, None]
            w = torch.stack([self.net.organ_feat(wins[k], tok, i, cache) for i in range(len(ORGANS))])
            feats[WHOLE] = F.normalize(w.mean(0), dim=-1)
        return feats, feats.nan_to_num(0.0).flatten()

    def similarity(self, state, t):
        """state (4, 256) organ features (NaN = organ absent), t (S, 258) -> (S,) logits / temp."""
        f, org, role = t[:, :256], t[:, 256].long(), t[:, 257]
        s = (state[org.clamp(min=0)] * f).sum(-1) / self.temp
        s = torch.where(org >= 0, s, torch.full_like(s, float('nan')))
        if self.kind == 'fvlm':                        # calc_metrics: missing organ -> P(pos) = 0
            gone = (org >= 0) & torch.isnan(s)
            s = torch.where(gone & (role > 0), torch.full_like(s, -1e4), s)
            s = torch.where(gone & (role < 0), torch.zeros_like(s), s)
        return s
