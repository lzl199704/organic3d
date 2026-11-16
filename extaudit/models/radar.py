"""RADAR+ (Alibaba DAMO, "RADAR: An Expert-Level Generalist AI for Abdominal CT Diagnosis", Science 2026;
github.com/alibaba-damo-academy/damo-radar, code LICENSE file Apache-2.0, README and weights CC BY-NC-SA 4.0; weights
huggingface.co/radar-generalist), the abdominal organ-level CLIP model. Run in $ORGANIC3D_BASELINES/radar/venv.
Local layout: repo cloned to $ORGANIC3D_BASELINES/radar/damo-radar; an empty working dir $ORGANIC3D_BASELINES/radar/work
(the authors' code resolves '../ckpt' from it); $ORGANIC3D_BASELINES/radar/ckpt holds the RADAR weights from HF
(bert-base-uncased/, checkpoint_radar_plus.pth, checkpoint_radar_plus_finetuned_on_merlin.pth, checkpoint_unet.pth) and
the repo's ckpt/infer_text_embedding_*.pt. ckpt 'plus' = checkpoint_radar_plus.pth (RADAR+ trained from scratch on the Merlin training split
only = the same data as ours; the fairest comparison) or 'plus_ft' = checkpoint_radar_plus_finetuned_on_merlin.pth (the
published RADAR+: RADAR pretrained on 400k private scans, fine-tuned on Merlin train). Both: English bert-base-uncased,
HU window [-1000, 1000]. (The Chinese-BERT 'pretrain' variant is not wired here.) Suffix ':nofov' (e.g. 'plus:nofov')
skips the field-of-view restoration below; it exists only to check this adapter against the earlier audit
(an earlier internal audit script, not distributed, which fed the body-cropped volume).

Preprocessing = the authors' RADAR_train/infer_merlin_anatomy.py (organ findings) and infer_merlin_whole.py (whole-scan
findings) on raw Merlin NIfTIs. Our Merlin 1 mm volumes are body-cropped, so the original FOV is restored first
(merlin.fov_pad, padded with -1000 HU; the CT table is the only thing lost; under the mirror condition the left/right
padding is swapped so the input is the mirrored full-FOV scan). Then our RAS array is flipped to LAS (= the array MONAI
LoadImaged reads from the raw NIfTI) with spacing (1, 1, 1), resized (trilinear, align_corners False) to 1 x 1 x 5 mm
and transposed to (z, y, x).
  Organ branch: HU clipped to [-1000, 1000], min-max to [0, 1], cropped to the non-zero box + 5 slices / 20 px, end-padded
  with 0 to the ROI 96 x 256 x 384; sliding windows (ROI, overlap 0.25, monai dense_patch_slices). RADAR's own U-Net
  (checkpoint_unet.pth, 37 classes) segments every window; each organ's feature comes from the FIRST window in which it
  is intact (not within 2 voxels of the window border): organ query attention-pooled over the tokens the organ touches
  -> organ projection -> l2-norm (= forward_test_win). Organs intact in no window: one more window centred on the
  organ in the stitched segmentation (authors' center_crop, padded to a multiple of 32). Organ absent from RADAR's own
  segmentation (e.g. after it was erased) -> no feature -> NaN delta. Our TotalSegmentator masks are NOT used: the
  authors' gating is part of what the erasure tests probe.
  Whole branch (RADAR+ whole-image/whole-report contrastive head): the resized HU volume end-padded with 0 (HU, as
  their SpatialPadd runs before clipping) to the ROI, centre-cropped to it, clipped, min-max; one backbone pass ->
  whole query attention over all tokens -> vision_projs_whole -> l2-norm.

Prompts and scoring rule (the authors'): P(present) = softmax over (mean logit to the negative prompts, mean logit to
the positive prompts), logit = cosine / temp. Here delta = (mean cos pos - mean cos neg) / temp (monotone in P; the
ensemble mean is taken in embedding space, which gives the same mean cosine). Per finding:
  11 organ findings with the authors' Merlin prompt ensembles (infer_merlin_anatomy.disease_prompts; 'normal' appended
     to the negatives as their evaluate() does), scored on the organ named by their 'region' field: hydronephrosis
     (kidney), gallstones (gallbladder), biliary dilatation / hepatomegaly / hepatic steatosis (liver), hiatal hernia
     (esophagus), pancreatic atrophy (pancreas), splenomegaly (spleen), appendicitis (large bowel), bowel obstruction
     (small bowel), aortic aneurysm (aorta). = the 11-finding set of the earlier audit.
   3 whole-scan findings with the authors' prompts (infer_merlin_whole.disease_prompts, no 'normal'), whole feature:
     lymphadenopathy, prostatomegaly, ascites. (Not in the earlier audit.)
   7 findings without an authors' prompt: harness default 'There is X.' / 'No X.' (meta phrase), on RADAR's organ for
     the 5 with one (renal calculus, kidney; gallbladder distension; pancreatic duct dilatation, pancreas; colonic
     diverticulosis, large bowel; aortic stent, aorta) and on the whole feature for uterine fibroid and ovarian cyst.
Missing organ: the authors' calc_metrics sets P = 0; here the delta is NaN (= the earlier audit; core/metrics drop NaN
scores per condition), so how often RADAR lost the organ is reported separately (feature flags below / NaN rate). The
authors' rule can be recovered offline (NaN of an organ finding -> -inf).
Free sentences (laterality pairs, retrieval sentences) go to the organ their first organ keyword names (as RADAR's
organ-wise report sentences), else / if that organ was not found to the whole feature; whole report sections (retrieval
'report' mode, core sets max_len = 512) always to the whole feature (the head trained on whole reports). Text: their
tokenizer + text_encoder.forward_text CLS -> text_proj -> l2-norm, max 100 tokens (their prompt length; 512 for reports),
padding to the longest sentence instead of max_length (same CLS output under the attention mask).
Image latent (probe feature) = 12 organ features (aorta, large bowel, duodenum, esophagus, gallbladder, kidney, liver,
pancreas, small bowel, spleen, stomach, bladder; zeros if not found) + whole feature + 12 organ-found flags (3,340-d).

Deviations: the fallback window records only the organ it is centred on (the authors' loop also takes any other
still-missing organ from it, which makes features depend on the order findings are listed; = the earlier audit); input
is our 1 mm resampling of the raw scan (then RADAR's own resize), not the raw grid; no CT table.
"""
import contextlib
import os
import re
import sys
import time
import types

# GPUs are shared: limit allocator fragmentation (read at the first CUDA allocation, which comes after this import)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from extaudit.models.merlin import fov_pad  # noqa: E402
from extaudit.paths import baseline  # noqa: E402

REPO = baseline('radar', 'damo-radar')
TRAIN = f'{REPO}/RADAR_train'
WORK = baseline('radar', 'work')                          # '../ckpt' (= radar/ckpt, the RADAR weights from HF) from here
CKPT = {'plus': 'checkpoint_radar_plus.pth', 'plus_ft': 'checkpoint_radar_plus_finetuned_on_merlin.pth'}
ROI, OVERLAP = (96, 256, 384), 0.25
HU = (-1000.0, 1000.0)
ORGANS = ['aorta', 'large bowel', 'duodenum', 'esophagus', 'gallbladder', 'kidney', 'liver', 'pancreas', 'small bowel',
          'spleen', 'stomach', 'bladder']                 # organ features kept (= ORGANS_CN of the earlier audit)
WHOLE = len(ORGANS)                                       # state row of the whole-scan feature
NATIVE_ORGAN = {'mr_hydronephrosis': 'hydronephrosis', 'mr_gallstones': 'gallstones',
                'mr_biliary_dilatation': 'biliary_ductal_dilation', 'mr_hepatomegaly': 'hepatomegaly',
                'mr_fatty_liver': 'hepatic_steatosis', 'mr_hiatal_hernia': 'hiatal_hernia',
                'mr_pancreatic_atrophy': 'pancreatic_atrophy', 'mr_splenomegaly': 'splenomegaly',
                'mr_appendicitis': 'appendicitis', 'mr_bowel_obstruction': 'bowel_obstruction',
                'mr_aortic_aneurysm': 'aortic_aneurysm'}
NATIVE_WHOLE = {'mr_lymphadenopathy': 'lymphadenopathy', 'mr_prostatomegaly': 'prostatomegaly', 'mr_ascites': 'ascites'}
DEFAULT_ORGAN = {'mr_renal_calculus': 'kidney', 'mr_gb_distension': 'gallbladder', 'mr_pancreatic_duct': 'pancreas',
                 'mr_diverticulosis': 'large bowel', 'mr_aortic_stent': 'aorta'}
KEYWORDS = [(r'gall ?bladder|cholecyst|gallstone|cholelith', 'gallbladder'),
            (r'kidney|renal|ureter|hydronephro|nephr', 'kidney'), (r'liver|hepat|biliary|bile duct', 'liver'),
            (r'pancrea', 'pancreas'), (r'splen', 'spleen'), (r'aort', 'aorta'), (r'esophag|hiatal', 'esophagus'),
            (r'stomach|gastric', 'stomach'), (r'duoden', 'duodenum'),
            (r'colon|colonic|appendi|diverticul|large bowel|sigmoid|cecum|cecal|rectum|rectal', 'large bowel'),
            (r'small bowel|jejun|ileum|ileal', 'small bowel'), (r'\bbladder', 'bladder')]


@contextlib.contextmanager
def _cwd(d):
    old = os.getcwd(); os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


def _stub(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs)
    sys.modules.setdefault(name, m)
    return m


def _import_radar_plus():
    """RadarPretrain from RADAR_train without running lavis/__init__ (dataset builders, training-only deps); call with
    cwd = WORK (relative '../ckpt' paths). Same as the earlier audit's radar_plus_import.py."""
    if 'lavis.models.radar_models.radar_pretrain' in sys.modules:
        return sys.modules['lavis.models.radar_models.radar_pretrain'].RadarPretrain
    sys.path.insert(0, TRAIN)                       # VisionBranch pydoc.locate()s dynamic_network_architectures
    import transformers  # noqa: F401  (its optional-dependency probes run before any stub exists)
    import transformers.utils.import_utils  # noqa: F401
    for name, path in [('lavis', f'{TRAIN}/lavis'), ('lavis.models', f'{TRAIN}/lavis/models')]:
        m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m
    try:
        import timm  # noqa: F401
    except ImportError:
        _stub('timm'); _stub('timm.models'); _stub('timm.models.hub', download_cached_file=lambda *a, **k: None,
                                                   get_cache_dir=lambda: '/tmp')
    try:
        import iopath  # noqa: F401
    except ImportError:
        class _PM:
            def __getattr__(self, k):
                return lambda *a, **kw: None
        _stub('iopath'); _stub('iopath.common')
        _stub('iopath.common.download', download=lambda *a, **k: None)
        _stub('iopath.common.file_io', PathManager=_PM, g_pathmgr=_PM(), file_lock=lambda *a, **k: None)
    try:
        import nltk  # noqa: F401
    except ImportError:                             # the regex nltk's wordpunct_tokenize uses
        _stub('nltk'); _stub('nltk.tokenize', wordpunct_tokenize=lambda s: re.findall(r'\w+|[^\w\s]+', s))
    from lavis.common.registry import registry
    registry.register_path('library_root', os.getcwd())
    from lavis.models.base_model import BaseModel
    sys.modules['lavis.models'].BaseModel = BaseModel
    from lavis.models.radar_models.radar_pretrain import RadarPretrain
    return RadarPretrain


def authors_prompts(script):
    """disease_prompts of RADAR_train/<script> (only its prompt-definition block is executed):
    key -> (region, positives, negatives), backslashes stripped as in their evaluate()."""
    src = open(f'{TRAIN}/{script}').read()
    ns = {}
    exec('from dataclasses import dataclass\nfrom typing import List\n' +
         src[src.index('@dataclass'):src.index('def masks_to_boxes_3d')], ns)
    return {k: (v.region, [p.replace('\\', '') for p in v.positive_prompts],
                [p.replace('\\', '') for p in v.negative_prompts]) for k, v in ns['disease_prompts'].items()}


def _masks_to_box(m):
    idx = torch.nonzero(m[0, 0])
    (z0, y0, x0), (z1, y1, x1) = idx.min(0).values.tolist(), idx.max(0).values.tolist()
    return x0, y0, z0, x1, y1, z1


def _center_crop(image, mask, crop_size):
    """Authors' center_crop: ROI-sized box centred on the organ (grown if the organ is larger)."""
    x_min, y_min, z_min, x_max, y_max, z_max = _masks_to_box(mask)
    cd, ch, cw = max(crop_size[0], z_max - z_min), max(crop_size[1], y_max - y_min), max(crop_size[2], x_max - x_min)
    cx, cy, cz = (x_min + x_max) // 2, (y_min + y_max) // 2, (z_min + z_max) // 2
    d, h, w = image.shape[-3:]
    x0 = max(0, cx - cw // 2); x1 = min(w, x0 + cw); x0 = max(0, x1 - cw) if x1 - x0 < cw else x0
    y0 = max(0, cy - ch // 2); y1 = min(h, y0 + ch); y0 = max(0, y1 - ch) if y1 - y0 < ch else y0
    z0 = max(0, cz - cd // 2); z1 = min(d, z0 + cd); z0 = max(0, z1 - cd) if z1 - z0 < cd else z0
    return image[..., z0:z1, y0:y1, x0:x1]


def _end_pad(x, size=None, k=None, value=0.0):
    """Pad the last 3 dims at the end: to at least `size` (SpatialPadd method='end') or to a multiple of k
    (DivisiblePadd method='end')."""
    n = x.shape[-3:]
    tgt = [max(s, t) for s, t in zip(n, size)] if size else [-(-s // k) * k for s in n]
    return F.pad(x, [p for s, t in zip(reversed(n), reversed(tgt)) for p in (0, t - s)], value=value)


class Adapter:
    def __init__(self, ckpt, device):
        kind, _, opt = (ckpt or 'plus').partition(':')
        assert kind in CKPT, f'ckpt must be one of {list(CKPT)} (optionally :nofov)'
        self.kind, self.use_fov, self.dev = kind, opt != 'nofov', device
        from omegaconf import OmegaConf
        with _cwd(WORK):
            os.environ['MODEL_ROOT'] = os.environ['CONFIGS_ROOT'] = os.path.join(os.path.dirname(WORK), 'ckpt')
            RadarPretrain = _import_radar_plus()
            cfg = OmegaConf.create({'med_config_path': '../ckpt/bert-base-uncased/config.json', 'max_txt_len': 512,
                                    'queue_size': 0, 'alpha': 0.4, 'radar_plus': True, 'radar_ft': False,
                                    'load_pretrained': False, 'load_finetuned': False})
            m = RadarPretrain.from_config(cfg)
            sd = torch.load(f'../ckpt/{CKPT[kind]}', map_location='cpu', weights_only=False)['model']
        r = m.load_state_dict(sd, strict=False)
        miss = [k for k in r.missing_keys if '_m.' not in k and not k.startswith(('text_encoder_m', 'text_proj_m'))]
        print(f'[radar {kind}] missing (excl. momentum copies): {len(miss)} {miss[:6]}', flush=True)
        self.m = m.to(device).eval()
        self.temp = float(self.m.temp)
        self.want = [self.m.organs.index(o) for o in ORGANS]
        self.max_len = 100
        self.anat, self.whole = authors_prompts('infer_merlin_anatomy.py'), authors_prompts('infer_merlin_whole.py')
        self.role = {}                       # prompt key -> (positive or negative texts, state row)
        self.pad, self.mirrored = None, False

    # ---------------------------------------------------------------- text
    def prompts(self, f, meta):
        if f in NATIVE_ORGAN:
            region, pos, neg = self.anat[NATIVE_ORGAN[f]]
            row, neg = ORGANS.index(region), neg + ['normal']           # evaluate(): 'normal' joins the negatives
        elif f in NATIVE_WHOLE:
            _, pos, neg = self.whole[NATIVE_WHOLE[f]]
            row = WHOLE
        else:
            p = meta['phrase'][f]
            pos, neg = [f'There is {p}.'], [f'No {p}.']
            row = ORGANS.index(DEFAULT_ORGAN[f]) if f in DEFAULT_ORGAN else WHOLE
        kp, kn = f'<radar {f} +>', f'<radar {f} ->'
        self.role[kp], self.role[kn] = (pos, row), (neg, row)
        return kp, kn

    @staticmethod
    def organ_of(s):
        """State row of the first organ keyword in a free sentence (whole feature if none)."""
        s, best = s.lower(), (len(s) + 1, WHOLE)
        for pat, organ in KEYWORDS:
            hit = re.search(pat, s)
            if hit and hit.start() < best[0]:
                best = (hit.start(), ORGANS.index(organ))
        return best[1]

    @torch.no_grad()
    def _embed(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            t = self.m.tokenizer(texts[i:i + bs], padding='longest', truncation=True, max_length=self.max_len,
                                 return_tensors='pt').to(self.dev)
            e = self.m.text_encoder.forward_text(t).last_hidden_state[:, 0, :]
            out.append(F.normalize(self.m.text_proj(e).float(), dim=-1))
        return torch.cat(out)

    def encode_text(self, sents):
        """-> (S, 258): 256-d text feature (prompt keys: mean of the ensemble's l2-normalised features), state row,
        free-sentence flag (1 = fall back to the whole feature when the organ was not found)."""
        flat, owner, meta = [], [], []
        for j, s in enumerate(sents):
            if s in self.role:
                texts, row = self.role[s]; free = 0.0
            else:
                texts, row, free = [s], (WHOLE if self.max_len > 100 else self.organ_of(s)), 1.0
            flat += texts; owner += [j] * len(texts); meta.append((row, free))
        e = self._embed(flat)
        own = torch.as_tensor(owner, device=self.dev)
        f = torch.zeros(len(sents), e.shape[1], device=self.dev).index_add_(0, own, e)
        f = f / torch.bincount(own, minlength=len(sents)).float()[:, None]
        return torch.cat([f, torch.tensor(meta, device=self.dev, dtype=torch.float32)], 1)

    def similarity(self, state, t):
        """state (13, 256) organ features (NaN = not found) + whole feature, t (S, 258) -> (S,) cosine / temp."""
        f, row, free = t[:, :256], t[:, 256].long(), t[:, 257] > 0
        row = torch.where(free & torch.isnan(state[row, 0]), torch.full_like(row, WHOLE), row)
        return (state[row] * f).sum(-1) / self.temp

    # ---------------------------------------------------------------- image
    def set_volume(self, volume):
        self.pad = fov_pad(volume) if self.use_fov else None
        self.mirrored = False
        if volume != getattr(self, '_last', None):       # new scan: hand cached blocks back to the shared GPU
            self._last = volume
            torch.cuda.empty_cache()

    def set_scan(self, seg):
        """core calls set_scan alone only for the mirror condition (and set_scan + set_volume otherwise), so a set_scan
        not followed by set_volume means the next image is the mirrored scan: swap the left/right FOV padding."""
        self.mirrored = True

    def _grid(self, hu):
        """(X, Y, Z) 1 mm RAS HU -> (1, z, y, x) at 1 x 1 x 5 mm, full FOV, LAS (their LoadImaged + Resized + Transposed)."""
        if self.pad:
            pad = [tuple(reversed(self.pad[0])) if self.mirrored else self.pad[0]] + list(self.pad[1:])
            hu = F.pad(hu, [p for d in reversed(pad) for p in d], value=-1000.0)
        v = hu.flip(0).float()
        h, w, d = v.shape
        x = F.interpolate(v[None, None], size=[h, w, int(d * 0.2)], mode='trilinear', align_corners=False)[0]
        return x.permute(0, 3, 2, 1)

    @staticmethod
    def _minmax(x):
        x = x.clamp(*HU)
        return (x - x.min()) / (x.max() - x.min())

    def _organ_input(self, x):
        x = self._minmax(x)
        nz = torch.nonzero(x[0])
        if len(nz) == 0:
            return None
        lo = (nz.min(0).values - torch.tensor([5, 20, 20], device=x.device)).clamp(min=0)
        hi = torch.minimum(nz.max(0).values + torch.tensor([5, 20, 20], device=x.device),
                           torch.tensor(x.shape[1:], device=x.device))
        x = x[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        return _end_pad(x, ROI)

    def _whole_input(self, x):
        x = _end_pad(x, ROI, value=0.0)                 # in HU, before clipping (their order)
        st = [(n - r) // 2 for n, r in zip(x.shape[1:], ROI)]   # = CenterSpatialCrop for even ROI
        x = x[:, st[0]:st[0] + ROI[0], st[1]:st[1] + ROI[1], st[2]:st[2] + ROI[2]]
        return self._minmax(x)

    def _window(self, patch, feats, skip_organ=None):
        """forward_test_win's feature part: organs intact in this window (and not yet seen) -> feats[organ id]."""
        m = self.m
        seg_probs, seg, e1, e2, e3, f1, f2, f3 = m.visual_encoder(patch, None)
        for i in range(len(seg)):
            mask = seg[i]
            bnd = []
            for d in range(mask.dim()):
                s0 = [slice(None)] * mask.dim(); s1 = [slice(None)] * mask.dim()
                s0[d] = slice(None, 2); s1[d] = slice(-2, None)
                bnd += [mask[tuple(s0)], mask[tuple(s1)]]
            bnd = torch.unique(torch.cat([b.flatten() for b in bnd]))
            bnd = bnd[bnd > 0]
            if skip_organ is not None:
                bnd = bnd[bnd != skip_organ + 1]
            bnd = set(bnd.tolist())
            present = torch.unique(mask).long()
            for oid in (present[present > 0] - 1).tolist():
                if oid not in self.want or oid in feats or (oid + 1) in bnd:
                    continue
                if skip_organ is not None and oid != skip_organ:
                    continue
                kv = torch.cat([e1[i][f1[i, oid]], e2[i][f2[i, oid]], e3[i][f3[i, oid]]], dim=0)[None]
                q, _ = m.attention(m.query_tokens[oid][None, None], kv, kv)
                feats[oid] = F.normalize(m.vision_projs[oid](q[0, 0]), dim=-1).float()
        return seg_probs

    def _organ_feats(self, image):
        from monai.data.utils import dense_patch_slices
        image = image[None]
        size = list(image.shape[2:])
        interval = tuple(int(r) if r == s else max(int(r * (1 - OVERLAP)), 1) for s, r in zip(size, ROI))
        feats = {}
        full = torch.zeros((1, 37) + tuple(size), device=image.device)
        cnt = torch.zeros((1, 1) + tuple(size), device=image.device)     # = their 37-channel count map (all equal)
        for sl in dense_patch_slices(size, ROI, interval):
            sl = [slice(0, 1), slice(None)] + list(sl)
            patch = image[sl]
            segp = self._window(patch, feats)
            full[sl] += F.interpolate(segp, size=patch.shape[2:], mode='trilinear')[0]; cnt[sl] += 1
        stitched = full.div_(cnt.clamp(min=1)).argmax(1).unsqueeze(0)
        del full, cnt
        for oid in self.want:
            if oid in feats:
                continue
            om = torch.eq(stitched, oid + 1)
            if not om.any():
                continue
            self._window(_end_pad(_center_crop(image, om, ROI), k=32), feats, skip_organ=oid)
        return feats

    def _whole_feat(self, x):
        m = self.m
        _, _, e1, e2, e3, _, _, _ = m.visual_encoder(x[None], None)
        kv = torch.cat([e1, e2, e3], dim=1)
        q, _ = m.attention_whole(m.query_tokens_whole[None].expand(kv.shape[0], -1, -1), kv, kv)
        return F.normalize(m.vision_projs_whole(q[:, 0]), dim=-1)[0].float()

    def encode_image(self, hu):
        """Retries on CUDA OOM (GPUs shared with other jobs) before giving up."""
        for attempt in range(6):
            try:
                return self._encode_image(hu)
            except torch.cuda.OutOfMemoryError:
                if attempt == 5:
                    raise
                torch.cuda.empty_cache()
                print(f'[radar] CUDA OOM, retry {attempt + 1} in 60 s', flush=True)
                time.sleep(60)

    @torch.inference_mode()
    def _encode_image(self, hu):
        x = self._grid(hu)
        state = torch.full((len(ORGANS) + 1, 256), float('nan'), device=self.dev)
        a = self._organ_input(x)
        if a is not None:
            for oid, f in self._organ_feats(a).items():
                state[self.want.index(oid)] = f
        state[WHOLE] = self._whole_feat(self._whole_input(x))
        found = (~torch.isnan(state[:WHOLE, 0])).float()
        return state, torch.cat([state.nan_to_num(0.0).flatten(), found])
