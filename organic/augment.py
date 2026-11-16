"""GPU-side counterfactual construction for stage A: organ erasure, mirroring, normalisation, text edits."""
import re

import numpy as np
import torch
import torch.nn.functional as F

from .vocab import FINDING_NAMES, SIZE_FINDINGS, finding_organ_ids, control_organ_ids

FILLS_HU = {'fat': -100.0, 'soft': 40.0, 'water': 0.0}   # water: evaluation only (no model trains with it)


def normalise(hu):
    """HU -> COLIPRI input range [-1, 1]."""
    return hu.clamp(-1000, 1000) / 1000.0


def organ_mask(organ, ids, dilate):
    """organ (192^3 uint8) -> bool mask of the organ groups `ids`, dilated by `dilate` voxels (2 mm each)."""
    if not ids:
        return torch.zeros_like(organ, dtype=torch.bool)
    m = torch.isin(organ, torch.as_tensor(ids, device=organ.device, dtype=organ.dtype)).float()[None, None]
    if dilate:
        m = F.max_pool3d(m, 2 * dilate + 1, stride=1, padding=dilate)
    return m[0, 0] > 0


def erase(vol_hu, mask, fill, noise_sd, gen):
    """Replace masked voxels with a fill (HU) plus Gaussian noise. fill='mean' uses the region's own mean HU."""
    out = vol_hu.clone()
    if not mask.any():
        return out
    val = vol_hu[mask].mean() if fill == 'mean' else torch.tensor(FILLS_HU[fill], device=vol_hu.device)
    noise = torch.randn(int(mask.sum()), device=vol_hu.device, generator=gen) * noise_sd
    out[mask] = val + noise
    return out


def present_organs(organ, min_vox):
    ids, counts = torch.unique(organ, return_counts=True)
    return {int(i) for i, c in zip(ids.tolist(), counts.tolist()) if i != 255 and c >= min_vox}


def plan_counterfactuals(batch, cfg, rng):
    """For each sample, choose (finding, target organs, control set U) for the evidence term.
    Returns list of dicts (or None when the sample is skipped)."""
    c = cfg.aug
    plans = []
    labels = batch['labels'].numpy()
    for i in range(len(labels)):
        if rng.random() > c.ev_prob:
            plans.append(None); continue
        organ = batch['organ'][i]
        present = present_organs(organ, c.min_organ_vox)
        known = [j for j, f in enumerate(FINDING_NAMES) if not np.isnan(labels[i, j])]
        cands = []
        for j in known:
            f = FINDING_NAMES[j]
            tgt = [o for o in finding_organ_ids(f) if o in present]
            if finding_organ_ids(f) and not tgt:
                continue                                   # target organ not in the crop
            if f in c.get('holdout_findings', []) or (c.get('ev_skip_size', False) and f in SIZE_FINDINGS):
                continue
            cands.append((j, tgt))
        if not cands:
            plans.append(None); continue
        j, tgt = cands[rng.integers(len(cands))]
        ctrl = [o for o in control_organ_ids(FINDING_NAMES[j]) if o in present and o not in c.get('holdout_organs_ids', [])]
        tgt = [o for o in tgt if o not in c.get('holdout_organs_ids', [])]
        if not ctrl and not tgt:
            plans.append(None); continue
        k_u = int(rng.integers(1, c.max_controls + 1)) if ctrl else 0
        U = list(rng.choice(ctrl, size=min(k_u, len(ctrl)), replace=False)) if ctrl else []
        U_t = U[:int(rng.integers(0, len(U) + 1))]          # |U| in {0..k} for the sensitivity pass
        plans.append({'finding': j, 'target': tgt, 'U': [int(u) for u in U], 'U_t': [int(u) for u in U_t],
                      'fill': c.fills[int(rng.integers(len(c.fills)))]})
    return plans


def build_image_batch(batch, plans, mirror_idx, cfg, device, seed):
    """Stack clean, sensitivity (target + U_t erased), invariance (U erased) and mirrored volumes.
    Returns normalised tensor (N, 1, 192, 192, 192) and an index map."""
    c = cfg.aug
    gen = torch.Generator(device=device).manual_seed(int(seed))
    hu = batch['img'].to(device, non_blocking=True).float()           # (B, 1, D, H, W)
    organ = batch['organ'].to(device, non_blocking=True)
    vols = [hu[i, 0] for i in range(hu.shape[0])]
    idx = {'clean': list(range(len(vols))), 'sens': {}, 'inv': {}, 'mirror': {}}
    for i, p in enumerate(plans):
        if p is None:
            continue
        if p['target']:
            m = organ_mask(organ[i], p['target'] + p['U_t'], c.dilate)
            idx['sens'][i] = len(vols); vols.append(erase(hu[i, 0], m, p['fill'], c.noise_sd, gen))
        if p['U']:
            m = organ_mask(organ[i], p['U'], c.dilate)
            idx['inv'][i] = len(vols); vols.append(erase(hu[i, 0], m, p['fill'], c.noise_sd, gen))
    for i in mirror_idx:
        idx['mirror'][i] = len(vols); vols.append(torch.flip(hu[i, 0], dims=[0]))   # axis 0 = x (R-L)
    x = normalise(torch.stack(vols)[:, None])
    return x, idx


# ---------------- rule-based sentence edits (LLM edits come from parsing)
_SIDE = re.compile(r'\b(left|right)\b', re.I)
_SEV = {'mild': 'severe', 'severe': 'mild', 'minimal': 'marked', 'marked': 'minimal', 'small': 'large',
        'large': 'small', 'mildly': 'severely', 'severely': 'mildly', 'trace': 'large', 'massive': 'small'}
_SEV_RE = re.compile(r'\b(' + '|'.join(_SEV) + r')\b', re.I)


def _case(src, word):
    return word.capitalize() if src[0].isupper() else word


_ANAT_SIZE = re.compile(r'\b(small|large)\s+(bowel|intestin\w*|saphenous|airways?|vessels?|cell)\b', re.I)


def lateral_swap(s):
    if not _SIDE.search(s):
        return None
    return _SIDE.sub(lambda m: _case(m.group(0), {'left': 'right', 'right': 'left'}[m.group(0).lower()]), s)


def severity_swap(s):
    """mild<->severe etc.; "small bowel" / "large bowel" are anatomy, not severity, and are left alone."""
    spans = [m.span(1) for m in _ANAT_SIZE.finditer(s)]
    hits = [m for m in _SEV_RE.finditer(s) if m.span() not in spans]
    if not hits:
        return None
    out, k = [], 0
    for m in hits:
        out += [s[k:m.start()], _case(m.group(0), _SEV[m.group(0).lower()])]; k = m.end()
    return ''.join(out + [s[k:]])


def one_side_only(s):
    """True if the sentence names exactly one side (swapping 'left and right' lists can keep the meaning)."""
    return len({w.lower() for w in _SIDE.findall(s)}) == 1
