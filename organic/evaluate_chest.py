"""Chest rungs (CT-RATE), evaluation only: WHERE (lobe) and WHAT (density-defined edits).

WHERE. Test sentences: report sentences that state something present in exactly one lung lobe ("Atelectasis
in the right middle lobe."). Each is re-written with each of the five lobe names (canonical
phrases), so the text varies only in the lobe.
  text test   on the clean scan, does the true-lobe version score highest of the five? (chance 20%; side only: 50%)
  image test  erase each lobe in turn (TotalSegmentator lobes 10-14, dilated 2 mm, fat fill + noise as in the organ
              rung) and score the true-lobe sentence. The named lobe should cause the largest drop. Larger lobes cause
              larger drops for every sentence, so each lobe's drop is adjusted by its mean drop over sentences naming
              OTHER lobes before taking the top-1.
Report location claims are themselves imperfect (wrong-side or wrong-lobe statements; nodule and opacity reports are
the least reliable), so these are lower bounds; the comparison between models is the point.

WHAT. Edits that remove exactly what a finding's radiological definition names, on every test scan:
  emph     lung voxels < -950 HU set to normal parenchyma (-860 HU + noise): emphysema should fall, nothing else
  ca_ao    aortic calcium (found at 1 mm: >= 130 HU after 3 mm smoothing, >= 3 connected voxels) set to 40 HU + noise
  ca_cor   the same in a 3 mm shell around the heart (outside heart, aorta and every other labelled structure)
Calcium edits are skipped on contrast-enhanced scans (median aorta > 120 HU). Output: score change of every finding
under every edit; the edit x finding matrix should be diagonal.

    torchrun --nproc_per_node 4 -m organic.evaluate_chest --env configs/env/local.yaml \
        --ckpt <work_root>/runs/enc4_organic3d/last.pt [--name enc4_organic3d] [--limit 0] [--parts lobes,edits]
Writes {work_root}/eval/<name>/ctrate_lobes.npz, ctrate_edits.npz and ctrate_chest_summary.json.
"""
import argparse
import json
import os
import re

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .augment import erase, normalise
from .config import load_config, read_manifest
from .data import SIZE, SPACING, CTDataset, _LUT, _crop_bounds, _crop_pad, crop_center
from .vocab import BACKGROUND, CTRATE_FINDINGS, FINDING_NAMES, ORGAN_ID
from .evaluate_stageA import Scorer, _NEG
from .ctrate_v2 import no_chest_volumes
from .train_stageA import auroc, setup_dist

HERE = os.path.dirname(__file__)
LOBES = ['right upper lobe', 'right middle lobe', 'right lower lobe', 'left upper lobe', 'left lower lobe']
LOBE_TS = [12, 13, 14, 10, 11]                                   # TotalSegmentator v2 ids, same order
SIDE = [0, 0, 0, 1, 1]                                           # 0 right, 1 left
_UML = r'(upper|middle|lower)'
_PAT = re.compile(r'\b(?:(right|left)\s+(?:lung\s+)?' + _UML + r'\s+lobes?\b|' + _UML + r'\s+lobes?\s+of\s+the\s+(right|left)\s+lung\b'
                  r'|lingula(?:r)?\b|\bmiddle\s+lobe\b)', re.I)   # "right middle lobe" matches first (earlier start)
_ANY_LOBE = re.compile(r'\b(?:upper|middle|lower)\s+lobes?\b|\blingula', re.I)    # every lobe mention, sided or not
_MULTI = re.compile(r'\blobes\b|\b(?:upper|middle|lower)\s*(?:and|or|,|/)\s*(?:upper|middle|lower)\b', re.I)
FINDING_WORDS = {'consolidation': r'consolidat', 'atelectasis': r'atelecta', 'nodule': r'nodul', 'opacity': r'opacit|ground[- ]glass',
                 'fibrotic': r'fibro|sequela', 'bronchiectasis': r'bronchiecta', 'septal': r'septal', 'emphysema': r'emphysem'}


def lobe_of(m):
    """regex match -> lobe index in LOBES, or None."""
    t = m.group(0).lower()
    if t.startswith('lingula'):
        return 3
    if m.group(1):
        side, level = m.group(1).lower(), m.group(2).lower()
    elif m.group(4):
        side, level = m.group(4).lower(), m.group(3).lower()
    else:                                                         # bare "middle lobe" = right middle lobe
        side, level = 'right', 'middle'
    name = f'{side} {level} lobe'
    return LOBES.index(name) if name in LOBES else None           # "left middle lobe" does not exist


def lobe_sentences(report, split, max_n=8):
    """[(sentence with a {LOBE} slot, true lobe index, finding word or '')] for present, single-lobe sentences."""
    out = []
    for s in split(report):
        if _NEG.search(s) or re.search(r'\b(both|bilateral)\b', s, re.I):
            continue
        ms = list(_PAT.finditer(s))
        lobes = {lobe_of(m) for m in ms} - {None}
        if len(ms) != 1 or len(lobes) != 1 or len(_ANY_LOBE.findall(s)) != 1 or _MULTI.search(s):
            continue                                              # one lobe, named once, with its side
        m = ms[0]
        tmpl = s[:m.start()] + '{LOBE}' + s[m.end():]
        word = next((k for k, p in FINDING_WORDS.items() if re.search(p, s, re.I)), '')
        out.append((tmpl, lobes.pop(), word))
    return out[:max_n]


def load_crop(ds, r, calcium=False):
    """Same crop as CTDataset (no cache), at 2 mm: image (1, 192^3) HU, lobe map (0-5), organ-group map; with
    calcium=True also the aortic and coronary calcium masks at 2 mm (found at 1 mm, see calcium_masks)."""
    img, seg = ds._load(r)
    organ1 = _LUT[seg]
    lobe1 = np.zeros_like(seg)
    for k, t in enumerate(LOBE_TS):
        lobe1[seg == t] = k + 1
    b1 = _crop_bounds(crop_center(organ1, r.dataset), img.shape, int(SIZE * SPACING))
    img_c = _crop_pad(img, b1, -1000.0)
    t = torch.from_numpy(img_c)[None, None]
    vol = F.interpolate(t, size=(SIZE, SIZE, SIZE), mode='trilinear', align_corners=False)[0]
    sub = lambda a: torch.from_numpy(np.ascontiguousarray(a[1::2, 1::2, 1::2]))
    org_c = _crop_pad(organ1, b1, BACKGROUND)
    out = [vol, sub(_crop_pad(lobe1, b1, 0)), sub(org_c)]
    if calcium:
        out.append(calcium_masks(img_c, org_c, _crop_pad(seg, b1, 0)))
    return out


def calcium_masks(img, organ, seg):
    """1 mm crop -> {'ca_ao', 'ca_cor'}: 2 mm bool masks (a 2 mm voxel is edited if any of its 1 mm voxels is calcium).
    Calcium = >= 130 HU on the 3x3x3 mean-filtered volume (image noise of 50-100 HU does not pass) and < 2000 HU raw (no
    metal), components >= 3 voxels, plus a 1 mm rim. Aorta region: aorta dilated 1 mm, unlabelled or aorta voxels only. Coronary region: 3 mm shell around the heart, outside
    heart and aorta (dilated 3 mm) and outside every other labelled structure. Empty on contrast-enhanced scans."""
    from scipy import ndimage
    aorta, heart = organ == ORGAN_ID['aorta'], organ == ORGAN_ID['heart']
    hs = ndimage.uniform_filter(img, size=3)
    empty = np.zeros([n // 2 for n in img.shape], bool)
    if aorta.sum() > 2000 and np.median(img[ndimage.binary_erosion(aorta, iterations=2)]) > 120:
        return {'ca_ao': empty, 'ca_cor': empty.copy(), 'contrast': True}

    def ca(region):
        m = region & (hs >= 130) & (img < 2000)                            # not metal (wires, stents)
        lab, n = ndimage.label(m)
        if n:
            size = ndimage.sum(m, lab, np.arange(1, n + 1))
            m = np.isin(lab, np.where(size >= 3)[0] + 1)
        m = ndimage.binary_dilation(m, iterations=1) & (img > 0)          # rim of the plaque, not surrounding fat
        return m.reshape(m.shape[0] // 2, 2, m.shape[1] // 2, 2, m.shape[2] // 2, 2).any(axis=(1, 3, 5))
    shell = ndimage.binary_dilation(heart, iterations=3) & ~heart & ~ndimage.binary_dilation(aorta, iterations=3) & (seg == 0)
    ao_region = ndimage.binary_dilation(aorta, iterations=1) & ((seg == 0) | (seg == 52))   # not the adjacent spine
    return {'ca_ao': ca(ao_region), 'ca_cor': ca(shell), 'contrast': False}


@torch.no_grad()
def score_matrix(scorer, x, sents):
    """x (V,1,D,H,W) normalised -> (V, S) similarities."""
    enc = scorer.tok(sents, max_length=96, padding='longest', truncation=True, return_tensors='pt')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        t = scorer.m.encode_text(enc['input_ids'].to(scorer.device), enc['attention_mask'].to(scorer.device))
        if scorer.native:
            e = scorer.m.encode_image(x, pool=True, project=True, normalize=True).float()
            return (e @ t.float().T).float()
        T, _, _ = scorer.m.encode_image(x)
        return scorer.m.score(T, scorer.m.attach_route(t, sents)).float()


def test_set(cfg, limit):
    m = read_manifest(cfg, 'ctrate')
    m = m[m.split == 'test']
    m = m[~m.volume.isin(no_chest_volumes(cfg.paths.work_root))]   # CT-RATE v2: drop the brain series (3,002 scans)
    if limit:
        m = m.sample(n=min(limit, len(m)), random_state=0)
    return CTDataset(cfg, [m], train=False)


def gather(recs, world):
    if world > 1:
        allr = [None] * world
        dist.all_gather_object(allr, recs)
        recs = [x for part in allr for x in part]
    return recs


def run_lobes(cfg, scorer, ds, rank, world, device):
    from .data import split_sentences
    gen = torch.Generator(device=device).manual_seed(rank)
    recs = []
    for i in range(rank, len(ds), world):
        r = ds.df.iloc[i]
        items = lobe_sentences(r.report, split_sentences)
        if not items:
            continue
        vol, lobe, _ = load_crop(ds, r)
        hu, lobe = vol[0].to(device).float(), lobe.to(device)
        nvox = [int((lobe == k + 1).sum()) for k in range(5)]
        vols = [hu]
        for k in range(5):
            mk = F.max_pool3d((lobe == k + 1).float()[None, None], 3, stride=1, padding=1)[0, 0] > 0
            vols.append(erase(hu, mk, 'fat', 15.0, gen))
        x = normalise(torch.stack(vols)[:, None])
        sents = [tmpl.replace('{LOBE}', LOBES[k]) for tmpl, _, _ in items for k in range(5)]
        S = score_matrix(scorer, x, sents).cpu().numpy().reshape(6, len(items), 5)     # (cond, sentence, text lobe)
        for s_i, (tmpl, true, word) in enumerate(items):
            recs.append(dict(index=i, patient=str(r.patient_id), true=true, word=word, nvox=nvox,
                             text=S[0, s_i].tolist(),                      # clean scan, five lobe texts
                             img=S[:, s_i, true].tolist()))                # true-lobe text, clean + five erasures
    return gather(recs, world)


def metrics_lobes(recs):
    true = np.array([r['true'] for r in recs])
    text = np.array([r['text'] for r in recs])
    img = np.array([r['img'] for r in recs])
    nvox = np.array([r['nvox'] for r in recs])
    side = np.array(SIDE)
    ok = nvox[np.arange(len(true)), true] > 0                        # named lobe visible in the crop
    drop = img[:, :1] - img[:, 1:]                                    # (n, 5) score drop when lobe k is erased
    drop = np.where(nvox > 0, drop, np.nan)
    generic = np.array([np.nanmean(drop[(true != k) & ok, k]) for k in range(5)])
    adj = drop - generic
    n = np.arange(len(true))
    others = lambda a: (np.nansum(a, 1) - a[n, true]) / (np.sum(~np.isnan(a), 1) - 1)
    top = np.nanargmax(np.where(np.isnan(adj), -np.inf, adj), 1)
    out = dict(n_sent=int(ok.sum()), n_scans=int(len({r['index'] for r, o in zip(recs, ok) if o})),
               text_top1=float((text.argmax(1) == true)[ok].mean()),
               text_side=float((side[text.argmax(1)] == side[true])[ok].mean()),
               img_top1=float((top == true)[ok].mean()),
               img_side=float((side[top] == side[true])[ok].mean()),
               img_margin=float(np.nanmean((drop[n, true] - others(drop))[ok])),
               img_margin_adj=float(np.nanmean((adj[n, true] - others(adj))[ok])))
    per = {}
    for w in sorted({r['word'] for r in recs}):
        sel = ok & np.array([r['word'] == w for r in recs])
        if sel.sum() >= 30:
            per[w or 'other'] = dict(n=int(sel.sum()), text_top1=float((text.argmax(1) == true)[sel].mean()),
                                     img_top1=float((top == true)[sel].mean()),
                                     img_margin_adj=float(np.nanmean((adj[n, true] - others(adj))[sel])))
    out['per_finding'] = per
    out['generic_drop'] = generic.tolist()
    return out


EDITS = {'emph': 'ct_emphysema', 'ca_ao': 'ct_arterial_calcification', 'ca_cor': 'ct_coronary_calcification'}


def edit_masks(hu, lobe, organ, ca):
    """2 mm crop (CPU numpy) + calcium masks -> {edit: (mask, fill HU)}; empty masks where an edit does not apply."""
    from scipy import ndimage
    lung = lobe > 0
    trachea = ndimage.binary_dilation(organ == ORGAN_ID['trachea'], iterations=2)
    low = (hu < -950) & (lung | trachea)
    lab, n = ndimage.label(low)
    airway = np.unique(lab[trachea & low]) if n else np.zeros(0, int)
    emph = (hu < -950) & lung & ~np.isin(lab, airway[airway > 0])     # low-attenuation lung, not connected airway air
    return {'emph': (emph, -860.0), 'ca_ao': (ca['ca_ao'], 40.0), 'ca_cor': (ca['ca_cor'], 40.0)}


def run_edits(cfg, scorer, ds, rank, world, device):
    gen = torch.Generator(device=device).manual_seed(1000 + rank)
    recs = []
    for i in range(rank, len(ds), world):
        r = ds.df.iloc[i]
        vol, lobe, organ, ca = load_crop(ds, r, calcium=True)
        hu = vol[0].float()
        masks = edit_masks(hu.numpy(), lobe.numpy(), organ.numpy(), ca)
        hu_d = hu.to(device)
        vols = [hu_d]
        for e in EDITS:
            mk, fill = masks[e]
            mk = torch.from_numpy(np.asarray(mk)).to(device)
            v = hu_d.clone()
            if mk.any():
                v[mk] = fill + torch.randn(int(mk.sum()), device=device, generator=gen) * 15.0
            vols.append(v)
        d, _ = scorer(normalise(torch.stack(vols)[:, None]))
        recs.append(dict(index=i, patient=str(r.patient_id), labels=ds.labels[i].tolist(),
                         nvox=[int(np.asarray(masks[e][0]).sum()) for e in EDITS], delta=d.cpu().numpy().tolist()))
    return gather(recs, world)


def metrics_edits(recs):
    """Edit x finding: AUROC change and mean score shift (in SD of the clean score) per finding, on scans where the
    edit changed something. The own finding should move most."""
    L = np.array([r['labels'] for r in recs]); D = np.array([r['delta'] for r in recs]); V = np.array([r['nvox'] for r in recs])
    out = {}
    for k, (e, own) in enumerate(EDITS.items()):
        sel = V[:, k] > 0
        row = {}
        for f in CTRATE_FINDINGS:
            j = FINDING_NAMES.index(f)
            y, d0, d1 = L[sel, j], D[sel, 0, j], D[sel, k + 1, j]
            if np.nansum(y) < 10 or np.nansum(1 - y) < 10:
                continue
            if not (np.isfinite(d0).all() and np.isfinite(d1).all()):   # finding the model does not score (NaN)
                continue
            sd = D[:, 0, j].std() + 1e-8
            row[f] = dict(d_auroc=float(auroc(y, d1) - auroc(y, d0)), shift_pos=float(np.mean((d1 - d0)[y == 1]) / sd),
                          shift_all=float(np.mean(d1 - d0) / sd))
        own_s = row.get(own, {}).get('shift_all', np.nan)
        oth = [v['shift_all'] for f, v in row.items() if f != own]
        out[e] = dict(own=own, n_scans=int(sel.sum()), own_shift=own_s, own_d_auroc=row.get(own, {}).get('d_auroc', np.nan),
                      others_mean_abs_shift=float(np.mean(np.abs(oth))) if oth else np.nan,
                      specificity=float(abs(own_s) / (np.mean(np.abs(oth)) + 1e-8)) if oth else np.nan, findings=row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(HERE, '../configs/stageA.yaml'))
    ap.add_argument('--env', required=True)
    ap.add_argument('--ckpt', required=True, help="checkpoint path, or 'colipri'")
    ap.add_argument('--name', default=None)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--parts', default='lobes,edits')
    ap.add_argument('overrides', nargs='*')
    a = ap.parse_args()
    run_cfg = os.path.join(os.path.dirname(a.ckpt), 'config.yaml') if a.ckpt != 'colipri' else None
    cfg = load_config(run_cfg or a.config, env=a.env, overrides=a.overrides)
    os.environ.setdefault('HF_HOME', cfg.paths.hf_home)
    rank, world, device = setup_dist(timeout_min=240)
    scorer = Scorer(cfg, a.ckpt, device)
    name = a.name or (os.path.basename(os.path.dirname(a.ckpt)) if a.ckpt != 'colipri' else 'COLIPRI')
    out_dir = os.path.join(cfg.paths.work_root, 'eval', name)
    ds = test_set(cfg, a.limit)
    summ = {'model': name}
    if 'lobes' in a.parts:
        recs = run_lobes(cfg, scorer, ds, rank, world, device)
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            np.savez_compressed(os.path.join(out_dir, 'ctrate_lobes.npz'),
                                **{k: np.array([r[k] for r in recs]) for k in ('index', 'patient', 'true', 'word', 'nvox', 'text', 'img')})
            summ['lobes'] = metrics_lobes(recs)
            print('LOBES', json.dumps({k: v for k, v in summ['lobes'].items() if k != 'per_finding'}), flush=True)
            for w, v in summ['lobes']['per_finding'].items():
                print(f'  {w:15s} {json.dumps(v)}', flush=True)
    if 'edits' in a.parts:
        recs = run_edits(cfg, scorer, ds, rank, world, device)
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            np.savez_compressed(os.path.join(out_dir, 'ctrate_edits.npz'), edits=np.array(list(EDITS)),
                                **{k: np.array([r[k] for r in recs]) for k in ('index', 'patient', 'labels', 'nvox', 'delta')})
            summ['edits'] = metrics_edits(recs)
            for e, v in summ['edits'].items():
                print('EDIT', e, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k != 'findings'}), flush=True)
    if rank == 0:
        with open(os.path.join(out_dir, 'ctrate_chest_summary.json'), 'w') as fh:
            json.dump(summ, fh, indent=1)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
