"""Lesion-erasure (heal) sets with a same-organ control (2026-11-03), for Table 1b LE on AbdomenAtlas 3.0 and on
ReXGroundingCT (extaudit/lesion_erasure.py), so both
three LE columns share one protocol: each expert-segmented lesion is replaced by the organ's normal tissue (median HU of
the organ minus every lesion dilated 3 mm, + N(0, 15)), and the control is the lesion's own shape translated inside the
same organ (>= 70% inside the parenchyma, < 5% overlap with any lesion dilated 10 mm; scripts/data/lesion_geometry.py
find_control), healed the same way.

  aa3  AbdomenAtlas 3.0 IID test (scripts/data/prepare_aa3.py: 1 mm img / lab / TotalSegmentator seg). Lesions >= 10 mm of
       liver (lab 1), pancreas (lab 2) and kidney (lab 3); the organ of a kidney lesion is the kidney (left or right)
       that holds most of it. Sentence "There is a {d} cm mass in the {liver|pancreas|right kidney|left kidney}."
  rex  ReXGroundingCT findings never used in training (our_split test or rexgrounding_holdout), radiologist masks
       (scripts/data/prepare_rexgrounding.py); the organ is the lung (TotalSegmentator lobes 10-14) on the side that holds
       >= 80% of the lesion, else both lungs; the sentence is the report finding itself.
One unit per (scan, organ): pseudo-volume '<scan>__<organ>.nii.gz' (img / seg point to the real files), masks
{W}/heal/<ds>/masks/<unit>.npz (k<j> lesion, c<j> control, par, shape), findings {W}/heal/<ds>/findings.parquet,
harness set {W}/extaudit/heal_<ds>_set.parquet. Run the test with LE_SET=<ds> python -m extaudit.core --task lesion_erasure.

    python scripts/data/prepare_heal.py --ds aa3 --workers 16
    python scripts/data/prepare_heal.py --ds rex --workers 16
"""
import argparse
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd

import lesion_geometry as P  # noqa: E402  (find_control, dilate, feret_axial, thresholds)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from organic.workroot import work_root                              # noqa: E402

W = work_root()
LUNG_L, LUNG_R = [10, 11], [12, 13, 14]          # TotalSegmentator v2: lung lobes (left upper / lower; right upper / middle / lower)


def load(p, dtype):
    return np.asarray(nib.load(p).dataobj).astype(dtype)


def heal_units(hu, units, shp):
    """units: list of (organ_name, word, organ mask, [lesion point arrays], [texts or None]) -> rows, masks"""
    rows, res = [], {}
    for organ, word, org, lesions, texts, extra in units:
        lab = np.zeros(shp, bool)
        for pts in lesions:
            lab[tuple(pts.T)] = True
        les_d3, les_d10 = P.dilate(lab, 3), P.dilate(lab, 10)
        par = org & ~les_d3
        par_hu = float(np.median(hu[par])) if par.sum() > 1000 else np.nan
        org_d = P.dilate(org, 10)
        R, M, k = [], {}, 0
        for j, pts in enumerate(lesions):
            diam = P.feret_axial(pts) if len(pts) > 1 else 1.0
            zext = float(pts[:, 2].max() - pts[:, 2].min() + 1)
            fio = float(org_d[tuple(pts.T)].mean())
            reason = ''
            if texts is None and max(diam, zext) < P.MIN_EXTENT_MM:
                continue
            if fio < 0.5:
                reason = 'outside organ'
            elif np.isnan(par_hu):
                reason = 'no parenchyma'
            elif texts is None and k >= P.MAX_PER_CASE:
                reason = 'beyond 5 largest'
            dcm = diam / 10.0
            text = texts[j] if texts is not None else f'There is a {dcm:.1f} cm mass in the {word}.'
            row = dict(organ=organ, k=k if not reason else -1, n_vox=int(len(pts)), size_cm=round(dcm, 1),
                       frac_in_organ=fio, text=text, lesion_hu=float(np.median(hu[tuple(pts.T)])), par_hu=par_hu,
                       excluded=reason, ctrl_valid=False, ctrl_in_organ=np.nan, ctrl_overlap=np.nan, **extra[j])
            if not reason:
                les = np.zeros(shp, bool)
                les[tuple(pts.T)] = True
                c = P.find_control(les, par, les_d10)
                M[f'k{k}'] = np.ravel_multi_index(pts.T, shp).astype(np.int64)
                if c is not None:
                    cm, fin, ov, _ = c
                    M[f'c{k}'] = np.flatnonzero(cm).astype(np.int64)
                    row.update(ctrl_valid=bool(fin >= P.MIN_IN_ORGAN and ov < P.MAX_OVERLAP), ctrl_in_organ=fin, ctrl_overlap=ov)
                k += 1
            R.append(row)
        if k:
            pf = np.flatnonzero(par)
            if len(pf) > 200000:
                pf = np.sort(np.random.default_rng(0).choice(pf, 200000, replace=False))
            M['par'] = pf.astype(np.int64)
            M['shape'] = np.asarray(shp)
        rows.append((organ, R, M))
    return rows


def aa3_one(case):
    from scipy import ndimage
    d = {k: os.path.join(W, 'aa3', '1mm', k) for k in ('img', 'lab', 'seg')}
    try:
        hu = load(os.path.join(d['img'], f'{case}.nii.gz'), np.float32)
        lab = load(os.path.join(d['lab'], f'{case}.nii.gz'), np.uint8)
        seg = load(os.path.join(d['seg'], f'{case}.nii.gz'), np.uint8)
        shp = hu.shape
        units = []
        for organ, lid, oids, word in (('liver', 1, [5], 'liver'), ('pancreas', 2, [7], 'pancreas')):
            lc, n = ndimage.label(lab == lid, structure=np.ones((3, 3, 3), bool))
            les = sorted([np.argwhere(lc == j) for j in range(1, n + 1)], key=lambda p: -len(p))
            if les:
                units.append((organ, word, np.isin(seg, oids), les, None, [{} for _ in les]))
        lc, n = ndimage.label(lab == 3, structure=np.ones((3, 3, 3), bool))
        kid = {'right kidney': seg == 2, 'left kidney': seg == 3}
        by = {'right kidney': [], 'left kidney': []}
        for j in range(1, n + 1):
            pts = np.argwhere(lc == j)
            side = max(kid, key=lambda s: P.dilate(kid[s], 3)[tuple(pts.T)].sum())
            by[side].append(pts)
        for side, les in by.items():
            if les:
                les = sorted(les, key=lambda p: -len(p))
                units.append((side.replace(' ', '_'), side, kid[side], les, None, [{} for _ in les]))
        return case, heal_units(hu, units, shp), None
    except Exception:                                                 # noqa: BLE001
        import traceback
        return case, [], traceback.format_exc()[-400:]


def rex_one(job):
    vol, img, segp, g = job
    try:
        hu = load(img, np.float32)
        seg = load(segp, np.uint8)
        shp = hu.shape
        L = dict(np.load(os.path.join(W, 'rexground', 'masks', vol.replace('.nii.gz', '.npz'))))
        left, right = np.isin(seg, LUNG_L), np.isin(seg, LUNG_R)
        groups = {'left_lung': [], 'right_lung': [], 'lungs': []}
        for r in g.itertuples():
            key = f'k{r.k}'
            if key not in L or not len(L[key]):
                continue
            pts = np.stack(np.unravel_index(L[key], shp), 1)
            fl, fr = left[tuple(pts.T)].mean(), right[tuple(pts.T)].mean()
            tot = max(fl + fr, 1e-6)
            side = 'left_lung' if fl / tot >= 0.8 else 'right_lung' if fr / tot >= 0.8 else 'lungs'
            groups[side].append((pts, r))
        units = []
        for side, items in groups.items():
            if items:
                org = {'left_lung': left, 'right_lung': right, 'lungs': left | right}[side]
                units.append((side, side, org, [p for p, _ in items], [r.text for _, r in items],
                              [dict(rex_k=int(r.k), category=r.category_name, our_split=r.our_split) for _, r in items]))
        return vol, heal_units(hu, units, shp), None
    except Exception:                                                 # noqa: BLE001
        import traceback
        return vol, [], traceback.format_exc()[-400:]


def write(ds, results, patient_of, real_paths):
    out = os.path.join(W, 'heal', ds)
    os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    rows = []
    for case, units, err in results:
        if err:
            print('ERROR', case, err)
            continue
        stem = case.replace('.nii.gz', '')
        for organ, R, M in units:
            unit = f'{stem}__{organ}'
            for r in R:
                rows.append(dict(volume=f'{unit}.nii.gz', case=stem, patient=patient_of(stem), source=ds, spectre_seen=False, **r))
            if M:
                np.savez_compressed(os.path.join(out, 'masks', f'{unit}.npz'), **M)
    t = pd.DataFrame(rows)
    t.to_parquet(os.path.join(out, 'findings_all.parquet'), index=False)
    f = t[t.excluded == ''].drop(columns='excluded')
    f.to_parquet(os.path.join(out, 'findings.parquet'), index=False)
    units = sorted(f.volume.unique())
    ref = pd.read_parquet(os.path.join(W, 'extaudit', 'merlin_set.parquet')).head(0)
    s = pd.DataFrame({'index': np.arange(len(units)), 'volume': units,
                      'patient_id': [patient_of(u.split('__')[0]) for u in units]})
    for col in ref.columns:
        if col.startswith('label_'):
            s[col] = np.nan
    s['img'] = [real_paths[u.split('__')[0]][0] for u in units]
    s['seg'] = [real_paths[u.split('__')[0]][1] for u in units]
    s['lat_pairs'] = '[]'
    s[list(ref.columns)].to_parquet(os.path.join(W, 'extaudit', f'heal_{ds}_set.parquet'), index=False)
    print(t.groupby(['organ', 'excluded']).size().unstack(fill_value=0).to_string())
    print(f'{ds}: {len(f)} lesions ({int(f.ctrl_valid.sum())} with a valid control) in {len(units)} units')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True, choices=['aa3', 'rex'])
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    if a.ds == 'aa3':
        cases = pd.read_parquet(os.path.join(W, 'aa3', 'cases.parquet'))
        pos = (cases[['liver_label', 'pancreas_label', 'kidney_label']] == 'positive').any(axis=1)     # scans with a lesion >= 10 mm
        ids = sorted(cases.loc[pos, 'case'].astype(str).str.replace('.nii.gz', '', regex=False).unique())
        with Pool(a.workers) as p:
            res = p.map(aa3_one, ids, chunksize=1)
        d = os.path.join(W, 'aa3', '1mm')
        write('aa3', res, lambda c: c, {c: (f'{d}/img/{c}.nii.gz', f'{d}/seg/{c}.nii.gz') for c in ids})
    else:
        gt = pd.read_parquet(os.path.join(W, 'rexground', 'findings.parquet'))
        gt = gt[gt.our_split.isin(['test', 'rexgrounding_holdout'])]
        st = pd.read_parquet(os.path.join(W, 'extaudit', 'ctrate_rex_set_v2.parquet')).set_index('volume')
        gt = gt[gt.volume.isin(st.index)]
        jobs = [(v, st.loc[v, 'img'], st.loc[v, 'seg'], g) for v, g in gt.groupby('volume')]
        with Pool(a.workers) as p:
            res = p.map(rex_one, jobs, chunksize=1)
        pat = gt.drop_duplicates('volume').set_index('volume').patient
        write('rex', res, lambda c: pat.get(f'{c}.nii.gz', c), {v.replace('.nii.gz', ''): (st.loc[v, 'img'], st.loc[v, 'seg']) for v, *_ in jobs})


if __name__ == '__main__':
    main()
