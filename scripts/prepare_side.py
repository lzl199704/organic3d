"""Expert-mask side sets for the unflipped side test (2026-11-03; extaudit/side.py, SIDE_SET=rex|aa3). Built from the
lesion-erasure units of scripts/prepare_heal.py, so the scans, crops and patients are those of Table 1b's LE columns.

  rex  held-out ReXGroundingCT findings whose radiologist mask lies >= 80% in one lung (heal organ left_lung /
       right_lung) and whose sentence names that side and only that side (no 'bilateral' / 'both'). Swapped sentence =
       left <-> right. Dropped when another finding of the same category in the scan (any split) has >= 20% of its
       mask in the other lung, since the swapped sentence could then be true.
  aa3  AbdomenAtlas 3.0 kidney tumours (heal organ left_kidney / right_kidney), sentence 'There is a {d} cm mass in
       the {side} kidney.'; dropped when the other kidney holds any tumour voxel (label 3, kidney dilated 3 mm).
Writes {W}/side/<ds>.parquet: volume (heal unit), case, organ, side, k, patient, text_true, text_swap.

    python scripts/prepare_side.py --ds rex --workers 16
"""
import argparse
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lesion_geometry as P  # noqa: E402
from prepare_heal import load, LUNG_L, LUNG_R, W  # noqa: E402

LEFT, RIGHT = re.compile(r'\bleft\b', re.I), re.compile(r'\bright\b', re.I)


def swap(t):
    return re.sub(r'\b(left|right)\b', lambda m: {'left': 'right', 'right': 'left'}[m.group(1).lower()]
                  if m.group(1).islower() else {'left': 'Right', 'right': 'Left'}[m.group(1).lower()], t, flags=re.I)


def rex_other_side(job):
    """-> {(rex_k): (frac left, frac right)} for every finding of the scan (all splits)."""
    vol, segp, ks = job
    seg = load(segp, np.uint8)
    left, right = np.isin(seg, LUNG_L), np.isin(seg, LUNG_R)
    L = dict(np.load(os.path.join(W, 'rexground', 'masks', vol.replace('.nii.gz', '.npz'))))
    out = {}
    for k in ks:
        key = f'k{k}'
        if key in L and len(L[key]):
            idx = np.unravel_index(L[key], seg.shape)
            out[k] = (float(left[idx].mean()), float(right[idx].mean()))
    return vol, out


def aa3_other(job):
    case, sides = job
    seg = load(os.path.join(W, 'aa3', '1mm', 'seg', f'{case}.nii.gz'), np.uint8)
    lab = load(os.path.join(W, 'aa3', '1mm', 'lab', f'{case}.nii.gz'), np.uint8) == 3
    kid = {'right_kidney': seg == 2, 'left_kidney': seg == 3}
    return case, {s: bool((lab & P.dilate(kid[s], 3)).any()) for s in kid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True, choices=['rex', 'aa3'])
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    f = pd.read_parquet(os.path.join(W, 'heal', a.ds, 'findings.parquet'))
    f = f[f.organ.isin(['left_lung', 'right_lung', 'left_kidney', 'right_kidney'])].copy()
    f['side'] = f.organ.str.split('_').str[0]
    n0 = len(f)
    st = pd.read_parquet(os.path.join(W, 'extaudit', f'heal_{a.ds}_set.parquet')).set_index('volume')
    if a.ds == 'rex':
        nl, nr = f.text.str.contains(LEFT), f.text.str.contains(RIGHT)
        one = (nl ^ nr) & ~f.text.str.contains(r'bilateral|\bboth\b', case=False)
        named = np.where(nl, 'left', 'right')
        print(f'rex: {n0} one-lung findings; names one side {int(one.sum())}; of these, mask side agrees '
              f'{int((one & (named == f.side)).sum())}, disagrees {int((one & (named != f.side)).sum())}')
        f = f[one & (named == f.side)].copy()
        gt = pd.read_parquet(os.path.join(W, 'rexground', 'findings.parquet'))
        gt = gt[gt.volume.isin(set(f.case + '.nii.gz'))]
        jobs = [(v, st.loc[f[f.case + '.nii.gz' == v].volume.iloc[0], 'seg'], list(g.k)) for v, g in gt.groupby('volume')]
        with Pool(a.workers) as p:
            fr = dict(p.map(rex_other_side, jobs, chunksize=1))
        cat = gt.set_index(['volume', 'k']).category
        keep = []
        for r in f.itertuples():
            v = r.case + '.nii.gz'
            other = 1 if r.side == 'left' else 0                      # index of the other lung's fraction
            clash = any(cat[(v, k)] == cat[(v, r.rex_k)] and fl[other] >= 0.2 for k, fl in fr[v].items() if k != r.rex_k)
            keep.append(not clash)
        f = f[np.asarray(keep)].copy()
    else:
        with Pool(a.workers) as p:
            oth = dict(p.map(aa3_other, [(c, None) for c in sorted(f.case.unique())], chunksize=1))
        other = f.organ.map({'left_kidney': 'right_kidney', 'right_kidney': 'left_kidney'})
        f = f[[not oth[c][o] for c, o in zip(f.case, other)]].copy()
    f['text_true'] = f.text
    f['text_swap'] = f.text.map(swap)
    assert (f.text_true != f.text_swap).all()
    os.makedirs(os.path.join(W, 'side'), exist_ok=True)
    cols = ['volume', 'case', 'organ', 'side', 'k', 'patient', 'text_true', 'text_swap']
    f[cols].to_parquet(os.path.join(W, 'side', f'{a.ds}.parquet'), index=False)
    print(f'{a.ds}: {len(f)} of {n0} one-side lesions kept; left {int((f.side == "left").sum())}, '
          f'right {int((f.side == "right").sum())}; {f.patient.nunique()} patients, {f.volume.nunique()} units')
    print(f.sample(min(4, len(f)), random_state=0)[['side', 'text_true', 'text_swap']].to_string())


if __name__ == '__main__':
    main()
