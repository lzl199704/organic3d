"""Lesion geometry shared by scripts/prepare_heal.py, prepare_side.py and prepare_aa3.py: TotalSegmentator on 1 mm
volumes, the largest axial Feret diameter, dilation, and the same-organ control of the lesion-erasure test (LE): the
lesion's own shape translated inside the same organ, on a 4 mm grid (FFT cross-correlation), valid when >= 70% lies inside
the parenchyma and < 5% overlaps any lesion dilated 10 mm; among valid translations the nearest one with >= 90% inside
(else the best inside fraction), then re-checked exactly at 1 mm.
"""
import os
import time

import numpy as np

MIN_EXTENT_MM, MIN_INPLANE_MM, MAX_PER_CASE = 10.0, 5.0, 5
MIN_IN_ORGAN, MAX_OVERLAP, PREF_IN_ORGAN = 0.7, 0.05, 0.9


def run_ts(W, device, cases, out_dirs):
    """TotalSegmentator v2 'total' (fast, multilabel) for each case of out_dirs(W)['img'] -> out_dirs(W)['seg']."""
    from totalsegmentator.python_api import totalsegmentator
    if device == 'cpu':
        import torch
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))
    d = out_dirs(W)
    lock = os.path.join(d['seg'], '.lock')
    os.makedirs(lock, exist_ok=True)
    done = 0
    for c in cases:
        src, dst = os.path.join(d['img'], f'{c}.nii.gz'), os.path.join(d['seg'], f'{c}.nii.gz')
        if os.path.exists(dst) or not os.path.exists(src):
            continue
        try:
            fd = os.open(os.path.join(lock, c), os.O_CREAT | os.O_EXCL)
            os.close(fd)
        except FileExistsError:
            continue
        t0 = time.time()
        try:
            totalsegmentator(src, dst + '.tmp.nii.gz', ml=True, fast=True, device=device, quiet=True,
                             nr_thr_resamp=2, nr_thr_saving=2)
            os.replace(dst + '.tmp.nii.gz', dst)
            done += 1
            print(f'{c} {device} {time.time() - t0:.0f} s', flush=True)
        except Exception as ex:                                       # noqa: BLE001
            print('ERROR', c, str(ex)[:300], flush=True)
            os.remove(os.path.join(lock, c))
    print(f'TS worker {device}: {done} cases', flush=True)


def feret_axial(pts):
    """pts (N, 3) voxel coords (1 mm) -> largest in-plane (axial: fixed axis 2) diameter in mm."""
    from scipy.spatial import ConvexHull
    best = 1.0
    for z in np.unique(pts[:, 2]):
        p = pts[pts[:, 2] == z][:, :2].astype(float)
        if len(p) >= 3:
            try:
                p = p[ConvexHull(p).vertices]
            except Exception:                                         # noqa: BLE001  (collinear)
                pass
        if len(p) >= 2:
            best = max(best, np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1)).max() + 1.0)
    return best


def dilate(mask, r):
    """Euclidean dilation by r voxels in the mask's padded bounding box."""
    from scipy import ndimage
    out = np.zeros_like(mask, bool)
    if not mask.any():
        return out
    idx = np.argwhere(mask)
    lo, hi = np.maximum(idx.min(0) - r - 1, 0), np.minimum(idx.max(0) + r + 2, mask.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    out[sl] = ndimage.distance_transform_edt(~mask[sl]) <= r
    return out


def pool4(m):
    """bool (X, Y, Z) at 1 mm -> fraction of each 4 mm block that is True (float32)."""
    s = [int(np.ceil(n / 4)) * 4 for n in m.shape]
    p = np.zeros(s, np.float32)
    p[:m.shape[0], :m.shape[1], :m.shape[2]] = m
    return p.reshape(s[0] // 4, 4, s[1] // 4, 4, s[2] // 4, 4).mean((1, 3, 5))


def find_control(les, par, excl):
    """les: lesion (bool, 1 mm); par: valid parenchyma; excl: every lesion dilated 10 mm.
    -> (ctrl mask at 1 mm, exact inside fraction, exact overlap, distance in mm) or None."""
    from scipy.signal import fftconvolve
    if not par.any():
        return None
    # search inside the parenchyma's bounding box (a valid control lies >= 70% inside it); lesion cut to its own box
    po, lo_l = np.argwhere(par), np.argwhere(les)
    lo_o, hi_o = po.min(0), po.max(0) + 1
    lo_l, hi_l = lo_l.min(0), lo_l.max(0) + 1
    box = tuple(slice(a, b) for a, b in zip(lo_o, hi_o))
    L = pool4(les[tuple(slice(a, b) for a, b in zip(lo_l, hi_l))])
    P, E = pool4(par[box]), pool4(excl[box])
    n = L.sum()
    Lf = L[::-1, ::-1, ::-1]
    fin = fftconvolve(P, Lf, mode='full') / n          # fin[j] = sum_p L(p) P(p + j - (|L| - 1)) / |L|, 4 mm units
    ov = fftconvolve(E, Lf, mode='full') / n
    c = np.asarray(L.shape) - 1
    grid = np.stack(np.meshgrid(*[np.arange(s) - o for s, o in zip(fin.shape, c)], indexing='ij'), -1)
    tr = grid * 4 + (lo_o - lo_l)                       # translation in mm (lesion box origin -> placement)
    dist = np.sqrt((tr.astype(np.float32) ** 2).sum(-1))
    ok = (fin >= MIN_IN_ORGAN) & (ov < MAX_OVERLAP)
    if not ok.any():
        return None
    pref = ok & (fin >= PREF_IN_ORGAN)
    if pref.any():
        j = np.unravel_index(np.where(pref, dist, np.inf).argmin(), fin.shape)
    else:
        j = np.unravel_index(np.where(ok, fin, -1).argmax(), fin.shape)
    t = tr[j]
    pts = np.argwhere(les) + t
    inb = ((pts >= 0) & (pts < np.asarray(les.shape))).all(1)
    m = np.zeros_like(les)
    m[tuple(pts[inb].T)] = True
    nl = max(int(les.sum()), 1)
    return m, float((m & par).sum()) / nl, float((m & excl).sum()) / nl, float(np.sqrt((t ** 2).sum()))
