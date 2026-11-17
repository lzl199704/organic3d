"""Reading CT volumes and masks in either NIfTI or a simple ZST0/RAW0 .bin format.

.bin = b"ZST0" (zstd-compressed) or b"RAW0", then 3 x uint32 shape, then int16 voxels in C order.
It stores no spacing or orientation: by convention these files are 1 mm, RAS, in the same axis order as the NIfTI
arrays they were converted from.
"""
import struct

import numpy as np


def load_bin(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        shape = struct.unpack('3I', f.read(12))
        data = f.read()
    if magic == b'ZST0':
        import zstandard
        data = zstandard.ZstdDecompressor().decompress(data)
    elif magic != b'RAW0':
        raise ValueError(f'{path}: unknown .bin header {magic!r}')
    return np.frombuffer(data, dtype=np.int16).reshape(shape)


def read_image(path):
    """-> float32 array in HU (x, y, z), RAS, 1 mm for our data."""
    if path.endswith('.bin'):
        return load_bin(path).astype(np.float32)
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)


def read_mask(path):
    """-> uint8 TotalSegmentator label array, same grid as the image."""
    if path.endswith('.bin'):
        return load_bin(path).astype(np.uint8)
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj).astype(np.uint8)


def ras_affine_1mm():
    return np.eye(4)


# ---------------------------------------------------------------- raw scans sampled onto the 1 mm mask grid
def image_space(cfg, dataset):
    """'raw' when the env layout reads the original scan and resamples it onto the mask grid (e.g. Merlin when only the raw scans and the 1 mm masks are available)."""
    lay = (cfg.paths.get('layout') or {}).get(dataset) if 'paths' in cfg else None
    return (lay or {}).get('img_space', 'mask')


def sample_raw_on_mask_grid(raw_path, seg_affine, seg_shape, starts, step, offset, out_shape, fill=-1000.0, chunk=24):
    """Trilinear sample of a raw scan at mask-grid voxel positions p = starts + step * k + offset (k = 0..out-1 per axis).
    Positions outside the mask grid get `fill`, as when the 1 mm image is crop-padded. Returns float32 (out_shape)."""
    import nibabel as nib
    import torch
    import torch.nn.functional as F
    nii = nib.load(raw_path)
    raw = torch.from_numpy(np.asarray(nii.dataobj, dtype=np.float32))[None, None]
    M = torch.as_tensor(np.linalg.inv(nii.affine) @ seg_affine, dtype=torch.float64)   # mask voxel -> raw voxel
    axes = [torch.arange(n, dtype=torch.float64) * step + s + offset for n, s in zip(out_shape, starts)]
    size = torch.tensor(raw.shape[2:], dtype=torch.float64)
    seg_hi = torch.tensor(seg_shape, dtype=torch.float64) - 1
    out = torch.empty(out_shape, dtype=torch.float32)
    for a0 in range(0, out_shape[0], chunk):
        P = torch.stack(torch.meshgrid(axes[0][a0:a0 + chunk], axes[1], axes[2], indexing='ij'), -1)     # (c, Y, Z, 3)
        R = P @ M[:3, :3].T + M[:3, 3]
        g = (2 * R / (size - 1) - 1)[..., [2, 1, 0]].float()
        v = F.grid_sample(raw, g[None], mode='bilinear', padding_mode='border', align_corners=True)[0, 0]
        outside = ((P < 0) | (P > seg_hi)).any(-1) | ((R < -0.5) | (R > size - 0.5)).any(-1)
        v[outside] = fill
        out[a0:a0 + chunk] = v
    return out.numpy()


def hu_checks(img, labels):
    """Image-mask alignment check without a reference image: tissue HU inside TotalSegmentator structures.
    labels are raw TotalSegmentator v2 ids on the same grid. Returns (ok, {structure: median HU})."""
    # lungs: abdominal CT shows only the lung bases, where 2 mm voxels mix lung with diaphragm or effusion, so the limit is
    # -300 HU and it needs a clear amount of lung (the other rules need 200 voxels)
    rules = {'liver': ([5], -50, 250, 200), 'spleen': ([1], -50, 250, 200), 'lungs': ([10, 11, 12, 13, 14], None, -300, 3000),
             'vertebrae': (list(range(26, 51)), 100, None, 200)}
    med, ok = {}, True
    for name, (ids, lo, hi, nmin) in rules.items():
        m = np.isin(labels, ids)
        if m.sum() < nmin:
            continue
        v = float(np.median(img[m])); med[name] = round(v, 1)
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            ok = False
    return ok, med
