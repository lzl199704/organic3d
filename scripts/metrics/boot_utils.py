"""Patient-level bootstrap helpers for scripts/metrics/heal_metrics.py and scripts/metrics/side_metrics.py."""
import numpy as np


def groups(pts):
    up, inv = np.unique(pts, return_inverse=True)
    return [np.where(inv == k)[0] for k in range(len(up))]


def boot(x, mem, B, rng):
    bs = [x[np.concatenate([mem[k] for k in rng.integers(0, len(mem), len(mem))])].mean() for _ in range(B)]
    return np.quantile(bs, [0.025, 0.975])
