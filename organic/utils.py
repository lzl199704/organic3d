"""Shared helpers: distributed training and evaluation (torchrun), and AUROC without sklearn."""
import datetime
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as all_gather_grad


# ------------------------------------------------------------------ distributed helpers
def setup_dist(timeout_min=30):
    """Evaluation scripts pass a long timeout: rank 0 alone fits the probes for one dataset (20+ min on an 8-core job)
    while the other ranks wait at the next gather, and NCCL's 10-minute default then aborts the job."""
    if 'RANK' in os.environ:
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=timeout_min))
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ['LOCAL_RANK'])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, torch.device('cuda', local)


def gather_padded(x, grad=True):
    """Concatenate variable-length first dimensions across ranks. Returns (all, counts)."""
    if not dist.is_initialized():
        return x, [len(x)]
    n = torch.tensor([len(x)], device=x.device)
    ns = [torch.zeros_like(n) for _ in range(dist.get_world_size())]
    dist.all_gather(ns, n)
    ns = [int(v) for v in ns]
    m = max(ns)
    pad = torch.zeros((m - len(x),) + tuple(x.shape[1:]), device=x.device, dtype=x.dtype)
    xp = torch.cat([x, pad])
    parts = all_gather_grad(xp) if grad else _gather_nograd(xp)
    return torch.cat([p[:k] for p, k in zip(parts, ns)]), ns


def _gather_nograd(x):
    out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out, x)
    return out


def offset(counts, rank):
    return sum(counts[:rank])


def allreduce_grads(params, world, bucket_mb=256):
    """Average gradients across ranks in fixed-order buckets; missing grads count as zeros on every rank."""
    if world == 1:
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    bucket, size = [], 0
    limit = bucket_mb * 2 ** 20 // 4
    for p in params + [None]:
        if p is not None:
            bucket.append(p); size += p.numel()
        if bucket and (p is None or size >= limit):
            flat = torch.cat([q.grad.reshape(-1).float() for q in bucket])
            dist.all_reduce(flat)
            flat /= world
            o = 0
            for q in bucket:
                n = q.numel(); q.grad.copy_(flat[o:o + n].view_as(q.grad)); o += n
            bucket, size = [], 0


# ------------------------------------------------------------------ AUROC without sklearn
def auroc(y, s):
    ok = ~(np.isnan(y) | np.isnan(s))
    y, s = y[ok], s[ok]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 < 1 or n0 < 1:
        return float('nan')
    r = np.empty(len(s)); r[np.argsort(s, kind='mergesort')] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
