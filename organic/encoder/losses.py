"""Stage-A objectives. Notation follows the method spec (v0.2):

  base : L_i2t, L_t2i (label-aware), L_txt (hard-negative sentences), L_cls (label prompts, anchors Delta_f)
  core : L_ev  = rho(Delta_f(E_{o(f)} E_{U_t} V) - b_f) + eta * rho(Delta_f(E_U V) - sg[Delta_f(V)])
  D1   : L_mir (mirror equivariance for sentences with a side)
  D2   : L_pair (label-matched ranking, pairs found inside the batch)
  D3   : L_anat (organ queries put attention mass inside their organ)
"""
import torch
import torch.nn.functional as F

from ..vocab import FINDING_NAMES


def i2t_loss(S, pos, true_mask, unknown=None):
    """S (V, S) logits; pos (V, S) bool own-report positives; true_mask (V, S) bool sentences consistent with V.
    Multi-positive InfoNCE; sentences true for V but not its own are dropped from the denominator, and so are
    sentences whose truth for V is unknown (unknown (V, S): V's label of the sentence's finding was never asked)."""
    drop = true_mask & ~pos
    if unknown is not None:
        drop = drop | (unknown & ~pos)
    logits = S.masked_fill(drop, float('-inf'))
    logp = logits.log_softmax(1)
    w = pos.float()
    has = w.sum(1) > 0
    return -(logp.masked_fill(~pos, 0).sum(1)[has] / w.sum(1)[has]).mean()


def t2i_loss(S, own, true_mask, lam, unknown=None):
    """Columns: each sentence retrieves the volumes it is true for. own (V, S) bool; soft targets. Volumes for which
    the sentence's truth is unknown are left out of that sentence's softmax (neither target nor negative)."""
    q = own.float() + lam * (true_mask & ~own).float()
    if unknown is not None:
        S = S.masked_fill(unknown & ~own & ~true_mask, float('-inf'))
    has = q.sum(0) > 0
    q = q[:, has] / q[:, has].sum(0, keepdim=True)
    ls = S[:, has].log_softmax(0)
    return -torch.where(q > 0, q * ls, torch.zeros_like(ls)).sum(0).mean()


def txt_loss(s_true, s_negs):
    """s_true (M,), s_negs (M, E) with -inf padding: the true sentence beats its edits for the same image."""
    logits = torch.cat([s_true[:, None], s_negs], 1)
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))


def cls_loss(delta, labels, bias):
    """BCE of sigma(Delta_f + c_f) against labels, only where labels are known."""
    known = ~torch.isnan(labels)
    if not known.any():
        return delta.sum() * 0
    return F.binary_cross_entropy_with_logits((delta + bias)[known], labels[known])


def evidence_loss(delta_clean, delta_sens, delta_inv, f_sens, f_inv, b, eta, huber_delta=1.0):
    """delta_* are the finding logits of the chosen finding for each counterfactual; f_* finding ids."""
    terms = []
    if len(delta_sens):
        terms.append(F.huber_loss(delta_sens, b[f_sens], delta=huber_delta))
    if len(delta_inv):
        terms.append(eta * F.huber_loss(delta_inv, delta_clean.detach(), delta=huber_delta))
    return sum(terms) if terms else torch.zeros((), device=b.device)


def pair_loss(delta, labels, tau):
    """Label-matched ranking inside the batch: for each finding, positive i and negative j whose other known
    labels are identical must satisfy Delta_f(i) > Delta_f(j)."""
    L = torch.nan_to_num(labels, nan=-1)
    losses = []
    for f in range(L.shape[1]):
        pos = torch.where(L[:, f] == 1)[0]; neg = torch.where(L[:, f] == 0)[0]
        if not len(pos) or not len(neg):
            continue
        others = torch.cat([L[:, :f], L[:, f + 1:]], 1)
        same = (others[pos][:, None] == others[neg][None]).all(-1)          # (P, N)
        if not same.any():
            continue
        pi, ni = torch.where(same)
        losses.append(F.softplus(-(delta[pos[pi], f] - delta[neg[ni], f]) / tau).mean())
    return torch.stack(losses).mean() if losses else delta.sum() * 0


def mirror_loss(s_swap, s_orig):
    """On the mirrored volume the side-swapped sentence must beat the original one."""
    logits = torch.stack([s_swap, s_orig], 1)
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))


def anatomy_loss(A, organ, organ_of_query, n_organs, patch=8, eps=1e-6):
    """A (B, K, N) last-layer query attention; organ (B, 192^3). Organ-assigned queries should put their
    attention mass on tokens whose 8^3 patch contains their organ (training-time masks only)."""
    B, g = A.shape[0], organ.shape[-1] // patch
    blocks = organ.reshape(B, g, patch, g, patch, g, patch)
    inside = torch.stack([(blocks == o).any(6).any(4).any(2).reshape(B, -1) for o in range(n_organs)], 1).float()  # (B, O, N)
    oq = organ_of_query.to(A.device)
    assigned = oq >= 0
    if not assigned.any():
        return A.sum() * 0
    tgt = inside[:, oq[assigned]]                                                              # (B, Ka, N)
    mass = (A[:, assigned].float() * tgt).sum(-1)                                              # (B, Ka)
    present = tgt.sum(-1) > 0
    if not present.any():
        return A.sum() * 0
    return -(mass[present] + eps).log().mean()
