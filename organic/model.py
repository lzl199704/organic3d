"""ORGANIC-3D stage-A model (the encoder; run id enc4_organic3d for the final model).

    H = E_phi(V)                      COLIPRI Primus backbone, 24^3 patch tokens of width 864
    T = AnatomyQueries(H)             K = n_organs * q_per_organ + q_global tokens of width d, plus PE(position)
    t_s = G_psi(s)                    COLIPRI text tower (CXR-BERT + attention pool), width 768
    S(V, s) = cos(g(v(V,s)), h(t_s)) / tau,   v(V,s) = sum_k beta_k(s) T_k   (sentence-conditioned pooling)

`pool='mean'` replaces v(V,s) by the mean of T (ablation), `queries='global'` uses only unassigned queries.
`pool='fixed'` (ablation, 2026-10-16): beta may only use the slots of the organs the sentence names (organic/routing.py);
the route travels as K extra columns appended to the text embedding by attach_route().
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .vocab import ORGAN_NAMES

GRID = 24


def fourier_pe(pos, dim, n_freq=16):
    """pos (..., 3) in [-1, 1] -> (..., dim) via sin/cos Fourier features + linear map done by caller."""
    freqs = 2.0 ** torch.arange(n_freq, device=pos.device, dtype=pos.dtype) * math.pi
    ang = pos[..., None] * freqs                                  # (..., 3, n_freq)
    return torch.cat([ang.sin(), ang.cos()], -1).flatten(-2)       # (..., 6 * n_freq)


def grid_positions(device, dtype=torch.float32):
    ax = (torch.arange(GRID, device=device, dtype=dtype) + 0.5) / GRID * 2 - 1
    gx, gy, gz = torch.meshgrid(ax, ax, ax, indexing='ij')         # token order (x y z), as in Primus
    return torch.stack([gx, gy, gz], -1).reshape(-1, 3)             # (N, 3)


class CrossBlock(nn.Module):
    def __init__(self, d, heads, kv_dim):
        super().__init__()
        self.nq, self.nkv = nn.LayerNorm(d), nn.LayerNorm(kv_dim)
        self.attn = nn.MultiheadAttention(d, heads, kdim=kv_dim, vdim=kv_dim, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, kv, need_weights=False):
        a, w = self.attn(self.nq(q), self.nkv(kv), self.nkv(kv), need_weights=need_weights, average_attn_weights=True)
        q = q + a
        return q + self.ffn(q), w


class AnatomyQueries(nn.Module):
    def __init__(self, kv_dim, d, n_organs, q_per_organ, q_global, layers=2, heads=12, use_pe=True):
        super().__init__()
        self.K_org, self.K_glob = n_organs * q_per_organ, q_global
        self.q_per_organ, self.use_pe = q_per_organ, use_pe
        self.queries = nn.Parameter(torch.randn(self.K_org + q_global, d))   # unit scale: the LayerNorm in front would amplify small inits
        self.kv_pe = nn.Linear(96, kv_dim)
        self.blocks = nn.ModuleList([CrossBlock(d, heads, kv_dim) for _ in range(layers)])
        self.out_pe = nn.Linear(96, d)

    def organ_of_query(self):
        """query index -> organ id (or -1 for global queries)."""
        o = torch.arange(self.K_org) // self.q_per_organ
        return torch.cat([o, torch.full((self.K_glob,), -1)])

    def forward(self, H, pos):
        """H (B, N, kv_dim), pos (N, 3). Returns T (B, K, d), A (B, K, N) last-layer attention, pbar (B, K, 3)."""
        pe = fourier_pe(pos, 96)
        kv = H + self.kv_pe(pe.to(H.dtype))[None]
        q = self.queries[None].expand(H.shape[0], -1, -1)
        A = None
        for i, blk in enumerate(self.blocks):
            q, A = blk(q, kv, need_weights=(i == len(self.blocks) - 1))
        pbar = torch.einsum('bkn,nc->bkc', A.float(), pos.float())
        if self.use_pe:
            q = q + self.out_pe(fourier_pe(pbar, 96).to(q.dtype))
        return q, A, pbar


class Organic3D(nn.Module):
    def __init__(self, cfg, colipri_model):
        super().__init__()
        m = cfg.model
        enc = colipri_model.image_encoder
        self.backbone = enc.backbone                               # Primus (decoder removed by COLIPRI)
        self.grad_ckpt = bool(m.grad_checkpointing)   # applied in run_eva (the library's own flag breaks on torch>=2.4)
        self.text = colipri_model.text_encoder                     # CXR-BERT + attention pool
        d, kv = m.dim, m.backbone_dim
        n_org = len(ORGAN_NAMES)
        qpo = m.q_per_organ if m.queries == 'anatomy' else 0
        qg = m.q_global if m.queries == 'anatomy' else m.q_per_organ * n_org + m.q_global
        self.queries = AnatomyQueries(kv, d, n_org, qpo, qg, layers=m.query_layers, heads=m.heads, use_pe=m.token_pe)
        self.pool_mode = m.pool
        self.W_a, self.W_b = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.g = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.h = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / m.init_temperature)))
        from .vocab import FINDING_NAMES
        self.ev_bias = nn.Parameter(torch.zeros(len(FINDING_NAMES)))    # b_f: prior target after erasure
        self.cls_bias = nn.Parameter(torch.zeros(len(FINDING_NAMES)))   # c_f: offset for label-prompt BCE
        self.register_buffer('pos', grid_positions('cpu'), persistent=False)

    # ---------------- image side
    def encode_image(self, vol):
        """vol (B, 1, 192, 192, 192) normalised to [-1, 1] -> T (B, K, d), A (B, K, N), pbar (B, K, 3)."""
        x = self.backbone.down_projection(vol)
        B = x.shape[0]
        x = x.flatten(2).transpose(1, 2)                           # (B, N, C) in (x y z) order
        x = self.run_eva(x)
        return self.queries(x, self.pos.to(x.device))

    def run_eva(self, x):
        eva = self.backbone.eva
        x, rope, _ = eva._pos_embed(x)
        for blk in eva.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                x = checkpoint(lambda t, b=blk: b(t, rope=rope), x, use_reentrant=False)
            else:
                x = blk(x, rope=rope)
        return eva.norm(x)

    def param_groups_for_logging(self):
        return {'backbone': list(self.backbone.parameters()), 'text': list(self.text.parameters()),
                'queries': list(self.queries.parameters()),
                'heads': [p for n, p in self.named_parameters() if n.split('.')[0] in ('W_a', 'W_b', 'g', 'h')],
                'scalars': [self.logit_scale, self.ev_bias, self.cls_bias]}

    # ---------------- text side
    def encode_text(self, token_ids, attn_mask):
        return self.text(token_ids, attn_mask, pool=True, normalize=False)   # (S, 768)

    def attach_route(self, t, strings):
        """pool='fixed' only: append each sentence's allowed-slot mask (K columns) to its text embedding; else no-op."""
        if self.pool_mode != 'fixed':
            return t
        from .routing import route_matrix
        R = route_matrix(strings, self.queries.organ_of_query(), t.device).to(t.dtype)
        return torch.cat([t, R], -1)

    # ---------------- scores
    def score(self, T, t):
        """T (V, K, d), t (S, d) -> S (V, S) similarity logits with sentence-conditioned pooling."""
        if self.pool_mode == 'mean':
            v = T.mean(1)                                          # (V, d)
            a = F.normalize(self.g(v), dim=-1) @ F.normalize(self.h(t), dim=-1).T
            return a * self.logit_scale.exp().clamp(max=100)
        d = T.shape[-1]
        R = None
        if self.pool_mode == 'fixed':                                   # route columns appended by attach_route
            t, R = t[:, :-T.shape[1]], t[:, -T.shape[1]:]
        logits = torch.einsum('vkd,sd->vsk', self.W_b(T), self.W_a(t)) / math.sqrt(d)
        if R is not None:
            logits = logits.masked_fill((R <= 0)[None], float('-inf'))
        beta = logits.float().softmax(-1).to(T.dtype)
        v = torch.einsum('vsk,vkd->vsd', beta, T)                  # (V, S, d)
        gv = F.normalize(self.g(v), dim=-1)
        ht = F.normalize(self.h(t), dim=-1)
        return (gv * ht[None]).sum(-1) * self.logit_scale.exp().clamp(max=100)


def build_model(cfg):
    from colipri import get_model
    base = get_model(pretrained=cfg.model.pretrained)
    model = Organic3D(cfg, base)
    if cfg.model.freeze_backbone_blocks:
        n = int(cfg.model.freeze_backbone_blocks)
        for p in model.backbone.down_projection.parameters():
            p.requires_grad = False
        for blk in model.backbone.eva.blocks[:n]:
            for p in blk.parameters():
                p.requires_grad = False
    return model
