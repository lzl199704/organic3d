"""Stage B model: 3D evidence tokens -> projector -> Qwen3.5-4B language model (LoRA).

The encoder is the stage-A Organic3D (or released COLIPRI with the same query head), producing K position-aware tokens.
Its 2D vision tower is dropped from Qwen3.5, as in NV-Reason-CT. Visual tokens sit between <|vision_start|> and
<|vision_end|> in the user turn; positions are plain 1D (our tokens carry their own 3D position encoding).

Sequence layout (identical prefix length for every sample, so generation needs no padding):
    <|im_start|>user\\n<|vision_start|> [K visual] <|vision_end|> {question}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n {answer}<|im_end|>
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

LLM_ID = 'Qwen/Qwen3.5-4B'
LORA_TARGETS = r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)'


class Organic3DVLM(nn.Module):
    def __init__(self, cfg, encoder):
        super().__init__()
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        from peft import LoraConfig, get_peft_model
        c = cfg.vlm
        self.cfg = cfg
        self.encoder = encoder
        # 'all': encoder fixed (stage-A checkpoint) · 'backbone': ViT fixed, query head trained (released COLIPRI) · 'none'
        self.freeze_mode = str(c.freeze_encoder)
        for n, p in self.encoder.named_parameters():
            p.requires_grad = self.freeze_mode == 'none' or (self.freeze_mode == 'backbone' and n.startswith('queries.'))
        base = Qwen3_5ForConditionalGeneration.from_pretrained(c.get('llm', LLM_ID), dtype=torch.bfloat16,
                                                                attn_implementation='sdpa')
        del base.model.visual                                   # 2D tower unused
        if c.get('grad_checkpointing', True):
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            base.config.use_cache = False
        lcfg = LoraConfig(r=c.lora_r, lora_alpha=c.lora_alpha, lora_dropout=c.lora_dropout, target_modules=LORA_TARGETS)
        self.llm = get_peft_model(base, lcfg)
        self.tok = AutoTokenizer.from_pretrained(c.get('llm', LLM_ID))
        d_llm = base.config.text_config.hidden_size
        d_enc = cfg.model.dim
        self.proj = nn.Sequential(nn.LayerNorm(d_enc), nn.Linear(d_enc, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))
        self.readout = nn.Linear(d_llm, c.get('n_findings', 39)) if c.get('readout_layer') else None
        # A (organ tags): a learned embedding per organ group (16 organs + 1 for global queries), added to each visual
        # token so the LLM knows which tokens belong to which organ. Zero-initialised: at step 0 the model equals S1.
        oq = self.encoder.queries.organ_of_query().clone()
        n_org = int(oq.max()) + 1 if (oq >= 0).any() else 0
        oq[oq < 0] = n_org
        self.register_buffer('token_organ', oq, persistent=False)                  # (K,) organ id per visual token
        self.organ_emb = nn.Embedding(n_org + 1, d_llm) if c.get('organ_tags', False) else None
        if self.organ_emb is not None:
            nn.init.zeros_(self.organ_emb.weight)
        self.att_layers = [int(x) for x in c.get('att_layers', [11, 15, 19, 23])]
        t = self.tok
        self.pre_ids = t('<|im_start|>user\n<|vision_start|>', add_special_tokens=False).input_ids
        self.mid_ids = t('<|vision_end|>', add_special_tokens=False).input_ids
        self.asst = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        self.eos = '<|im_end|>'
        self.pad_to = int(c.get('pad_to', 64))
        self.yes_id = t('Yes', add_special_tokens=False).input_ids[0]
        self.no_id = t('No', add_special_tokens=False).input_ids[0]

    # ---------------- vision
    def visual_tokens(self, x):
        """x (B,1,192,192,192) normalised -> projected tokens (B, K, d_llm) and raw tokens (B, K, d_enc)."""
        enc = self.encoder
        if self.freeze_mode == 'none':
            T, _, _ = enc.encode_image(x)
        else:
            with torch.no_grad():
                h = enc.backbone.down_projection(x).flatten(2).transpose(1, 2)
                h = enc.run_eva(h)
            if self.freeze_mode == 'all':
                with torch.no_grad():
                    T, _, _ = enc.queries(h, enc.pos.to(h.device))
            else:
                T, _, _ = enc.queries(h, enc.pos.to(h.device))
        V = self.proj(T.to(self.proj[1].weight.dtype))
        if self.organ_emb is not None:
            V = V + self.organ_emb(self.token_organ.to(V.device))[None].to(V.dtype)
        return V, T

    def embed_ids(self, ids, device):
        emb = self.llm.get_input_embeddings()
        return emb(torch.as_tensor(ids, device=device))

    # ---------------- sequences
    def build(self, V, questions, answers, device, pad=True):
        """V (B,K,d) projected tokens; questions/answers lists of str (answer may be '' for generation).
        Returns inputs_embeds (B,L,d), attention_mask (B,L), labels (B,L) with -100 outside the answer, ans_start (B,)."""
        B, K, d = V.shape
        pre = self.embed_ids(self.pre_ids, device)
        mid = self.embed_ids(self.mid_ids, device)
        rows, labs, starts = [], [], []
        for b in range(B):
            q_ids = self.tok(questions[b] + self.asst, add_special_tokens=False).input_ids
            a_ids = self.tok(answers[b] + self.eos, add_special_tokens=False).input_ids if answers[b] else []
            q = self.embed_ids(q_ids, device)
            a = self.embed_ids(a_ids, device) if a_ids else V.new_zeros(0, d)
            e = torch.cat([pre, V[b].to(pre.dtype), mid, q, a], 0)
            n_ctx = len(self.pre_ids) + K + len(self.mid_ids) + len(q_ids)
            lab = torch.full((e.shape[0],), -100, dtype=torch.long, device=device)
            if a_ids:
                lab[n_ctx:] = torch.as_tensor(a_ids, device=device)
            rows.append(e); labs.append(lab); starts.append(n_ctx)
        L = max(r.shape[0] for r in rows)
        L = -(-L // self.pad_to) * self.pad_to if pad else L                       # few distinct shapes: Triton autotunes per shape
        emb = pre.new_zeros(B, L, d)
        mask = torch.zeros(B, L, dtype=torch.long, device=device)
        labels = torch.full((B, L), -100, dtype=torch.long, device=device)
        for b, (r, l) in enumerate(zip(rows, labs)):                 # right padding (causal: content unaffected)
            emb[b, :r.shape[0]] = r; mask[b, :r.shape[0]] = 1; labels[b, :l.shape[0]] = l
        return emb, mask, labels, torch.as_tensor(starts, device=device)

    def hidden(self, emb, mask, all_layers=False):
        """Final hidden states (B, L, d) of the language model, without the vocabulary projection (248k vocab)."""
        base = self.llm.get_base_model()
        out = base.model.language_model(inputs_embeds=emb, attention_mask=mask, use_cache=False,
                                        output_hidden_states=all_layers)
        return out.last_hidden_state, (out.hidden_states if all_layers else None)

    def head(self):
        return self.llm.get_base_model().lm_head

    def visual_attention(self, hs, mask, q_pos):
        """Attention of the position that predicts the first answer token over the K visual tokens, per full-attention
        layer in self.att_layers: (B, n_layers, heads, K), each row normalised over ALL keys up to that position (so a
        row sums to the share of attention the visual tokens get). Recomputed from each layer's input hidden state with
        the layer's own norm, projections and rotary embedding, outside gradient checkpointing (no hooks)."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        lm = self.llm.get_base_model().model.language_model
        B, L, _ = hs[0].shape
        K, v0 = self.token_organ.numel(), len(self.pre_ids)
        pos = torch.arange(L, device=mask.device).view(1, 1, -1).expand(3, B, -1)
        keep = (torch.arange(L, device=mask.device)[None] <= q_pos[:, None]) & mask.bool()       # causal + padding
        out = []
        for li in self.att_layers:
            layer = lm.layers[li]
            att = layer.self_attn
            x = layer.input_layernorm(hs[li])
            cos, sin = lm.rotary_emb(x, pos)
            q, _gate = torch.chunk(att.q_proj(x).view(B, L, -1, att.head_dim * 2), 2, dim=-1)
            q = att.q_norm(q).transpose(1, 2)                                                        # (B, h, L, d)
            k = att.k_norm(att.k_proj(x).view(B, L, -1, att.head_dim)).transpose(1, 2)               # (B, kv, L, d)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            q = q[torch.arange(B, device=q.device), :, q_pos]                                         # (B, h, d)
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)                                 # (B, h, L, d)
            logit = torch.einsum('bhd,bhld->bhl', q.float(), k.float()) * att.scaling
            p = logit.masked_fill(~keep[:, None], float('-inf')).softmax(-1)
            out.append(p[:, :, v0:v0 + K])
        return torch.stack(out, 1)

    def token_logprob(self, hs, tgt, chunk=512):
        """log p(tgt | hs) per token. Short answers in one pass; long ones (reports) in checkpointed chunks, so the
        248k-vocabulary logits of a whole report are never stored for backward (same values, less memory)."""
        def f(x, t):
            return torch.log_softmax(self.head()(x).float(), -1).gather(-1, t[:, None])[:, 0]
        if hs.shape[0] <= chunk:
            return f(hs, tgt)
        from torch.utils.checkpoint import checkpoint
        return torch.cat([checkpoint(f, hs[s:s + chunk], tgt[s:s + chunk], use_reentrant=False)
                          for s in range(0, hs.shape[0], chunk)])

    def answer_logprob(self, V, questions, answers, device, hidden_layer=None, attention=False):
        """Sum of teacher-forced log-probs of each answer (B,), mean token CE (scalar), optional hidden states.
        The vocabulary projection is applied only at answer positions. attention=True also returns the visual attention
        of the first answer prediction (see visual_attention) as a 4th value."""
        emb, mask, labels, starts = self.build(V, questions, answers, device)
        h, hs = self.hidden(emb, mask, all_layers=(hidden_layer is not None) or attention)
        tgt = labels[:, 1:]
        valid = tgt != -100
        b_idx, t_idx = valid.nonzero(as_tuple=True)
        tok_lp = self.token_logprob(h[:, :-1][b_idx, t_idx], tgt[b_idx, t_idx])
        seq_lp = torch.zeros(V.shape[0], device=device, dtype=tok_lp.dtype).index_add(0, b_idx, tok_lp)
        ce = -tok_lp.mean() if len(tok_lp) else seq_lp.sum() * 0
        hl = hs[hidden_layer] if hidden_layer is not None else None
        if attention:
            return seq_lp, ce, hl, self.visual_attention(hs, mask, starts - 1)
        return seq_lp, ce, hl

    @torch.no_grad()
    def p_yes(self, V, questions, device):
        """P(Yes) against No at the first answer position (B,)."""
        emb, mask, _, _ = self.build(V, questions, [''] * len(questions), device)
        h, _ = self.hidden(emb, mask)
        last = mask.sum(1) - 1
        w = self.head().weight[[self.yes_id, self.no_id]]
        two = (h[torch.arange(len(questions), device=device), last] @ w.T).float()
        return two.softmax(-1)[:, 0]

    @torch.no_grad()
    def generate(self, V, prompt, device, max_new_tokens=256, prefix=None):
        """Greedy continuation. prefix: text forced at the start of the assistant turn (the same for every row, e.g. the
        answer-then-write finding list); the returned text is the continuation only."""
        emb, mask, _, _ = self.build(V, [prompt] * V.shape[0], [''] * V.shape[0], device, pad=False)
        if prefix:
            p = self.embed_ids(self.tok(prefix, add_special_tokens=False).input_ids, device)
            emb = torch.cat([emb, p[None].expand(emb.shape[0], -1, -1).to(emb.dtype)], 1)
            mask = torch.cat([mask, mask.new_ones(mask.shape[0], p.shape[0])], 1)
        extra = {}                                     # repetition control (2026-10-25 test); unset = plain greedy
        if int(os.environ.get('GEN_NO_REPEAT_NGRAM', '0')) > 0:
            extra['no_repeat_ngram_size'] = int(os.environ['GEN_NO_REPEAT_NGRAM'])
        if float(os.environ.get('GEN_REP_PENALTY', '1')) != 1.0:
            extra['repetition_penalty'] = float(os.environ['GEN_REP_PENALTY'])
        out = self.llm.generate(inputs_embeds=emb, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                                eos_token_id=self.tok.convert_tokens_to_ids('<|im_end|>'), pad_token_id=self.tok.pad_token_id,
                                use_cache=True, **extra)
        return self.tok.batch_decode(out, skip_special_tokens=True)

    def trainable_state(self):
        """Only what stage B trains: LoRA, projector, read-out (and the encoder if unfrozen)."""
        keep = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                keep[n] = p.detach().cpu()
        return keep


def sync_encoder_cfg(cfg):
    """Use the model section of the stage-A run that produced the encoder checkpoint (queries, pooling, K ...)."""
    ck = cfg.vlm.get('encoder_ckpt')
    if ck and ck != 'colipri':
        p = os.path.join(os.path.dirname(ck), 'config.yaml')
        if os.path.exists(p):
            from omegaconf import OmegaConf
            cfg.model = OmegaConf.merge(cfg.model, OmegaConf.load(p).model)
    return cfg


def build_encoder(cfg, device):
    """Stage-A encoder: Organic3D from a checkpoint (cfg.vlm.encoder_ckpt) or released COLIPRI + fresh query head."""
    from .model import build_model
    enc = build_model(cfg)
    ck = cfg.vlm.get('encoder_ckpt')
    if ck and ck != 'colipri':
        state = torch.load(ck, map_location='cpu', weights_only=False)['model']
        missing, unexpected = enc.load_state_dict(state, strict=False)
        print('encoder loaded', ck, 'missing', len(missing), 'unexpected', len(unexpected), flush=True)
    return enc.to(device)
