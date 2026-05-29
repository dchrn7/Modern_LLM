"""Causal Language Model (SmolLM2-360M) — student skeleton.

Architecture overview
─────────────────────
  Input embeddings  [B, T, 960]   (provided by the VLM wrapper)
      │
  32 × LMBlock      RMSNorm → LMAttention (GQA + RoPE + KV cache) → residual
                    RMSNorm → LMMLP (SiLU gate)                    → residual
      │  [B, T, 960]
  RMSNorm
      │
  Output  [B, T, 960]   (caller applies self.head for logits)

Key numbers (from LMConfig):
  hidden_dim  = 960
  inter_dim   = 2560
  n_heads     = 15   (query heads)   head_dim = 960/15 = 64
  n_kv_heads  = 5    (KV heads)      n_kv_groups = 15/5 = 3
  n_blocks    = 32
  vocab_size  = 49153

Pretrained weights: LanguageModel.from_pretrained(cfg: LMConfig)  ← PROVIDED.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalises WITHOUT subtracting the mean:
        out = x * rsqrt( mean(x²) + eps ) * weight
    """
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.hidden_dim))
        self.rms_eps = cfg.rms_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args/Returns: [B, T, hidden_dim]

        Hint: torch.rsqrt(t) computes 1/sqrt(t) element-wise.
              Take the mean over the last dimension (keepdim=True).
        """
        # TODO: irms = rsqrt( mean(x², dim=-1, keepdim=True) + self.rms_eps )
        #       return x * irms * self.weight
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE).

    Instead of adding a positional vector, RoPE *rotates* the query and key
    vectors by an angle proportional to the token's position.  Relative
    distance is preserved in the dot-product, enabling length generalisation.

    How it works
    ────────────
    1.  Assign a frequency to each dimension pair (2i, 2i+1):
            inv_freq[i] = 1 / (base ^ (2i / dim))   i = 0…dim/2-1
        These decay geometrically so early dims rotate fast and later dims
        rotate slowly.

    2.  For position t, compute:
            freqs[t, i] = t × inv_freq[i]            shape [T, dim/2]

    3.  Duplicate along the feature axis:
            emb = cat([freqs, freqs], dim=-1)         shape [T, dim]
        This interleaving is required by rotate_half() below.

    4.  Return cos(emb) and sin(emb) — passed to apply_rotary_pos_embd()
        which rotates Q and K inside each attention layer.

    Key numbers (from LMConfig):
        dim  = hidden_dim / n_heads = 960 / 15 = 64
        base = re_base = 100_000
        inv_freq shape: [dim/2] = [32]
        cos/sin output shape: [B, T, dim] = [B, T, 64]
    """
    def __init__(self, cfg):
        """PROVIDED — precomputes inv_freq and stores config scalars."""
        super().__init__()
        assert cfg.hidden_dim % cfg.n_heads == 0
        self.dim = cfg.hidden_dim // cfg.n_heads  # head_dim = 64
        self.re_base = cfg.re_base                # 100_000
        self.max_position_embeddings = cfg.max_position_embeddings
        self.attn_scaling = cfg.attn_scaling      # 1.0

        # inv_freq[i] = 1 / (re_base ^ (2i / dim)),  shape [dim/2]
        inv_freq = 1.0 / (
            self.re_base ** (
                torch.arange(0, self.dim, 2).float() / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor):
        """
        Args:
            position_ids: [B, T]  integer positions, e.g. [[0,1,…,T-1]]
        Returns:
            cos: [B, T, dim]
            sin: [B, T, dim]

        TODO 1 — Dynamic scaling for long sequences:
            max_seq = position_ids.max() + 1
            if max_seq > self.max_position_embeddings:
                scale = max_seq / self.max_position_embeddings
                inv_freq = self.inv_freq / scale
            else:
                inv_freq = self.inv_freq

        TODO 2 — Per-position frequencies:
            flat = position_ids.reshape(-1).float()  → [B*T]
            freqs = flat.unsqueeze(-1) * inv_freq.unsqueeze(0)  → [B*T, dim/2]
            freqs = freqs.reshape(B, T, -1)          → [B, T, dim/2]

        TODO 3 — Duplicate to full head dim:
            emb = torch.cat([freqs, freqs], dim=-1)  → [B, T, dim]

        TODO 4 — Scale and return:
            cos = torch.cos(emb) * self.attn_scaling
            sin = torch.sin(emb) * self.attn_scaling
            return cos, sin
        """
        raise NotImplementedError


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_embd(q, k, cos, sin, unsqueeze_dim=1):
    """Apply RoPE to query and key tensors.  PROVIDED — do not modify.

    cos/sin: [B, T, dim]  →  unsqueezed to [B, 1, T, dim] for broadcasting.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), \
           (k * cos) + (rotate_half(k) * sin)


# ─────────────────────────────────────────────────────────────────────────────
class LMAttention(nn.Module):
    """Grouped-Query Attention (GQA) with KV caching and RoPE.

    GQA uses fewer KV heads than query heads to save memory.
    n_heads=15, n_kv_heads=5  →  n_kv_groups=3 (each KV head shared by 3 Q).

    KV cache: during generation, past K/V are stored and concatenated with
    the new token's K/V so only one token is processed per decode step.

    Key attribute names (DO NOT rename — from_pretrained depends on them):
        q_proj   nn.Linear(960, 960,          bias=False)
        k_proj   nn.Linear(960, 5×64=320,     bias=False)
        v_proj   nn.Linear(960, 5×64=320,     bias=False)
        out_proj nn.Linear(960, 960,          bias=False)
    """
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads       # 15
        self.n_kv_heads = cfg.n_kv_heads  # 5
        self.hidden_dim = cfg.hidden_dim  # 960
        self.dropout = cfg.dropout

        assert self.n_heads % self.n_kv_heads == 0
        self.n_kv_groups = self.n_heads // self.n_kv_heads  # 3
        self.head_dim = self.hidden_dim // self.n_heads      # 64

        self.q_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )  # 960 → 960
        self.k_proj = nn.Linear(
            self.hidden_dim, self.head_dim * self.n_kv_heads, bias=False
        )  # 960 → 320
        self.v_proj = nn.Linear(
            self.hidden_dim, self.head_dim * self.n_kv_heads, bias=False
        )  # 960 → 320
        self.out_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )  # 960 → 960

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.sdpa = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, cos, sin, attention_mask=None, block_kv_cache=None):
        """
        Args:
            x:              [B, T_curr, 960]
            cos, sin:       [B, T_curr, 64]
            attention_mask: [B, T_total] 1=attend, 0=pad  (optional)
            block_kv_cache: dict {'key': …, 'value': …} or None (prefill)
        Returns:
            output:         [B, T_curr, 960]
            block_kv_cache: updated cache dict

        TODO 1 — Project:
            q: [B, n_heads,    T_curr, head_dim]  via q_proj + reshape + T
            k: [B, n_kv_heads, T_curr, head_dim]  via k_proj + reshape + T
            v: same shape as k                     via v_proj

        TODO 2 — Rotary embeddings:
            q, k_rotated = apply_rotary_pos_embd(q, k, cos, sin)

        TODO 3 — KV cache:
            Prefill  (block_kv_cache is None):
                k = k_rotated;  v = v_curr
                block_kv_cache = {'key': k, 'value': v}
            Decode (cache exists):
                k = cat([block_kv_cache['key'],   k_rotated], dim=2)
                v = cat([block_kv_cache['value'], v_curr],    dim=2)
                block_kv_cache = {'key': k, 'value': v}

        TODO 4 — Expand K/V for GQA:
            k_exp = k.repeat_interleave(n_kv_groups, dim=1) → [B, 15, T_kv, 64]
            v_exp = v.repeat_interleave(n_kv_groups, dim=1)

        TODO 5 — Additive padding mask (optional):
            if attention_mask is not None:
                T_kv = k_exp.size(2)
                mask = attention_mask[:, :T_kv]           # [B, T_kv]
                additive = (1.0 - mask.unsqueeze(1).unsqueeze(2).float())
                           * torch.finfo(q.dtype).min     # [B, 1, 1, T_kv]

        TODO 6 — SDPA:
            is_causal = (T_curr == T_kv and T_curr > 1)
            F.scaled_dot_product_attention(q, k_exp, v_exp,
                attn_mask=additive, dropout_p=…, is_causal=is_causal)

        TODO 7 — Merge heads + project:
            y = y.transpose(1,2).contiguous().view(B, T_curr, C)
            return self.resid_dropout(self.out_proj(y)), block_kv_cache
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class LMMLP(nn.Module):
    """Gated MLP (SwiGLU-style).

    gate_proj: hidden_dim → inter_dim   (960 → 2560)
    up_proj:   hidden_dim → inter_dim   (960 → 2560)
    down_proj: inter_dim  → hidden_dim  (2560 → 960)

    Output = down_proj( silu(gate_proj(x)) × up_proj(x) )
    """
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_dim, cfg.inter_dim, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_dim, cfg.inter_dim, bias=False)
        self.down_proj = nn.Linear(cfg.inter_dim, cfg.hidden_dim, bias=False)

    def forward(self, x):
        """
        Args/Returns: [B, T, hidden_dim]

        TODO: gate = F.silu(self.gate_proj(x))
              return self.down_proj(gate * self.up_proj(x))
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class LMBlock(nn.Module):
    """Pre-norm residual block.  Attention returns (output, cache) — unpack!"""
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg)
        self.attn = LMAttention(cfg)
        self.norm2 = RMSNorm(cfg)
        self.mlp = LMMLP(cfg)

    def forward(self, x, cos, sin, attention_mask=None, block_kv_cache=None):
        """
        Args:
            x:              [B, T, hidden_dim]
            cos, sin:       [B, T, head_dim]
            attention_mask: [B, T_total] or None
            block_kv_cache: dict or None
        Returns:
            x:              [B, T, hidden_dim]
            block_kv_cache: updated dict

        Pattern (same as ViTBlock, but attention also returns the cache):
            res = x
            x = self.norm1(x)
            x, block_kv_cache = self.attn(x, cos, sin, attention_mask, …)
            x = res + x
            …
        """
        # TODO: two pre-norm residual sub-layers.
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class LanguageModel(nn.Module):
    """Full causal language model.

    forward() receives embeddings (not token ids) when called from the VLM,
    because image placeholder tokens have already been replaced with visual
    embeddings by VisionLanguageModel._replace_img_tokens_with_embd().

    self.head is applied externally by the VLM only when computing loss.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tie_weights = cfg.tie_weights

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.rotary_embd = RotaryEmbedding(cfg)
        self.blocks = nn.ModuleList(
            [LMBlock(cfg) for _ in range(cfg.n_blocks)]
        )
        self.norm = RMSNorm(cfg)
        self.head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

        if self.tie_weights:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask=None,
        kv_cache=None,
        start_pos: int = 0,
    ):
        """Run the language model on a sequence of embeddings.

        Args:
            x:              [B, T, hidden_dim]  — embeddings (NOT token ids)
            attention_mask: [B, T_total] 1=attend, 0=pad  (or None)
            kv_cache:       list of per-block dicts, or None (prefill)
            start_pos:      position of first token in x
                            (0 during prefill; total_seq_len-1 during decode)
        Returns:
            hidden:   [B, T, hidden_dim]
            kv_cache: updated list of per-block dicts

        TODO 1: B, T_curr, _ = x.size()

        TODO 2: position_ids = arange(start_pos, start_pos+T_curr)
                broadcast to [B, T_curr], call self.rotary_embd(position_ids)

        TODO 3: if kv_cache is None: kv_cache = [None] * len(self.blocks)

        TODO 4: for i, block in enumerate(self.blocks):
                    x, kv_cache[i] = block(x, cos, sin, attention_mask, …)

        TODO 5: x = self.norm(x)

        TODO 6: return x, kv_cache
        """
        raise NotImplementedError

    # ── Provided: greedy generation for the standalone LM ────────────────────
    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor, max_new_tokens: int = 20):
        """Greedy autoregressive generation from token ids.  PROVIDED."""
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        generated = inputs.clone()
        out, kv = self.forward(
            self.token_embedding(generated), kv_cache=None, start_pos=0
        )
        last = self.head(out[:, -1, :])
        for i in range(max_new_tokens):
            next_tok = torch.argmax(last, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            pos = generated.size(1) - 1
            if i == max_new_tokens - 1:
                break
            out, kv = self.forward(
                self.token_embedding(next_tok), kv_cache=kv, start_pos=pos
            )
            last = self.head(out[:, -1, :])
        return generated

    # ── Provided: loads pretrained SmolLM2-360M weights ──────────────────────
    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        import safetensors
        import json
        import torch.nn.init as init

        hf = AutoConfig.from_pretrained(cfg.model_type)
        original_vocab = hf.vocab_size

        cfg.hidden_dim = hf.hidden_size
        cfg.inter_dim = hf.intermediate_size
        cfg.rms_eps = hf.rms_norm_eps
        cfg.re_base = hf.rope_parameters.get('rope_theta', 100000)
        cfg.max_position_embeddings = hf.max_position_embeddings
        cfg.n_heads = hf.num_attention_heads
        cfg.n_kv_heads = hf.num_key_value_heads
        cfg.dropout = hf.attention_dropout
        cfg.n_blocks = hf.num_hidden_layers

        if cfg.vocab_size < original_vocab:
            raise ValueError(
                f"cfg.vocab_size ({cfg.vocab_size}) < pretrained "
                f"({original_vocab})"
            )

        model = cls(cfg)

        try:
            idx = hf_hub_download(
                repo_id=cfg.model_type,
                filename="model.safetensors.index.json",
            )
            with open(idx) as f:
                index = json.load(f)
            fnames = sorted(set(index['weight_map'].values()))
            sf_files = [
                hf_hub_download(repo_id=cfg.model_type, filename=fn)
                for fn in fnames
            ]
        except EntryNotFoundError:
            sf_files = [
                hf_hub_download(
                    repo_id=cfg.model_type, filename="model.safetensors"
                )
            ]

        sd = model.state_dict()
        mapping = {
            'model.embed_tokens.weight': 'token_embedding.weight',
            'model.norm.weight':         'norm.weight',
        }
        for i in range(cfg.n_blocks):
            lp = f'model.layers.{i}.'
            bp = f'blocks.{i}.'
            mapping.update({
                f'{lp}self_attn.q_proj.weight': f'{bp}attn.q_proj.weight',
                f'{lp}self_attn.k_proj.weight': f'{bp}attn.k_proj.weight',
                f'{lp}self_attn.v_proj.weight': f'{bp}attn.v_proj.weight',
                f'{lp}self_attn.o_proj.weight': f'{bp}attn.out_proj.weight',
                f'{lp}mlp.gate_proj.weight': f'{bp}mlp.gate_proj.weight',
                f'{lp}mlp.up_proj.weight': f'{bp}mlp.up_proj.weight',
                f'{lp}mlp.down_proj.weight': f'{bp}mlp.down_proj.weight',
                f'{lp}input_layernorm.weight': f'{bp}norm1.weight',
                f'{lp}post_attention_layernorm.weight': f'{bp}norm2.weight',
            })

        loaded = set()
        for sf in sf_files:
            with safetensors.safe_open(sf, framework="pt", device="cpu") as f:
                for hf_key, our_key in mapping.items():
                    if our_key in loaded or hf_key not in f.keys():
                        continue
                    if our_key not in sd:
                        continue
                    t = f.get_tensor(hf_key)
                    if (hf_key == 'model.embed_tokens.weight'
                            and t.shape[0] != sd[our_key].shape[0]):
                        sd[our_key][:t.shape[0]].copy_(t)
                        init.normal_(
                            sd[our_key][t.shape[0]:], mean=0.0, std=0.02
                        )
                        print(
                            f"Extended embeddings: "
                            f"{t.shape[0]} → {sd[our_key].shape[0]}"
                        )
                    elif t.shape == sd[our_key].shape:
                        sd[our_key].copy_(t)
                    loaded.add(our_key)

        model.load_state_dict(sd)
        if cfg.tie_weights:
            model.head.weight = model.token_embedding.weight

        n = sum(p.numel() for p in model.parameters())
        print(f"Loaded {cfg.model_type} — {n:,} parameters")
        return model
