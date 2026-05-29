"""Vision Transformer (ViT) encoder — student skeleton.

Architecture overview
─────────────────────
  Input image  [B, 3, 512, 512]
      │
  ViTPatchEmbeddings   Conv2d(16×16) → flatten → + positional embedding
      │  [B, 1024, 768]
  12 × ViTBlock        LayerNorm → ViTAttention → residual
                       LayerNorm → ViTMLP       → residual
      │  [B, 1024, 768]
  LayerNorm
      │
  Output  [B, 1024, 768]   (one 768-dim token per 16×16 patch)

Key numbers (from ViTConfig):
  img_size   = 512   → (512/16)² = 1024 patches
  hidden_dim = 768
  inter_dim  = 3072  (= 4 × 768)
  n_heads    = 12    → head_dim = 768/12 = 64
  n_blocks   = 12

Pretrained weights: ViT.from_pretrained(cfg: ViTConfig)  ← PROVIDED.
That function expects:
  • a single qkv_proj  nn.Linear(768, 3×768)  per block
  • the attribute names below to match exactly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
class ViTPatchEmbeddings(nn.Module):
    """Convert a batch of images into patch embeddings + positional encoding.

    Step 1 — patch extraction:
        Conv2d(kernel=patch_size, stride=patch_size) extracts patches.
        Output: [B, hidden_dim, img_size/p, img_size/p]

    Step 2 — flatten + transpose:
        Flatten spatial dims → [B, hidden_dim, num_patches]
        Transpose            → [B, num_patches, hidden_dim]

    Step 3 — add position embeddings:
        Broadcast-add a learnable [1, num_patches, hidden_dim] tensor.
    """
    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.img_size                                 # 512
        self.patch_size = cfg.patch_size                             # 16
        self.num_patches = (self.img_size // self.patch_size) ** 2  # 1024
        self.hidden_dim = cfg.hidden_dim                             # 768

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.position_embedding = nn.Parameter(
            torch.rand(1, self.num_patches, self.hidden_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [B, 3, img_size, img_size]
        Returns:
            [B, num_patches, hidden_dim]  =  [B, 1024, 768]
        """
        # TODO 1: Apply self.conv(x)
        #         Input:  [B, 3, 512, 512]
        #         Output: [B, hidden_dim, 32, 32]

        # TODO 2: Flatten spatial dims (dim 2 and 3).
        #         Output: [B, hidden_dim, 1024]

        # TODO 3: Transpose dims 1 and 2.
        #         Output: [B, 1024, hidden_dim]

        # TODO 4: Add self.position_embedding (broadcasts over batch dim).
        #         Output: [B, 1024, hidden_dim]

        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class ViTAttention(nn.Module):
    """Multi-head self-attention for the ViT encoder (bidirectional).

    Uses a single combined qkv_proj so that SigLIP2 weights can be loaded
    directly (from_pretrained concatenates the separate Q, K, V matrices).

    Key attributes (DO NOT rename — weight loading depends on them):
        qkv_proj : nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        out_proj  : nn.Linear(hidden_dim, hidden_dim, bias=True)
    """
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads       # 12
        self.hidden_dim = cfg.hidden_dim  # 768
        assert self.hidden_dim % self.n_heads == 0
        self.head_dim = self.hidden_dim // self.n_heads  # 64
        self.dropout = cfg.dropout

        self.qkv_proj = nn.Linear(
            self.hidden_dim, 3 * self.hidden_dim, bias=True
        )
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.sdpa = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x):
        """
        Args:
            x: [B, T, C]  where T=1024, C=768
        Returns:
            [B, T, C]
        """
        B, T, C = x.size()

        # TODO 1: qkv = self.qkv_proj(x)   → [B, T, 3*C]
        #         split into q, k, v, each  → [B, T, C]

        # TODO 2: reshape each to [B, T, n_heads, head_dim]
        #         then transpose to [B, n_heads, T, head_dim]

        # TODO 3: attend.
        #   If self.sdpa:
        #       F.scaled_dot_product_attention(q, k, v,
        #           dropout_p=self.dropout if self.training else 0.0,
        #           is_causal=False)   ← ViT is BIDIRECTIONAL
        #   Else (fallback):
        #       scores = q @ k.T / sqrt(head_dim), softmax, dropout, @ v

        # TODO 4: [B, n_heads, T, head_dim] → [B, T, C]
        #         apply self.out_proj then self.resid_dropout

        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class ViTMLP(nn.Module):
    """Two-layer MLP with GELU used inside each ViT block.

    fc1: hidden_dim → inter_dim   (768 → 3072)
    fc2: inter_dim  → hidden_dim  (3072 → 768)
    """
    def __init__(self, cfg):
        super().__init__()
        self.activation_fn = nn.GELU(approximate='tanh')
        self.fc1 = nn.Linear(cfg.hidden_dim, cfg.inter_dim)
        self.fc2 = nn.Linear(cfg.inter_dim, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        """
        Args/Returns: [B, T, hidden_dim]

        TODO: fc1 → activation_fn → fc2 → dropout
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class ViTBlock(nn.Module):
    """Pre-norm residual block: attention sub-layer then MLP sub-layer."""
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim, eps=cfg.ln_eps)
        self.attn = ViTAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim, eps=cfg.ln_eps)
        self.mlp = ViTMLP(cfg)

    def forward(self, x):
        """
        Args/Returns: [B, T, hidden_dim]

        Pre-norm residual pattern:
            x = x + attn(ln1(x))
            x = x + mlp(ln2(x))
        """
        # TODO: apply the two sub-layers.
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class ViT(nn.Module):
    """Full Vision Transformer encoder."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = ViTPatchEmbeddings(cfg)
        self.cls_flag = cfg.cls_flag
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [ViTBlock(cfg) for _ in range(cfg.n_blocks)]
        )
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim, eps=cfg.ln_eps)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 512, 512]
        Returns:
            [B, 1024, 768]

        TODO 1: self.patch_embedding(x)  → [B, 1024, 768]
        TODO 2: self.dropout
        TODO 3: loop over self.blocks
        TODO 4: self.layer_norm
        TODO 5: return  (cls_flag=False → return all tokens)
        """
        raise NotImplementedError

    # ── Provided: loads pretrained SigLIP2 weights ────────────────────────────
    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import SiglipVisionConfig
        from huggingface_hub import hf_hub_download
        import safetensors

        hf = SiglipVisionConfig.from_pretrained(cfg.model_type)
        cfg.dropout = hf.attention_dropout
        cfg.hidden_dim = hf.hidden_size
        cfg.img_size = hf.image_size
        cfg.inter_dim = hf.intermediate_size
        cfg.ln_eps = hf.layer_norm_eps
        cfg.n_heads = hf.num_attention_heads
        cfg.n_blocks = hf.num_hidden_layers
        cfg.patch_size = hf.patch_size
        model = cls(cfg)

        sf = hf_hub_download(repo_id=cfg.model_type, filename="model.safetensors")
        sd = model.state_dict()

        mapping = {
            'vision_model.embeddings.patch_embedding.weight':
                'patch_embedding.conv.weight',
            'vision_model.embeddings.patch_embedding.bias':
                'patch_embedding.conv.bias',
            'vision_model.embeddings.position_embedding.weight':
                'patch_embedding.position_embedding',
            'vision_model.post_layernorm.weight': 'layer_norm.weight',
            'vision_model.post_layernorm.bias':   'layer_norm.bias',
        }
        for i in range(cfg.n_blocks):
            p = f'vision_model.encoder.layers.{i}'
            b = f'blocks.{i}'
            mapping.update({
                f'{p}.layer_norm1.weight': f'{b}.ln1.weight',
                f'{p}.layer_norm1.bias':   f'{b}.ln1.bias',
                f'{p}.layer_norm2.weight': f'{b}.ln2.weight',
                f'{p}.layer_norm2.bias':   f'{b}.ln2.bias',
                f'{p}.mlp.fc1.weight':     f'{b}.mlp.fc1.weight',
                f'{p}.mlp.fc1.bias':       f'{b}.mlp.fc1.bias',
                f'{p}.mlp.fc2.weight':     f'{b}.mlp.fc2.weight',
                f'{p}.mlp.fc2.bias':       f'{b}.mlp.fc2.bias',
                f'{p}.self_attn.out_proj.weight': f'{b}.attn.out_proj.weight',
                f'{p}.self_attn.out_proj.bias':   f'{b}.attn.out_proj.bias',
            })

        with safetensors.safe_open(sf, framework="pt", device="cpu") as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    t = f.get_tensor(hf_key)
                    if t.shape == sd[our_key].shape:
                        sd[our_key].copy_(t)
                    elif 'position_embedding' in hf_key:
                        sd[our_key].copy_(t.unsqueeze(0))

            for i in range(cfg.n_blocks):
                p = f'vision_model.encoder.layers.{i}.self_attn'
                q = f.get_tensor(f'{p}.q_proj.weight')
                k = f.get_tensor(f'{p}.k_proj.weight')
                v = f.get_tensor(f'{p}.v_proj.weight')
                sd[f'blocks.{i}.attn.qkv_proj.weight'].copy_(
                    torch.cat([q, k, v], dim=0)
                )
                qb = f.get_tensor(f'{p}.q_proj.bias')
                kb = f.get_tensor(f'{p}.k_proj.bias')
                vb = f.get_tensor(f'{p}.v_proj.bias')
                sd[f'blocks.{i}.attn.qkv_proj.bias'].copy_(
                    torch.cat([qb, kb, vb], dim=0)
                )

        model.load_state_dict(sd)
        n = sum(p.numel() for p in model.parameters())
        print(f"Loaded {cfg.model_type} — {n:,} parameters")
        return model
