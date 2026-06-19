"""
This code was vibe-coded with Claude Sonnet4.6 (Free version)
But it was our idea to use a C-Abstractor.

Modality Projector — C-Abstractor style (ResBlock -> AdaptiveAvgPool -> ResBlock).

Follows Honeybee's C-Abstractor design (https://arxiv.org/abs/2312.06742):
local feature refinement happens BEFORE any lossy downsampling, average
pooling does the actual compression (parameter-free, position-independent,
and decoupled from the exact patch-grid size), and a second stack of
ResBlocks refines the already-compressed tokens afterward.

    [B, 1024, 768]                                  (ViT output, 32x32 grid)
        -> reshape to [B, 768, 32, 32]
        -> n_pre  x ResBlock (stride 1, full 32x32 res -- expensive, keep small)
        -> AdaptiveAvgPool2d(8)                       -> [B, 768, 8, 8]
        -> 1x1 conv: 768 -> 960
        -> n_post x ResBlock (stride 1, 8x8 res -- cheap, fine to stack more)
        -> flatten                                    -> [B, 64, 960]

Each ResBlock is zero-init on its last conv, so every block starts as the
identity function -- the whole module starts out close to "just
average-pool the raw ViT features," which is a safe, stable starting point
for end-to-end training with no freezing.
"""
import torch
import torch.nn as nn


""" Modified from Claude : 
    Switch normalisation to after the layer (beacause ViT token are normalized already).
    And no gelu after the last layer (we want it to be linear before reaching LM tokens)"""
class ResBlock(nn.Module):
    """3x3 conv residual block: out = x + GN(Conv(GELU(GN(Conv(x)))))"""

    def __init__(self, dim, groups=32, is_last=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),  
            nn.GroupNorm(groups, dim),
        )
        self.is_last = is_last

    def forward(self, x):
        x = x + self.net(x)
        if not self.is_last: # added condition
            x = nn.functional.gelu(x)
        return x

class CAbstractor(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: VLMConfig with:
                cfg.projector.image_token_length = 64   -> target grid = sqrt(64) = 8
                cfg.vit.hidden_dim                = 768
                cfg.lm.hidden_dim                 = 960
            n_pre:  ResBlocks BEFORE pooling, at full ViT resolution (32x32).
                    ~16x more spatial positions than post-pool -> dominant
                    compute cost. Keep this small (1-2) on a tight budget.
            n_post: ResBlocks AFTER pooling, at the compressed resolution (8x8).
                    Cheap -- safe to stack more here for extra capacity.
        """
        super().__init__()
        self.vit_hidden_dim = cfg.vit.hidden_dim
        self.lm_hidden_dim = cfg.lm.hidden_dim

        target_tokens = cfg.projector.image_token_length
        self.target_side = int(target_tokens ** 0.5)
        assert self.target_side ** 2 == target_tokens, "image_token_length must be a perfect square"

        self.pre_blocks = nn.Sequential(*[ResBlock(self.vit_hidden_dim, is_last=False) for _ in range(cfg.n_pre)])
        self.pool = nn.AdaptiveAvgPool2d(self.target_side)
        self.channel_proj = nn.Conv2d(self.vit_hidden_dim, self.lm_hidden_dim, kernel_size=1, bias=False)
        self.post_blocks = nn.Sequential(*[ResBlock(self.lm_hidden_dim, is_last=(i==cfg.n_post - 1)) 
                                           for i in range(cfg.n_post)])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, ResBlock):
            # zero the last conv in the residual branch -> block starts as identity.
            # .apply() hits children before parents, so this runs *after* the
            # kaiming init above and correctly overrides it.
            nn.init.zeros_(module.net[-2].weight)

    def forward(self, x):
        """
        Args:
            x: [B, seq, 768]   (ViT output, seq must be a perfect square)
        Returns:
            [B, target_tokens, 960]
        """
        bsz, seq, embed_dim = x.shape
        side = int(seq ** 0.5)
        assert side * side == seq, "seq must be a perfect square"

        x = x.transpose(1, 2).reshape(bsz, embed_dim, side, side)   # [B, 768, 32, 32]
        x = self.pre_blocks(x)                                      # [B, 768, 32, 32]
        x = self.pool(x)                                            # [B, 768, 8, 8]
        x = self.channel_proj(x)                                    # [B, 960, 8, 8]
        x = self.post_blocks(x)                                     # [B, 960, 8, 8]
        x = x.flatten(2).transpose(1, 2)                            # [B, 64, 960]
        return x