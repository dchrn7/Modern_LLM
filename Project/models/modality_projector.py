"""Modality Projector — student skeleton.

Bridges the Vision Encoder and the Language Model by reducing the number of
visual tokens and mapping their dimension to the LM's hidden dimension.

Pixel shuffle (PROVIDED below) groups neighbouring ViT patches spatially:
    [B, 1024, 768]  →  [B, 64, 768 × 16]  =  [B, 64, 12288]

A single linear layer then maps each token to the LM dimension:
    [B, 64, 12288]  →  [B, 64, 960]

The output (64 tokens per image, each 960-dim) is used to replace the 64
<|image|> placeholder tokens in the LM's input sequence.
"""

import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: VLMConfig with:
                cfg.projector.pixel_shuffle_factor = 4
                cfg.projector.image_token_length   = 64   (= 1024 / 4²)
                cfg.vit.hidden_dim                 = 768
                cfg.lm.hidden_dim                  = 960

        Hint — pixel_shuffle groups (scale_factor × scale_factor) neighbouring
        ViT patches.  Each output token therefore combines scale_factor² patches,
        so its embedding dimension becomes:

            input_dim = vit.hidden_dim × (projector.pixel_shuffle_factor²)
                      = 768 × 16 = 12288

        After pixel_shuffle:  [B, projector.image_token_length, input_dim]
        After self.proj:       [B, projector.image_token_length, lm.hidden_dim]

        TODO: store the necessary attributes and create self.proj (nn.Linear).
              No bias is needed.
        """
        super().__init__()
        self.pixel_shuffle_factor = cfg.projector.pixel_shuffle_factor  # 4

        # TODO: compute self.input_dim  = cfg.vit.hidden_dim × (self.pixel_shuffle_factor²)
        # TODO: set    self.output_dim  = cfg.lm.hidden_dim
        # TODO: create self.proj        = nn.Linear(self.input_dim, self.output_dim, bias=False)

        raise NotImplementedError

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── PROVIDED — pixel shuffle spatial downsampling ─────────────────────────
    # Do NOT modify this method.  It rearranges ViT patch tokens so that
    # pixel_shuffle_factor² neighbouring patches are merged into a single token.
    def pixel_shuffle(self, x):
        """[B, seq, d]  →  [B, seq/f², d*f²]  where f = pixel_shuffle_factor"""
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq ** 0.5)
        assert seq_root ** 2 == seq, "seq must be a perfect square"
        assert seq_root % self.pixel_shuffle_factor == 0

        h = w = seq_root
        h_out = h // self.pixel_shuffle_factor
        w_out = w // self.pixel_shuffle_factor

        x = x.view(bsz, h, w, embed_dim)
        x = x.reshape(bsz, h_out, self.pixel_shuffle_factor, w_out, self.pixel_shuffle_factor, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.pixel_shuffle_factor ** 2)
        return x

    def forward(self, x):
        """
        Args:
            x: [B, 1024, 768]   (ViT output)
        Returns:
            [B, 64, 960]        (projected tokens ready for LM)

        TODO: apply self.pixel_shuffle(x), then self.proj.
        """
        # TODO: x = self.pixel_shuffle(x)   → [B, 64, 12288]
        # TODO: x = self.proj(x)             → [B, 64, 960]
        # TODO: return x
        raise NotImplementedError
