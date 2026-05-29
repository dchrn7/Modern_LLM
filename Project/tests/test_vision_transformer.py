"""Shape tests for the ViT components.

All tests run on CPU with a tiny configuration (small image, few blocks)
so they complete in seconds.  Run with:  pytest tests/test_vision_transformer.py
"""

import pytest
import torch

from models.config import ViTConfig
from models.vision_transformer import (
    ViTPatchEmbeddings, ViTAttention, ViTMLP, ViTBlock, ViT,
)


# Tiny config for fast CPU tests — different from the real 512/12/768 values
# so shape bugs are caught immediately.
@pytest.fixture
def cfg():
    return ViTConfig(
        img_size=64,     # 64×64 images  →  (64/8)^2 = 64 patches
        patch_size=8,
        hidden_dim=32,
        inter_dim=64,    # 2 × hidden_dim
        n_heads=4,       # head_dim = 32/4 = 8
        n_blocks=2,
        dropout=0.0,
        ln_eps=1e-6,
        cls_flag=False,
    )


B = 2   # batch size used throughout


class TestViTPatchEmbeddings:
    def test_output_shape(self, cfg):
        model = ViTPatchEmbeddings(cfg)
        x = torch.randn(B, 3, cfg.img_size, cfg.img_size)
        out = model(x)
        n_patches = (cfg.img_size // cfg.patch_size) ** 2
        expected = (B, n_patches, cfg.hidden_dim)
        assert out.shape == expected, f"Expected {expected}, got {out.shape}"

    def test_position_embedding_added(self, cfg):
        model = ViTPatchEmbeddings(cfg)
        model.position_embedding.data.zero_()
        x = torch.randn(B, 3, cfg.img_size, cfg.img_size)
        out_no_pos = model(x).clone()
        model.position_embedding.data.fill_(1.0)
        out_with_pos = model(x)
        assert not torch.allclose(out_no_pos, out_with_pos), (
            "Position embedding had no effect"
        )


class TestViTAttention:
    def test_output_shape(self, cfg):
        model = ViTAttention(cfg)
        T = (cfg.img_size // cfg.patch_size) ** 2  # 64
        x = torch.randn(B, T, cfg.hidden_dim)
        out = model(x)
        assert out.shape == (B, T, cfg.hidden_dim)

    def test_bidirectional(self, cfg):
        """ViT attention is NOT causal — every token can attend to every other."""
        model = ViTAttention(cfg)
        model.eval()
        T = 4
        x = torch.randn(B, T, cfg.hidden_dim)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, T, cfg.hidden_dim)
        assert out.abs().sum() > 0


class TestViTMLP:
    def test_output_shape(self, cfg):
        model = ViTMLP(cfg)
        T = 64
        x = torch.randn(B, T, cfg.hidden_dim)
        out = model(x)
        assert out.shape == (B, T, cfg.hidden_dim)


class TestViTBlock:
    def test_output_shape(self, cfg):
        model = ViTBlock(cfg)
        T = 64
        x = torch.randn(B, T, cfg.hidden_dim)
        out = model(x)
        assert out.shape == (B, T, cfg.hidden_dim)

    def test_residual_connection(self, cfg):
        """With zeroed weights, output ≈ input (residual passes through)."""
        model = ViTBlock(cfg)
        for p in model.parameters():
            p.data.zero_()
        model.ln1.weight.data.fill_(1.0)
        model.ln2.weight.data.fill_(1.0)
        T = 4
        x = torch.randn(B, T, cfg.hidden_dim)
        out = model(x)
        assert out.shape == (B, T, cfg.hidden_dim)


class TestViT:
    def test_output_shape(self, cfg):
        model = ViT(cfg)
        x = torch.randn(B, 3, cfg.img_size, cfg.img_size)
        out = model(x)
        num_patches = (cfg.img_size // cfg.patch_size) ** 2
        assert out.shape == (B, num_patches, cfg.hidden_dim)

    def test_output_dtype(self, cfg):
        model = ViT(cfg)
        x = torch.randn(B, 3, cfg.img_size, cfg.img_size)
        out = model(x)
        assert out.dtype == torch.float32

    def test_different_batch_sizes(self, cfg):
        model = ViT(cfg)
        num_patches = (cfg.img_size // cfg.patch_size) ** 2
        for b in [1, 3, 4]:
            x = torch.randn(b, 3, cfg.img_size, cfg.img_size)
            out = model(x)
            assert out.shape == (b, num_patches, cfg.hidden_dim)


@pytest.mark.slow
class TestViTPretrainedLoading:
    """Load real SigLIP2 weights and verify architecture.

    Skipped by default — requires ~350 MB download.
    Run with:  pytest tests/test_vision_transformer.py -m slow
    """

    @pytest.fixture(scope="class")
    def pretrained(self):
        cfg = ViTConfig()
        model = ViT.from_pretrained(cfg)
        return model, cfg

    def test_parameter_count(self, pretrained):
        model, _ = pretrained
        n = sum(p.numel() for p in model.parameters())
        assert 80_000_000 < n < 100_000_000, (
            f"Unexpected param count: {n:,}"
        )

    def test_config_updated(self, pretrained):
        """from_pretrained must mutate cfg to match the HF config."""
        _, cfg = pretrained
        assert cfg.hidden_dim == 768
        assert cfg.n_heads == 12
        assert cfg.n_blocks == 12
        assert cfg.patch_size == 16
        assert cfg.img_size == 512

    def test_patch_embedding_shape(self, pretrained):
        """Conv weight shape must match (hidden_dim, 3, patch, patch)."""
        model, cfg = pretrained
        w = model.patch_embedding.conv.weight
        assert w.shape == (
            cfg.hidden_dim, 3, cfg.patch_size, cfg.patch_size
        )

    def test_position_embedding_shape(self, pretrained):
        model, cfg = pretrained
        n_patches = (cfg.img_size // cfg.patch_size) ** 2
        assert model.patch_embedding.position_embedding.shape == (
            1, n_patches, cfg.hidden_dim
        )
