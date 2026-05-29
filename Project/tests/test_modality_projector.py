"""Shape tests for the ModalityProjector.

Run with:  pytest tests/test_modality_projector.py
"""

import pytest
import torch
import torch.nn as nn

from models.config import VLMConfig, ViTConfig, LMConfig, ProjectorConfig
from models.modality_projector import ModalityProjector


@pytest.fixture
def cfg():
    return VLMConfig(
        vit=ViTConfig(hidden_dim=768),
        lm=LMConfig(hidden_dim=960),
        projector=ProjectorConfig(pixel_shuffle_factor=4, image_token_length=64),
    )


B = 2
VIT_PATCHES = 1024   # (512/16)^2


class TestModalityProjector:
    def test_pixel_shuffle_shape(self, cfg):
        """pixel_shuffle: [B, 1024, 768] → [B, 64, 768×16=12288]"""
        mp = ModalityProjector(cfg)
        x = torch.randn(B, VIT_PATCHES, cfg.vit.hidden_dim)
        shuffled = mp.pixel_shuffle(x)
        factor = cfg.projector.pixel_shuffle_factor
        expected_dim = cfg.vit.hidden_dim * (factor ** 2)
        expected_tokens = VIT_PATCHES // (factor ** 2)
        assert shuffled.shape == (B, expected_tokens, expected_dim), (
            f"pixel_shuffle output shape mismatch: {shuffled.shape}"
        )

    def test_forward_shape(self, cfg):
        """Full projector: [B, 1024, 768] → [B, 64, 960]"""
        mp = ModalityProjector(cfg)
        x = torch.randn(B, VIT_PATCHES, cfg.vit.hidden_dim)
        out = mp(x)
        assert out.shape == (
            B, cfg.projector.image_token_length, cfg.lm.hidden_dim
        ), f"Projector output shape mismatch: {out.shape}"

    def test_input_dim_attribute(self, cfg):
        """Students must set self.input_dim = vit.hidden_dim × factor²."""
        mp = ModalityProjector(cfg)
        factor = cfg.projector.pixel_shuffle_factor
        expected = cfg.vit.hidden_dim * (factor ** 2)
        assert hasattr(mp, 'input_dim'), (
            "ModalityProjector must have attribute 'input_dim'"
        )
        assert mp.input_dim == expected, (
            f"input_dim should be {expected} (768×16), got {mp.input_dim}"
        )

    def test_proj_is_linear(self, cfg):
        mp = ModalityProjector(cfg)
        assert hasattr(mp, 'proj'), (
            "ModalityProjector must have attribute 'proj'"
        )
        assert isinstance(mp.proj, nn.Linear)
        assert mp.proj.weight.shape == (cfg.lm.hidden_dim, mp.input_dim)

    def test_dtype_preserved(self, cfg):
        mp = ModalityProjector(cfg)
        x = torch.randn(B, VIT_PATCHES, cfg.vit.hidden_dim, dtype=torch.float32)
        out = mp(x)
        assert out.dtype == torch.float32
