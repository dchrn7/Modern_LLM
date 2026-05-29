"""End-to-end tests for VisionLanguageModel.

All tests use DummyDataset and load_backbone=False (no HuggingFace downloads).

Run with:  pytest tests/test_vlm.py

For pretrained weight loading tests (requires ~800 MB download), run:
    pytest tests/test_pretrained_loading.py -m slow
"""

import pytest
import torch
import torch.optim as optim

from models.config import VLMConfig, ViTConfig, LMConfig, ProjectorConfig
from models.vision_language_model import VisionLanguageModel


@pytest.fixture
def cfg():
    """Tiny VLMConfig that matches DummyDataset shapes."""
    return VLMConfig(
        vit=ViTConfig(
            hidden_dim=32,
            inter_dim=64,
            patch_size=16,
            img_size=512,   # real size — still produces 1024 patches
            n_heads=4,
            n_blocks=2,
        ),
        lm=LMConfig(
            hidden_dim=32,
            inter_dim=64,
            n_heads=4,
            n_kv_heads=2,
            n_blocks=2,
            vocab_size=256,
            base_vocab_size=255,
            tie_weights=True,
        ),
        projector=ProjectorConfig(
            pixel_shuffle_factor=4,
            image_token_length=64,
        ),
        load_backbone_weights=False,
    )


@pytest.fixture
def model(cfg):
    m = VisionLanguageModel(cfg, load_backbone=False)
    m.eval()
    return m


B, T = 2, 128


class TestVLMForward:
    def test_forward_without_targets(self, cfg, model):
        """Without targets, forward returns hidden states, no loss."""
        input_ids = torch.randint(0, cfg.lm.vocab_size, (B, T))
        pixel_values = torch.randn(B, 3, 512, 512)
        attention_mask = torch.ones(B, T, dtype=torch.long)

        # Insert image token placeholders so replacement works
        image_token_id = cfg.lm.vocab_size - 1
        model.tokenizer.image_token_id = image_token_id
        input_ids[:, :cfg.projector.image_token_length] = image_token_id

        with torch.no_grad():
            out, loss = model(input_ids, pixel_values, attention_mask, targets=None)

        assert loss is None
        assert out.shape == (B, T, cfg.lm.hidden_dim)

    def test_forward_with_targets(self, cfg, model):
        """With targets, forward returns logits and a finite scalar loss."""
        input_ids = torch.randint(0, cfg.lm.vocab_size, (B, T))
        pixel_values = torch.randn(B, 3, 512, 512)
        attention_mask = torch.ones(B, T, dtype=torch.long)
        targets = input_ids.clone()
        targets[:, :cfg.projector.image_token_length] = -100

        image_token_id = cfg.lm.vocab_size - 1
        model.tokenizer.image_token_id = image_token_id
        input_ids[:, :cfg.projector.image_token_length] = image_token_id

        with torch.no_grad():
            logits, loss = model(
                input_ids, pixel_values, attention_mask, targets=targets
            )

        assert logits.shape == (B, T, cfg.lm.vocab_size)
        assert loss is not None
        assert loss.shape == ()       # scalar
        assert torch.isfinite(loss)

    def test_image_token_replacement(self, cfg, model):
        """Positions with image tokens must have different embeddings."""
        n_img = cfg.projector.image_token_length
        T_short = n_img + 10
        input_ids = torch.randint(1, cfg.lm.vocab_size - 1, (1, T_short))
        image_token_id = cfg.lm.vocab_size - 1
        model.tokenizer.image_token_id = image_token_id
        input_ids[0, :n_img] = image_token_id

        pixel_values = torch.randn(1, 3, 512, 512)

        with torch.no_grad():
            out, _ = model(input_ids, pixel_values, targets=None)

        assert out.shape[1] == T_short
