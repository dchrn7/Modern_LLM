"""Integration tests: load pretrained weights and verify.

These tests download ~800 MB from HuggingFace and run on GPU (or CPU).
They are marked @pytest.mark.slow and skipped by default.

Run with:  pytest tests/test_pretrained_loading.py -m slow

If your architecture is correct, from_pretrained() will succeed and the
model will run a forward pass.  A wrong parameter name, wrong Linear size,
or wrong weight tying will raise an error here.
"""

import pytest
import torch

from models.config import VLMConfig


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cfg():
    return VLMConfig()


@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TestViTPretrainedLoading:
    def test_loads_without_error(self, cfg):
        from models.vision_transformer import ViT
        model = ViT.from_pretrained(cfg.vit)
        assert model is not None

    def test_parameter_count(self, cfg):
        from models.vision_transformer import ViT
        model = ViT.from_pretrained(cfg.vit)
        n = sum(p.numel() for p in model.parameters())
        # SigLIP2-base has ~86M parameters
        assert 80_000_000 < n < 100_000_000, f"Unexpected param count: {n:,}"

    def test_forward_shape(self, cfg, device):
        from models.vision_transformer import ViT
        model = ViT.from_pretrained(cfg.vit).to(device).eval()
        x = torch.randn(1, 3, 512, 512, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1024, cfg.vit.hidden_dim)


class TestLMPretrainedLoading:
    def test_loads_without_error(self, cfg):
        from models.language_model import LanguageModel
        model = LanguageModel.from_pretrained(cfg.lm)
        assert model is not None

    def test_parameter_count(self, cfg):
        from models.language_model import LanguageModel
        model = LanguageModel.from_pretrained(cfg.lm)
        n = sum(p.numel() for p in model.parameters())
        # SmolLM2-360M has ~360M parameters
        assert 300_000_000 < n < 420_000_000, f"Unexpected param count: {n:,}"

    def test_forward_shape(self, cfg, device):
        from models.language_model import LanguageModel
        model = LanguageModel.from_pretrained(cfg.lm).to(device).eval()
        x = torch.randn(1, 32, cfg.lm.hidden_dim, device=device)
        with torch.no_grad():
            hidden, _ = model(x)
        assert hidden.shape == (1, 32, cfg.lm.hidden_dim)

    def test_kv_cache_consistency_with_pretrained(self, cfg, device):
        """After loading real weights, prefill+decode must equal full forward."""
        from models.language_model import LanguageModel
        torch.manual_seed(0)
        model = LanguageModel.from_pretrained(cfg.lm).to(device).eval()
        T = 16
        x = torch.randn(1, T, cfg.lm.hidden_dim, device=device)
        with torch.no_grad():
            hidden_full, _ = model(x, kv_cache=None, start_pos=0)
            logits_full = model.head(hidden_full)
            hidden_prefix, kv = model(x[:, :-1], kv_cache=None, start_pos=0)
            hidden_last, _ = model(x[:, -1:], kv_cache=kv, start_pos=T - 1)
            logits_last = model.head(hidden_last)
        torch.testing.assert_close(
            logits_full[:, -1:], logits_last, atol=1e-3, rtol=1e-3
        )


class TestVLMPretrainedForward:
    def test_vlm_forward_with_pretrained_weights(self, cfg, device):
        """Full VLM forward pass with pretrained ViT + LM."""
        from models.vision_language_model import VisionLanguageModel
        model = VisionLanguageModel(cfg, load_backbone=True).to(device).eval()
        tokenizer = model.tokenizer

        T = 128
        n_img = cfg.projector.image_token_length
        image_string = tokenizer.image_token * n_img
        prompt = image_string + "What is in this image?"
        input_ids = torch.tensor(
            tokenizer.encode(prompt, add_special_tokens=False)[:T]
        ).unsqueeze(0).to(device)
        if input_ids.size(1) < T:
            pad = torch.full(
                (1, T - input_ids.size(1)),
                tokenizer.pad_token_id,
                device=device,
            )
            input_ids = torch.cat([input_ids, pad], dim=1)

        pixel_values = torch.randn(1, 3, 512, 512, device=device)
        targets = input_ids.clone()
        targets[:, :n_img] = -100

        with torch.no_grad():
            logits, loss = model(input_ids, pixel_values, targets=targets)

        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert logits.shape[-1] == cfg.lm.vocab_size
