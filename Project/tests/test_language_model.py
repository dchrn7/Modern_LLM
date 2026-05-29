"""Shape + correctness tests for the Language Model components.

Key test: test_kv_cache_consistency — verifies that running the full sequence
at once (prefill) gives the same logits as running the prefix first and then
appending one token at a time (decode with KV cache).  If this test fails,
there is a bug in your KV cache or RoPE implementation.

Run with:  pytest tests/test_language_model.py
"""

import pytest
import torch

from models.config import LMConfig
from models.language_model import (
    RMSNorm, LMAttention, LMMLP, LMBlock, LanguageModel, RotaryEmbedding,
)


@pytest.fixture
def cfg():
    return LMConfig(
        hidden_dim=32,
        inter_dim=64,
        rms_eps=1e-5,
        re_base=10000,
        max_position_embeddings=128,
        vocab_size=256,
        base_vocab_size=255,
        n_heads=4,        # head_dim = 32/4 = 8
        n_kv_heads=2,     # n_kv_groups = 2
        n_blocks=2,
        dropout=0.0,
        attn_scaling=1.0,
        tie_weights=True,
    )


B, T = 2, 10


class TestRMSNorm:
    def test_output_shape(self, cfg):
        norm = RMSNorm(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        assert norm(x).shape == (B, T, cfg.hidden_dim)

    def test_scale_invariance(self, cfg):
        """RMSNorm output should not depend on the scale of the input."""
        norm = RMSNorm(cfg)
        norm.weight.data.fill_(1.0)
        norm.eval()
        x = torch.randn(B, T, cfg.hidden_dim)
        with torch.no_grad():
            out1 = norm(x)
            out2 = norm(x * 10)
        assert torch.allclose(out1, out2, atol=1e-5), (
            "RMSNorm output changed when input was scaled by 10"
        )

    def test_weight_effect(self, cfg):
        norm = RMSNorm(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        with torch.no_grad():
            norm.weight.data.fill_(1.0)
            out1 = norm(x)
            norm.weight.data.fill_(2.0)
            out2 = norm(x)
        assert torch.allclose(out2, 2 * out1, atol=1e-6)


class TestLMAttention:
    def test_output_shape_prefill(self, cfg):
        attn = LMAttention(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        rope = RotaryEmbedding(cfg)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        cos, sin = rope(pos_ids)
        out, cache = attn(x, cos, sin, block_kv_cache=None)
        assert out.shape == (B, T, cfg.hidden_dim)
        assert 'key' in cache and 'value' in cache

    def test_kv_cache_shape(self, cfg):
        attn = LMAttention(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        rope = RotaryEmbedding(cfg)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        cos, sin = rope(pos_ids)
        _, cache = attn(x, cos, sin, block_kv_cache=None)
        head_dim = cfg.hidden_dim // cfg.n_heads
        assert cache['key'].shape == (B, cfg.n_kv_heads, T, head_dim)
        assert cache['value'].shape == (B, cfg.n_kv_heads, T, head_dim)


class TestLMMLp:
    def test_output_shape(self, cfg):
        mlp = LMMLP(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        assert mlp(x).shape == (B, T, cfg.hidden_dim)


class TestLMBlock:
    def test_output_shape(self, cfg):
        block = LMBlock(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        rope = RotaryEmbedding(cfg)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        cos, sin = rope(pos_ids)
        out, cache = block(x, cos, sin)
        assert out.shape == (B, T, cfg.hidden_dim)
        assert cache is not None


class TestLanguageModel:
    def test_forward_shape(self, cfg):
        model = LanguageModel(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)  # embeddings, not token ids
        hidden, kv = model(x)
        assert hidden.shape == (B, T, cfg.hidden_dim)
        assert len(kv) == cfg.n_blocks

    def test_head_shape(self, cfg):
        model = LanguageModel(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        hidden, _ = model(x)
        logits = model.head(hidden)
        assert logits.shape == (B, T, cfg.vocab_size)

    def test_kv_cache_consistency(self, cfg):
        """Prefill + single-token decode must match full-sequence forward.

        This is the key correctness test for KV caching + RoPE.

        Strategy:
          1. Run the full sequence [tok_0, ..., tok_{T-1}] in one shot  → logits_full
          2. Run the prefix  [tok_0, ..., tok_{T-2}] to build the KV cache
          3. Run just        [tok_{T-1}]  with start_pos=T-1             → logits_last
          4. logits_last should equal logits_full[:, -1, :]
        """
        torch.manual_seed(42)
        model = LanguageModel(cfg)
        model.eval()

        x = torch.randn(1, T, cfg.hidden_dim)

        with torch.no_grad():
            hidden_full, _ = model(x, kv_cache=None, start_pos=0)
            logits_full = model.head(hidden_full)           # [1, T, vocab]

            hidden_prefix, kv = model(
                x[:, :-1, :], kv_cache=None, start_pos=0
            )
            hidden_last, _ = model(
                x[:, -1:, :], kv_cache=kv, start_pos=T - 1
            )
            logits_last = model.head(hidden_last)           # [1, 1, vocab]

        torch.testing.assert_close(
            logits_full[:, -1:, :], logits_last,
            atol=1e-4, rtol=1e-4,
            msg=(
                "KV cache decode does not match full-sequence forward — "
                "check your KV concatenation and start_pos handling in RoPE."
            ),
        )


@pytest.mark.slow
class TestLanguageModelPretrainedLoading:
    """Load real SmolLM2-360M weights and verify architecture.

    Skipped by default — requires ~720 MB download.
    Run with:  pytest tests/test_language_model.py -m slow
    """

    @pytest.fixture(scope="class")
    def pretrained(self):
        from models.config import LMConfig
        cfg = LMConfig()
        model = LanguageModel.from_pretrained(cfg)
        return model, cfg

    def test_parameter_count(self, pretrained):
        model, _ = pretrained
        n = sum(p.numel() for p in model.parameters())
        assert 300_000_000 < n < 420_000_000, (
            f"Unexpected param count: {n:,}"
        )

    def test_config_updated(self, pretrained):
        """from_pretrained must mutate cfg to match the HF config."""
        _, cfg = pretrained
        assert cfg.hidden_dim == 960
        assert cfg.n_heads == 15
        assert cfg.n_kv_heads == 5
        assert cfg.n_blocks == 32
        assert cfg.inter_dim == 2560

    def test_inv_freq_shape(self, pretrained):
        """inv_freq must have shape [head_dim / 2]."""
        model, cfg = pretrained
        head_dim = cfg.hidden_dim // cfg.n_heads  # 64
        assert model.rotary_embd.inv_freq.shape == (head_dim // 2,)

    def test_weight_tying(self, pretrained):
        """Embedding and output head must share the same tensor."""
        model, _ = pretrained
        assert (
            model.head.weight.data_ptr()
            == model.token_embedding.weight.data_ptr()
        )

    def test_vocab_extended(self, pretrained):
        """Embedding table must have vocab_size rows (49152 + 1 image token)."""
        model, cfg = pretrained
        assert model.token_embedding.weight.shape[0] == cfg.vocab_size
        assert cfg.vocab_size == 49153
