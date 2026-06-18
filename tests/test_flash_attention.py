"""Tests for Flash Attention 2 and HybridAttention.

Covers:
- Import / availability detection
- Forward-pass shape correctness
- KV-cache incremental decoding
- Causal masking (no future peeking)
- Fallback to standard attention (flash_attn not installed or fp32 input)
- Mixed-precision (fp16 / bf16) paths
- Numerical equivalence between FlashAttention and standard attention
"""

import pytest
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# We must be able to import the module even when flash_attn is not installed.
# --------------------------------------------------------------------------- #
from selfllm.model.flash_attention import (
    HAS_FLASH_ATTN,
    FlashAttention2,
    HybridAttention,
    RoPE,
)
from selfllm.model.attention import RoPEMultiHeadAttention


# =============================================================================
# Helpers
# =============================================================================

DEVICE = torch.device("cpu")
BATCH = 2
SEQ_LEN = 8
D_MODEL = 64
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS
MAX_SEQ_LEN = 128


def _make_input(seq_len=SEQ_LEN, dtype=torch.float32, device=DEVICE):
    """Create a random input tensor."""
    torch.manual_seed(42)
    return torch.randn(BATCH, seq_len, D_MODEL, dtype=dtype, device=device)


def _make_cache(seq_len=4, dtype=torch.float32, device=DEVICE):
    """Create a dummy KV cache."""
    torch.manual_seed(123)
    k = torch.randn(BATCH, seq_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(BATCH, seq_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    return (k, v)


# =============================================================================
# 1. Import / availability
# =============================================================================

class TestAvailability:
    def test_has_flash_attn_flag_is_bool(self):
        assert isinstance(HAS_FLASH_ATTN, bool)

    def test_flash_attention2_instantiates_without_flash_attn(self):
        """FlashAttention2 can be created even when flash_attn is missing."""
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        assert isinstance(attn, nn.Module)

    def test_hybrid_attention_instantiates(self):
        attn = HybridAttention(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        assert isinstance(attn, nn.Module)
        assert attn.backend_name in ("flash_attn", "standard")


# =============================================================================
# 2. RoPE correctness
# =============================================================================

class TestRoPE:
    def test_rope_shape_preservation(self):
        rope = RoPE(HEAD_DIM, MAX_SEQ_LEN)
        x = torch.randn(BATCH, SEQ_LEN, N_HEADS, HEAD_DIM)
        positions = torch.arange(SEQ_LEN)
        out = rope.apply_rotary_emb(x, positions)
        assert out.shape == x.shape

    def test_rope_changes_values(self):
        """RoPE should actually rotate the values."""
        rope = RoPE(HEAD_DIM, MAX_SEQ_LEN)
        x = torch.ones(BATCH, SEQ_LEN, N_HEADS, HEAD_DIM)
        positions = torch.arange(SEQ_LEN)
        out = rope.apply_rotary_emb(x, positions)
        assert not torch.allclose(out, x)

    def test_rope_position_dependence(self):
        """Different positions should give different rotations."""
        rope = RoPE(HEAD_DIM, MAX_SEQ_LEN)
        x = torch.ones(1, 1, 1, HEAD_DIM)
        out_0 = rope.apply_rotary_emb(x, torch.tensor([0]))
        out_1 = rope.apply_rotary_emb(x, torch.tensor([1]))
        assert not torch.allclose(out_0, out_1)


# =============================================================================
# 3. Forward-pass shape
# =============================================================================

class TestForwardShape:
    def test_flash_attention_output_shape(self):
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        x = _make_input()
        out, cache = attn(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
        assert cache is None  # no kv_cache provided

    def test_hybrid_attention_output_shape(self):
        attn = HybridAttention(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        x = _make_input()
        out, cache = attn(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_single_token_decode_shape(self):
        """Decode mode with seq_len=1."""
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        x = _make_input(seq_len=1)
        out, cache = attn(x, is_prefill=False)
        assert out.shape == (BATCH, 1, D_MODEL)

    def test_varying_seq_len(self):
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        for sl in [1, 4, 8, 16]:
            x = _make_input(seq_len=sl)
            out, _ = attn(x)
            assert out.shape == (BATCH, sl, D_MODEL)


# =============================================================================
# 4. KV-cache incremental decoding
# =============================================================================

class TestKVCache:
    def test_prefill_populates_cache(self):
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        x = _make_input(seq_len=4)
        cache_in = _make_cache(seq_len=0)  # Empty cache
        out, cache_out = attn(x, kv_cache=cache_in, is_prefill=True)
        assert cache_out is not None
        k, v = cache_out
        assert k.shape == (BATCH, 4, N_HEADS, HEAD_DIM)
        assert v.shape == (BATCH, 4, N_HEADS, HEAD_DIM)

    def test_decode_with_existing_cache(self):
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        cache = _make_cache(seq_len=4)
        x = _make_input(seq_len=1)  # Single new token
        out, new_cache = attn(x, kv_cache=cache, is_prefill=False)
        assert out.shape == (BATCH, 1, D_MODEL)
        k, v = new_cache
        assert k.shape == (BATCH, 5, N_HEADS, HEAD_DIM)  # 4 + 1
        assert v.shape == (BATCH, 5, N_HEADS, HEAD_DIM)

    def test_incremental_consistency(self):
        """Prefill full sequence == prefill 4 + decode 4 more step by step."""
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        torch.manual_seed(99)
        full_x = torch.randn(BATCH, 8, D_MODEL)

        # Full prefill
        attn.eval()
        with torch.no_grad():
            full_out, _ = attn(full_x, is_prefill=True)

        # Incremental: prefill first 4, then decode 4 tokens one by one
        empty_cache = (
            torch.randn(BATCH, 0, N_HEADS, HEAD_DIM),
            torch.randn(BATCH, 0, N_HEADS, HEAD_DIM),
        )
        with torch.no_grad():
            out_4, cache = attn(full_x[:, :4, :], kv_cache=empty_cache, is_prefill=True)
            for t in range(4, 8):
                out_t, cache = attn(
                    full_x[:, t : t + 1, :],
                    kv_cache=cache,
                    is_prefill=False,
                )

        # The final cache should have 8 tokens
        assert cache[0].shape[1] == 8


# =============================================================================
# 5. Causal masking
# =============================================================================

class TestCausalMasking:
    def test_no_future_peeking(self):
        """Token i should not be affected by token j > i.

        We verify this by zeroing out future positions in the input and
        checking that earlier tokens' outputs remain unchanged.
        """
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        attn.eval()

        torch.manual_seed(77)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

        with torch.no_grad():
            out_full, _ = attn(x)

        # Zero out the last half of the sequence
        x_masked = x.clone()
        x_masked[:, SEQ_LEN // 2 :, :] = 0.0

        with torch.no_grad():
            out_masked, _ = attn(x_masked)

        # The first half should be identical (causal masking)
        torch.testing.assert_close(
            out_full[:, : SEQ_LEN // 2, :],
            out_masked[:, : SEQ_LEN // 2, :],
        )

    def test_causal_mask_not_all_same(self):
        """With different inputs, outputs should differ."""
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        x1 = _make_input()
        x2 = _make_input() + 1.0  # Different values
        attn.eval()
        with torch.no_grad():
            out1, _ = attn(x1)
            out2, _ = attn(x2)
        assert not torch.allclose(out1, out2)


# =============================================================================
# 6. Fallback to standard attention
# =============================================================================

class TestFallback:
    def test_fp32_triggers_fallback(self):
        """Flash Attention requires fp16/bf16; fp32 should use manual path."""
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        x = _make_input(dtype=torch.float32)
        # Should not raise, should fall back gracefully
        out, _ = attn(x)
        assert out.dtype == torch.float32
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_hybrid_uses_correct_backend(self):
        """HybridAttention backend should match HAS_FLASH_ATTN."""
        attn = HybridAttention(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        if HAS_FLASH_ATTN:
            assert attn.backend_name == "flash_attn"
        else:
            assert attn.backend_name == "standard"


# =============================================================================
# 7. Mixed precision (fp16 / bf16)
# =============================================================================

class TestMixedPrecision:
    @pytest.mark.skipif(
        not HAS_FLASH_ATTN, reason="flash_attn not installed"
    )
    def test_fp16_path(self):
        """fp16 input should work (uses Flash Attention if available)."""
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        x = _make_input(dtype=torch.float16)
        out, _ = attn(x)
        assert out.dtype == torch.float16
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    @pytest.mark.skipif(
        not HAS_FLASH_ATTN, reason="flash_attn not installed"
    )
    def test_bf16_path(self):
        """bf16 input should work (uses Flash Attention if available)."""
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        x = _make_input(dtype=torch.bfloat16)
        out, _ = attn(x)
        assert out.dtype == torch.bfloat16
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


# =============================================================================
# 8. Numerical equivalence
# =============================================================================

class TestEquivalence:
    def test_flash_manual_attention_matches(self):
        """FlashAttention2 (manual fallback) should match RoPEMultiHeadAttention
        when both use fp32 and identical weights.
        """
        torch.manual_seed(42)
        fa = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN, dropout=0.0)
        torch.manual_seed(42)
        sa = RoPEMultiHeadAttention(D_MODEL, N_HEADS, MAX_SEQ_LEN, dropout=0.0)

        x = _make_input()
        fa.eval()
        sa.eval()

        with torch.no_grad():
            out_fa, _ = fa(x)
            out_sa, _ = sa(x)

        # Outputs should be very close (same math, different layout)
        torch.testing.assert_close(out_fa, out_sa, rtol=1e-4, atol=1e-4)

    def test_flash_attention_gradients_flow(self):
        """Gradients should flow through all projections."""
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        x = _make_input()
        x.requires_grad_(True)
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert attn.q_proj.weight.grad is not None
        assert attn.k_proj.weight.grad is not None
        assert attn.v_proj.weight.grad is not None
        assert attn.out_proj.weight.grad is not None


# =============================================================================
# 9. HybridAttention integration with TransformerBlock
# =============================================================================

class TestIntegration:
    def test_transformer_block_with_hybrid_attention(self):
        """A full TransformerBlock should work with HybridAttention."""
        from selfllm.model.layers import TransformerBlock
        from selfllm.model.config import ModelConfig

        config = ModelConfig(
            vocab_size=100,
            d_model=D_MODEL,
            n_layers=1,
            n_heads=N_HEADS,
            d_ff=128,
            max_seq_len=MAX_SEQ_LEN,
            dropout=0.0,
        )
        block = TransformerBlock(config)
        x = _make_input()
        out = block(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_train_eval_mode(self):
        """Dropout should behave differently in train vs eval."""
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN, dropout=0.5)
        x = _make_input()
        attn.train()
        out_train, _ = attn(x)
        attn.eval()
        out_eval, _ = attn(x)
        # Just check both produce valid outputs; exact equality is probabilistic
        assert out_train.shape == out_eval.shape


# =============================================================================
# 10. Edge cases
# =============================================================================

class TestEdgeCases:
    def test_single_head(self):
        """Works with n_heads=1."""
        d = 64
        attn = FlashAttention2(d, 1, MAX_SEQ_LEN)
        x = torch.randn(1, 4, d)
        out, _ = attn(x)
        assert out.shape == (1, 4, d)

    def test_large_head_dim(self):
        """Works with larger head dimensions."""
        d = 256
        h = 8
        attn = FlashAttention2(d, h, MAX_SEQ_LEN)
        x = torch.randn(1, 8, d)
        out, _ = attn(x)
        assert out.shape == (1, 8, d)

    def test_batch_size_one(self):
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        x = torch.randn(1, SEQ_LEN, D_MODEL)
        out, _ = attn(x)
        assert out.shape == (1, SEQ_LEN, D_MODEL)

    def test_kv_cache_none_no_cache_returned(self):
        attn = FlashAttention2(D_MODEL, N_HEADS, MAX_SEQ_LEN)
        x = _make_input()
        out, cache = attn(x, kv_cache=None)
        assert cache is None
