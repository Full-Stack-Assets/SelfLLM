"""Tests for quantization (GPTQ 4-bit and AWQ activation-aware)."""

import math
import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.quantization import (
    AWQQuantizer,
    QuantizedLinear,
    apply_awq_forward_patch,
    load_quantized,
    quantize_model,
    save_quantized,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def tiny_config():
    """Very small config for fast tests."""
    return ModelConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
        tie_weights=False,
    )


@pytest.fixture
def tiny_model(tiny_config):
    return SelfImprovingLLM(tiny_config)


@pytest.fixture
def simple_linear():
    """Simple linear layer for unit tests."""
    torch.manual_seed(42)
    return nn.Linear(128, 64, bias=False)


# --------------------------------------------------------------------------- #
# QuantizedLinear tests
# --------------------------------------------------------------------------- #

class TestQuantizedLinear:

    def test_quantized_linear_shape(self, simple_linear):
        """QuantizedLinear produces correct output shape."""
        qlinear = QuantizedLinear.quantize_from_linear(simple_linear, bits=4, group_size=64)
        x = torch.randn(4, 16, 128)
        out = qlinear(x)
        assert out.shape == (4, 16, 64), f"Expected (4, 16, 64), got {out.shape}"

    def test_quantized_linear_4bit(self, simple_linear):
        """4-bit weights stored as uint8 (2 weights per byte)."""
        qlinear = QuantizedLinear.quantize_from_linear(simple_linear, bits=4, group_size=64)

        # qweight should be uint8
        assert qlinear.qweight.dtype == torch.uint8
        # For 4-bit: pack_dim = padded_elements // 2
        expected_pack_dim = math.ceil(128 * 64 / 64) * 64 // 2
        assert qlinear.qweight.numel() == expected_pack_dim

    def test_quantized_linear_8bit(self, simple_linear):
        """8-bit weights stored as uint8 (1 weight per byte)."""
        qlinear = QuantizedLinear.quantize_from_linear(simple_linear, bits=8, group_size=64)

        assert qlinear.qweight.dtype == torch.uint8
        expected_pack_dim = math.ceil(128 * 64 / 64) * 64
        assert qlinear.qweight.numel() == expected_pack_dim

    def test_quantize_from_linear_roundtrip(self, simple_linear):
        """Quantization + dequantization roundtrip is close."""
        torch.manual_seed(42)
        original_weight = simple_linear.weight.data.clone()

        qlinear = QuantizedLinear.quantize_from_linear(simple_linear, bits=4, group_size=64)
        dequantized = qlinear.dequantize()

        # Should be close but not exact
        rel_error = (original_weight - dequantized).abs().mean() / original_weight.abs().mean()
        assert rel_error < 0.10, f"Relative error too large: {rel_error:.4f}"

    def test_quantize_from_linear_roundtrip_8bit(self, simple_linear):
        """8-bit roundtrip should be more accurate than 4-bit."""
        torch.manual_seed(42)
        original_weight = simple_linear.weight.data.clone()

        qlinear = QuantizedLinear.quantize_from_linear(simple_linear, bits=8, group_size=64)
        dequantized = qlinear.dequantize()

        rel_error = (original_weight - dequantized).abs().mean() / original_weight.abs().mean()
        assert rel_error < 0.05, f"8-bit relative error too large: {rel_error:.4f}"

    def test_forward_produces_output(self, simple_linear):
        """Forward pass produces non-zero output."""
        qlinear = QuantizedLinear.quantize_from_linear(simple_linear, bits=4, group_size=64)
        x = torch.randn(2, 8, 128)
        out = qlinear(x)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_bias_support(self):
        """QuantizedLinear handles bias correctly."""
        linear = nn.Linear(32, 16, bias=True)
        linear.bias.data.fill_(1.0)
        qlinear = QuantizedLinear.quantize_from_linear(linear, bits=4, group_size=32)

        x = torch.zeros(1, 4, 32)
        out = qlinear(x)
        # With zero input, output should equal bias
        assert torch.allclose(out[0, 0, :], torch.ones(16), atol=1e-5)

    def test_no_gradient_for_quantized_params(self, simple_linear):
        """Quantized weights should not require gradients."""
        qlinear = QuantizedLinear.quantize_from_linear(simple_linear, bits=4, group_size=64)
        assert not qlinear.qweight.requires_grad
        assert not qlinear.scales.requires_grad
        assert not qlinear.zeros.requires_grad


# --------------------------------------------------------------------------- #
# Model-level quantization tests
# --------------------------------------------------------------------------- #

class TestQuantizeModel:

    def test_all_linears_replaced(self, tiny_model):
        """All nn.Linear layers are replaced with QuantizedLinear."""
        original_linears = [
            name for name, m in tiny_model.named_modules()
            if isinstance(m, nn.Linear) and not isinstance(m, QuantizedLinear)
        ]
        assert len(original_linears) > 0, "Model should have Linear layers"

        quantize_model(tiny_model, bits=4, group_size=32)

        remaining_linears = [
            name for name, m in tiny_model.named_modules()
            if isinstance(m, nn.Linear) and not isinstance(m, QuantizedLinear)
        ]
        assert len(remaining_linears) == 0, (
            f"Some Linear layers were not replaced: {remaining_linears}"
        )

        # All QuantizedLinear layers present
        qlinears = [
            name for name, m in tiny_model.named_modules()
            if isinstance(m, QuantizedLinear)
        ]
        assert len(qlinears) == len(original_linears)

    def test_memory_savings_4bit(self, tiny_model):
        """4-bit quantization should reduce model size by at least 50%."""
        import io
        from selfllm.model.quantization import _get_model_size_mb

        orig_size = _get_model_size_mb(tiny_model)
        quantize_model(tiny_model, bits=4, group_size=32)
        new_size = _get_model_size_mb(tiny_model)

        savings_pct = (orig_size - new_size) / orig_size * 100
        assert savings_pct >= 50.0, (
            f"Expected >= 50% size reduction, got {savings_pct:.1f}%"
        )

    def test_quantized_forward(self, tiny_config, tiny_model):
        """Quantized model forward produces output similar to original."""
        tiny_model.eval()
        token_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))

        with torch.no_grad():
            orig_output = tiny_model(token_ids)["logits"]

        quantize_model(tiny_model, bits=4, group_size=32)
        tiny_model.eval()

        with torch.no_grad():
            q_output = tiny_model(token_ids)["logits"]

        assert orig_output.shape == q_output.shape

        # Relative error should be within ~25% for 4-bit on tiny models
        # (logits can differ significantly but argmax/ranking stays stable)
        rel_error = (orig_output - q_output).abs().mean() / orig_output.abs().mean()
        assert rel_error < 0.30, f"Forward output differs too much: {rel_error:.4f}"

    def test_quantized_forward_8bit(self, tiny_config, tiny_model):
        """8-bit quantization should be more accurate than 4-bit."""
        tiny_model.eval()
        token_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))

        with torch.no_grad():
            orig_output = tiny_model(token_ids)["logits"]

        quantize_model(tiny_model, bits=8, group_size=32)
        tiny_model.eval()

        with torch.no_grad():
            q_output = tiny_model(token_ids)["logits"]

        rel_error = (orig_output - q_output).abs().mean() / orig_output.abs().mean()
        assert rel_error < 0.05, f"8-bit forward error too large: {rel_error:.4f}"

    def test_embedding_not_quantized(self, tiny_model):
        """Embedding layers should NOT be quantized."""
        quantize_model(tiny_model, bits=4, group_size=32)
        assert isinstance(tiny_model.token_embedding, nn.Embedding)


# --------------------------------------------------------------------------- #
# Save / Load tests
# --------------------------------------------------------------------------- #

class TestSaveLoad:

    def test_save_load_roundtrip(self, tiny_config, tiny_model):
        """Save and load preserves quantized weights."""
        tiny_model.eval()
        quantize_model(tiny_model, bits=4, group_size=32)

        token_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
        with torch.no_grad():
            pre_save_output = tiny_model(token_ids)["logits"]

        # Collect all qweights before save
        pre_qweights = {}
        for name, m in tiny_model.named_modules():
            if isinstance(m, QuantizedLinear):
                pre_qweights[name] = m.qweight.clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        try:
            save_quantized(tiny_model, tmp_path)
            assert os.path.exists(tmp_path)

            # Load into a fresh model
            fresh_model = SelfImprovingLLM(tiny_config)
            load_quantized(tmp_path, fresh_model)

            # Verify qweights match
            for name, m in fresh_model.named_modules():
                if isinstance(m, QuantizedLinear):
                    assert name in pre_qweights
                    assert torch.equal(m.qweight, pre_qweights[name])

            # Verify forward output matches
            fresh_model.eval()
            with torch.no_grad():
                post_load_output = fresh_model(token_ids)["logits"]

            assert torch.allclose(pre_save_output, post_load_output, atol=1e-5)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_save_file_smaller_than_full(self, tiny_config, tiny_model):
        """Saved quantized file is smaller than full checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            full_path = os.path.join(tmpdir, "full.pt")
            quant_path = os.path.join(tmpdir, "quantized.pt")

            torch.save(tiny_model.state_dict(), full_path)
            quantize_model(tiny_model, bits=4, group_size=32)
            save_quantized(tiny_model, quant_path)

            full_size = os.path.getsize(full_path)
            quant_size = os.path.getsize(quant_path)

            assert quant_size < full_size, (
                f"Quantized file ({quant_size}) should be smaller than "
                f"full file ({full_size})"
            )


# --------------------------------------------------------------------------- #
# AWQ tests
# --------------------------------------------------------------------------- #

class TestAWQ:

    def test_awq_calibrate(self, tiny_config, tiny_model):
        """AWQ calibrate collects activation statistics."""
        quantizer = AWQQuantizer()
        token_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))

        stats = quantizer.calibrate(tiny_model, token_ids)

        # Should have stats for linear layers
        assert len(stats) > 0
        for name, mag in stats.items():
            assert isinstance(mag, torch.Tensor)
            assert mag.dim() == 1
            assert mag.numel() > 0
            assert (mag >= 0).all(), "Activation magnitudes should be non-negative"

    def test_awq_quantize(self, tiny_config, tiny_model):
        """AWQ quantize produces a valid quantized model."""
        quantizer = AWQQuantizer()
        token_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))

        quantizer.calibrate(tiny_model, token_ids)
        quantizer.quantize(tiny_model, bits=4, group_size=32)

        # All nn.Linear should be replaced with QuantizedLinear
        # (QuantizedLinear is NOT a subclass of nn.Linear, so plain nn.Linear count should be 0)
        plain_linears = [m for m in tiny_model.modules() if type(m) is nn.Linear]
        qlinears = [m for m in tiny_model.modules() if isinstance(m, QuantizedLinear)]
        assert len(plain_linears) == 0, f"Found {len(plain_linears)} unquantized Linear layers"
        assert len(qlinears) > 0, "No QuantizedLinear layers found"

    def test_awq_forward(self, tiny_config, tiny_model):
        """AWQ-quantized model can run forward pass."""
        quantizer = AWQQuantizer()
        token_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))

        quantizer.calibrate(tiny_model, token_ids)
        quantizer.quantize(tiny_model, bits=4, group_size=32)
        apply_awq_forward_patch(tiny_model)

        tiny_model.eval()
        with torch.no_grad():
            out = tiny_model(token_ids)

        assert "logits" in out
        assert out["logits"].shape == (2, 8, tiny_config.vocab_size)

    def test_awq_calibrate_without_quantize_raises(self, tiny_config, tiny_model):
        """Quantizing without calibration raises RuntimeError."""
        quantizer = AWQQuantizer()
        with pytest.raises(RuntimeError, match="calibrate"):
            quantizer.quantize(tiny_model, bits=4, group_size=32)

    def test_awq_has_awq_scale(self, tiny_config, tiny_model):
        """AWQ-quantized layers have awq_scale buffer."""
        quantizer = AWQQuantizer()
        token_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))

        quantizer.calibrate(tiny_model, token_ids)
        quantizer.quantize(tiny_model, bits=4, group_size=32)

        awq_layers = [
            m for m in tiny_model.modules()
            if isinstance(m, QuantizedLinear) and hasattr(m, "awq_scale")
        ]
        assert len(awq_layers) > 0


# --------------------------------------------------------------------------- #
# Gradient / training tests
# --------------------------------------------------------------------------- #

class TestGradientFlow:

    def test_quantized_weights_no_grad(self, simple_linear):
        """Quantized weights don't require gradients."""
        qlinear = QuantizedLinear.quantize_from_linear(simple_linear, bits=4, group_size=64)

        # All quantized params should be buffers (no grad)
        assert not qlinear.qweight.requires_grad
        assert not qlinear.scales.requires_grad
        assert not qlinear.zeros.requires_grad

    def test_forward_does_not_track_grad_for_weights(self, simple_linear):
        """Forward pass does not track gradients for quantized params."""
        qlinear = QuantizedLinear.quantize_from_linear(simple_linear, bits=4, group_size=64)
        x = torch.randn(2, 4, 128, requires_grad=True)

        out = qlinear(x)
        loss = out.sum()
        loss.backward()

        # Input gradient should exist
        assert x.grad is not None
        # Quantized params should not have grad
        assert qlinear.qweight.grad is None
        assert qlinear.scales.grad is None


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

class TestEdgeCases:

    def test_invalid_bits_raises(self):
        """Only 4 and 8 bits are supported."""
        with pytest.raises(ValueError, match="4-bit and 8-bit"):
            QuantizedLinear(64, 64, bits=2, group_size=32)

    def test_invalid_group_size_raises(self):
        """Negative group_size raises ValueError."""
        with pytest.raises(ValueError, match="group_size"):
            QuantizedLinear(64, 64, bits=4, group_size=-1)

    def test_group_size_larger_than_weights(self):
        """group_size larger than total weight elements."""
        linear = nn.Linear(8, 8, bias=False)
        qlinear = QuantizedLinear.quantize_from_linear(linear, bits=4, group_size=256)

        x = torch.randn(1, 4, 8)
        out = qlinear(x)
        assert out.shape == (1, 4, 8)

    def test_small_group_size(self):
        """Very small group_size works correctly."""
        linear = nn.Linear(32, 16, bias=False)
        qlinear = QuantizedLinear.quantize_from_linear(linear, bits=4, group_size=8)

        x = torch.randn(2, 4, 32)
        out = qlinear(x)
        assert out.shape == (2, 4, 16)


# --------------------------------------------------------------------------- #
# Integration: quantization + generation
# --------------------------------------------------------------------------- #

class TestQuantizedGeneration:

    def test_quantized_model_can_generate(self, tiny_config, tiny_model):
        """A quantized model can still generate text."""
        quantize_model(tiny_model, bits=4, group_size=32)
        tiny_model.eval()

        prompt = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            result = tiny_model.generate(prompt, max_new_tokens=5, temperature=0.0)

        assert "sequences" in result
        assert result["sequences"].shape[1] > prompt.shape[1]
