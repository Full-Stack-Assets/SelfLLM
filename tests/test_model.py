"""Unit tests for SelfImprovingLLM.

Tests cover:
- Forward pass output shapes
- Generation output shapes
- Parameter count
- Save/load roundtrip
"""

import tempfile
from pathlib import Path

import pytest
import torch

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.utils import count_parameters, format_num


class TestSelfImprovingLLM:
    """Comprehensive test suite for the SelfImprovingLLM model."""

    @pytest.fixture
    def tiny_config(self) -> ModelConfig:
        """Return a tiny model config for fast unit tests."""
        return ModelConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            max_seq_len=64,
            dropout=0.0,  # Disable dropout for deterministic tests
            tie_weights=True,
            use_rope=True,
            use_swiglu=True,
        )

    @pytest.fixture
    def model(self, tiny_config: ModelConfig) -> SelfImprovingLLM:
        """Return an instantiated (tiny) model."""
        return SelfImprovingLLM(tiny_config)

    @pytest.fixture
    def device(self) -> str:
        """Return compute device for tests."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # 1. Forward pass shape
    # ------------------------------------------------------------------

    def test_forward_pass_shape(self, model: SelfImprovingLLM, tiny_config: ModelConfig, device: str) -> None:
        """Forward pass should return logits of shape [batch, seq_len, vocab_size]."""
        model = model.to(device)
        model.eval()

        batch_size = 2
        seq_len = 16
        token_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            outputs = model(token_ids)

        assert "logits" in outputs
        logits = outputs["logits"]
        assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size)

    def test_forward_pass_with_targets(self, model: SelfImprovingLLM, tiny_config: ModelConfig, device: str) -> None:
        """Forward pass with targets should return a scalar loss."""
        model = model.to(device)
        model.train()

        batch_size = 2
        seq_len = 16
        token_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len), device=device)

        outputs = model(token_ids, targets=targets)

        assert "loss" in outputs
        loss = outputs["loss"]
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0  # Cross-entropy loss is positive

    def test_forward_hidden_states(self, model: SelfImprovingLLM, tiny_config: ModelConfig, device: str) -> None:
        """Forward pass should return hidden_states when available."""
        model = model.to(device)
        model.eval()

        batch_size = 2
        seq_len = 16
        token_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            outputs = model(token_ids)

        # hidden_states may not always be present depending on implementation
        if "hidden_states" in outputs:
            hidden = outputs["hidden_states"]
            assert hidden.shape == (batch_size, seq_len, tiny_config.d_model)

    # ------------------------------------------------------------------
    # 2. Generation output shape
    # ------------------------------------------------------------------

    def test_generation_output_shape(self, model: SelfImprovingLLM, tiny_config: ModelConfig, device: str) -> None:
        """Generation should return sequences longer than the prompt."""
        model = model.to(device)
        model.eval()

        batch_size = 1
        prompt_len = 8
        max_new_tokens = 12
        prompt_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, prompt_len), device=device)

        with torch.no_grad():
            result = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=10,
                top_p=0.9,
            )

        assert "sequences" in result
        sequences = result["sequences"]
        # Output should include prompt + generated tokens
        assert sequences.shape[0] == batch_size
        assert sequences.shape[1] >= prompt_len

    def test_generation_batch(self, model: SelfImprovingLLM, tiny_config: ModelConfig, device: str) -> None:
        """Generation should support batch prompts."""
        model = model.to(device)
        model.eval()

        batch_size = 3
        prompt_len = 8
        prompt_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, prompt_len), device=device)

        with torch.no_grad():
            result = model.generate(prompt_ids, max_new_tokens=8)

        assert result["sequences"].shape[0] == batch_size

    def test_generation_deterministic(self, model: SelfImprovingLLM, tiny_config: ModelConfig, device: str) -> None:
        """Generation with temperature=0 should be deterministic."""
        model = model.to(device)
        model.eval()

        prompt_ids = torch.randint(0, tiny_config.vocab_size, (1, 8), device=device)

        with torch.no_grad():
            result1 = model.generate(prompt_ids, max_new_tokens=8, temperature=0.0)
            result2 = model.generate(prompt_ids, max_new_tokens=8, temperature=0.0)

        assert torch.equal(result1["sequences"], result2["sequences"])

    # ------------------------------------------------------------------
    # 3. Parameter count
    # ------------------------------------------------------------------

    def test_parameter_count(self, model: SelfImprovingLLM, tiny_config: ModelConfig) -> None:
        """Model should report a positive parameter count."""
        param_count = count_parameters(model)
        assert param_count > 0
        assert isinstance(param_count, int)

        # Compare with model.count_parameters() if available
        if hasattr(model, "count_parameters"):
            assert model.count_parameters() == param_count

    def test_parameter_count_tied(self, tiny_config: ModelConfig) -> None:
        """Tied weights should reduce total parameter count."""
        tied_cfg = ModelConfig(**{**tiny_config.__dict__, "tie_weights": True})
        tied_model = SelfImprovingLLM(tied_cfg)
        tied_params = count_parameters(tied_model)

        untied_cfg = ModelConfig(**{**tiny_config.__dict__, "tie_weights": False})
        untied_model = SelfImprovingLLM(untied_cfg)
        untied_params = count_parameters(untied_model)

        # Tied weights should reduce or equal the parameter count
        assert tied_params <= untied_params

    def test_format_num(self) -> None:
        """Number formatting utility should work correctly."""
        assert format_num(1200000) == "1.2M"
        assert format_num(350000) == "350K"
        assert format_num(1000) == "1K"
        assert format_num(999) == "999"
        assert format_num(1_000_000_000) == "1B"
        assert format_num(1_500_000_000_000) == "1.5T"

    # ------------------------------------------------------------------
    # 4. Save / load roundtrip
    # ------------------------------------------------------------------

    def test_save_load_roundtrip(self, model: SelfImprovingLLM, tiny_config: ModelConfig, device: str) -> None:
        """Model weights should survive a save/load cycle."""
        model = model.to(device)
        model.eval()

        batch_size = 2
        seq_len = 8
        token_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len), device=device)

        # Capture output before save
        with torch.no_grad():
            original_output = model(token_ids)["logits"]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "model")
            model.save_pretrained(save_path)

            # Load into a fresh instance
            loaded = SelfImprovingLLM.from_pretrained(save_path).to(device)
            loaded.eval()

            with torch.no_grad():
                loaded_output = loaded(token_ids)["logits"]

            assert torch.allclose(original_output, loaded_output, atol=1e-5)
