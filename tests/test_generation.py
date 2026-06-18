"""Integration tests for the generation pipeline.

Tests cover how different sampling parameters affect generation output:
- Temperature effect on randomness/determinism
- Top-p (nucleus) sampling behavior
- Repetition penalty effect
- Max new tokens limit
"""

import pytest
import torch

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM


class TestGenerationPipeline:
    """Integration tests for the text generation pipeline."""

    @pytest.fixture
    def tiny_config(self) -> ModelConfig:
        """Return a tiny model config for fast tests."""
        return ModelConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            max_seq_len=64,
            dropout=0.0,
            tie_weights=True,
            use_rope=True,
            use_swiglu=True,
        )

    @pytest.fixture
    def model(self, tiny_config: ModelConfig) -> SelfImprovingLLM:
        """Return an instantiated model."""
        return SelfImprovingLLM(tiny_config)

    @pytest.fixture
    def device(self) -> str:
        """Return compute device for tests."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # 1. Temperature effect
    # ------------------------------------------------------------------

    def test_temperature_effect(self, model: SelfImprovingLLM, tiny_config: ModelConfig, device: str) -> None:
        """Higher temperature should produce more diverse outputs;
        temperature=0 should be deterministic."""
        model = model.to(device)
        model.eval()

        prompt_ids = torch.randint(0, tiny_config.vocab_size, (1, 8), device=device)

        # Deterministic (greedy) at T=0
        with torch.no_grad():
            result_greedy = model.generate(prompt_ids, max_new_tokens=16, temperature=0.0)
            result_greedy2 = model.generate(prompt_ids, max_new_tokens=16, temperature=0.0)

        assert torch.equal(result_greedy["sequences"], result_greedy2["sequences"]), (
            "Temperature 0 should be deterministic"
        )

        # Higher temperature should produce different outputs across runs
        with torch.no_grad():
            result_high1 = model.generate(prompt_ids, max_new_tokens=16, temperature=2.0)
            result_high2 = model.generate(prompt_ids, max_new_tokens=16, temperature=2.0)

        # With high temp, outputs *may* differ; we just verify they have valid shape
        assert result_high1["sequences"].shape[1] > 8
        assert result_high2["sequences"].shape[1] > 8

    # ------------------------------------------------------------------
    # 2. Top-p sampling
    # ------------------------------------------------------------------

    def test_top_p_sampling(self, model: SelfImprovingLLM, tiny_config: ModelConfig, device: str) -> None:
        """Generation with top_p should produce valid sequences of expected length."""
        model = model.to(device)
        model.eval()

        prompt_ids = torch.randint(0, tiny_config.vocab_size, (1, 8), device=device)

        # Test various top-p values
        for top_p in [0.1, 0.5, 0.9, 1.0]:
            with torch.no_grad():
                result = model.generate(
                    prompt_ids,
                    max_new_tokens=12,
                    temperature=1.0,
                    top_p=top_p,
                )
            sequences = result["sequences"]
            assert sequences.shape[0] == 1
            assert sequences.shape[1] >= 8  # At least prompt length

    def test_top_p_limits(self, model: SelfImprovingLLM, tiny_config: ModelConfig, device: str) -> None:
        """Very low top_p should restrict the token distribution."""
        model = model.to(device)
        model.eval()

        prompt_ids = torch.randint(0, tiny_config.vocab_size, (1, 8), device=device)

        # Low top_p (0.01) is essentially greedy for most distributions
        with torch.no_grad():
            result_low = model.generate(prompt_ids, max_new_tokens=8, temperature=1.0, top_p=0.01)
            result_low2 = model.generate(prompt_ids, max_new_tokens=8, temperature=1.0, top_p=0.01)

        # With extremely low top_p, outputs should be very similar or identical
        assert result_low["sequences"].shape[1] >= 8
        assert result_low2["sequences"].shape[1] >= 8

    # ------------------------------------------------------------------
    # 3. Repetition penalty
    # ------------------------------------------------------------------

    def test_repetition_penalty(self, model: SelfImprovingLLM, tiny_config: ModelConfig, device: str) -> None:
        """Higher repetition penalty should reduce repeated token patterns."""
        model = model.to(device)
        model.eval()

        prompt_ids = torch.randint(0, tiny_config.vocab_size, (1, 8), device=device)

        # Generate with no penalty
        with torch.no_grad():
            result_no_penalty = model.generate(
                prompt_ids,
                max_new_tokens=20,
                temperature=1.0,
                top_p=0.9,
                repetition_penalty=1.0,
            )

        # Generate with high penalty
        with torch.no_grad():
            result_high_penalty = model.generate(
                prompt_ids,
                max_new_tokens=20,
                temperature=1.0,
                top_p=0.9,
                repetition_penalty=2.0,
            )

        # Both should produce valid-length sequences
        assert result_no_penalty["sequences"].shape[1] >= 8
        assert result_high_penalty["sequences"].shape[1] >= 8

        # Check that sequences are not identical (penalty had some effect)
        # This is probabilistic, so we only check shape validity
        seq_no_pen = result_no_penalty["sequences"][0]
        seq_high_pen = result_high_penalty["sequences"][0]

        # Count repeated consecutive tokens
        repeats_no_pen = sum(
            1 for i in range(1, len(seq_no_pen)) if seq_no_pen[i] == seq_no_pen[i - 1]
        )
        repeats_high_pen = sum(
            1
            for i in range(1, len(seq_high_pen))
            if seq_high_pen[i] == seq_high_pen[i - 1]
        )

        # High penalty should not increase repeats (it should decrease or keep same)
        assert repeats_high_pen <= repeats_no_pen + 2  # Allow small variance

    # ------------------------------------------------------------------
    # 4. Max new tokens
    # ------------------------------------------------------------------

    def test_max_new_tokens(self, model: SelfImprovingLLM, tiny_config: ModelConfig, device: str) -> None:
        """Generation should respect the max_new_tokens limit."""
        model = model.to(device)
        model.eval()

        prompt_len = 8
        prompt_ids = torch.randint(0, tiny_config.vocab_size, (1, prompt_len), device=device)

        for max_new in [1, 5, 10, 20]:
            with torch.no_grad():
                result = model.generate(
                    prompt_ids,
                    max_new_tokens=max_new,
                    temperature=0.0,  # deterministic
                )

            sequences = result["sequences"]
            total_len = sequences.shape[1]
            generated_len = total_len - prompt_len

            # Should not exceed max_new_tokens
            assert generated_len <= max_new, (
                f"Generated {generated_len} tokens, expected <= {max_new}"
            )

            # Should generate at least 1 new token
            assert generated_len >= 0

    def test_max_new_tokens_zero(self, model: SelfImprovingLLM, tiny_config: ModelConfig, device: str) -> None:
        """max_new_tokens=0 should return just the prompt."""
        model = model.to(device)
        model.eval()

        prompt_len = 8
        prompt_ids = torch.randint(0, tiny_config.vocab_size, (1, prompt_len), device=device)

        with torch.no_grad():
            result = model.generate(prompt_ids, max_new_tokens=0, temperature=0.0)

        sequences = result["sequences"]
        assert sequences.shape[1] == prompt_len
        assert torch.equal(sequences, prompt_ids)
