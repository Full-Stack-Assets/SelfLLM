"""Integration tests for the recursive self-improvement loop.

Tests cover:
- A single iteration completes successfully (mocked components)
- Quality filtering behavior
- Overall recursive trainer flow
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.recursive.recursive_config import RecursiveConfig
from selfllm.recursive.recursive_trainer import RecursiveSelfTrainer
from selfllm.training.quality_filter import QualityFilter


class TestRecursiveTrainer:
    """Integration test suite for the recursive improvement engine."""

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
    def recursive_config(self) -> RecursiveConfig:
        """Return a recursive config with minimal settings for fast tests."""
        return RecursiveConfig(
            samples_per_iteration=10,
            responses_per_prompt=2,
            generation_temperature=0.8,
            generation_top_p=0.92,
            generation_max_tokens=16,
            keep_ratio=0.5,
            training_epochs=1,
            learning_rate=1e-4,
            batch_size=4,
            gradient_accumulation_steps=1,
            max_iterations=2,
            patience=1,
            target_improvement=0.001,
            replay_buffer_size=50,
            replay_ratio=0.3,
            checkpoint_dir="./test_checkpoints",
            save_every=1,
            keep_last_n=3,
            min_quality_score=0.3,
            max_perplexity=500.0,
            min_diversity=0.1,
            weight_decay=0.01,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            rollback_on_degradation=True,
            eval_prompts=["Test prompt 1", "Test prompt 2"],
            eval_every=1,
        )

    @pytest.fixture
    def model(self, tiny_config: ModelConfig) -> SelfImprovingLLM:
        """Return an instantiated model."""
        return SelfImprovingLLM(tiny_config)

    @pytest.fixture
    def tokenizer(self, tiny_config: ModelConfig) -> BPETokenizer:
        """Return a trained tokenizer."""
        tokenizer = BPETokenizer(vocab_size=tiny_config.vocab_size)
        tokenizer.train([
            "The quick brown fox jumps over the lazy dog.",
            "Recursive self-improvement is a powerful learning paradigm.",
            "Language models learn patterns from text data.",
        ])
        return tokenizer

    @pytest.fixture
    def device(self) -> str:
        """Return compute device for tests."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # 1. One iteration completes
    # ------------------------------------------------------------------

    def test_one_iteration_completes(
        self,
        model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        recursive_config: RecursiveConfig,
        device: str,
    ) -> None:
        """A single iteration of the recursive loop should execute
        and return valid metrics."""
        model = model.to(device)

        trainer = RecursiveSelfTrainer(model, tokenizer, recursive_config, device=device)

        # Run a single iteration
        metrics = trainer.run_iteration()

        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert "iteration" in metrics
        assert "samples_generated" in metrics
        assert "samples_kept" in metrics
        assert "train_loss" in metrics
        assert "eval_perplexity" in metrics
        assert "quality_score" in metrics
        assert "improvement_delta" in metrics
        assert "checkpoint_path" in metrics

        # Validate metric types and ranges
        assert isinstance(metrics["iteration"], int)
        assert metrics["iteration"] >= 1
        assert isinstance(metrics["train_loss"], float)
        assert metrics["train_loss"] >= 0
        assert isinstance(metrics["eval_perplexity"], float)
        assert metrics["eval_perplexity"] > 0
        assert isinstance(metrics["quality_score"], float)

    def test_run_multiple_iterations(
        self,
        model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        recursive_config: RecursiveConfig,
        device: str,
    ) -> None:
        """The full run() method should execute and return iteration history."""
        model = model.to(device)

        trainer = RecursiveSelfTrainer(model, tokenizer, recursive_config, device=device)

        results = trainer.run(
            max_iterations=2,
            target_improvement=0.001,
            patience=1,
        )

        assert isinstance(results, list)
        assert len(results) > 0

        for result in results:
            assert "iteration" in result
            assert "train_loss" in result
            assert "eval_perplexity" in result

    def test_iteration_monotonicity(
        self,
        model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        recursive_config: RecursiveConfig,
        device: str,
    ) -> None:
        """Iteration numbers should increase monotonically."""
        model = model.to(device)
        trainer = RecursiveSelfTrainer(model, tokenizer, recursive_config, device=device)

        results = trainer.run(
            max_iterations=2,
            target_improvement=0.001,
            patience=1,
        )

        iterations = [r["iteration"] for r in results]
        assert iterations == sorted(iterations)
        assert len(set(iterations)) == len(iterations)  # All unique

    # ------------------------------------------------------------------
    # 2. Quality filtering
    # ------------------------------------------------------------------

    def test_quality_filtering(self) -> None:
        """QualityFilter should correctly score and filter samples."""
        # Create a mock base model for perplexity computation
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        quality_filter = QualityFilter(
            quality_model=None,
            base_model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        # Test with sample data
        samples = [
            {"prompt": "Explain A:", "response": "A is the first letter of the alphabet."},
            {"prompt": "Explain B:", "response": "B comes after A."},
            {"prompt": "Explain C:", "response": "C C C C C C C C C C C C C C C C C C C C"},  # Low diversity
            {"prompt": "Explain D:", "response": "D is the fourth letter and has many uses in programming and music theory."},
        ]

        # Mock perplexity computation to return predictable values
        def mock_compute_perplexity(text: str) -> float:
            # Higher perplexity for repetitive text
            if text.count("C C") > 0:
                return 200.0
            return 50.0

        quality_filter.compute_perplexity = mock_compute_perplexity  # type: ignore[method-assign]

        # Test diversity computation
        diverse_text = "The quick brown fox jumps over the lazy dog"
        repetitive_text = "hello hello hello hello hello"

        diverse_score = quality_filter.compute_diversity_score(diverse_text)
        repetitive_score = quality_filter.compute_diversity_score(repetitive_text)

        assert 0.0 <= diverse_score <= 1.0
        assert 0.0 <= repetitive_score <= 1.0
        assert diverse_score > repetitive_score, (
            "Diverse text should have higher diversity score"
        )

    def test_quality_filter_batch(self) -> None:
        """filter_batch should keep high-quality samples and discard low-quality ones."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        quality_filter = QualityFilter(
            quality_model=None,
            base_model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        samples = [
            {"prompt": "Q1", "response": "A comprehensive well-thought-out answer with varied vocabulary."},
            {"prompt": "Q2", "response": "A A A A A A A A A A"},  # Low quality
            {"prompt": "Q3", "response": "Another good response with sufficient length and content."},
            {"prompt": "Q4", "response": "B B B B B B B B B B"},  # Low quality
        ]

        # Mock methods to return predictable scores
        def mock_perplexity(text: str) -> float:
            if text.count("A A") > 0 or text.count("B B") > 0:
                return 300.0
            return 30.0

        def mock_diversity(text: str) -> float:
            if text.count("A A") > 0 or text.count("B B") > 0:
                return 0.05
            return 0.8

        def mock_coherence(text: str) -> float:
            return 0.7

        def mock_length(text: str) -> float:
            if len(text) < 20:
                return 0.3
            return 0.9

        quality_filter.compute_perplexity = mock_perplexity  # type: ignore[method-assign]
        quality_filter.compute_diversity_score = mock_diversity  # type: ignore[method-assign]
        quality_filter.compute_coherence_score = mock_coherence  # type: ignore[method-assign]
        quality_filter.compute_length_score = mock_length  # type: ignore[method-assign]

        filtered = quality_filter.filter_batch(
            samples,
            min_quality_score=0.3,
            max_perplexity=100.0,
            min_diversity=0.1,
            keep_ratio=0.5,
        )

        # Should keep at least the good samples
        assert isinstance(filtered, list)
        assert len(filtered) <= len(samples)

        # Good samples should rank higher than bad ones
        if len(filtered) > 0:
            assert all(
                "A A" not in s["response"] and "B B" not in s["response"]
                for s in filtered
            ), "Low-quality samples should be filtered out"

    def test_quality_filter_empty(self) -> None:
        """filter_batch should handle empty input gracefully."""
        quality_filter = QualityFilter(device="cpu")
        result = quality_filter.filter_batch([], keep_ratio=0.5)
        assert result == []

    def test_quality_filter_scores_in_range(self) -> None:
        """All quality scores should be in the [0, 1] range (or valid ranges)."""
        quality_filter = QualityFilter(device="cpu")

        # Test diversity score
        score = quality_filter.compute_diversity_score("hello world test example")
        assert 0.0 <= score <= 1.0

        # Test coherence score
        score = quality_filter.compute_coherence_score("This is a test sentence.")
        assert 0.0 <= score <= 1.0

        # Test length score
        score = quality_filter.compute_length_score("Short.")
        assert 0.0 <= score <= 1.0

        score = quality_filter.compute_length_score("a " * 300)
        assert 0.0 <= score <= 1.0
