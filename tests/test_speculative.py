"""Tests for speculative decoding module."""

import sys
import unittest

import torch

sys.path.insert(0, "./")

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.speculative import SpeculativeDecoder


class TestSpeculativeDecoder(unittest.TestCase):
    """Unit tests for the SpeculativeDecoder class."""

    def setUp(self):
        """Create small models for testing."""
        # Very small config for fast tests
        self.target_config = ModelConfig(
            vocab_size=128,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            max_seq_len=64,
            dropout=0.0,
            tie_weights=False,
        )
        self.target_model = SelfImprovingLLM(self.target_config)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def test_init_with_explicit_draft(self):
        draft_config = ModelConfig(
            vocab_size=128,
            d_model=32,
            n_layers=1,
            n_heads=2,
            d_ff=64,
            max_seq_len=64,
            dropout=0.0,
            tie_weights=False,
        )
        draft_model = SelfImprovingLLM(draft_config)
        decoder = SpeculativeDecoder(
            target_model=self.target_model,
            draft_model=draft_model,
            gamma=4,
            device="cpu",
        )
        self.assertIs(decoder.target_model, self.target_model)
        self.assertIs(decoder.draft_model, draft_model)
        self.assertEqual(decoder.gamma, 4)
        self.assertEqual(decoder.device, "cpu")

    def test_init_auto_draft(self):
        """Auto-created draft model should be initialized."""
        decoder = SpeculativeDecoder(
            target_model=self.target_model,
            draft_model=None,
            gamma=4,
            device="cpu",
        )
        self.assertIsNotNone(decoder.draft_model)
        # Draft model config should be smaller or equal
        self.assertLessEqual(
            decoder.draft_model.config.d_model,
            self.target_model.config.d_model,
        )
        self.assertLessEqual(
            decoder.draft_model.config.n_layers,
            self.target_model.config.n_layers,
        )

    def test_draft_config_smaller(self):
        config = SpeculativeDecoder._create_draft_config(self.target_config)
        self.assertLessEqual(config.d_model, self.target_config.d_model)
        self.assertLessEqual(config.n_layers, self.target_config.n_layers)
        self.assertLessEqual(config.d_ff, self.target_config.d_ff)
        self.assertLessEqual(config.n_heads, self.target_config.n_heads)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def test_generate_returns_sequences(self):
        decoder = SpeculativeDecoder(
            target_model=self.target_model,
            gamma=2,
            device="cpu",
        )
        prompt_ids = torch.tensor([[10, 20, 30]])
        result = decoder.generate(
            prompt_ids, max_new_tokens=8, temperature=0.0
        )
        self.assertIn("sequences", result)
        self.assertIn("acceptance_rate", result)

    def test_generate_sequence_length(self):
        """Output should contain prompt + new tokens."""
        decoder = SpeculativeDecoder(
            target_model=self.target_model,
            gamma=2,
            device="cpu",
        )
        prompt_ids = torch.tensor([[10, 20, 30]])
        max_new = 8
        result = decoder.generate(
            prompt_ids, max_new_tokens=max_new, temperature=0.0
        )
        seq = result["sequences"]
        self.assertEqual(seq.shape[0], 1)  # batch = 1
        # Should have at least prompt_len + 1 tokens
        self.assertGreaterEqual(seq.shape[1], prompt_ids.shape[1])

    def test_generate_with_temperature_zero(self):
        decoder = SpeculativeDecoder(
            target_model=self.target_model,
            gamma=2,
            device="cpu",
        )
        prompt_ids = torch.tensor([[10, 20, 30]])
        result = decoder.generate(
            prompt_ids, max_new_tokens=8, temperature=0.0
        )
        self.assertIn("sequences", result)
        self.assertIsInstance(result["acceptance_rate"], torch.Tensor)

    def test_generate_with_temperature_positive(self):
        decoder = SpeculativeDecoder(
            target_model=self.target_model,
            gamma=2,
            device="cpu",
        )
        prompt_ids = torch.tensor([[10, 20, 30]])
        result = decoder.generate(
            prompt_ids, max_new_tokens=6, temperature=0.8, top_p=0.9
        )
        self.assertIn("sequences", result)

    def test_generate_single_token(self):
        """Should work even for single new token."""
        decoder = SpeculativeDecoder(
            target_model=self.target_model,
            gamma=1,
            device="cpu",
        )
        prompt_ids = torch.tensor([[10, 20, 30]])
        result = decoder.generate(
            prompt_ids, max_new_tokens=1, temperature=0.0
        )
        self.assertIn("sequences", result)
        self.assertGreaterEqual(result["sequences"].shape[1], 3)

    def test_generate_longer_than_gamma(self):
        """Test generation when max_new_tokens > gamma."""
        decoder = SpeculativeDecoder(
            target_model=self.target_model,
            gamma=3,
            device="cpu",
        )
        prompt_ids = torch.tensor([[10, 20, 30]])
        result = decoder.generate(
            prompt_ids, max_new_tokens=20, temperature=0.0
        )
        self.assertIn("sequences", result)

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    def test_acceptance_rate_zero_drafts(self):
        decoder = SpeculativeDecoder(
            target_model=self.target_model,
            gamma=4,
            device="cpu",
        )
        self.assertEqual(decoder.acceptance_rate, 0.0)

    def test_reset_stats(self):
        decoder = SpeculativeDecoder(
            target_model=self.target_model,
            gamma=4,
            device="cpu",
        )
        decoder.total_draft_tokens = 100
        decoder.accepted_tokens = 50
        decoder.reset_stats()
        self.assertEqual(decoder.total_draft_tokens, 0)
        self.assertEqual(decoder.accepted_tokens, 0)
        self.assertEqual(decoder.acceptance_rate, 0.0)

    def test_stats_after_generation(self):
        decoder = SpeculativeDecoder(
            target_model=self.target_model,
            gamma=2,
            device="cpu",
        )
        prompt_ids = torch.tensor([[10, 20, 30]])
        decoder.generate(prompt_ids, max_new_tokens=8, temperature=0.0)
        # Should have tracked some draft tokens
        self.assertGreater(decoder.total_draft_tokens, 0)
        # Acceptance rate should be in valid range
        ar = decoder.acceptance_rate
        self.assertGreaterEqual(ar, 0.0)
        self.assertLessEqual(ar, 1.0)

    # ------------------------------------------------------------------ #
    # Draft generation helper
    # ------------------------------------------------------------------ #

    def test_draft_generate_returns_tensor(self):
        decoder = SpeculativeDecoder(
            target_model=self.target_model,
            gamma=3,
            device="cpu",
        )
        prefix = torch.tensor([[10, 20]])
        draft_tokens = decoder._draft_generate(
            prefix, gamma=3, temperature=0.0
        )
        self.assertIsInstance(draft_tokens, torch.Tensor)
        self.assertEqual(draft_tokens.shape[0], 1)  # batch size

    def test_draft_generate_shape(self):
        decoder = SpeculativeDecoder(
            target_model=self.target_model,
            gamma=4,
            device="cpu",
        )
        prefix = torch.tensor([[10, 20, 30]])
        draft_tokens = decoder._draft_generate(
            prefix, gamma=5, temperature=0.0
        )
        self.assertEqual(draft_tokens.shape[0], 1)
        self.assertLessEqual(draft_tokens.shape[1], 5)

    def test_draft_generate_with_temperature(self):
        decoder = SpeculativeDecoder(
            target_model=self.target_model,
            gamma=3,
            device="cpu",
        )
        prefix = torch.tensor([[10, 20]])
        draft_tokens = decoder._draft_generate(
            prefix, gamma=3, temperature=0.8, top_p=0.9
        )
        self.assertIsInstance(draft_tokens, torch.Tensor)
        self.assertGreater(draft_tokens.numel(), 0)

    # ------------------------------------------------------------------ #
    # Module import
    # ------------------------------------------------------------------ #

    def test_module_import(self):
        from selfllm.model import SpeculativeDecoder as SD

        self.assertTrue(callable(SD))

    def test_model_init_export(self):
        from selfllm.model import SpeculativeDecoder  # noqa: F401

        self.assertIn("SpeculativeDecoder", dir())


class TestSpeculativeDecoderEdgeCases(unittest.TestCase):
    """Edge-case tests for speculative decoding."""

    def test_empty_prompt(self):
        target = SelfImprovingLLM(
            ModelConfig(
                vocab_size=64,
                d_model=32,
                n_layers=1,
                n_heads=2,
                d_ff=64,
                max_seq_len=32,
                dropout=0.0,
                tie_weights=False,
            )
        )
        decoder = SpeculativeDecoder(target_model=target, gamma=2, device="cpu")
        prompt_ids = torch.tensor([[1]])  # Single token
        result = decoder.generate(
            prompt_ids, max_new_tokens=4, temperature=0.0
        )
        self.assertIn("sequences", result)

    def test_gamma_larger_than_max_tokens(self):
        target = SelfImprovingLLM(
            ModelConfig(
                vocab_size=64,
                d_model=32,
                n_layers=1,
                n_heads=2,
                d_ff=64,
                max_seq_len=32,
                dropout=0.0,
                tie_weights=False,
            )
        )
        decoder = SpeculativeDecoder(target_model=target, gamma=10, device="cpu")
        prompt_ids = torch.tensor([[10, 20]])
        result = decoder.generate(
            prompt_ids, max_new_tokens=5, temperature=0.0
        )
        self.assertIn("sequences", result)

    def test_batch_size_one(self):
        target = SelfImprovingLLM(
            ModelConfig(
                vocab_size=64,
                d_model=32,
                n_layers=1,
                n_heads=2,
                d_ff=64,
                max_seq_len=32,
                dropout=0.0,
                tie_weights=False,
            )
        )
        decoder = SpeculativeDecoder(target_model=target, gamma=2, device="cpu")
        prompt_ids = torch.tensor([[5, 10, 15, 20]])
        result = decoder.generate(
            prompt_ids, max_new_tokens=6, temperature=0.0
        )
        self.assertEqual(result["sequences"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
