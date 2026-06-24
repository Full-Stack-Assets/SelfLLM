"""Tests for Chain-of-Thought reasoning module."""

import sys
import unittest
from unittest.mock import MagicMock

import torch

# Ensure the project root is on the path
sys.path.insert(0, "./")

from selfllm.cot.cot_generator import (
    ChainOfThoughtGenerator,
    ChainOfThoughtPrompts,
)


class MockTokenizer:
    """Mock tokenizer for unit testing."""

    PAD_TOKEN_ID = 0
    EOS_TOKEN_ID = 1
    UNK_TOKEN_ID = 2

    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
        # Pre-populate some tokens
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
            self.vocab[c] = 3 + i
        for i, c in enumerate("0123456789"):
            self.vocab[c] = 29 + i
        for w in [" ", ".", ",", "?", "!", "<", ">", "/", "=", "+", "-", "*", "%"]:
            self.vocab[w] = len(self.vocab)
        self.inverse = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_token_id(self):
        return self.PAD_TOKEN_ID

    @property
    def eos_token_id(self):
        return self.EOS_TOKEN_ID

    def encode(self, text):
        return [self.vocab.get(c, self.UNK_TOKEN_ID) for c in text.lower()]

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(
            self.inverse.get(t, "<unk>") for t in token_ids if t != self.PAD_TOKEN_ID
        )


class MockModel:
    """Mock SelfImprovingLLM for testing CoT generation."""

    def __init__(self, device="cpu"):
        self.config = MagicMock()
        self.config.vocab_size = 50
        self.config.max_seq_len = 512
        self.device = device

    def generate(self, prompt_ids, max_new_tokens=128, temperature=1.0, top_p=0.95):
        """Return a mock CoT response."""
        batch_size, prompt_len = prompt_ids.shape
        # Generate a mock sequence that looks like:
        # <think>reasoning...</think><answer>answer</answer>
        num_tokens = min(max_new_tokens, 50)
        vocab_size = self.config.vocab_size
        seq = torch.randint(3, vocab_size, (batch_size, prompt_len + num_tokens))
        return {"sequences": seq, "scores": torch.ones(batch_size, num_tokens)}

    def to(self, device):
        return self

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, *args, **kwargs):
        batch_size = args[0].shape[0] if args else 1
        seq_len = args[0].shape[1] if args else 10
        vocab = self.config.vocab_size
        logits = torch.randn(batch_size, seq_len, vocab)
        return {
            "logits": logits,
            "hidden_states": torch.randn(batch_size, seq_len, 64),
        }


class TestChainOfThoughtPrompts(unittest.TestCase):
    """Test the prompt template collections."""

    def test_templates_not_empty(self):
        self.assertTrue(len(ChainOfThoughtPrompts.TEMPLATES) > 0)

    def test_math_problems_not_empty(self):
        self.assertTrue(len(ChainOfThoughtPrompts.MATH_PROBLEMS) > 0)

    def test_logic_problems_not_empty(self):
        self.assertTrue(len(ChainOfThoughtPrompts.LOGIC_PROBLEMS) > 0)

    def test_template_formatting(self):
        for tmpl in ChainOfThoughtPrompts.TEMPLATES:
            formatted = tmpl.format(problem="What is 2+2?")
            self.assertIn("2+2", formatted)

    def test_all_math_problems_are_strings(self):
        for p in ChainOfThoughtPrompts.MATH_PROBLEMS:
            self.assertIsInstance(p, str)
            self.assertTrue(len(p) > 0)

    def test_all_logic_problems_are_strings(self):
        for p in ChainOfThoughtPrompts.LOGIC_PROBLEMS:
            self.assertIsInstance(p, str)
            self.assertTrue(len(p) > 0)


class TestChainOfThoughtGenerator(unittest.TestCase):
    """Test the CoT generator."""

    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.model = MockModel()
        self.generator = ChainOfThoughtGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device="cpu",
        )

    def test_init(self):
        self.assertIs(self.generator.model, self.model)
        self.assertIs(self.generator.tokenizer, self.tokenizer)
        self.assertEqual(self.generator.device, "cpu")

    def test_generate_cot_response_structure(self):
        result = self.generator.generate_cot_response(
            "What is 2+2?", template_idx=0
        )
        self.assertIn("thinking", result)
        self.assertIn("answer", result)
        self.assertIn("full_text", result)
        self.assertIsInstance(result["thinking"], str)
        self.assertIsInstance(result["answer"], str)
        self.assertIsInstance(result["full_text"], str)

    def test_generate_cot_response_different_templates(self):
        results = []
        for i in range(min(3, len(ChainOfThoughtPrompts.TEMPLATES))):
            r = self.generator.generate_cot_response("Test", template_idx=i)
            results.append(r)
            self.assertIn("full_text", r)

    def test_self_consistency_vote_structure(self):
        result = self.generator.self_consistency_vote(
            "What is 5*5?", num_paths=3
        )
        self.assertIn("answer", result)
        self.assertIn("confidence", result)
        self.assertIn("all_paths", result)
        self.assertIn("answer_counts", result)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["all_paths"], list)
        self.assertEqual(len(result["all_paths"]), 3)

    def test_self_consistency_vote_confidence_range(self):
        result = self.generator.self_consistency_vote(
            "Test problem", num_paths=3
        )
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_generate_cot_training_data(self):
        samples = self.generator.generate_cot_training_data(num_samples=5)
        self.assertEqual(len(samples), 5)
        for s in samples:
            self.assertIn("prompt", s)
            self.assertIn("response", s)
            self.assertIn("cot_text", s)
            self.assertIn("answer", s)
            self.assertIn("token_ids", s)
            self.assertTrue(s["response"].startswith("<think>"))
            self.assertIn("</think><answer>", s["response"])
            self.assertIn("</answer>", s["response"])

    def test_generate_cot_training_data_empty(self):
        samples = self.generator.generate_cot_training_data(num_samples=0)
        self.assertEqual(len(samples), 0)

    def test_generate_cot_training_data_wraps_problems(self):
        num_samples = len(ChainOfThoughtPrompts.MATH_PROBLEMS) + 3
        samples = self.generator.generate_cot_training_data(
            num_samples=num_samples
        )
        self.assertEqual(len(samples), num_samples)


class TestChainOfThoughtIntegration(unittest.TestCase):
    """Integration-style tests with real model components."""

    def test_cot_module_import(self):
        """Verify the CoT module can be imported correctly."""
        from selfllm.cot import ChainOfThoughtGenerator, ChainOfThoughtPrompts

        self.assertTrue(callable(ChainOfThoughtGenerator))
        self.assertTrue(hasattr(ChainOfThoughtPrompts, "TEMPLATES"))
        self.assertTrue(hasattr(ChainOfThoughtPrompts, "MATH_PROBLEMS"))
        self.assertTrue(hasattr(ChainOfThoughtPrompts, "LOGIC_PROBLEMS"))

    def test_cot_module_all_exports(self):
        import selfllm.cot

        self.assertIn("ChainOfThoughtGenerator", selfllm.cot.__all__)
        self.assertIn("ChainOfThoughtPrompts", selfllm.cot.__all__)


if __name__ == "__main__":
    unittest.main()
