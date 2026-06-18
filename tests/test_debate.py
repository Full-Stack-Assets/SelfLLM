"""Tests for the multi-agent debate module."""

import sys
import unittest

import torch

# Ensure the project root is on the path
sys.path.insert(0, "./")

from selfllm.debate import (
    DebateAgent,
    DebateOrchestrator,
    DebateResult,
    agreement_score,
    jaccard_similarity,
    majority_vote,
    normalize_answer,
    select_consensus,
)
from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer


class ScriptedModel:
    """Deterministic model stub returning a canned per-agent continuation.

    Lets us test agent prompt-handling/decoding without a real network or
    weights. ``generate`` appends a fixed sequence of token ids to the prompt.
    """

    def __init__(self, reply_ids, vocab_size=50, max_seq_len=512):
        self.config = type("C", (), {"vocab_size": vocab_size, "max_seq_len": max_seq_len})()
        self.reply_ids = reply_ids

    def generate(self, prompt_ids, max_new_tokens=48, temperature=1.0,
                 top_p=0.9, top_k=50, stop_token_id=None):
        batch = prompt_ids.shape[0]
        reply = torch.tensor([self.reply_ids[:max_new_tokens]] * batch)
        seq = torch.cat([prompt_ids, reply], dim=1)
        return {"sequences": seq, "scores": torch.ones(batch, reply.shape[1])}


# --------------------------------------------------------------------- #
# Voting / consensus / confidence logic (model-independent)
# --------------------------------------------------------------------- #


class TestVotingLogic(unittest.TestCase):
    """Unit tests for the deterministic consensus/confidence functions."""

    def test_normalize_answer(self):
        self.assertEqual(normalize_answer("  The Answer. "), "the answer")
        self.assertEqual(normalize_answer("Yes!"), "yes")
        self.assertEqual(normalize_answer("a\n  b"), "a b")

    def test_majority_vote_basic(self):
        best, conf, counts = majority_vote(["cat", "cat", "dog"])
        self.assertEqual(normalize_answer(best), "cat")
        self.assertAlmostEqual(conf, 2 / 3)
        self.assertEqual(counts["cat"], 2)
        self.assertEqual(counts["dog"], 1)

    def test_majority_vote_normalization(self):
        # Differing case/punctuation/whitespace count as the same answer.
        best, conf, _ = majority_vote(["Paris.", "paris", " PARIS "])
        self.assertEqual(normalize_answer(best), "paris")
        self.assertAlmostEqual(conf, 1.0)

    def test_unanimous_confidence_is_one(self):
        best, conf, counts = majority_vote(["42", "42", "42", "42"])
        self.assertAlmostEqual(conf, 1.0)
        self.assertEqual(len(counts), 1)

    def test_full_disagreement_low_confidence(self):
        _, conf, _ = majority_vote(["a", "b", "c", "d"])
        self.assertAlmostEqual(conf, 0.25)

    def test_majority_vote_tie_break_deterministic(self):
        # Tie between "a" and "b"; earliest-appearing wins, deterministically.
        for _ in range(5):
            best, conf, _ = majority_vote(["a", "b", "b", "a"])
            self.assertEqual(normalize_answer(best), "a")
            self.assertAlmostEqual(conf, 0.5)

    def test_majority_vote_empty(self):
        best, conf, counts = majority_vote([])
        self.assertEqual(best, "")
        self.assertEqual(conf, 0.0)
        self.assertEqual(counts, {})

    def test_jaccard_similarity(self):
        self.assertAlmostEqual(jaccard_similarity("the cat", "the cat"), 1.0)
        self.assertAlmostEqual(jaccard_similarity("the cat", "the dog"), 1 / 3)
        self.assertAlmostEqual(jaccard_similarity("cat", "dog"), 0.0)
        self.assertAlmostEqual(jaccard_similarity("", ""), 1.0)
        self.assertAlmostEqual(jaccard_similarity("cat", ""), 0.0)

    def test_agreement_score(self):
        self.assertAlmostEqual(agreement_score(["x"]), 1.0)
        self.assertAlmostEqual(agreement_score(["same", "same", "same"]), 1.0)
        self.assertAlmostEqual(agreement_score(["a", "b", "c"]), 0.0)

    def test_select_consensus_majority(self):
        consensus, conf, counts = select_consensus(["yes", "yes", "no"])
        self.assertEqual(normalize_answer(consensus), "yes")
        self.assertAlmostEqual(conf, 2 / 3)

    def test_select_consensus_similarity_blend(self):
        # Two identical + one overlapping answer: similarity lifts confidence
        # above the pure majority fraction.
        answers = ["the sky is blue", "the sky is blue", "the sky is grey"]
        _, maj_conf, _ = select_consensus(answers, use_similarity=False)
        _, sim_conf, _ = select_consensus(answers, use_similarity=True)
        self.assertAlmostEqual(maj_conf, 2 / 3)
        self.assertGreater(sim_conf, 0.0)
        self.assertLessEqual(sim_conf, 1.0)

    def test_confidence_bounds(self):
        for answers in (["a"], ["a", "b"], ["a", "a", "b", "c"]):
            for use_sim in (True, False):
                _, conf, _ = select_consensus(answers, use_similarity=use_sim)
                self.assertGreaterEqual(conf, 0.0)
                self.assertLessEqual(conf, 1.0)


# --------------------------------------------------------------------- #
# Agent prompt construction
# --------------------------------------------------------------------- #


class TestDebateAgentPrompts(unittest.TestCase):
    """Test that an agent's prompts embed its perspective."""

    def setUp(self):
        self.tokenizer = BPETokenizer(vocab_size=200)
        self.tokenizer.train(
            ["the quick brown fox jumps over the lazy dog 0123456789",
             "you are a skeptic optimist question answer critique revise"]
        )
        self.agent = DebateAgent(
            name="skeptic",
            model=ScriptedModel(reply_ids=[5, 6, 7]),
            tokenizer=self.tokenizer,
            perspective="You are a skeptical analyst.",
            temperature=0.5,
        )

    def test_propose_prompt_includes_perspective(self):
        prompt = self.agent.build_propose_prompt("What is 2 + 2?")
        self.assertIn("You are a skeptical analyst.", prompt)
        self.assertIn("What is 2 + 2?", prompt)

    def test_critique_prompt_includes_perspective_and_other(self):
        prompt = self.agent.build_critique_prompt("Q?", "the moon is cheese")
        self.assertIn("You are a skeptical analyst.", prompt)
        self.assertIn("the moon is cheese", prompt)

    def test_revise_prompt_includes_critiques(self):
        prompt = self.agent.build_revise_prompt(
            "Q?", "my first answer", ["too vague", "no evidence"]
        )
        self.assertIn("You are a skeptical analyst.", prompt)
        self.assertIn("my first answer", prompt)
        self.assertIn("too vague", prompt)
        self.assertIn("no evidence", prompt)

    def test_propose_returns_string(self):
        out = self.agent.propose("Hello?")
        self.assertIsInstance(out, str)


# --------------------------------------------------------------------- #
# End-to-end debate with a tiny real model
# --------------------------------------------------------------------- #


class TestDebateEndToEnd(unittest.TestCase):
    """Run a real (tiny) model through the full debate loop."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.tokenizer = BPETokenizer(vocab_size=120)
        cls.tokenizer.train(
            ["the quick brown fox jumps over the lazy dog",
             "you are an optimist a skeptic question answer critique revise it"]
        )
        cls.config = ModelConfig(
            vocab_size=max(cls.tokenizer.vocab_size, 40),
            d_model=32,
            n_layers=2,
            n_heads=2,
            d_ff=64,
            max_seq_len=128,
            dropout=0.0,
        )
        cls.model = SelfImprovingLLM(cls.config)

    def _make_agents(self):
        return [
            DebateAgent("optimist", self.model, self.tokenizer,
                        perspective="You are an optimist.",
                        temperature=0.8, max_new_tokens=6, device="cpu"),
            DebateAgent("skeptic", self.model, self.tokenizer,
                        perspective="You are a skeptic.",
                        temperature=0.8, max_new_tokens=6, device="cpu"),
        ]

    def test_debate_completes_and_is_well_formed(self):
        orchestrator = DebateOrchestrator(self._make_agents(), num_rounds=2)
        result = orchestrator.run("Is the sky blue?")

        self.assertIsInstance(result, DebateResult)
        self.assertEqual(result.question, "Is the sky blue?")
        self.assertIsInstance(result.consensus, str)
        self.assertIsInstance(result.confidence, float)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

        # Two agents, both have final answers.
        self.assertEqual(set(result.final_answers), {"optimist", "skeptic"})

        # Two rounds in the transcript: propose, then critique+revise.
        self.assertEqual(len(result.transcript), 2)
        self.assertEqual(result.transcript[0]["round"], 1)
        actions_r1 = {e["action"] for e in result.transcript[0]["events"]}
        self.assertEqual(actions_r1, {"propose"})
        actions_r2 = {e["action"] for e in result.transcript[1]["events"]}
        self.assertEqual(actions_r2, {"critique", "revise"})

    def test_single_round_is_propose_only(self):
        orchestrator = DebateOrchestrator(self._make_agents(), num_rounds=1)
        result = orchestrator.run("Hello?")
        self.assertEqual(len(result.transcript), 1)
        self.assertEqual(
            {e["action"] for e in result.transcript[0]["events"]}, {"propose"}
        )

    def test_orchestrator_rejects_bad_args(self):
        agents = self._make_agents()
        with self.assertRaises(ValueError):
            DebateOrchestrator([], num_rounds=1)
        with self.assertRaises(ValueError):
            DebateOrchestrator(agents, num_rounds=0)


class TestDebateExports(unittest.TestCase):
    """Verify the package exports its public API."""

    def test_all_exports(self):
        import selfllm.debate as d

        for name in ("DebateAgent", "DebateOrchestrator", "DebateResult"):
            self.assertIn(name, d.__all__)
            self.assertTrue(hasattr(d, name))


if __name__ == "__main__":
    unittest.main()
