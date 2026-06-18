"""Tests for V6 Wave 2 reasoning: self-consistency, verifiers, best-of-N."""

import pytest
import torch

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.reasoning import (
    BeamSearchReasoner,
    BestOfNStrategy,
    Candidate,
    QualityVerifier,
    ReasoningResult,
    SelfConsistencyStrategy,
    SelfConsistencyVerifier,
    Verifier,
)


class _PickAnswer(Verifier):
    """Verifier that scores 1.0 for candidates whose answer equals `target`."""

    def __init__(self, target):
        self.target = target

    def score(self, question, candidates):
        return [1.0 if (c.answer or "") == self.target else 0.0 for c in candidates]


@pytest.fixture
def tiny_model_tok():
    cfg = ModelConfig(
        vocab_size=256, d_model=64, n_layers=2, n_heads=4, d_ff=128,
        max_seq_len=128, dropout=0.0,
    )
    model = SelfImprovingLLM(cfg).eval()
    tok = BPETokenizer(vocab_size=256)
    tok.train([
        "Let's think step by step. The answer is 4.",
        "First compute the sum. The result is seven.",
        "Reasoning carefully and concluding the solution.",
    ])
    return model, tok


def _traces(answers):
    """Build fake CoT trace dicts with the given answers (full_text varies len)."""
    return [
        {"answer": a, "full_text": ("word " * (i + 1)) + f"answer {a}"}
        for i, a in enumerate(answers)
    ]


class _StubQuality:
    """Quality model stub: scores a sequence by its length (prefers longer)."""

    def eval(self):
        return self

    def score_sequences(self, seq):
        return torch.tensor([seq.shape[1] / 100.0])


# --------------------------------------------------------------------------- #
# SelfConsistencyVerifier
# --------------------------------------------------------------------------- #


def test_self_consistency_verifier_scores():
    cands = [Candidate("t", "4"), Candidate("t", "4"),
             Candidate("t", "4"), Candidate("t", "7")]
    scores = SelfConsistencyVerifier().score("q", cands)
    assert scores == [0.75, 0.75, 0.75, 0.25]


def test_self_consistency_verifier_none_answer_scores_zero():
    cands = [Candidate("t", "4"), Candidate("t", None)]
    scores = SelfConsistencyVerifier().score("q", cands)
    assert scores[1] == 0.0


def test_self_consistency_verifier_empty():
    assert SelfConsistencyVerifier().score("q", []) == []


# --------------------------------------------------------------------------- #
# QualityVerifier
# --------------------------------------------------------------------------- #


def test_quality_verifier_prefers_longer(tiny_model_tok):
    _, tok = tiny_model_tok
    verifier = QualityVerifier(_StubQuality(), tok, device="cpu")
    cands = [Candidate("a", "4"), Candidate("a b c d e f", "7")]
    scores = verifier.score("q", cands)
    assert scores[1] > scores[0]  # longer text scores higher under the stub


# --------------------------------------------------------------------------- #
# SelfConsistencyStrategy (deterministic via patched sampler)
# --------------------------------------------------------------------------- #


def test_self_consistency_majority(monkeypatch, tiny_model_tok):
    model, tok = tiny_model_tok
    monkeypatch.setattr(
        "selfllm.reasoning.self_consistency.sample_cot_traces",
        lambda *a, **k: _traces(["4", "4", "7"]),
    )
    strat = SelfConsistencyStrategy(model, tok, answer_type="free",
                                    num_samples=3, device="cpu")
    r = strat.solve("what is 2+2?")
    assert isinstance(r, ReasoningResult)
    assert r.answer == "4"
    assert r.confidence == pytest.approx(2 / 3)
    assert r.num_samples == 3
    assert len(r.traces) == 3


def test_self_consistency_all_unextractable(monkeypatch, tiny_model_tok):
    model, tok = tiny_model_tok
    monkeypatch.setattr(
        "selfllm.reasoning.self_consistency.sample_cot_traces",
        lambda *a, **k: [{"answer": "", "full_text": "   "} for _ in range(3)],
    )
    strat = SelfConsistencyStrategy(model, tok, answer_type="numeric",
                                    num_samples=3, device="cpu")
    r = strat.solve("q")
    assert r.answer is None
    assert r.confidence == 0.0


# --------------------------------------------------------------------------- #
# BestOfNStrategy
# --------------------------------------------------------------------------- #


def test_best_of_n_self_consistency(monkeypatch, tiny_model_tok):
    model, tok = tiny_model_tok
    monkeypatch.setattr(
        "selfllm.reasoning.best_of_n.sample_cot_traces",
        lambda *a, **k: _traces(["4", "4", "7"]),
    )
    strat = BestOfNStrategy(model, tok, SelfConsistencyVerifier(),
                            answer_type="free", num_samples=3, device="cpu")
    r = strat.solve("q")
    assert r.answer == "4"  # majority answer wins the verifier
    assert r.details["best_idx"] in (0, 1)
    assert r.confidence == pytest.approx(2 / 3)


def test_best_of_n_quality_verifier_picks_longest(monkeypatch, tiny_model_tok):
    model, tok = tiny_model_tok
    # answers ascend with length; the longest (index 2, answer "7") should win.
    monkeypatch.setattr(
        "selfllm.reasoning.best_of_n.sample_cot_traces",
        lambda *a, **k: _traces(["4", "5", "7"]),
    )
    strat = BestOfNStrategy(
        model, tok, QualityVerifier(_StubQuality(), tok, device="cpu"),
        answer_type="free", num_samples=3, device="cpu",
    )
    r = strat.solve("q")
    assert r.details["best_idx"] == 2
    assert r.answer == "7"


# --------------------------------------------------------------------------- #
# End-to-end with the tiny real model (shape only)
# --------------------------------------------------------------------------- #


def test_self_consistency_end_to_end(tiny_model_tok):
    model, tok = tiny_model_tok
    strat = SelfConsistencyStrategy(model, tok, answer_type="free", num_samples=2,
                                    device="cpu", max_think_tokens=6, max_answer_tokens=4)
    r = strat.solve("What is 1 + 1?")
    assert isinstance(r, ReasoningResult)
    assert r.num_samples == 2 and len(r.traces) == 2
    assert 0.0 <= r.confidence <= 1.0


def test_best_of_n_end_to_end(tiny_model_tok):
    model, tok = tiny_model_tok
    strat = BestOfNStrategy(model, tok, SelfConsistencyVerifier(),
                            answer_type="free", num_samples=2, device="cpu",
                            max_think_tokens=6, max_answer_tokens=4)
    r = strat.solve("What is 1 + 1?")
    assert isinstance(r, ReasoningResult)
    assert r.num_samples == 2 and len(r.traces) == 2
    assert 0.0 <= r.confidence <= 1.0


# --------------------------------------------------------------------------- #
# BeamSearchReasoner
# --------------------------------------------------------------------------- #


def test_beam_search_follows_verifier(tiny_model_tok):
    model, tok = tiny_model_tok
    strat = BeamSearchReasoner(
        model, tok, _PickAnswer("4"), answer_type="free", device="cpu",
        beam_width=1, num_expansions=2, max_steps=2,
    )
    # Each expansion yields a "4" chain and a "7" chain; the verifier prefers "4".
    strat._expand = lambda prefix: ["reasoning answer 4", "reasoning answer 7"]
    r = strat.solve("q")
    assert r.answer == "4"
    assert r.confidence == 1.0
    assert len(r.traces) == 1  # beam_width


def test_beam_search_end_to_end(tiny_model_tok):
    model, tok = tiny_model_tok
    strat = BeamSearchReasoner(
        model, tok, SelfConsistencyVerifier(), answer_type="free", device="cpu",
        beam_width=2, num_expansions=2, max_steps=2, tokens_per_step=4,
    )
    r = strat.solve("What is 1 + 1?")
    assert isinstance(r, ReasoningResult)
    assert 1 <= len(r.traces) <= 2
    assert 0.0 <= r.confidence <= 1.0
    assert r.details["beam_width"] == 2
