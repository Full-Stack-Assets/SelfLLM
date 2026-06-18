"""Tests for V6 Wave 3: reasoning strategies wired into the eval harness."""

import pytest
import torch

from selfllm.eval import GSM8KBenchmark, MMLUBenchmark
from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.reasoning import (
    CoTStrategy,
    ReasoningResult,
    compare_strategies,
    format_lift_report,
    measure_lift,
)


# --------------------------------------------------------------------------- #
# Stubs for deterministic tests
# --------------------------------------------------------------------------- #


class _StubStrategy:
    """Always returns a fixed answer, regardless of the prompt."""

    def __init__(self, answer):
        self._answer = answer

    def solve(self, prompt):
        return ReasoningResult(answer=self._answer, confidence=1.0,
                               traces=["trace"], num_samples=1)


class _StubTok:
    eos_token_id = 1

    def encode(self, s):
        return [2, 3, 4]

    def decode(self, ids):
        return "no number or choice here"


class _StubModel:
    """Greedy baseline that decodes to a non-answer (so baseline is wrong)."""

    def __init__(self):
        self.called = False

    def generate(self, input_ids, **kwargs):
        self.called = True
        return {"sequences": torch.cat([input_ids, torch.tensor([[0, 0]])], dim=1)}


GSM8K_RECORDS = [
    {"question": "What is 2 + 3?", "answer": "Add them. #### 5"},
    {"question": "What is 1 + 4?", "answer": "#### 5"},
]
MMLU_RECORDS = [
    {"question": "Capital of France?",
     "choices": ["Rome", "Paris", "Berlin", "Madrid"], "answer": "B"},
]


# --------------------------------------------------------------------------- #
# Strategy path inside evaluate()
# --------------------------------------------------------------------------- #


def test_gsm8k_strategy_path_scores_and_skips_model():
    bench = GSM8KBenchmark.from_source(GSM8K_RECORDS)
    model = _StubModel()
    res = bench.evaluate(model, _StubTok(), strategy=_StubStrategy("5"))
    assert res.score == 1.0  # both gold answers are "5"
    assert model.called is False  # strategy replaced the model decode


def test_gsm8k_strategy_wrong_answer():
    bench = GSM8KBenchmark.from_source(GSM8K_RECORDS)
    res = bench.evaluate(_StubModel(), _StubTok(), strategy=_StubStrategy("99"))
    assert res.score == 0.0


def test_mmlu_strategy_path():
    bench = MMLUBenchmark.from_source(MMLU_RECORDS)
    res = bench.evaluate(_StubModel(), _StubTok(), strategy=_StubStrategy("B"))
    assert res.score == 1.0


# --------------------------------------------------------------------------- #
# measure_lift / compare_strategies
# --------------------------------------------------------------------------- #


def test_measure_lift():
    bench = GSM8KBenchmark.from_source(GSM8K_RECORDS)
    out = measure_lift(bench, _StubModel(), _StubTok(), _StubStrategy("5"))
    assert out["benchmark"] == "gsm8k"
    assert out["baseline_score"] == 0.0   # stub model never produces "5"
    assert out["strategy_score"] == 1.0
    assert out["lift"] == 1.0
    assert out["n"] == 2


def test_compare_strategies_and_report():
    bench = GSM8KBenchmark.from_source(GSM8K_RECORDS)
    comp = compare_strategies(
        bench, _StubModel(), _StubTok(),
        {"self_consistency": _StubStrategy("5"), "bad": _StubStrategy("9")},
    )
    assert comp["baseline_score"] == 0.0
    assert comp["strategies"]["self_consistency"] == {"score": 1.0, "lift": 1.0}
    assert comp["strategies"]["bad"] == {"score": 0.0, "lift": 0.0}

    report = format_lift_report(comp)
    assert "gsm8k" in report
    assert "greedy (baseline)" in report
    assert "self_consistency" in report


# --------------------------------------------------------------------------- #
# End-to-end with the tiny real model (no crash, well-formed lift)
# --------------------------------------------------------------------------- #


def test_measure_lift_end_to_end():
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                      d_ff=128, max_seq_len=128, dropout=0.0)
    model = SelfImprovingLLM(cfg).eval()
    tok = BPETokenizer(vocab_size=256)
    tok.train(["The answer is 5.", "Step by step reasoning to a number."])

    bench = GSM8KBenchmark.from_source(GSM8K_RECORDS)
    strat = CoTStrategy(model, tok, answer_type="numeric", device="cpu",
                        max_think_tokens=6, max_answer_tokens=4, temperature=0.0)
    out = measure_lift(bench, model, tok, strat, limit=1, max_new_tokens=4)
    assert out["n"] == 1
    assert isinstance(out["lift"], float)
    assert -1.0 <= out["lift"] <= 1.0
