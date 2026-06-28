"""Tests for the V6 reasoning core (selfllm.reasoning)."""

import pytest

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.reasoning import (
    CoTStrategy,
    ReasoningResult,
    ReasoningStrategy,
    extract_answer,
    extract_tagged_answer,
    normalize,
)


# --------------------------------------------------------------------------- #
# Extraction layer
# --------------------------------------------------------------------------- #


def test_extract_numeric():
    assert extract_answer("So the total is 42.", "numeric") == "42"
    assert extract_answer("Reasoning... #### 6", "numeric") == "6"


def test_extract_choice():
    assert extract_answer("Therefore the answer is B.", "choice") == "B"


def test_extract_free_tagged():
    text = "<think>blah</think><answer>Paris</answer>"
    assert extract_answer(text, "free") == "Paris"
    assert extract_tagged_answer(text) == "Paris"


def test_extract_free_last_tag_wins():
    text = "<answer>first</answer> ... <answer>second</answer>"
    assert extract_tagged_answer(text) == "second"


def test_extract_free_phrase():
    assert extract_answer("I think. Final answer: blue", "free") == "blue"


def test_extract_free_last_line():
    assert extract_answer("step one\nstep two\nDone here", "free") == "Done here"


def test_extract_free_none_on_empty():
    assert extract_answer("   \n  ", "free") is None
    assert extract_tagged_answer("no tags here") is None


def test_extract_unknown_type_raises():
    with pytest.raises(ValueError):
        extract_answer("x", "bogus")


def test_normalize():
    assert normalize(None) == ""
    assert normalize("  Paris. ") == normalize("paris")


# --------------------------------------------------------------------------- #
# ReasoningResult
# --------------------------------------------------------------------------- #


def test_reasoning_result_str():
    r = ReasoningResult(answer="42", confidence=0.8, traces=["t"], num_samples=3)
    assert "42" in str(r) and "0.8" in str(r) and "n=3" in str(r)


# --------------------------------------------------------------------------- #
# CoTStrategy end-to-end (tiny model)
# --------------------------------------------------------------------------- #


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
        "First, compute the sum. The result is seven.",
        "Reasoning about the problem carefully and concluding.",
    ])
    return model, tok


def test_cot_strategy_is_reasoning_strategy(tiny_model_tok):
    model, tok = tiny_model_tok
    strat = CoTStrategy(model, tok, answer_type="free", device="cpu",
                        max_think_tokens=8, max_answer_tokens=4, temperature=0.0)
    assert isinstance(strat, ReasoningStrategy)


def test_cot_strategy_solve_shape(tiny_model_tok):
    model, tok = tiny_model_tok
    strat = CoTStrategy(model, tok, answer_type="free", device="cpu",
                        max_think_tokens=8, max_answer_tokens=4, temperature=0.0)
    result = strat.solve("What is 2 + 2?")
    assert isinstance(result, ReasoningResult)
    assert result.num_samples == 1
    assert len(result.traces) == 1
    assert isinstance(result.traces[0], str)
    assert result.confidence in (0.0, 1.0)
    assert result.answer is None or isinstance(result.answer, str)
    assert "thinking" in result.details


def test_cot_strategy_numeric_answer_type(tiny_model_tok):
    model, tok = tiny_model_tok
    strat = CoTStrategy(model, tok, answer_type="numeric", device="cpu",
                        max_think_tokens=8, max_answer_tokens=4, temperature=0.0)
    result = strat.solve("What is 2 + 2?")
    # Numeric extraction returns a numeric string or None; never raises.
    assert result.answer is None or result.answer.lstrip("-").replace(".", "", 1).isdigit()
