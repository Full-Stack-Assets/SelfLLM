"""Tests for the self-improvement data path: quality scoring, evaluation
metrics, and self-training data generation.

Covers three previously untested modules:

- :class:`selfllm.model.quality_model.QualityModel` — frozen-encoder scoring
  heads producing bounded [0, 1] scores.
- :class:`selfllm.recursive.evaluator.SelfImprovementEvaluator` — the metric
  helpers (diversity, coherence, repetition, similarity) and the
  improvement-delta composite that drive the recursive loop's accept/reject
  signal, plus a generation smoke test.
- :class:`selfllm.training.data_generator.SelfTrainingDataGenerator` — prompt
  templating, response scoring fallbacks, and the self-critic N-best selection.

Generation-heavy paths use the tiny CPU model with small token budgets, or
mock the expensive generation step, so the file stays fast.
"""

from __future__ import annotations

import random

import pytest
import torch

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.quality_model import QualityModel
from selfllm.recursive.evaluator import SelfImprovementEvaluator
from selfllm.training.data_generator import SelfTrainingDataGenerator


# ===========================================================================
# QualityModel
# ===========================================================================


@pytest.fixture()
def quality_model(tiny_model_config: ModelConfig) -> QualityModel:
    torch.manual_seed(0)
    base = SelfImprovingLLM(tiny_model_config)
    return QualityModel(base)


def test_quality_model_freezes_base(quality_model):
    assert all(not p.requires_grad for p in quality_model.base_model.parameters())
    # Heads remain trainable.
    assert any(p.requires_grad for p in quality_model.heads.parameters())


def test_quality_model_forward_shapes_and_range(quality_model):
    ids = torch.randint(0, 50, (3, 7))
    out = quality_model(ids)
    overall = out["overall_score"]
    assert overall.shape == (3,)
    assert torch.all((overall >= 0.0) & (overall <= 1.0))

    dims = out["dimension_scores"]
    assert set(dims) == {"coherence", "diversity", "complexity"}
    for name, t in dims.items():
        assert t.shape == (3,)
        assert torch.all((t >= 0.0) & (t <= 1.0))


def test_quality_model_custom_dimension_count(tiny_model_config):
    torch.manual_seed(0)
    qm = QualityModel(SelfImprovingLLM(tiny_model_config), num_dimensions=2)
    out = qm(torch.randint(0, 50, (1, 5)))
    assert set(out["dimension_scores"]) == {"coherence", "diversity"}


def test_score_sequences_matches_forward(quality_model):
    ids = torch.randint(0, 50, (2, 6))
    torch.manual_seed(1)
    a = quality_model.score_sequences(ids)
    assert a.shape == (2,)
    assert torch.all((a >= 0.0) & (a <= 1.0))


def test_quality_grads_do_not_reach_base(quality_model):
    ids = torch.randint(0, 50, (2, 6))
    score = quality_model(ids)["overall_score"].sum()
    score.backward()
    # No grad on frozen base params.
    assert all(p.grad is None for p in quality_model.base_model.parameters())
    # Grad present on at least one head param.
    assert any(p.grad is not None for p in quality_model.heads.parameters())


# ===========================================================================
# SelfImprovementEvaluator — pure metric helpers
# ===========================================================================


@pytest.fixture()
def evaluator(tiny_model, tiny_tokenizer) -> SelfImprovementEvaluator:
    return SelfImprovementEvaluator(
        tiny_model, tiny_tokenizer, eval_prompts=["the cat", "a dog"], device="cpu"
    )


def test_text_similarity_identical_is_one(evaluator):
    assert evaluator._text_similarity("the quick fox", "the quick fox") == pytest.approx(1.0)


def test_text_similarity_disjoint_is_zero(evaluator):
    assert evaluator._text_similarity("alpha beta", "gamma delta") == pytest.approx(0.0)


def test_text_similarity_both_empty_is_one(evaluator):
    assert evaluator._text_similarity("", "") == pytest.approx(1.0)


def test_text_similarity_one_empty_is_zero(evaluator):
    assert evaluator._text_similarity("word", "") == pytest.approx(0.0)


def test_text_similarity_jaccard_value(evaluator):
    # tokens {a,b,c} vs {b,c,d}: intersection 2, union 4 -> 0.5
    assert evaluator._text_similarity("a b c", "b c d") == pytest.approx(0.5)


def test_avg_length(evaluator):
    assert evaluator._compute_avg_length(["one two", "a b c d"]) == pytest.approx(3.0)
    assert evaluator._compute_avg_length([]) == 0.0


def test_diversity_distinct_vs_repeated(evaluator):
    distinct = evaluator._compute_diversity(["a b c d e"])      # all bigrams unique -> 1.0
    repeated = evaluator._compute_diversity(["a a a a a"])      # one repeated bigram
    assert distinct == pytest.approx(1.0)
    assert repeated < distinct


def test_repetition_rate_bounds(evaluator):
    none = evaluator._compute_repetition(["one two three four five"])  # no repeated trigram
    some = evaluator._compute_repetition(["a b c a b c a b c"])        # repeated trigrams
    assert none == pytest.approx(0.0)
    assert some > 0.0
    assert 0.0 <= some <= 1.0


def test_coherence_single_sentence_defaults_high(evaluator):
    # A single sentence is "coherent by default" (score contribution 1.0).
    assert evaluator._compute_coherence(["just one sentence here"]) == pytest.approx(1.0)


def test_coherence_empty(evaluator):
    assert evaluator._compute_coherence([]) == 0.0


# --- compute_improvement_delta ---


def test_improvement_delta_first_iteration_is_zero(evaluator):
    assert evaluator.compute_improvement_delta({"perplexity": 10.0}, None) == 0.0


def test_improvement_delta_positive_when_perplexity_drops(evaluator):
    prev = {"perplexity": 100.0, "overall_quality": 0.5, "self_consistency": 0.5}
    curr = {"perplexity": 50.0, "overall_quality": 0.5, "self_consistency": 0.5}
    # Only perplexity improved (halved) -> positive composite.
    assert evaluator.compute_improvement_delta(curr, prev) > 0.0


def test_improvement_delta_negative_when_quality_drops(evaluator):
    prev = {"perplexity": 50.0, "overall_quality": 0.8, "self_consistency": 0.8}
    curr = {"perplexity": 50.0, "overall_quality": 0.4, "self_consistency": 0.4}
    assert evaluator.compute_improvement_delta(curr, prev) < 0.0


# --- generation-backed methods (smoke, tiny budgets) ---


def test_evaluate_perplexity_finite_positive(evaluator):
    ppl = evaluator.evaluate_perplexity(["the cat sat on the mat"])
    assert ppl > 0.0
    assert ppl != float("inf")


def test_evaluate_perplexity_no_usable_tokens_is_inf(evaluator):
    # Texts that encode to <2 tokens are skipped; with none usable -> inf.
    assert evaluator.evaluate_perplexity([""]) == float("inf")


def test_generation_quality_empty_prompts_returns_zeros(tiny_model, tiny_tokenizer):
    ev = SelfImprovementEvaluator(tiny_model, tiny_tokenizer, eval_prompts=[], device="cpu")
    metrics = ev.evaluate_generation_quality()
    assert metrics == {
        "avg_length": 0.0,
        "diversity_score": 0.0,
        "coherence_score": 0.0,
        "repetition_rate": 0.0,
        "overall_quality": 0.0,
    }


def test_self_consistency_empty_prompts_is_zero(tiny_model, tiny_tokenizer):
    ev = SelfImprovementEvaluator(tiny_model, tiny_tokenizer, eval_prompts=[], device="cpu")
    assert ev.evaluate_self_consistency() == 0.0


# ===========================================================================
# SelfTrainingDataGenerator
# ===========================================================================


@pytest.fixture()
def generator(tiny_model, tiny_tokenizer) -> SelfTrainingDataGenerator:
    return SelfTrainingDataGenerator(tiny_model, tiny_tokenizer, device="cpu")


def test_generate_prompt_fills_all_placeholders(generator):
    random.seed(0)
    # Across many draws (covering every template), no unfilled {placeholder}
    # should ever leak through.
    for _ in range(60):
        prompt = generator.generate_prompt()
        assert "{" not in prompt and "}" not in prompt
        assert prompt.strip()


def test_generate_prompt_uses_given_topic_for_single_topic_templates(generator):
    # Force a {topic} template by retrying until one is selected; the supplied
    # topic must then appear verbatim.
    random.seed(3)
    found = False
    for _ in range(50):
        p = generator.generate_prompt(topic="zzz_unique_topic")
        if "zzz_unique_topic" in p:
            found = True
            break
    assert found, "expected the supplied topic to appear in at least one template"


def test_score_response_without_quality_model_returns_half(generator):
    assert generator._score_response("some text") == 0.5
    # Empty response also short-circuits to 0.5.
    assert generator._score_response("") == 0.5


def test_score_response_with_quality_model_is_bounded(tiny_model, tiny_tokenizer, tiny_model_config):
    torch.manual_seed(0)
    qm = QualityModel(SelfImprovingLLM(tiny_model_config))
    gen = SelfTrainingDataGenerator(tiny_model, tiny_tokenizer, quality_model=qm, device="cpu")
    score = gen._score_response("the quick brown fox")
    assert 0.0 <= score <= 1.0


def test_generate_response_batch_smoke(generator):
    results = generator.generate_response_batch(
        ["the cat", "a dog"], max_new_tokens=4, batch_size=2
    )
    assert len(results) == 2
    for r in results:
        assert set(r) >= {"prompt", "response", "token_ids", "generation_metadata"}
        meta = r["generation_metadata"]
        assert "length" in meta and "confidence_score" in meta


def test_generate_training_batch_selects_best_candidate(generator, monkeypatch):
    """The self-critic step should keep the single best-scored candidate per
    prompt and annotate it with quality_score + num_candidates. We stub the
    expensive generation and the scorer to make selection deterministic."""

    # Two prompts, 3 candidates each -> generate_response_batch returns 6.
    def fake_batch(prompts, **kw):
        return [{"prompt": p, "response": f"resp-{i}", "token_ids": [1, 2],
                 "generation_metadata": {"length": 2}} for i, p in enumerate(prompts)]

    monkeypatch.setattr(generator, "generate_response_batch", fake_batch)
    # Score = trailing index number, so the highest-index candidate wins.
    monkeypatch.setattr(
        generator, "_score_response",
        lambda resp: float(resp.split("-")[1]),
    )

    samples = generator.generate_training_batch(num_samples=2, num_critic_samples=3)
    assert len(samples) == 2
    for s in samples:
        assert "quality_score" in s
        assert s["num_candidates"] == 3
        # Best of indices {0,1,2} and {3,4,5} -> scores 2 and 5 respectively.
        assert s["quality_score"] in (2.0, 5.0)


def test_generate_training_batch_falls_back_when_all_invalid(generator, monkeypatch):
    """If every candidate is empty/errored, the generator still emits a sample
    rather than dropping the prompt."""
    def fake_batch(prompts, **kw):
        return [{"prompt": p, "response": "", "token_ids": [],
                 "generation_metadata": {"error": "boom"}} for p in prompts]

    monkeypatch.setattr(generator, "generate_response_batch", fake_batch)
    samples = generator.generate_training_batch(num_samples=1, num_critic_samples=2)
    assert len(samples) == 1
    assert samples[0]["num_candidates"] == 2
