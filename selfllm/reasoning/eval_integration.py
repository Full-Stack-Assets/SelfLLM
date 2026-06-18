"""Measure the accuracy lift of a reasoning strategy over greedy decoding.

Ties the V6 reasoning strategies to the eval suite: run a benchmark with plain
greedy decoding (baseline) and again *under* a reasoning strategy, and report
the delta. This is the headline V6 metric -- does spending test-time compute
actually improve accuracy?

The strategy's ``answer_type`` must match the benchmark:
``"numeric"`` for GSM8K, ``"choice"`` for MMLU.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from selfllm.eval.harness import EvalResult


def measure_lift(
    benchmark: Any,
    model: Any,
    tokenizer: Any,
    strategy: Any,
    limit: Optional[int] = None,
    **baseline_kwargs: Any,
) -> Dict[str, Any]:
    """Run ``benchmark`` greedy vs under ``strategy`` and return the lift.

    Args:
        benchmark: An eval-suite ``Benchmark`` whose ``evaluate`` accepts a
            ``strategy=`` argument (MMLU / GSM8K).
        model: The model (used for the greedy baseline).
        tokenizer: The tokenizer.
        strategy: A reasoning strategy with a matching ``answer_type``.
        limit: Optional cap on the number of examples (applied to both runs).
        **baseline_kwargs: Extra args for the greedy baseline run (e.g.
            ``max_new_tokens``).

    Returns:
        Dict with ``benchmark``, ``baseline_score``, ``strategy_score``,
        ``lift`` (strategy - baseline), ``n``, and the two ``EvalResult`` s.
    """
    baseline: EvalResult = benchmark.evaluate(
        model, tokenizer, limit=limit, **baseline_kwargs
    )
    with_strategy: EvalResult = benchmark.evaluate(
        model, tokenizer, strategy=strategy, limit=limit
    )
    return {
        "benchmark": baseline.name,
        "baseline_score": baseline.score,
        "strategy_score": with_strategy.score,
        "lift": with_strategy.score - baseline.score,
        "n": baseline.n,
        "baseline": baseline,
        "strategy": with_strategy,
    }


def compare_strategies(
    benchmark: Any,
    model: Any,
    tokenizer: Any,
    strategies: Dict[str, Any],
    limit: Optional[int] = None,
    **baseline_kwargs: Any,
) -> Dict[str, Any]:
    """Compare several strategies against the greedy baseline on one benchmark.

    Args:
        strategies: Mapping ``name -> strategy``.

    Returns:
        Dict with ``benchmark``, ``baseline_score``, ``n``, and a ``strategies``
        mapping of ``name -> {"score", "lift"}``.
    """
    baseline: EvalResult = benchmark.evaluate(
        model, tokenizer, limit=limit, **baseline_kwargs
    )
    results: Dict[str, Dict[str, float]] = {}
    for name, strat in strategies.items():
        res: EvalResult = benchmark.evaluate(
            model, tokenizer, strategy=strat, limit=limit
        )
        results[name] = {
            "score": res.score,
            "lift": res.score - baseline.score,
        }
    return {
        "benchmark": baseline.name,
        "baseline_score": baseline.score,
        "n": baseline.n,
        "strategies": results,
    }


def format_lift_report(comparison: Dict[str, Any]) -> str:
    """Pretty-print a :func:`compare_strategies` result as a table."""
    lines = [
        f"Benchmark: {comparison['benchmark']} (n={comparison['n']})",
        f"  {'greedy (baseline)':<28} {comparison['baseline_score']:.4f}",
    ]
    for name, r in comparison["strategies"].items():
        lines.append(
            f"  {name:<28} {r['score']:.4f}  ({r['lift']:+.4f})"
        )
    return "\n".join(lines)
