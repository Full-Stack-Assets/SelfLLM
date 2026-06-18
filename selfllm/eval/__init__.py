"""Evaluation suite for SelfLLM.

Provides MMLU, GSM8K and HumanEval benchmarks behind a common
:class:`Evaluator` / :class:`Benchmark` interface, plus loaders, answer
extractors and a pass@k estimator.

Example:
    >>> from selfllm.eval import Evaluator, MMLUBenchmark, comparison_report
    >>> evaluator = Evaluator([MMLUBenchmark.from_source(records)])
    >>> results = evaluator.run(model, tokenizer)
    >>> print(comparison_report(results, as_string=True))
"""

from .harness import (
    Benchmark,
    EvalResult,
    Evaluator,
    comparison_report,
    run_benchmark,
)
from .mmlu import (
    MMLUBenchmark,
    MMLUExample,
    build_mmlu_prompt,
    extract_choice,
    load_mmlu,
)
from .gsm8k import (
    GSM8KBenchmark,
    GSM8KExample,
    build_gsm8k_prompt,
    extract_final_number,
    load_gsm8k,
)
from .humaneval import (
    HumanEvalBenchmark,
    HumanEvalProblem,
    execute_candidate,
    load_humaneval,
    pass_at_k,
)


def run_all(
    model,
    tokenizer,
    mmlu_source=None,
    gsm8k_source=None,
    humaneval_source=None,
    per_benchmark_kwargs=None,
    as_string=False,
    **common_kwargs,
):
    """Run all available benchmarks and return a comparison report.

    Only the benchmarks whose source is supplied are run.

    Args:
        model: A model exposing ``generate``.
        tokenizer: A tokenizer.
        mmlu_source: MMLU JSONL path or in-memory records (or ``None`` to skip).
        gsm8k_source: GSM8K JSONL path or in-memory records (or ``None``).
        humaneval_source: HumanEval JSONL path or in-memory records (or ``None``).
        per_benchmark_kwargs: Optional per-benchmark kwargs overrides.
        as_string: If ``True``, return a pretty string report; else a dict.
        **common_kwargs: Forwarded to every benchmark's ``evaluate``.

    Returns:
        A tuple ``(results, report)`` where ``results`` is the list of
        :class:`EvalResult` and ``report`` is the comparison report.
    """
    evaluator = Evaluator()
    if mmlu_source is not None:
        evaluator.add(MMLUBenchmark.from_source(mmlu_source))
    if gsm8k_source is not None:
        evaluator.add(GSM8KBenchmark.from_source(gsm8k_source))
    if humaneval_source is not None:
        evaluator.add(HumanEvalBenchmark.from_source(humaneval_source))

    results = evaluator.run(
        model,
        tokenizer,
        per_benchmark_kwargs=per_benchmark_kwargs,
        **common_kwargs,
    )
    return results, comparison_report(results, as_string=as_string)


__all__ = [
    # harness
    "Benchmark",
    "EvalResult",
    "Evaluator",
    "comparison_report",
    "run_benchmark",
    "run_all",
    # mmlu
    "MMLUBenchmark",
    "MMLUExample",
    "build_mmlu_prompt",
    "extract_choice",
    "load_mmlu",
    # gsm8k
    "GSM8KBenchmark",
    "GSM8KExample",
    "build_gsm8k_prompt",
    "extract_final_number",
    "load_gsm8k",
    # humaneval
    "HumanEvalBenchmark",
    "HumanEvalProblem",
    "execute_candidate",
    "load_humaneval",
    "pass_at_k",
]
