"""Common evaluation harness: result dataclass and benchmark runner interface.

This module defines the shared plumbing used by every benchmark in
:mod:`selfllm.eval`:

- :class:`EvalResult`: a standardized, serializable result record.
- :class:`Benchmark`: the abstract base class every benchmark implements.
- :class:`Evaluator`: a thin orchestrator that runs one or more benchmarks
  against a model/tokenizer and produces a comparison report.
- :func:`run_benchmark`: a convenience wrapper for running a single benchmark.

The benchmarks themselves (MMLU, GSM8K, HumanEval) live in sibling modules and
subclass :class:`Benchmark`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol


# --------------------------------------------------------------------------- #
# Lightweight structural typing for the model / tokenizer
# --------------------------------------------------------------------------- #


class SupportsGenerate(Protocol):
    """Structural type for a model exposing ``generate``.

    Matches :class:`selfllm.model.model.SelfImprovingLLM`.
    """

    def generate(self, prompt_ids: Any, **kwargs: Any) -> Dict[str, Any]:
        ...


class SupportsTokenize(Protocol):
    """Structural type for a tokenizer.

    Matches :class:`selfllm.model.tokenizer.BPETokenizer`.
    """

    def encode(self, text: str) -> List[int]:
        ...

    def decode(self, ids: List[int]) -> str:
        ...

    @property
    def eos_token_id(self) -> int:
        ...


# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #


@dataclass
class EvalResult:
    """Standardized result for a single benchmark run.

    Attributes:
        name: Benchmark name (e.g. ``"mmlu"``, ``"gsm8k"``, ``"humaneval"``).
        score: Primary scalar metric in ``[0, 1]`` (accuracy or pass@k).
        n: Number of examples evaluated.
        details: Free-form benchmark-specific breakdown (per-subject accuracy,
            pass@k for multiple k, per-example predictions, etc.).
    """

    name: str
    score: float
    n: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-``dict`` representation suitable for JSON dumping."""
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.name}: score={self.score:.4f} (n={self.n})"


# --------------------------------------------------------------------------- #
# Benchmark base class
# --------------------------------------------------------------------------- #


class Benchmark(ABC):
    """Abstract base class for an evaluation benchmark.

    Concrete benchmarks load a dataset and implement :meth:`evaluate`, which
    runs the supplied model over the dataset and returns an :class:`EvalResult`.

    Args:
        name: Human-readable benchmark name.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def evaluate(
        self,
        model: SupportsGenerate,
        tokenizer: SupportsTokenize,
        **kwargs: Any,
    ) -> EvalResult:
        """Run the benchmark and return an :class:`EvalResult`.

        Args:
            model: A model exposing ``generate``.
            tokenizer: A tokenizer exposing ``encode``/``decode``/``eos_token_id``.
            **kwargs: Benchmark-specific options (e.g. ``max_new_tokens``,
                ``limit``).

        Returns:
            An :class:`EvalResult` describing the run.
        """
        raise NotImplementedError


def run_benchmark(
    benchmark: Benchmark,
    model: SupportsGenerate,
    tokenizer: SupportsTokenize,
    **kwargs: Any,
) -> EvalResult:
    """Run a single benchmark and return its :class:`EvalResult`.

    Args:
        benchmark: The benchmark instance to run.
        model: A model exposing ``generate``.
        tokenizer: A tokenizer.
        **kwargs: Forwarded to :meth:`Benchmark.evaluate`.

    Returns:
        The benchmark's :class:`EvalResult`.
    """
    return benchmark.evaluate(model, tokenizer, **kwargs)


# --------------------------------------------------------------------------- #
# Evaluator orchestrator
# --------------------------------------------------------------------------- #


class Evaluator:
    """Runs a collection of benchmarks and aggregates their results.

    Args:
        benchmarks: The benchmarks to run. May be omitted and populated later
            via :meth:`add`.
    """

    def __init__(self, benchmarks: Optional[List[Benchmark]] = None) -> None:
        self.benchmarks: List[Benchmark] = list(benchmarks) if benchmarks else []

    def add(self, benchmark: Benchmark) -> "Evaluator":
        """Append a benchmark and return ``self`` for chaining."""
        self.benchmarks.append(benchmark)
        return self

    def run(
        self,
        model: SupportsGenerate,
        tokenizer: SupportsTokenize,
        per_benchmark_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        **common_kwargs: Any,
    ) -> List[EvalResult]:
        """Run every registered benchmark.

        Args:
            model: A model exposing ``generate``.
            tokenizer: A tokenizer.
            per_benchmark_kwargs: Optional mapping of benchmark name to a kwargs
                dict applied only to that benchmark (overrides ``common_kwargs``
                on key collision).
            **common_kwargs: Keyword arguments forwarded to every benchmark.

        Returns:
            A list of :class:`EvalResult`, one per benchmark, in registration
            order.
        """
        per_benchmark_kwargs = per_benchmark_kwargs or {}
        results: List[EvalResult] = []
        for bench in self.benchmarks:
            kwargs = dict(common_kwargs)
            kwargs.update(per_benchmark_kwargs.get(bench.name, {}))
            results.append(bench.evaluate(model, tokenizer, **kwargs))
        return results


def comparison_report(
    results: List[EvalResult],
    as_string: bool = False,
) -> Any:
    """Build a comparison report from a list of :class:`EvalResult`.

    Args:
        results: The results to summarize.
        as_string: If ``True``, return a pretty-printed table string; otherwise
            return a ``dict`` keyed by benchmark name.

    Returns:
        Either a ``dict`` (``{name: {"score", "n", "details"}}``) or a
        formatted multi-line string.
    """
    report: Dict[str, Any] = {
        r.name: {"score": r.score, "n": r.n, "details": r.details} for r in results
    }
    if not as_string:
        return report

    if not results:
        return "(no benchmark results)"

    name_w = max(len("Benchmark"), max(len(r.name) for r in results))
    lines = [
        f"{'Benchmark'.ljust(name_w)}  {'Score':>8}  {'N':>6}",
        f"{'-' * name_w}  {'-' * 8}  {'-' * 6}",
    ]
    for r in results:
        lines.append(f"{r.name.ljust(name_w)}  {r.score:>8.4f}  {r.n:>6d}")
    return "\n".join(lines)
