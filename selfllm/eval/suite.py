"""
Benchmark suite runner for SelfLLM.

:class:`BenchmarkSuite` is a high-level orchestrator that:

1. Runs MMLU, GSM8K, and HumanEval evaluations against any model/tokenizer
   pair that conforms to the :class:`~selfllm.eval.harness.SupportsGenerate` /
   :class:`~selfllm.eval.harness.SupportsTokenize` protocols.
2. Collects :class:`~selfllm.eval.harness.EvalResult` objects from each
   benchmark.
3. Writes a timestamped JSON snapshot to ``results_dir`` (optional).
4. Returns a structured :class:`SuiteResult` for downstream comparison and
   regression detection.

This module is deliberately thin — the benchmark logic lives in
:mod:`selfllm.eval.mmlu`, :mod:`selfllm.eval.gsm8k`, and
:mod:`selfllm.eval.humaneval`.  The suite only wires them together.

Typical usage in a training script::

    from selfllm.eval.suite import BenchmarkSuite, SuiteResult

    suite = BenchmarkSuite(
        mmlu_records=mmlu_data,
        gsm8k_records=gsm8k_data,
        results_dir="./run_output/eval",
    )
    result: SuiteResult = suite.run(model, tokenizer, tag="after_iter_5")
    print(result.summary_str())

Typical usage in the recursive self-improvement loop::

    suite = BenchmarkSuite(mmlu_records=records)
    result = suite.run(model, tokenizer, tag=f"iter_{n}")
    if result.regression_vs(baseline):
        logger.warning("Performance regression detected!")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .harness import EvalResult, Evaluator, comparison_report
from .mmlu import MMLUBenchmark, MMLUExample
from .gsm8k import GSM8KBenchmark, GSM8KExample
from .humaneval import HumanEvalBenchmark, HumanEvalProblem

logger = logging.getLogger(__name__)

__all__ = ["BenchmarkSuite", "SuiteResult"]


# ---------------------------------------------------------------------------
# SuiteResult
# ---------------------------------------------------------------------------


@dataclass
class SuiteResult:
    """Aggregated result from a single :class:`BenchmarkSuite` run.

    Attributes:
        tag:          Human-readable label for this run (e.g. ``"iter_3"``).
        timestamp:    Unix timestamp of the run start.
        results:      Per-benchmark :class:`~selfllm.eval.harness.EvalResult` list.
        elapsed_s:    Wall-clock seconds for the entire suite.
        snapshot_path: Path to the JSON snapshot file, or ``None``.
    """

    tag: str
    timestamp: float
    results: List[EvalResult]
    elapsed_s: float
    snapshot_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def score(self, benchmark_name: str) -> Optional[float]:
        """Return the primary score for a named benchmark, or ``None``."""
        for r in self.results:
            if r.name == benchmark_name:
                return r.score
        return None

    @property
    def average_score(self) -> float:
        """Mean primary score across all benchmarks that ran."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def summary_str(self) -> str:
        """Return a short human-readable summary line."""
        parts = [f"[{self.tag}]"]
        for r in self.results:
            parts.append(f"{r.name}={r.score:.3f}(n={r.n})")
        parts.append(f"avg={self.average_score:.3f}")
        parts.append(f"elapsed={self.elapsed_s:.1f}s")
        return "  ".join(parts)

    def regression_vs(
        self,
        baseline: "SuiteResult",
        threshold: float = 0.02,
    ) -> bool:
        """Return ``True`` if any benchmark dropped by more than *threshold*.

        A drop of 0.02 (2 percentage points) on a 0-1 scale is considered a
        regression.  Set *threshold* to 0 for zero-tolerance.
        """
        for r in self.results:
            base_score = baseline.score(r.name)
            if base_score is not None and (base_score - r.score) > threshold:
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "timestamp": self.timestamp,
            "elapsed_s": self.elapsed_s,
            "average_score": self.average_score,
            "snapshot_path": self.snapshot_path,
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """High-level orchestrator that runs multiple benchmarks and saves snapshots.

    Args:
        mmlu_records:       In-memory MMLU records or a JSONL path, or ``None``
                            to skip MMLU.
        gsm8k_records:      In-memory GSM8K records or a JSONL path, or ``None``
                            to skip GSM8K.
        humaneval_records:  In-memory HumanEval records or a JSONL path, or
                            ``None`` to skip HumanEval.
        results_dir:        Directory for timestamped JSON snapshots.  Set to
                            ``None`` to disable snapshot writing.
        max_examples:       Maximum examples to evaluate per benchmark.  ``None``
                            means evaluate all.
        max_new_tokens:     Token budget for each model generation call.
        temperature:        Sampling temperature (0.0 = greedy).

    Example::

        suite = BenchmarkSuite(mmlu_records=records, results_dir="./eval_out")
        result = suite.run(model, tokenizer, tag="baseline")
    """

    def __init__(
        self,
        mmlu_records=None,
        gsm8k_records=None,
        humaneval_records=None,
        results_dir: Optional[str] = None,
        max_examples: Optional[int] = None,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> None:
        self._mmlu       = mmlu_records
        self._gsm8k      = gsm8k_records
        self._humaneval  = humaneval_records
        self._results_dir = results_dir
        self._max_examples = max_examples
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, model, tokenizer, tag: str = "eval") -> SuiteResult:
        """Run all configured benchmarks and return a :class:`SuiteResult`.

        Args:
            model:     Any model exposing ``generate(prompt_ids, **kwargs)``.
            tokenizer: Any tokenizer exposing ``encode``/``decode``.
            tag:       Label used in snapshot filenames and summary output.

        Returns:
            A :class:`SuiteResult` with per-benchmark scores and elapsed time.
        """
        t0 = time.perf_counter()
        ts = time.time()

        evaluator = Evaluator()
        common_kwargs = dict(
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
        )

        if self._mmlu is not None:
            bench = MMLUBenchmark.from_source(self._mmlu)
            if self._max_examples:
                bench.examples = bench.examples[: self._max_examples]
            evaluator.add(bench)
            logger.debug("MMLU: %d examples", len(bench.examples))

        if self._gsm8k is not None:
            bench = GSM8KBenchmark.from_source(self._gsm8k)
            if self._max_examples:
                bench.examples = bench.examples[: self._max_examples]
            evaluator.add(bench)
            logger.debug("GSM8K: %d examples", len(bench.examples))

        if self._humaneval is not None:
            bench = HumanEvalBenchmark.from_source(self._humaneval)
            if self._max_examples:
                bench.problems = bench.problems[: self._max_examples]
            evaluator.add(bench)
            logger.debug("HumanEval: %d problems", len(bench.problems))

        if not evaluator.benchmarks:
            logger.warning("BenchmarkSuite.run() called with no benchmarks configured.")
            return SuiteResult(
                tag=tag, timestamp=ts, results=[],
                elapsed_s=time.perf_counter() - t0,
            )

        results = evaluator.run(model, tokenizer, **common_kwargs)
        elapsed = time.perf_counter() - t0

        suite_result = SuiteResult(
            tag=tag,
            timestamp=ts,
            results=results,
            elapsed_s=elapsed,
        )

        # Persist snapshot
        if self._results_dir is not None:
            snapshot_path = self._write_snapshot(suite_result)
            suite_result.snapshot_path = snapshot_path

        logger.info(suite_result.summary_str())
        return suite_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_snapshot(self, result: SuiteResult) -> str:
        """Write a JSON snapshot to ``results_dir`` and return the path."""
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(result.timestamp))
        filename = f"eval_{result.tag}_{ts_str}.json"
        path = os.path.join(self._results_dir, filename)  # type: ignore[arg-type]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        logger.debug("Eval snapshot → %s", path)
        return path
