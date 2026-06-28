"""
Regression detector for SelfLLM evaluation metrics.

:class:`RegressionDetector` tracks benchmark scores across iterations of
the recursive self-improvement loop and raises alerts when performance drops
below a configurable threshold.  It is designed to integrate cleanly with
the training monitor GitHub Actions workflow and with the
:class:`~selfllm.eval.suite.BenchmarkSuite`.

Design goals
~~~~~~~~~~~~
- **Zero false negatives on hard regressions** — any drop ≥ ``hard_threshold``
  always triggers an alert, regardless of history length.
- **Noise-tolerant on small fluctuations** — drops below ``soft_threshold``
  on a single step are ignored; only persistent downtrends (``patience``
  consecutive drops) trigger a soft alert.
- **Serialisable** — the full history can be written to / loaded from JSON so
  the detector survives across CI runs.

Example (inside a recursive training loop)::

    from selfllm.eval.regression import RegressionDetector
    from selfllm.eval.suite import BenchmarkSuite

    suite   = BenchmarkSuite(mmlu_records=records)
    detector = RegressionDetector(hard_threshold=0.05, soft_threshold=0.02)

    baseline = suite.run(model, tokenizer, tag="baseline")
    detector.update(baseline)

    for i in range(10):
        # ... training step ...
        result = suite.run(model, tokenizer, tag=f"iter_{i}")
        alerts = detector.update(result)
        if alerts:
            for a in alerts:
                logger.warning("REGRESSION: %s", a)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .suite import SuiteResult

logger = logging.getLogger(__name__)

__all__ = ["RegressionAlert", "RegressionDetector"]


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------


@dataclass
class RegressionAlert:
    """Represents a single detected regression event.

    Attributes:
        benchmark:   Name of the benchmark that regressed (e.g. ``"mmlu"``).
        tag:         The iteration tag where the regression was detected.
        prev_score:  Score at the previous step.
        curr_score:  Score at the current step.
        delta:       Absolute drop (``prev_score - curr_score``, always > 0).
        severity:    ``"hard"`` (≥ hard_threshold) or ``"soft"`` (persistent trend).
        message:     Human-readable alert message.
    """

    benchmark: str
    tag: str
    prev_score: float
    curr_score: float
    delta: float
    severity: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "tag": self.tag,
            "prev_score": round(self.prev_score, 4),
            "curr_score": round(self.curr_score, 4),
            "delta": round(self.delta, 4),
            "severity": self.severity,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# RegressionDetector
# ---------------------------------------------------------------------------


class RegressionDetector:
    """Stateful detector that tracks benchmark scores over time.

    Args:
        hard_threshold: Any single-step drop ≥ this value triggers a ``"hard"``
                        alert immediately.  Default 0.05 (5 pp).
        soft_threshold: Drops below this are ignored for soft-trend analysis.
                        Default 0.02 (2 pp).
        patience:       Number of consecutive soft drops before a ``"soft"``
                        alert is raised.  Default 3.
        history_path:   Optional JSON file path for persisting history across
                        CI runs.  If the file exists on construction, history
                        is loaded from it.
    """

    def __init__(
        self,
        hard_threshold: float = 0.05,
        soft_threshold: float = 0.02,
        patience: int = 3,
        history_path: Optional[str] = None,
    ) -> None:
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.patience       = patience
        self.history_path   = history_path

        # benchmark_name → list of (tag, score) in order
        self._history: Dict[str, List[tuple[str, float]]] = {}
        # benchmark_name → consecutive soft-drop count
        self._soft_streak: Dict[str, int] = {}

        if history_path and os.path.exists(history_path):
            self._load(history_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, result: SuiteResult) -> List[RegressionAlert]:
        """Record new benchmark scores and return any triggered alerts.

        Args:
            result: A :class:`~selfllm.eval.suite.SuiteResult` from the
                    current training step.

        Returns:
            List of :class:`RegressionAlert` objects (empty if no regression).
        """
        alerts: List[RegressionAlert] = []

        for eval_result in result.results:
            name  = eval_result.name
            score = eval_result.score
            tag   = result.tag

            history = self._history.setdefault(name, [])

            if history:
                prev_tag, prev_score = history[-1]
                delta = prev_score - score  # positive = drop

                if delta >= self.hard_threshold:
                    # Hard regression — alert immediately
                    msg = (
                        f"[HARD] {name}: {prev_score:.3f} → {score:.3f} "
                        f"(Δ−{delta:.3f}) at '{tag}' — exceeds hard threshold {self.hard_threshold}"
                    )
                    alerts.append(RegressionAlert(
                        benchmark=name, tag=tag,
                        prev_score=prev_score, curr_score=score,
                        delta=delta, severity="hard", message=msg,
                    ))
                    logger.warning(msg)
                    self._soft_streak[name] = 0

                elif delta >= self.soft_threshold:
                    # Soft drop — increment streak
                    streak = self._soft_streak.get(name, 0) + 1
                    self._soft_streak[name] = streak
                    logger.debug(
                        "[%s] soft drop %.3f → %.3f (streak=%d/%d)",
                        name, prev_score, score, streak, self.patience,
                    )
                    if streak >= self.patience:
                        msg = (
                            f"[SOFT] {name}: persistent drop over {streak} steps, "
                            f"now {score:.3f} (was {history[-self.patience][1]:.3f}) "
                            f"at '{tag}'"
                        )
                        alerts.append(RegressionAlert(
                            benchmark=name, tag=tag,
                            prev_score=history[-self.patience][1], curr_score=score,
                            delta=history[-self.patience][1] - score,
                            severity="soft", message=msg,
                        ))
                        logger.warning(msg)
                        self._soft_streak[name] = 0  # reset after alert

                else:
                    # Improvement or flat — reset soft streak
                    if delta < 0:
                        logger.debug("[%s] improved %.3f → %.3f", name, prev_score, score)
                    self._soft_streak[name] = 0

            history.append((tag, score))

        if self.history_path:
            self._save(self.history_path)

        return alerts

    def reset(self) -> None:
        """Clear all history and streaks."""
        self._history.clear()
        self._soft_streak.clear()

    def history(self, benchmark_name: str) -> List[tuple[str, float]]:
        """Return the score history for a named benchmark as (tag, score) pairs."""
        return list(self._history.get(benchmark_name, []))

    def best_score(self, benchmark_name: str) -> Optional[float]:
        """Return the highest score ever recorded for a benchmark."""
        hist = self._history.get(benchmark_name)
        if not hist:
            return None
        return max(s for _, s in hist)

    def latest_score(self, benchmark_name: str) -> Optional[float]:
        """Return the most recent score for a benchmark."""
        hist = self._history.get(benchmark_name)
        if not hist:
            return None
        return hist[-1][1]

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary of tracked benchmarks."""
        out: Dict[str, Any] = {}
        for name, hist in self._history.items():
            out[name] = {
                "n_steps": len(hist),
                "latest": hist[-1][1] if hist else None,
                "best": max(s for _, s in hist) if hist else None,
                "history": [{"tag": t, "score": round(s, 4)} for t, s in hist],
            }
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "hard_threshold": self.hard_threshold,
                "soft_threshold": self.soft_threshold,
                "patience": self.patience,
                "history": {
                    name: [list(entry) for entry in hist]
                    for name, hist in self._history.items()
                },
                "soft_streak": self._soft_streak,
            }, fh, indent=2)

    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self._history = {
            name: [tuple(entry) for entry in hist]   # type: ignore[misc]
            for name, hist in data.get("history", {}).items()
        }
        self._soft_streak = data.get("soft_streak", {})
        logger.info("RegressionDetector: loaded history from %s (%d benchmarks)", path, len(self._history))
