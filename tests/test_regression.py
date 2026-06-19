"""Unit tests for :mod:`selfllm.eval.regression`.

The :class:`RegressionDetector` is the gate that decides whether a
self-improvement iteration regressed benchmark performance. Its decisions
are high-stakes (a missed hard regression silently accepts a worse model),
so this suite exercises the full decision matrix:

- hard regressions always alert on the first occurrence
- sub-soft fluctuations never alert
- soft drops only alert after ``patience`` consecutive steps
- improvements / recoveries reset the soft streak
- multi-benchmark histories are tracked independently
- history survives a JSON save/load round-trip
"""

from __future__ import annotations

import json

import pytest

from selfllm.eval.harness import EvalResult
from selfllm.eval.regression import RegressionAlert, RegressionDetector
from selfllm.eval.suite import SuiteResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _suite(tag: str, **scores: float) -> SuiteResult:
    """Build a SuiteResult with one EvalResult per ``name=score`` kwarg."""
    results = [
        EvalResult(name=name, score=score, n=10)
        for name, score in scores.items()
    ]
    return SuiteResult(tag=tag, timestamp=0.0, results=results, elapsed_s=0.0)


def _feed(detector: RegressionDetector, name: str, scores, tag_prefix="iter"):
    """Feed a sequence of scores for a single benchmark, returning all alerts."""
    all_alerts = []
    for i, s in enumerate(scores):
        all_alerts.append(detector.update(_suite(f"{tag_prefix}_{i}", **{name: s})))
    return all_alerts


# ---------------------------------------------------------------------------
# Baseline behaviour
# ---------------------------------------------------------------------------


def test_first_update_never_alerts():
    """The very first observation has no predecessor, so it can't regress."""
    det = RegressionDetector(hard_threshold=0.05)
    alerts = det.update(_suite("baseline", mmlu=0.10))
    assert alerts == []
    assert det.latest_score("mmlu") == pytest.approx(0.10)


def test_improvement_does_not_alert():
    det = RegressionDetector()
    alerts = _feed(det, "mmlu", [0.40, 0.50, 0.60])
    assert all(a == [] for a in alerts)
    assert det.best_score("mmlu") == pytest.approx(0.60)


def test_flat_scores_do_not_alert():
    det = RegressionDetector(soft_threshold=0.02)
    alerts = _feed(det, "mmlu", [0.50, 0.50, 0.50, 0.50])
    assert all(a == [] for a in alerts)


# ---------------------------------------------------------------------------
# Hard regressions
# ---------------------------------------------------------------------------


def test_hard_regression_alerts_immediately():
    det = RegressionDetector(hard_threshold=0.05)
    det.update(_suite("baseline", mmlu=0.60))
    alerts = det.update(_suite("iter_1", mmlu=0.50))  # 0.10 drop >= 0.05

    assert len(alerts) == 1
    alert = alerts[0]
    assert isinstance(alert, RegressionAlert)
    assert alert.severity == "hard"
    assert alert.benchmark == "mmlu"
    assert alert.tag == "iter_1"
    assert alert.prev_score == pytest.approx(0.60)
    assert alert.curr_score == pytest.approx(0.50)
    assert alert.delta == pytest.approx(0.10)


def test_drop_exactly_at_hard_threshold_alerts():
    """The threshold is inclusive (``delta >= hard_threshold``).

    Uses values whose difference is exactly representable in binary floating
    point (1.0 - 0.75 == 0.25) to test the boundary itself, not float noise.
    """
    det = RegressionDetector(hard_threshold=0.25)
    det.update(_suite("baseline", mmlu=1.0))
    alerts = det.update(_suite("iter_1", mmlu=0.75))
    assert len(alerts) == 1
    assert alerts[0].severity == "hard"


def test_threshold_boundary_is_float_fragile():
    """Documents a known limitation: because the comparison is a raw float
    subtraction with no epsilon, a drop that is *nominally* equal to the
    threshold (0.60 - 0.55 = 0.05) can fall just under it due to binary
    floating-point representation and fail to alert.

    If the detector is ever hardened with an epsilon/rounding tolerance,
    this test should flip to asserting an alert IS produced.
    """
    assert (0.60 - 0.55) < 0.05  # the float reality this guards against
    det = RegressionDetector(hard_threshold=0.05)
    det.update(_suite("baseline", mmlu=0.60))
    alerts = det.update(_suite("iter_1", mmlu=0.55))
    assert alerts == []


def test_hard_regression_resets_soft_streak():
    """A hard alert should clear any accumulated soft streak."""
    det = RegressionDetector(hard_threshold=0.10, soft_threshold=0.02, patience=3)
    det.update(_suite("b", mmlu=1.00))
    det.update(_suite("i1", mmlu=0.97))  # soft drop, streak=1
    det.update(_suite("i2", mmlu=0.94))  # soft drop, streak=2
    alerts = det.update(_suite("i3", mmlu=0.80))  # 0.14 drop -> hard

    assert len(alerts) == 1
    assert alerts[0].severity == "hard"
    # Streak was reset, so the next soft drop must start counting from 1 again.
    det.update(_suite("i4", mmlu=0.77))  # soft, streak=1
    det.update(_suite("i5", mmlu=0.74))  # soft, streak=2
    final = det.update(_suite("i6", mmlu=0.71))  # soft, streak=3 -> alert
    assert len(final) == 1
    assert final[0].severity == "soft"


# ---------------------------------------------------------------------------
# Soft regressions / patience
# ---------------------------------------------------------------------------


def test_subthreshold_drop_never_alerts():
    """Drops below soft_threshold are pure noise and must be ignored forever."""
    det = RegressionDetector(hard_threshold=0.05, soft_threshold=0.02)
    # each step drops 0.01 (< 0.02) for many steps
    alerts = _feed(det, "mmlu", [0.50 - 0.01 * i for i in range(8)])
    assert all(a == [] for a in alerts)


def test_soft_alert_only_after_patience():
    det = RegressionDetector(hard_threshold=0.10, soft_threshold=0.02, patience=3)
    # consistent 0.03 drops: soft each step
    a0 = det.update(_suite("b", mmlu=1.00))
    a1 = det.update(_suite("i1", mmlu=0.97))  # streak 1
    a2 = det.update(_suite("i2", mmlu=0.94))  # streak 2
    a3 = det.update(_suite("i3", mmlu=0.91))  # streak 3 -> alert
    assert a0 == [] and a1 == [] and a2 == []
    assert len(a3) == 1
    assert a3[0].severity == "soft"
    # delta is measured across the patience window (3 steps), not one step.
    assert a3[0].delta == pytest.approx(1.00 - 0.91)


def test_soft_streak_resets_after_alert():
    det = RegressionDetector(hard_threshold=0.10, soft_threshold=0.02, patience=2)
    det.update(_suite("b", mmlu=1.00))
    det.update(_suite("i1", mmlu=0.97))             # streak 1
    a2 = det.update(_suite("i2", mmlu=0.94))        # streak 2 -> alert, reset
    assert len(a2) == 1
    a3 = det.update(_suite("i3", mmlu=0.91))        # streak 1 again, no alert
    assert a3 == []


def test_improvement_breaks_soft_streak():
    det = RegressionDetector(hard_threshold=0.10, soft_threshold=0.02, patience=3)
    det.update(_suite("b", mmlu=1.00))
    det.update(_suite("i1", mmlu=0.97))   # streak 1
    det.update(_suite("i2", mmlu=0.94))   # streak 2
    det.update(_suite("i3", mmlu=0.99))   # improvement -> reset
    a4 = det.update(_suite("i4", mmlu=0.96))  # streak 1, no alert
    a5 = det.update(_suite("i5", mmlu=0.93))  # streak 2, no alert
    assert a4 == [] and a5 == []


# ---------------------------------------------------------------------------
# Multi-benchmark independence
# ---------------------------------------------------------------------------


def test_benchmarks_tracked_independently():
    det = RegressionDetector(hard_threshold=0.05)
    det.update(_suite("b", mmlu=0.60, gsm8k=0.30))
    alerts = det.update(_suite("i1", mmlu=0.50, gsm8k=0.35))  # mmlu drops, gsm8k up
    assert len(alerts) == 1
    assert alerts[0].benchmark == "mmlu"
    assert det.latest_score("gsm8k") == pytest.approx(0.35)


def test_multiple_simultaneous_hard_regressions():
    det = RegressionDetector(hard_threshold=0.05)
    det.update(_suite("b", mmlu=0.60, gsm8k=0.60))
    alerts = det.update(_suite("i1", mmlu=0.50, gsm8k=0.40))
    assert len(alerts) == 2
    assert {a.benchmark for a in alerts} == {"mmlu", "gsm8k"}
    assert all(a.severity == "hard" for a in alerts)


# ---------------------------------------------------------------------------
# Accessors & summary
# ---------------------------------------------------------------------------


def test_accessors_on_empty_detector():
    det = RegressionDetector()
    assert det.best_score("nope") is None
    assert det.latest_score("nope") is None
    assert det.history("nope") == []


def test_history_and_best_and_latest():
    det = RegressionDetector()
    _feed(det, "mmlu", [0.4, 0.6, 0.5])
    hist = det.history("mmlu")
    assert [round(s, 3) for _, s in hist] == [0.4, 0.6, 0.5]
    assert det.best_score("mmlu") == pytest.approx(0.6)
    assert det.latest_score("mmlu") == pytest.approx(0.5)


def test_reset_clears_state():
    det = RegressionDetector()
    _feed(det, "mmlu", [0.4, 0.5])
    det.reset()
    assert det.history("mmlu") == []
    assert det.latest_score("mmlu") is None


def test_summary_is_json_serialisable():
    det = RegressionDetector()
    _feed(det, "mmlu", [0.4, 0.6, 0.5])
    summary = det.summary()
    # Round-trips through JSON without error.
    encoded = json.dumps(summary)
    decoded = json.loads(encoded)
    assert decoded["mmlu"]["n_steps"] == 3
    assert decoded["mmlu"]["best"] == pytest.approx(0.6)
    assert decoded["mmlu"]["latest"] == pytest.approx(0.5)


def test_alert_to_dict_rounds_values():
    det = RegressionDetector(hard_threshold=0.05)
    det.update(_suite("b", mmlu=0.601234))
    alerts = det.update(_suite("i1", mmlu=0.501234))
    d = alerts[0].to_dict()
    assert d["severity"] == "hard"
    assert d["prev_score"] == 0.6012
    assert d["curr_score"] == 0.5012
    # Whole dict must be JSON serialisable.
    json.dumps(d)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_history_persists_across_save_and_reload(tmp_path):
    path = str(tmp_path / "nested" / "history.json")  # nested dir must be created
    det = RegressionDetector(hard_threshold=0.05, history_path=path)
    det.update(_suite("b", mmlu=0.60, gsm8k=0.30))
    det.update(_suite("i1", mmlu=0.58, gsm8k=0.31))

    # A fresh detector pointed at the same file restores history.
    reloaded = RegressionDetector(history_path=path)
    assert reloaded.latest_score("mmlu") == pytest.approx(0.58)
    assert reloaded.latest_score("gsm8k") == pytest.approx(0.31)
    assert len(reloaded.history("mmlu")) == 2


def test_reloaded_detector_continues_detecting():
    """A detector restored from disk should still flag a regression vs. the
    last persisted score (state, not just data, survives)."""
    import tempfile, os

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hist.json")
        det = RegressionDetector(hard_threshold=0.05, history_path=path)
        det.update(_suite("b", mmlu=0.60))

        reloaded = RegressionDetector(hard_threshold=0.05, history_path=path)
        alerts = reloaded.update(_suite("i1", mmlu=0.50))
        assert len(alerts) == 1
        assert alerts[0].severity == "hard"
