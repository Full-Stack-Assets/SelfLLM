#!/usr/bin/env python3
"""
Benchmark Suite Smoke Test
===========================
Validates the full BenchmarkSuite → training_summary.json → RegressionDetector
pipeline using a tiny in-memory model and 5-example reduced datasets.

This script is called by the ``benchmark-smoke`` job in ci.yml and is
designed to be completely self-contained: no weights files, no network
access, no GPU required.

Exit codes:
    0  All assertions passed
    1  At least one assertion failed

Tests performed
---------------
1.  BenchmarkSuite runs MMLU (5 ex) + GSM8K (5 ex) and returns 2 EvalResults
    with scores in [0, 1] and n == 5.
2.  A timestamped JSON snapshot is written to a temp results_dir.
3.  A training_summary.json-style file is constructed and verified for
    required keys (model_params, pretrain, lora, self_improvement).
4.  RegressionDetector emits ZERO alerts on the first (baseline) run.
5.  RegressionDetector emits at least ONE hard alert when a second run is
    injected with score=0.0 on every benchmark (catastrophic-drop simulation).
6.  SuiteResult.regression_vs(baseline, threshold=0.05) returns True for the
    crashed run and False for the baseline vs itself.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

# Ensure the repo root is importable regardless of cwd
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import torch

from selfllm.eval.harness import EvalResult
from selfllm.eval.regression import RegressionDetector
from selfllm.eval.suite import BenchmarkSuite, SuiteResult
from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer

# ---------------------------------------------------------------------------
# Inline reduced datasets
# ---------------------------------------------------------------------------

MMLU_RECORDS = [
    {"question": "What is 2+2?",       "choices": ["3", "4", "5", "6"],             "answer": "B"},
    {"question": "Capital of France?", "choices": ["Rome", "Berlin", "Paris", "Madrid"], "answer": "C"},
    {"question": "H2O is?",            "choices": ["salt", "water", "acid", "base"], "answer": "B"},
    {"question": "Speed of light units?", "choices": ["km/s", "m/s", "m/h", "km/h"], "answer": "B"},
    {"question": "DNA contains?",      "choices": ["ATP", "RNA", "genes", "codons"], "answer": "C"},
]

GSM8K_RECORDS = [
    {"question": "If x=2, what is 2x?",                       "answer": "####4"},
    {"question": "Tom has 3 apples, gets 2 more. How many?",  "answer": "####5"},
    {"question": "What is 10 minus 3?",                        "answer": "####7"},
    {"question": "What is 4 times 4?",                         "answer": "####16"},
    {"question": "What is 100 divided by 5?",                  "answer": "####20"},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model() -> tuple[SelfImprovingLLM, BPETokenizer]:
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=256, d_model=32, n_layers=2, n_heads=2,
        d_ff=64, max_seq_len=64, dropout=0.0,
        use_rope=False, use_swiglu=False,
    )
    model = SelfImprovingLLM(cfg)
    model.eval()

    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning and artificial intelligence",
        "once upon a time in a land far far away",
    ] * 20
    tok = BPETokenizer(vocab_size=256)
    tok.train(corpus)
    if not hasattr(tok, "eos_token_id"):
        tok.eos_token_id = 1
    return model, tok


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------


def main() -> None:
    results_dir = tempfile.mkdtemp(prefix="selfllm_smoke_")
    print(f"Smoke test results dir: {results_dir}")

    # ------------------------------------------------------------------ #
    # Step 1 — Build model + tokenizer
    # ------------------------------------------------------------------ #
    model, tok = _tiny_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params  vocab={tok.vocab_size}")

    # ------------------------------------------------------------------ #
    # Step 2 — Run BenchmarkSuite with reduced MMLU + GSM8K
    # ------------------------------------------------------------------ #
    suite = BenchmarkSuite(
        mmlu_records=MMLU_RECORDS,
        gsm8k_records=GSM8K_RECORDS,
        results_dir=results_dir,
        max_examples=5,
        max_new_tokens=8,
        temperature=0.0,
    )

    baseline = suite.run(model, tok, tag="smoke_baseline")

    _assert(
        len(baseline.results) == 2,
        f"Expected 2 benchmark results (mmlu+gsm8k), got {len(baseline.results)}"
    )
    for r in baseline.results:
        _assert(0.0 <= r.score <= 1.0, f"{r.name} score {r.score:.4f} out of [0,1]")
        _assert(r.n == 5, f"{r.name} n={r.n}, expected 5")
    print(f"[PASS] BenchmarkSuite: {baseline.summary_str()}")

    # ------------------------------------------------------------------ #
    # Step 3 — Verify JSON snapshot in results_dir
    # ------------------------------------------------------------------ #
    snapshots = [f for f in os.listdir(results_dir) if f.startswith("eval_smoke_baseline")]
    _assert(len(snapshots) == 1, f"Expected 1 snapshot file, found: {snapshots}")

    snap_path = os.path.join(results_dir, snapshots[0])
    with open(snap_path) as fh:
        snap = json.load(fh)

    _assert(snap["tag"] == "smoke_baseline", f"Snapshot tag mismatch: {snap['tag']}")
    _assert(len(snap["results"]) == 2,       f"Snapshot results count mismatch")
    _assert("average_score" in snap,         "Snapshot missing average_score key")
    _assert("elapsed_s" in snap,             "Snapshot missing elapsed_s key")
    print(f"[PASS] Snapshot written: {snapshots[0]}")

    # ------------------------------------------------------------------ #
    # Step 4 — Write training_summary.json-style file
    # ------------------------------------------------------------------ #
    summary = {
        "model_params": n_params,
        "model_config": {"vocab_size": 256, "d_model": 32, "n_layers": 2},
        "pretrain": {
            "epochs": 3,
            "loss_history": [6.1698, 3.6141, 2.8626],
            "note": "Smoke test — synthetic pretrain metrics",
        },
        "lora": {
            "lora": 94176,
            "lora_pct": 1.37,
            "frozen": 6766848,
        },
        "self_improvement": {
            "iterations_run": 10,
            "first_iter_loss": 1.4501,
            "last_iter_loss": 2.1733,
            "duration_s": 44.2,
        },
        "eval_smoke": baseline.to_dict(),
    }
    summary_path = os.path.join(results_dir, "training_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    # Verify required keys present
    for key in ("model_params", "pretrain", "lora", "self_improvement"):
        _assert(key in summary, f"training_summary.json missing key: {key}")

    # Verify loss is monotonically decreasing
    lh = summary["pretrain"]["loss_history"]
    for i in range(1, len(lh)):
        _assert(lh[i] < lh[i - 1],
                f"Pretrain loss should decrease: epoch {i}: {lh[i-1]:.4f} -> {lh[i]:.4f}")

    # Verify LoRA pct < 10%
    _assert(summary["lora"]["lora_pct"] < 10.0,
            f"LoRA trainable% {summary['lora']['lora_pct']:.2f}% should be <10%")

    print(f"[PASS] training_summary.json: valid schema, monotone loss, LoRA pct OK")

    # ------------------------------------------------------------------ #
    # Step 5 — RegressionDetector: no alert on baseline
    # ------------------------------------------------------------------ #
    detector = RegressionDetector(
        hard_threshold=0.05,
        soft_threshold=0.02,
        patience=3,
        history_path=os.path.join(results_dir, "regression_history.json"),
    )
    # Seed the detector with a known-good, non-zero baseline. The smoke model is
    # tiny and untrained, so its real benchmark scores are 0.0 — feeding those as
    # the baseline would make the Step 6 "drop to 0.0" a no-op (0.0 → 0.0) that
    # the detector can never alert on. An explicit synthetic baseline exercises
    # the alerting logic deterministically, independent of the throwaway model.
    seed_baseline = SuiteResult(
        tag="smoke_seed",
        timestamp=time.time(),
        results=[
            EvalResult(name="mmlu", score=0.60, n=5),
            EvalResult(name="gsm8k", score=0.50, n=5),
        ],
        elapsed_s=0.1,
    )
    alerts = detector.update(seed_baseline)
    _assert(len(alerts) == 0, f"Expected 0 alerts on first run, got: {alerts}")
    print("[PASS] RegressionDetector: 0 alerts on baseline (correct)")

    # ------------------------------------------------------------------ #
    # Step 6 — RegressionDetector: hard alert on catastrophic drop
    # ------------------------------------------------------------------ #
    crashed_results = [
        EvalResult(name="mmlu",  score=0.00, n=5),
        EvalResult(name="gsm8k", score=0.00, n=5),
    ]
    crashed = SuiteResult(
        tag="smoke_crash",
        timestamp=time.time(),
        results=crashed_results,
        elapsed_s=0.1,
    )
    alerts = detector.update(crashed)
    hard_alerts = [a for a in alerts if a.severity == "hard"]

    _assert(len(hard_alerts) >= 1,
            f"Expected >=1 hard alert on drop to 0.0, got {len(alerts)} total, {len(hard_alerts)} hard")

    for a in hard_alerts:
        print(f"  [{a.severity.upper()}] {a.benchmark}: "
              f"{a.prev_score:.3f} → {a.curr_score:.3f} (Δ={a.delta:.3f})")
    print(f"[PASS] RegressionDetector: {len(hard_alerts)} hard alert(s) fired correctly")

    # Verify history was persisted
    hist_path = os.path.join(results_dir, "regression_history.json")
    _assert(os.path.exists(hist_path), "regression_history.json was not written")
    with open(hist_path) as fh:
        hist = json.load(fh)
    _assert("mmlu" in hist.get("history", {}),    "mmlu not in persisted history")
    _assert("gsm8k" in hist.get("history", {}),   "gsm8k not in persisted history")
    print("[PASS] Regression history persisted to JSON")

    # ------------------------------------------------------------------ #
    # Step 7 — SuiteResult.regression_vs() helper
    # ------------------------------------------------------------------ #
    # Compare against the synthetic non-zero baseline (the real smoke baseline
    # scores 0.0, so a 0.0 → 0.0 comparison could never be a regression).
    _assert(
        crashed.regression_vs(seed_baseline, threshold=0.05),
        "crashed.regression_vs(seed_baseline) should be True for score drop to 0"
    )
    _assert(
        not seed_baseline.regression_vs(seed_baseline, threshold=0.05),
        "seed_baseline.regression_vs(seed_baseline) should be False (same run)"
    )
    print("[PASS] SuiteResult.regression_vs() helper correct")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("  ALL BENCHMARK SMOKE TESTS PASSED")
    print("=" * 60)
    print(f"  Python        : {sys.version.split()[0]}")
    print(f"  Results dir   : {results_dir}")
    print(f"  Snapshot      : {snapshots[0]}")
    print(f"  MMLU score    : {baseline.score('mmlu'):.3f}")
    print(f"  GSM8K score   : {baseline.score('gsm8k'):.3f}")
    print(f"  Hard alerts   : {len(hard_alerts)}/2 benchmarks fired on crash")


if __name__ == "__main__":
    main()
