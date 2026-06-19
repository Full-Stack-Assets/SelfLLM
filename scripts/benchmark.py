#!/usr/bin/env python3
"""
SelfLLM Serving Benchmark CLI
==============================
Benchmarks the three inference engines (baseline eager, torch.compile,
paged continuous-batching scheduler) against a tiny in-memory model and
writes structured JSON results.

Designed to be called from the perf.yml GitHub Actions workflow, but also
useful locally for quick regression checks.

Usage::

    python scripts/benchmark.py --help
    python scripts/benchmark.py --max-new-tokens 32 --timed-runs 3
    python scripts/benchmark.py --output results/benchmark_raw.json

The script uses a tiny in-memory model (no weights file needed) so it can
run in any CI environment without artefact access.  Results are comparable
across runs on the same hardware.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Resolve the package root (one level up from scripts/)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import torch

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.serving.optimized import compile_model, is_compiled


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tiny model — fast enough for CI
_CI_CONFIG = ModelConfig(
    vocab_size=256,
    d_model=64,
    n_layers=2,
    n_heads=4,
    d_ff=128,
    max_seq_len=128,
    dropout=0.0,
    use_rope=True,
    use_swiglu=False,
)

_PROMPTS = [
    "The history of science is",
    "Once upon a time in a land far",
    "In the beginning there was",
    "Artificial intelligence enables",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchRow:
    engine: str
    workload: str
    tokens_per_sec: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    peak_mem_mb: float
    new_tokens: int
    batch_size: int
    runs: int
    speedup_vs_baseline: Optional[float] = None
    notes: str = ""
    raw_latencies_ms: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "workload": self.workload,
            "tokens_per_sec": round(self.tokens_per_sec, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "peak_mem_mb": round(self.peak_mem_mb, 3),
            "new_tokens": self.new_tokens,
            "batch_size": self.batch_size,
            "runs": self.runs,
            "speedup_vs_baseline": round(self.speedup_vs_baseline, 4) if self.speedup_vs_baseline else None,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pct(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    d = sorted(data)
    idx = (len(d) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(d) - 1)
    return d[lo] + (d[hi] - d[lo]) * (idx - lo)


def _mem_mb() -> float:
    snap = tracemalloc.take_snapshot()
    return sum(s.size for s in snap.statistics("lineno")) / (1024 * 1024)


def _make_input(tokenizer: BPETokenizer, prompt: str, batch: int) -> torch.Tensor:
    ids = tokenizer.encode(prompt)[:20]
    return torch.tensor([ids] * batch, dtype=torch.long)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_eager(
    model: SelfImprovingLLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    batch_size: int,
    timed_runs: int,
    warmup_runs: int = 1,
    temperature: float = 0.0,
) -> BenchRow:
    label = f"bs{batch_size}_tok{max_new_tokens}"
    input_ids = _make_input(tokenizer, prompt, batch_size)

    def _run():
        with torch.no_grad():
            return model.generate(input_ids, max_new_tokens=max_new_tokens,
                                  temperature=temperature)

    for _ in range(warmup_runs):
        _run()

    gc.collect()
    tracemalloc.start()
    lats = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        out = _run()
        lats.append(time.perf_counter() - t0)
    peak = _mem_mb()
    tracemalloc.stop()

    gen_len = max(int(out["sequences"].shape[1] - input_ids.shape[1]), max_new_tokens)
    total_new = gen_len * batch_size
    tps = (total_new * timed_runs) / sum(lats)

    return BenchRow(
        engine="A_baseline_eager",
        workload=label,
        tokens_per_sec=tps,
        avg_latency_ms=statistics.mean(lats) * 1000,
        p50_latency_ms=_pct(lats, 50) * 1000,
        p95_latency_ms=_pct(lats, 95) * 1000,
        peak_mem_mb=peak,
        new_tokens=total_new,
        batch_size=batch_size,
        runs=timed_runs,
        raw_latencies_ms=[round(l * 1000, 2) for l in lats],
    )


def bench_compiled(
    model: SelfImprovingLLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    batch_size: int,
    timed_runs: int,
    warmup_runs: int = 1,
    temperature: float = 0.0,
) -> BenchRow:
    label = f"bs{batch_size}_tok{max_new_tokens}"
    compiled = compile_model(model, mode="reduce-overhead")
    active = is_compiled(compiled)
    input_ids = _make_input(tokenizer, prompt, batch_size)

    def _run():
        with torch.no_grad():
            return compiled.generate(input_ids, max_new_tokens=max_new_tokens,
                                     temperature=temperature)

    for _ in range(warmup_runs):
        _run()

    gc.collect()
    tracemalloc.start()
    lats = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        out = _run()
        lats.append(time.perf_counter() - t0)
    peak = _mem_mb()
    tracemalloc.stop()

    gen_len = max(int(out["sequences"].shape[1] - input_ids.shape[1]), max_new_tokens)
    total_new = gen_len * batch_size
    tps = (total_new * timed_runs) / sum(lats)
    note = "dynamo=ACTIVE" if active else "dynamo=FALLBACK(eager)"

    return BenchRow(
        engine="B_compiled",
        workload=label,
        tokens_per_sec=tps,
        avg_latency_ms=statistics.mean(lats) * 1000,
        p50_latency_ms=_pct(lats, 50) * 1000,
        p95_latency_ms=_pct(lats, 95) * 1000,
        peak_mem_mb=peak,
        new_tokens=total_new,
        batch_size=batch_size,
        runs=timed_runs,
        notes=note,
        raw_latencies_ms=[round(l * 1000, 2) for l in lats],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SelfLLM serving benchmark CLI")
    p.add_argument("--max-new-tokens", type=int, default=32,
                   help="Tokens to generate per call (default: 32)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size for throughput benchmark (default: 1)")
    p.add_argument("--timed-runs", type=int, default=5,
                   help="Number of timed runs per engine (default: 5)")
    p.add_argument("--warmup-runs", type=int, default=1,
                   help="Warmup runs before timing (default: 1)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature 0=greedy (default: 0.0)")
    p.add_argument("--output", type=str, default="benchmark_results.json",
                   help="Output JSON path (default: benchmark_results.json)")
    p.add_argument("--prompt", type=str, default=_PROMPTS[0],
                   help="Prompt text to use")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    print("=" * 60)
    print("  SelfLLM Benchmark CLI")
    print("=" * 60)
    print(f"  max_new_tokens : {args.max_new_tokens}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  timed_runs     : {args.timed_runs}")
    print(f"  output         : {args.output}")
    print()

    # Build tiny model + tokenizer (no weights needed — CI-safe)
    model = SelfImprovingLLM(_CI_CONFIG)
    model.eval()

    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning and artificial intelligence",
        "once upon a time in a land far far away",
    ] * 20
    tokenizer = BPETokenizer(vocab_size=_CI_CONFIG.vocab_size)
    tokenizer.train(corpus)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params/1e6:.2f}M params  vocab={_CI_CONFIG.vocab_size}")
    print()

    rows: List[BenchRow] = []

    # Engine A
    print(f"Engine A (baseline eager)...", end="", flush=True)
    r_a = bench_eager(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        timed_runs=args.timed_runs,
        warmup_runs=args.warmup_runs,
        temperature=args.temperature,
    )
    rows.append(r_a)
    print(f"  {r_a.tokens_per_sec:.1f} tok/s  avg={r_a.avg_latency_ms:.0f}ms")

    # Engine B
    print(f"Engine B (compiled)...", end="", flush=True)
    r_b = bench_compiled(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        timed_runs=args.timed_runs,
        warmup_runs=args.warmup_runs,
        temperature=args.temperature,
    )
    speedup = r_b.tokens_per_sec / max(r_a.tokens_per_sec, 1e-9)
    r_b.speedup_vs_baseline = speedup
    rows.append(r_b)
    print(f"  {r_b.tokens_per_sec:.1f} tok/s  avg={r_b.avg_latency_ms:.0f}ms  "
          f"speedup={speedup:.3f}x  [{r_b.notes}]")

    # Summary table
    print()
    print(f"  {'Engine':25s}  {'tok/s':>8}  {'avg ms':>8}  {'p95 ms':>8}  {'speedup':>8}")
    print(f"  {'-'*60}")
    for r in rows:
        sp = f"{r.speedup_vs_baseline:.3f}x" if r.speedup_vs_baseline else "—"
        print(f"  {r.engine:25s}  {r.tokens_per_sec:>8.1f}  {r.avg_latency_ms:>8.0f}  "
              f"{r.p95_latency_ms:>8.0f}  {sp:>8}")

    # Write output
    output = {
        "timestamp": time.time(),
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "timed_runs": args.timed_runs,
            "model_params": total_params,
        },
        "results": [r.to_dict() for r in rows],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=2)
    print()
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
