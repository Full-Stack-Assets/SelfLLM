"""Formatting helpers for recursive self-improvement metrics.

Turns a ``metrics_history`` (the per-iteration records produced by
:class:`~selfllm.recursive.recursive_trainer.RecursiveSelfTrainer`) into a
human-readable table and summary. Any benchmark scores recorded by the
eval-suite hook -- stored under ``benchmark_<name>`` keys -- are surfaced
automatically alongside the core metrics, so MMLU / GSM8K / HumanEval results
show up in reports and the dashboard without any per-benchmark wiring.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

# Core per-iteration columns shown first, in this order. Each tuple is
# (metric key, display header, float format).
CORE_COLUMNS = [
    ("iteration", "iter", "{:d}"),
    ("train_loss", "train_loss", "{:.4f}"),
    ("eval_perplexity", "perplexity", "{:.2f}"),
    ("quality_score", "quality", "{:.4f}"),
]

BENCHMARK_PREFIX = "benchmark_"


def load_metrics(path: str) -> List[Dict[str, Any]]:
    """Load a ``metrics_history.json`` file into a list of records."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of metric records, got {type(data).__name__}")
    return data


def detect_benchmark_keys(history: Sequence[Dict[str, Any]]) -> List[str]:
    """Return the sorted set of ``benchmark_*`` keys present in the history."""
    keys = {
        k
        for record in history
        for k in record
        if k.startswith(BENCHMARK_PREFIX)
    }
    return sorted(keys)


def _benchmark_header(key: str) -> str:
    """Strip the ``benchmark_`` prefix for display."""
    return key[len(BENCHMARK_PREFIX):]


def _fmt(value: Any, fmt: str) -> str:
    """Format a numeric value, tolerating missing/non-numeric entries."""
    if value is None:
        return "-"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def format_metrics_table(history: Sequence[Dict[str, Any]]) -> str:
    """Render the metrics history as a GitHub-flavored Markdown table.

    Columns are the core metrics followed by one column per detected benchmark.
    """
    if not history:
        return "_(no metrics recorded)_"

    benchmark_keys = detect_benchmark_keys(history)
    headers = [h for _, h, _ in CORE_COLUMNS] + [
        _benchmark_header(k) for k in benchmark_keys
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in history:
        cells = [_fmt(record.get(key), fmt) for key, _, fmt in CORE_COLUMNS]
        cells += [_fmt(record.get(k), "{:.4f}") for k in benchmark_keys]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize_metrics(history: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute first -> last trajectory for core metrics and each benchmark.

    Returns a mapping ``metric -> {"first", "last", "delta"}``. Benchmarks that
    are only recorded on some iterations use their first and last *recorded*
    values.
    """
    summary: Dict[str, Dict[str, float]] = {}
    if not history:
        return summary

    for key in ("eval_perplexity", "quality_score", "train_loss"):
        present = [r[key] for r in history if isinstance(r.get(key), (int, float))]
        if present:
            summary[key] = {
                "first": float(present[0]),
                "last": float(present[-1]),
                "delta": float(present[-1] - present[0]),
            }

    for key in detect_benchmark_keys(history):
        present = [r[key] for r in history if isinstance(r.get(key), (int, float))]
        if present:
            summary[key] = {
                "first": float(present[0]),
                "last": float(present[-1]),
                "delta": float(present[-1] - present[0]),
            }
    return summary


def format_metrics_report(history: Sequence[Dict[str, Any]]) -> str:
    """Build a full Markdown report: a summary section plus the table.

    The summary lists first -> last (delta) for perplexity, quality, train
    loss, and every benchmark recorded by the eval-suite hook.
    """
    if not history:
        return "_(no metrics recorded)_"

    summary = summarize_metrics(history)
    lines = [
        f"### Recursive self-improvement metrics ({len(history)} iterations)",
        "",
        "**Trajectory (first -> last):**",
        "",
    ]
    label = {
        "eval_perplexity": "Perplexity",
        "quality_score": "Quality",
        "train_loss": "Train loss",
    }
    for key in ("eval_perplexity", "quality_score", "train_loss"):
        if key in summary:
            s = summary[key]
            lines.append(
                f"- {label[key]}: {s['first']:.4f} -> {s['last']:.4f} "
                f"({s['delta']:+.4f})"
            )

    benchmark_keys = detect_benchmark_keys(history)
    if benchmark_keys:
        lines.append("")
        lines.append("**Benchmarks (first -> last):**")
        lines.append("")
        for key in benchmark_keys:
            if key in summary:
                s = summary[key]
                lines.append(
                    f"- {_benchmark_header(key)}: {s['first']:.4f} -> "
                    f"{s['last']:.4f} ({s['delta']:+.4f})"
                )

    lines.append("")
    lines.append(format_metrics_table(history))
    return "\n".join(lines)


def metrics_table_rows(
    history: Sequence[Dict[str, Any]],
) -> List[List[Any]]:
    """Return ``[headers, *rows]`` suitable for a ``gr.Dataframe``.

    Numeric cells are rounded; benchmark columns are appended after the core
    columns (header without the ``benchmark_`` prefix).
    """
    benchmark_keys = detect_benchmark_keys(history)
    headers = [h for _, h, _ in CORE_COLUMNS] + [
        _benchmark_header(k) for k in benchmark_keys
    ]
    rows: List[List[Any]] = [headers]
    for record in history:
        row: List[Any] = []
        for key, _, _ in CORE_COLUMNS:
            v = record.get(key)
            row.append(round(v, 4) if isinstance(v, float) else v)
        for k in benchmark_keys:
            v = record.get(k)
            row.append(round(v, 4) if isinstance(v, (int, float)) else v)
        rows.append(row)
    return rows
