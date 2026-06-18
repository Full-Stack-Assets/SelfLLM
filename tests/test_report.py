"""Tests for the recursive-metrics report utility (selfllm.recursive.report)."""

import json

import pytest

from selfllm.recursive.report import (
    detect_benchmark_keys,
    format_metrics_report,
    format_metrics_table,
    load_metrics,
    metrics_table_rows,
    summarize_metrics,
)


def _history():
    """A small metrics history with benchmark scores on each iteration."""
    return [
        {
            "iteration": 1, "train_loss": 4.4, "eval_perplexity": 1037.7,
            "quality_score": 0.74, "benchmark_mmlu": 0.25, "benchmark_gsm8k": 0.10,
        },
        {
            "iteration": 2, "train_loss": 3.8, "eval_perplexity": 700.0,
            "quality_score": 0.72, "benchmark_mmlu": 0.30, "benchmark_gsm8k": 0.15,
        },
        {
            "iteration": 3, "train_loss": 3.1, "eval_perplexity": 223.8,
            "quality_score": 0.70, "benchmark_mmlu": 0.40, "benchmark_gsm8k": 0.20,
        },
    ]


def test_detect_benchmark_keys():
    keys = detect_benchmark_keys(_history())
    assert keys == ["benchmark_gsm8k", "benchmark_mmlu"]  # sorted


def test_detect_benchmark_keys_none():
    assert detect_benchmark_keys([{"iteration": 1, "train_loss": 2.0}]) == []


def test_summarize_includes_core_and_benchmarks():
    s = summarize_metrics(_history())
    assert s["eval_perplexity"]["first"] == 1037.7
    assert s["eval_perplexity"]["last"] == 223.8
    assert s["eval_perplexity"]["delta"] == pytest.approx(223.8 - 1037.7)
    # benchmarks improve
    assert s["benchmark_mmlu"]["delta"] == pytest.approx(0.15)
    assert s["benchmark_gsm8k"]["delta"] == pytest.approx(0.10)


def test_table_has_benchmark_columns():
    table = format_metrics_table(_history())
    # headers include benchmark names without the prefix
    header = table.splitlines()[0]
    assert "mmlu" in header and "gsm8k" in header
    assert "perplexity" in header and "quality" in header
    # one row per iteration + header + separator
    assert len(table.splitlines()) == 2 + 3


def test_report_mentions_benchmarks():
    report = format_metrics_report(_history())
    assert "Benchmarks (first -> last)" in report
    assert "mmlu:" in report and "gsm8k:" in report
    assert "Perplexity:" in report


def test_table_rows_for_dataframe():
    rows = metrics_table_rows(_history())
    headers = rows[0]
    assert headers[:4] == ["iter", "train_loss", "perplexity", "quality"]
    assert "mmlu" in headers and "gsm8k" in headers
    assert len(rows) == 1 + 3  # header + 3 iterations


def test_empty_history():
    assert format_metrics_table([]) == "_(no metrics recorded)_"
    assert format_metrics_report([]) == "_(no metrics recorded)_"
    assert summarize_metrics([]) == {}


def test_missing_benchmark_on_some_iterations():
    history = [
        {"iteration": 1, "train_loss": 4.0, "eval_perplexity": 900.0,
         "quality_score": 0.7},  # no benchmark
        {"iteration": 2, "train_loss": 3.0, "eval_perplexity": 400.0,
         "quality_score": 0.7, "benchmark_mmlu": 0.5},  # benchmark only here
    ]
    keys = detect_benchmark_keys(history)
    assert keys == ["benchmark_mmlu"]
    # table renders a placeholder for the missing cell, no crash
    table = format_metrics_table(history)
    assert " - " in table or "| - " in table
    s = summarize_metrics(history)
    assert s["benchmark_mmlu"]["first"] == 0.5  # first recorded value


def test_load_metrics(tmp_path):
    p = tmp_path / "metrics_history.json"
    p.write_text(json.dumps(_history()), encoding="utf-8")
    loaded = load_metrics(str(p))
    assert len(loaded) == 3
    assert loaded[0]["benchmark_mmlu"] == 0.25


def test_load_metrics_rejects_non_list(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_metrics(str(p))
