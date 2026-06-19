"""Tests for the `selfllm benchmark` CLI command (cmd_benchmark)."""

import json

import torch

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.train import cmd_benchmark


class _Logger:
    def info(self, *a):
        pass

    def warning(self, *a):
        pass

    def error(self, *a):  # surface misconfiguration clearly
        raise AssertionError(f"logger.error: {a}")


def _setup(tmp_path):
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                      d_ff=128, max_seq_len=64, dropout=0.0)
    model = SelfImprovingLLM(cfg)
    model_dir = tmp_path / "model"
    model.save_pretrained(str(model_dir))

    tok = BPETokenizer(vocab_size=256)
    tok.train(["the answer is 4", "Paris is the capital", "step by step reasoning"])
    tok_path = tmp_path / "tok.json"
    tok.save(str(tok_path))

    gsm8k = tmp_path / "gsm8k.jsonl"
    gsm8k.write_text(
        json.dumps({"question": "2+3?", "answer": "#### 5"}) + "\n"
        + json.dumps({"question": "1+4?", "answer": "#### 5"}) + "\n",
        encoding="utf-8",
    )
    mmlu = tmp_path / "mmlu.jsonl"
    mmlu.write_text(
        json.dumps({"question": "Capital of France?",
                    "choices": ["Rome", "Paris", "Berlin", "Madrid"],
                    "answer": "B"}) + "\n",
        encoding="utf-8",
    )
    return str(model_dir), str(tok_path), str(gsm8k), str(mmlu)


def _base_cfg(model_dir, tok_path):
    return {
        "model_path": model_dir, "tokenizer_path": tok_path,
        "device": "cpu", "model": {"vocab_size": 256},
    }


def test_benchmark_greedy(tmp_path):
    model_dir, tok_path, gsm8k, mmlu = _setup(tmp_path)
    out = tmp_path / "results.json"
    cfg = _base_cfg(model_dir, tok_path)
    cfg.update({"gsm8k": gsm8k, "mmlu": mmlu, "limit": 1,
                "output_path": str(out)})
    cmd_benchmark(cfg, _Logger())

    report = json.loads(out.read_text())
    assert "greedy" in report
    assert "gsm8k" in report["greedy"] and "mmlu" in report["greedy"]
    assert 0.0 <= report["greedy"]["gsm8k"]["score"] <= 1.0


def test_benchmark_with_reasoning_lift(tmp_path):
    model_dir, tok_path, gsm8k, _ = _setup(tmp_path)
    out = tmp_path / "results.json"
    cfg = _base_cfg(model_dir, tok_path)
    cfg.update({"gsm8k": gsm8k, "limit": 1, "reasoning": "self_consistency",
                "reasoning_samples": 2, "output_path": str(out)})
    cmd_benchmark(cfg, _Logger())

    report = json.loads(out.read_text())
    assert "greedy" in report and "self_consistency" in report
    assert "gsm8k" in report["self_consistency"]


def test_benchmark_requires_a_dataset(tmp_path):
    model_dir, tok_path, _, _ = _setup(tmp_path)
    cfg = _base_cfg(model_dir, tok_path)  # no mmlu/gsm8k/humaneval
    # No benchmark selected -> logger.error -> our stub raises.
    try:
        cmd_benchmark(cfg, _Logger())
        assert False, "expected an error when no dataset is given"
    except AssertionError as e:
        assert "logger.error" in str(e)
