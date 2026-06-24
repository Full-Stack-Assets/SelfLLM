"""Tests for the real model training pipeline."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.real_training import (
    get_full_config,
    get_medium_config,
    get_small_config,
    train_real_model,
)
from selfllm.utils import count_parameters, format_num


# --------------------------------------------------------------------------- #
# Config tests
# --------------------------------------------------------------------------- #


class TestModelConfigs:
    """Verify each scale produces models with the expected parameter counts."""

    def test_small_config_parameter_count(self):
        """Small config should produce ~5M parameters."""
        config = get_small_config()
        model = SelfImprovingLLM(config)
        params = count_parameters(model)

        assert 4_000_000 <= params <= 7_000_000, (
            f"Small model has {format_num(params)} parameters, "
            f"expected ~5M (4M-7M range)"
        )
        # Verify config values
        assert config.d_model == 256
        assert config.n_layers == 6
        assert config.n_heads == 8
        assert config.vocab_size == 8000
        assert config.max_seq_len == 512

    def test_medium_config_parameter_count(self):
        """Medium config should produce ~50M parameters."""
        config = get_medium_config()
        model = SelfImprovingLLM(config)
        params = count_parameters(model)

        assert 40_000_000 <= params <= 70_000_000, (
            f"Medium model has {format_num(params)} parameters, "
            f"expected ~50M (40M-70M range)"
        )
        assert config.d_model == 512
        assert config.n_layers == 12
        assert config.n_heads == 8
        assert config.vocab_size == 10000
        assert config.max_seq_len == 1024
        assert config.grad_checkpoint is True

    def test_full_config_parameter_count(self):
        """Full config should produce ~350M parameters."""
        config = get_full_config()
        model = SelfImprovingLLM(config)
        params = count_parameters(model)

        assert 300_000_000 <= params <= 400_000_000, (
            f"Full model has {format_num(params)} parameters, "
            f"expected ~350M (300M-400M range)"
        )
        assert config.d_model == 1024
        assert config.n_layers == 24
        assert config.n_heads == 16
        assert config.vocab_size == 32000
        assert config.max_seq_len == 2048
        assert config.grad_checkpoint is True

    def test_config_consistency(self):
        """All configs should use consistent settings."""
        for name, fn in [
            ("small", get_small_config),
            ("medium", get_medium_config),
            ("full", get_full_config),
        ]:
            config = fn()
            assert config.use_rope is True, f"{name}: RoPE should be enabled"
            assert config.use_swiglu is True, f"{name}: SwiGLU should be enabled"
            assert config.dropout == 0.1, f"{name}: dropout should be 0.1"
            # d_ff should be 4x d_model
            assert config.d_ff == config.d_model * 4, (
                f"{name}: d_ff ({config.d_ff}) should be 4x d_model ({config.d_model})"
            )


# --------------------------------------------------------------------------- #
# Pipeline integration tests
# --------------------------------------------------------------------------- #


class TestPipelineIntegration:
    """Test the data pipeline, tokenizer training, and dataset creation."""

    def test_data_pipeline_creation(self):
        """DataPipeline can be created with and without a tokenizer."""
        from selfllm.data.pipeline import DataPipeline
        from selfllm.model.tokenizer import BPETokenizer

        # Without tokenizer
        pipeline = DataPipeline(tokenizer=None, max_seq_len=512)
        assert pipeline.tokenizer is None
        assert pipeline.max_seq_len == 512

        # With tokenizer
        tokenizer = BPETokenizer(vocab_size=1000)
        pipeline = DataPipeline(tokenizer=tokenizer, max_seq_len=256)
        assert pipeline.tokenizer is tokenizer
        assert pipeline.max_seq_len == 256

    def test_preprocess_text(self):
        """Text preprocessing removes Gutenberg headers and normalizes."""
        from selfllm.data.pipeline import DataPipeline

        raw_text = """
*** START OF THIS PROJECT GUTENBERG EBOOK TEST ***
This is the actual content.
It spans multiple lines.


With extra whitespace.
*** END OF THIS PROJECT GUTENBERG EBOOK TEST ***
"""
        cleaned = DataPipeline.preprocess_text(raw_text)
        assert "START OF" not in cleaned
        assert "END OF" not in cleaned
        assert "actual content" in cleaned
        # Should not have excessive newlines
        assert "\n\n\n" not in cleaned

    def test_chunk_text(self):
        """Text is split into overlapping chunks."""
        from selfllm.data.pipeline import DataPipeline
        from selfllm.model.tokenizer import BPETokenizer

        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(["The quick brown fox jumps over the lazy dog. " * 50])

        pipeline = DataPipeline(tokenizer=tokenizer, max_seq_len=32)
        text = "The quick brown fox jumps over the lazy dog. " * 50
        chunks = pipeline.chunk_text(text, overlap=8)

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_discover_gutenberg_ids_extends_curated_list(self, monkeypatch):
        """Discovery fills requests larger than the curated seed list."""
        from selfllm.data.pipeline import DataPipeline

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'<a href="/ebooks/3">A</a><a href="/ebooks/4">B</a>'

        monkeypatch.setattr(DataPipeline, "GUTENBERG_POPULAR", [1, 2])
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *args, **kwargs: FakeResponse()
        )

        assert DataPipeline._discover_gutenberg_book_ids(4) == [1, 2, 3, 4]

    def test_download_uses_discovered_ids_beyond_curated_list(self, monkeypatch, tmp_path):
        """download_gutenberg_books is no longer capped by GUTENBERG_POPULAR."""
        from selfllm.data.pipeline import DataPipeline

        monkeypatch.setattr(
            DataPipeline,
            "_discover_gutenberg_book_ids",
            classmethod(lambda cls, target_count: [1, 2, 3, 4][:target_count]),
        )

        def fake_download(cls, book_id, path):
            Path(path).write_text(f"book {book_id}", encoding="utf-8")
            return True

        monkeypatch.setattr(DataPipeline, "_download_book", classmethod(fake_download))

        paths = DataPipeline(tokenizer=None).download_gutenberg_books(
            output_dir=str(tmp_path), num_books=4
        )

        assert [Path(path).stem for path in paths] == ["1", "2", "3", "4"]

    def test_train_tokenizer_uses_full_corpus_when_sample_size_is_none(
        self, monkeypatch, tmp_path
    ):
        """Tokenizer training can consume every downloaded text file."""
        from selfllm.data.pipeline import DataPipeline
        from selfllm.model.tokenizer import BPETokenizer

        (tmp_path / "a.txt").write_text("alpha " * 20, encoding="utf-8")
        (tmp_path / "b.txt").write_text("beta " * 20, encoding="utf-8")
        seen_texts = []

        def fake_train(self, texts):
            seen_texts.extend(texts)

        monkeypatch.setattr(BPETokenizer, "train", fake_train)

        DataPipeline(tokenizer=None).train_tokenizer_on_corpus(
            corpus_dir=str(tmp_path), vocab_size=128, sample_size=None
        )

        assert len(seen_texts) == 2
        assert any("alpha" in text for text in seen_texts)
        assert any("beta" in text for text in seen_texts)


# --------------------------------------------------------------------------- #
# Training pipeline test (mocked)
# --------------------------------------------------------------------------- #


class TestTrainingPipeline:
    """Test the full training pipeline with mocked components."""

    @pytest.fixture
    def temp_dir(self):
        """Provide a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_train_small_mock(self, temp_dir):
        """Small-scale training runs without errors on tiny data."""
        data_dir = os.path.join(temp_dir, "data")
        output_dir = os.path.join(temp_dir, "output")
        books_dir = os.path.join(data_dir, "books")
        os.makedirs(books_dir, exist_ok=True)

        # Write a few small text files as "books"
        for i in range(3):
            with open(os.path.join(books_dir, f"book_{i}.txt"), "w") as f:
                f.write("The quick brown fox jumps. " * 20)

        # Patch download, trainer, and recursive self-trainer to avoid
        # heavy compute / network calls.
        with patch(
            "selfllm.data.pipeline.DataPipeline.download_gutenberg_books",
            return_value=[os.path.join(books_dir, f"book_{i}.txt") for i in range(3)],
        ):
            with patch(
                "selfllm.training.trainer.LLMTrainer.train",
                return_value={"train_loss": [2.5]},
            ):
                with patch(
                    "selfllm.recursive.recursive_trainer.RecursiveSelfTrainer.run",
                    return_value=[
                        {
                            "iteration": 1,
                            "samples_generated": 10,
                            "samples_kept": 5,
                            "train_loss": 2.5,
                            "quality_score": 0.6,
                        }
                    ],
                ):
                    result = train_real_model(
                        scale="small",
                        data_dir=data_dir,
                        output_dir=output_dir,
                        num_books=3,
                        pretrain_epochs=1,
                        pretrain_batch_size=2,
                        pretrain_lr=1e-3,
                        self_improve_iterations=1,
                        use_lora=True,
                        lora_rank=4,
                        use_dpo=False,
                        device="cpu",
                        seed=42,
                    )

        assert result["output_dir"] == output_dir
        assert result["total_parameters"] > 0
        assert result["num_books"] == 3
        assert result["dataset_size"] > 0
        assert result["self_improve_iterations"] == 1
        assert os.path.exists(os.path.join(output_dir, "config.json"))
        assert os.path.exists(os.path.join(output_dir, "tokenizer.json"))
        assert os.path.exists(os.path.join(output_dir, "pretrained"))
        assert os.path.exists(os.path.join(output_dir, "final"))

        with open(os.path.join(output_dir, "config.json"), "r") as f:
            saved_config = json.load(f)
        assert saved_config["d_model"] == 256
        assert saved_config["n_layers"] == 6

    def test_train_no_self_improve(self, temp_dir):
        """Training works with self-improvement disabled."""
        data_dir = os.path.join(temp_dir, "data")
        output_dir = os.path.join(temp_dir, "output_no_improve")
        books_dir = os.path.join(data_dir, "books")
        os.makedirs(books_dir, exist_ok=True)

        for i in range(2):
            with open(os.path.join(books_dir, f"book_{i}.txt"), "w") as f:
                f.write("Hello world. " * 20)

        with patch(
            "selfllm.data.pipeline.DataPipeline.download_gutenberg_books",
            return_value=[os.path.join(books_dir, f"book_{i}.txt") for i in range(2)],
        ):
            with patch(
                "selfllm.training.trainer.LLMTrainer.train",
                return_value={"train_loss": [2.3]},
            ):
                result = train_real_model(
                    scale="small",
                    data_dir=data_dir,
                    output_dir=output_dir,
                    num_books=2,
                    pretrain_epochs=1,
                    pretrain_batch_size=2,
                    self_improve_iterations=0,
                    use_lora=False,
                    use_dpo=False,
                    device="cpu",
                    seed=42,
                )

        assert result["self_improve_iterations"] == 0
        assert os.path.exists(os.path.join(output_dir, "final"))


# --------------------------------------------------------------------------- #
# LoRA integration
# --------------------------------------------------------------------------- #


class TestLoRAIntegration:
    """Test LoRA injection works correctly with the pipeline."""

    def test_lora_injection_on_small_model(self):
        """LoRA can be injected into the small model."""
        from selfllm.model.lora import (
            count_lora_parameters,
            inject_lora,
        )

        config = get_small_config()
        model = SelfImprovingLLM(config)
        base_params = count_parameters(model)

        # Inject LoRA
        model = inject_lora(model, rank=4, alpha=8)
        lora_info = count_lora_parameters(model)

        assert lora_info["lora"] > 0, "LoRA parameters should be present"
        assert lora_info["lora_pct"] < 10.0, (
            f"LoRA should be <10% of total params, got {lora_info['lora_pct']:.2f}%"
        )
        # Base model should be frozen
        assert lora_info["frozen"] == base_params, (
            "All base parameters should be frozen after LoRA injection"
        )

    def test_lora_trainable_params(self):
        """Only LoRA parameters are trainable after injection."""
        from selfllm.model.lora import get_lora_parameters, inject_lora

        config = get_small_config()
        model = SelfImprovingLLM(config)
        model = inject_lora(model, rank=4, alpha=8)

        lora_params = get_lora_parameters(model)
        assert len(lora_params) > 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert "lora_" in name, (
                    f"Non-LoRA parameter '{name}' is trainable after LoRA injection"
                )


# --------------------------------------------------------------------------- #
# CLI argument tests
# --------------------------------------------------------------------------- #


class TestCLIArguments:
    """Test CLI argument parsing."""

    def test_default_args(self):
        """Default arguments should be sensible."""
        from selfllm.real_training import build_parser

        parser = build_parser()
        args = parser.parse_args([])

        assert args.scale == "small"
        assert args.num_books == 100
        assert args.tokenizer_sample_size == 0
        assert args.max_chunks == 0
        assert args.pretrain_epochs == 3
        assert args.pretrain_batch_size == 8
        assert args.pretrain_lr == 1e-3
        assert args.self_improve_iterations == 5
        assert args.use_lora is True
        assert args.lora_rank == 8
        assert args.use_dpo is True
        assert args.seed == 42

    def test_full_scale_args(self):
        """Full scale arguments should be parsed correctly."""
        from selfllm.real_training import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--scale", "full",
            "--num-books", "1000",
            "--tokenizer-sample-size", "5000000",
            "--max-chunks", "250000",
            "--pretrain-epochs", "5",
            "--self-improve-iterations", "10",
            "--lora-rank", "16",
            "--no-lora",
            "--no-dpo",
            "--seed", "123",
        ])

        assert args.scale == "full"
        assert args.num_books == 1000
        assert args.tokenizer_sample_size == 5000000
        assert args.max_chunks == 250000
        assert args.pretrain_epochs == 5
        assert args.self_improve_iterations == 10
        assert args.lora_rank == 16
        assert args.use_lora is False
        assert args.use_dpo is False
        assert args.seed == 123

    def test_medium_scale_args(self):
        """Medium scale with custom device."""
        from selfllm.real_training import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--scale", "medium",
            "--device", "cpu",
            "--pretrain-batch-size", "4",
        ])

        assert args.scale == "medium"
        assert args.device == "cpu"
        assert args.pretrain_batch_size == 4
