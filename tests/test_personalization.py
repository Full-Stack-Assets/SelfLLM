"""Tests for the personalization pipeline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from selfllm.model.model import SelfImprovingLLM
from selfllm.personalization.frontier_config import (
    get_frontier_config,
    get_frontier_full_config,
    get_frontier_medium_config,
    get_frontier_small_config,
)
from selfllm.personalization.personalize import _prepare_corpus_dir, personalize_model
from selfllm.personalization.profile import UserProfile, load_profile
from selfllm.utils import count_parameters


class TestFrontierConfigs:
    def test_small_frontier_enables_moe(self):
        config = get_frontier_small_config()
        assert config.use_moe is True
        assert config.moe_num_experts == 4

    def test_medium_frontier_enables_moe_and_sliding_window(self):
        config = get_frontier_medium_config()
        assert config.use_moe is True
        assert config.sliding_window == 512
        assert config.attention_sinks == 4

    def test_full_frontier_enables_moe_and_long_context(self):
        config = get_frontier_full_config()
        assert config.use_moe is True
        assert config.sliding_window == 1024

    def test_get_frontier_config_factory(self):
        config = get_frontier_config("small")
        assert config.use_moe is True

    def test_unknown_scale_raises(self):
        with pytest.raises(ValueError, match="Unknown frontier scale"):
            get_frontier_config("tiny")

    def test_frontier_small_model_instantiates(self):
        model = SelfImprovingLLM(get_frontier_small_config())
        params = count_parameters(model)
        assert params > 0


class TestUserProfile:
    def test_load_profile_from_yaml(self, tmp_path):
        profile_file = tmp_path / "profile.yaml"
        profile_file.write_text(
            "profile:\n"
            "  name: Taylor\n"
            "  topics:\n"
            "    - robotics\n"
            "  style: concise\n"
            "  eval_prompts:\n"
            "    - What is your focus?\n"
        )
        profile = load_profile(str(profile_file))
        assert profile.name == "Taylor"
        assert profile.topics == ["robotics"]
        assert profile.eval_prompts == ["What is your focus?"]

    def test_default_eval_prompts_from_topics(self):
        profile = UserProfile(name="Sam", topics=["cooking", "travel"])
        prompts = profile.default_eval_prompts()
        assert len(prompts) >= 2
        assert any("cooking" in p for p in prompts)

    def test_profile_roundtrip_dict(self):
        profile = UserProfile(name="Casey", topics=["math"], style="formal")
        restored = UserProfile.from_dict(profile.to_dict())
        assert restored.name == "Casey"
        assert restored.topics == ["math"]


class TestCorpusPreparation:
    def test_prepare_single_file_corpus(self, tmp_path):
        src = tmp_path / "notes.txt"
        src.write_text("Personal writing sample " * 20)
        work = tmp_path / "work"
        corpus_dir = _prepare_corpus_dir(str(src), str(work))
        assert Path(corpus_dir, "notes.txt").exists()

    def test_prepare_directory_corpus(self, tmp_path):
        src = tmp_path / "corpus"
        src.mkdir()
        (src / "a.txt").write_text("First document " * 30)
        (src / "b.txt").write_text("Second document " * 30)
        work = tmp_path / "work"
        corpus_dir = _prepare_corpus_dir(str(src), str(work))
        assert Path(corpus_dir, "a.txt").exists()
        assert Path(corpus_dir, "b.txt").exists()

    def test_missing_corpus_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _prepare_corpus_dir(str(tmp_path / "missing"), str(tmp_path / "work"))


class TestPersonalizePipeline:
    @pytest.fixture
    def sample_corpus(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        text = (
            "I write about software, machine learning, and practical engineering. "
            "My style is direct and example-driven. "
        ) * 50
        (corpus / "sample.txt").write_text(text)
        return corpus

    @pytest.fixture
    def sample_profile(self, tmp_path):
        profile = tmp_path / "profile.yaml"
        profile.write_text(
            "profile:\n"
            "  name: TestUser\n"
            "  topics:\n"
            "    - software\n"
            "  eval_prompts:\n"
            "    - Describe your approach.\n"
        )
        return profile

    def test_personalize_model_end_to_end_cpu(self, sample_corpus, sample_profile, tmp_path):
        output_dir = tmp_path / "out"
        results = personalize_model(
            corpus_path=str(sample_corpus),
            output_dir=str(output_dir),
            profile_path=str(sample_profile),
            scale="small",
            pretrain_epochs=1,
            pretrain_batch_size=4,
            self_improve_iterations=1,
            use_lora=True,
            use_dpo=True,
            max_chunks=8,
            device="cpu",
            seed=0,
        )

        assert Path(results["final_model_path"]).exists()
        assert Path(results["tokenizer_path"]).exists()
        assert Path(results["manifest_path"]).exists()
        assert results["dataset_size"] > 0
        assert results["self_improve_iterations"] == 1

        with open(results["manifest_path"], "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        assert manifest["profile"]["name"] == "TestUser"
        assert manifest["architecture"]["type"] == "frontier"


class TestPersonalizeCLI:
    def test_personalize_subcommand_parsed(self):
        from selfllm import train as cli

        args = cli.build_parser().parse_args(
            [
                "personalize",
                "--corpus-path",
                "./examples/personal_corpus",
                "--profile",
                "personalize.yaml",
                "--personalize-scale",
                "small",
                "--self-improve-iterations",
                "2",
            ]
        )
        assert args.command == "personalize"
        assert args.corpus_path == "./examples/personal_corpus"
        assert args.profile == "personalize.yaml"
        assert args.personalize_scale == "small"
        assert args.self_improve_iterations == 2

    def test_merge_config_personalization_section(self):
        from argparse import Namespace

        from selfllm import train as cli

        yaml_cfg = {
            "personalization": {"scale": "medium", "use_dpo": False},
            "output_dir": "/tmp/out",
        }
        merged = cli.merge_config(
            yaml_cfg,
            Namespace(
                corpus_path="/data/me",
                profile="profile.yaml",
                personalize_scale=None,
                use_dpo=None,
                output_dir=None,
            ),
        )
        assert merged["corpus_path"] == "/data/me"
        assert merged["profile"] == "profile.yaml"
        assert merged["personalization"]["scale"] == "medium"
        assert merged["output_dir"] == "/tmp/out"
