"""Tests for Vertex AI Gemini tuning preparation helpers."""

from __future__ import annotations

import json

import pytest

from selfllm.cloud.vertex_tuning import (
    build_vertex_tuning_request,
    export_vertex_sft_dataset,
    load_prompt_response_samples,
)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_load_prompt_response_samples_from_selfllm_json(tmp_path):
    path = tmp_path / "samples.json"
    path.write_text(
        json.dumps({"samples": [{"prompt": "Question?", "response": "Answer."}]}),
        encoding="utf-8",
    )

    samples = load_prompt_response_samples(str(path))

    assert samples == [{"prompt": "Question?", "response": "Answer."}]


def test_export_vertex_sft_dataset_writes_gemini_jsonl(tmp_path):
    src = tmp_path / "samples.jsonl"
    src.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "Summarize A", "response": "A summary"}),
                json.dumps({"prompt": "", "response": "missing prompt"}),
                json.dumps({"prompt": "Summarize B", "response": ""}),
                json.dumps({"prompt": "Summarize C", "response": "C summary"}),
            ]
        ),
        encoding="utf-8",
    )
    train_path = tmp_path / "train.jsonl"

    manifest = export_vertex_sft_dataset(
        str(src),
        str(train_path),
        system_instruction="Answer in one sentence.",
    )

    rows = _read_jsonl(train_path)
    assert manifest["examples_read"] == 4
    assert manifest["examples_exported"] == 2
    assert manifest["skipped_empty_prompt"] == 1
    assert manifest["skipped_empty_response"] == 1
    assert rows[0]["contents"][0]["role"] == "user"
    assert rows[0]["contents"][1]["role"] == "model"
    assert rows[0]["contents"][0]["parts"][0]["text"].startswith(
        "Answer in one sentence."
    )


def test_export_vertex_sft_dataset_can_split_validation(tmp_path):
    src = tmp_path / "samples.json"
    samples = [
        {"prompt": f"Prompt {i}", "response": f"Response {i}"}
        for i in range(10)
    ]
    src.write_text(json.dumps(samples), encoding="utf-8")
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"

    manifest = export_vertex_sft_dataset(
        str(src),
        str(train_path),
        validation_output_path=str(val_path),
        validation_ratio=0.2,
        seed=1,
    )

    assert manifest["train_examples"] == 8
    assert manifest["validation_examples"] == 2
    assert len(_read_jsonl(train_path)) == 8
    assert len(_read_jsonl(val_path)) == 2


def test_export_vertex_sft_dataset_rejects_empty_export(tmp_path):
    src = tmp_path / "samples.json"
    src.write_text(json.dumps([{"prompt": "", "response": "Only response"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="No valid"):
        export_vertex_sft_dataset(str(src), str(tmp_path / "train.jsonl"))


def test_build_vertex_tuning_request_full_payload():
    plan = build_vertex_tuning_request(
        project_id="my-project",
        location="us-central1",
        base_model="gemini-2.5-flash",
        training_dataset_uri="gs://bucket/train.jsonl",
        validation_dataset_uri="gs://bucket/val.jsonl",
        tuned_model_display_name="selfllm-gemini-adapter",
        adapter_size=4,
        epoch_count=3,
        learning_rate_multiplier=1.5,
    )

    assert plan["endpoint"] == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/"
        "my-project/locations/us-central1/tuningJobs"
    )
    request = plan["request"]
    assert request["baseModel"] == "gemini-2.5-flash"
    assert request["tunedModelDisplayName"] == "selfllm-gemini-adapter"
    spec = request["supervisedTuningSpec"]
    assert spec["trainingDatasetUri"] == "gs://bucket/train.jsonl"
    assert spec["validationDatasetUri"] == "gs://bucket/val.jsonl"
    assert spec["hyperParameters"]["adapterSize"] == "ADAPTER_SIZE_FOUR"
    assert spec["hyperParameters"]["epochCount"] == 3
    assert spec["hyperParameters"]["learningRateMultiplier"] == 1.5


def test_build_vertex_tuning_request_requires_gcs_uri():
    with pytest.raises(ValueError, match="gs://"):
        build_vertex_tuning_request(
            project_id="my-project",
            location="us-central1",
            base_model="gemini-2.5-flash",
            training_dataset_uri="/tmp/train.jsonl",
        )
