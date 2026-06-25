"""Utilities for preparing Vertex AI Gemini supervised tuning jobs.

The helpers in this module are intentionally dependency-free. They create
Gemini-compatible supervised tuning JSONL and REST request payloads locally;
uploading to GCS and submitting the job are separate Google Cloud operations.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

VERTEX_ADAPTER_SIZES = {1, 4, 8, 16}


def _clean_text(value: Any) -> str:
    """Normalize user/model text for JSONL export."""
    return str(value or "").strip()


def load_prompt_response_samples(path: str) -> List[Dict[str, Any]]:
    """Load prompt/response samples from SelfLLM JSON or JSONL.

    Supported inputs:
    - ``{"samples": [{"prompt": "...", "response": "..."}]}``
    - ``[{"prompt": "...", "response": "..."}]``
    - JSONL with one prompt/response object per line
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")

    if src.suffix.lower() == ".jsonl":
        samples: List[Dict[str, Any]] = []
        with open(src, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"JSONL line {line_no} is not an object")
                samples.append(item)
        return samples

    with open(src, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        return list(data["samples"])
    if isinstance(data, list):
        return list(data)
    raise ValueError("Expected a JSON list or object with a 'samples' list")


def _to_vertex_record(
    sample: Dict[str, Any],
    *,
    system_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert one prompt/response sample into Gemini SFT JSONL format."""
    prompt = _clean_text(sample.get("prompt"))
    response = _clean_text(sample.get("response"))
    if system_instruction:
        prompt = f"{system_instruction.strip()}\n\n{prompt}".strip()

    return {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]},
            {"role": "model", "parts": [{"text": response}]},
        ]
    }


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    """Write rows to JSONL and return the row count."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(dest, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _split_records(
    records: List[Dict[str, Any]],
    *,
    validation_ratio: float,
    validation_output_path: Optional[str],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records into train/validation sets deterministically."""
    if validation_ratio < 0 or validation_ratio >= 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    if not validation_output_path or validation_ratio == 0:
        return records, []

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * validation_ratio))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def export_vertex_sft_dataset(
    input_path: str,
    train_output_path: str,
    *,
    validation_output_path: Optional[str] = None,
    validation_ratio: float = 0.0,
    max_examples: Optional[int] = None,
    min_prompt_chars: int = 1,
    min_response_chars: int = 1,
    system_instruction: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Export prompt/response samples to Vertex AI Gemini SFT JSONL."""
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")
    if min_prompt_chars < 0 or min_response_chars < 0:
        raise ValueError("minimum character lengths must be non-negative")

    samples = load_prompt_response_samples(input_path)
    records: List[Dict[str, Any]] = []
    skipped_empty_prompt = 0
    skipped_empty_response = 0

    for sample in samples:
        prompt = _clean_text(sample.get("prompt"))
        response = _clean_text(sample.get("response"))
        if len(prompt) < min_prompt_chars:
            skipped_empty_prompt += 1
            continue
        if len(response) < min_response_chars:
            skipped_empty_response += 1
            continue
        records.append(
            _to_vertex_record(sample, system_instruction=system_instruction)
        )
        if max_examples is not None and len(records) >= max_examples:
            break

    if not records:
        raise ValueError("No valid prompt/response samples were exportable")

    train_records, validation_records = _split_records(
        records,
        validation_ratio=validation_ratio,
        validation_output_path=validation_output_path,
        seed=seed,
    )
    if not train_records:
        raise ValueError("Training split is empty")

    train_count = _write_jsonl(train_output_path, train_records)
    validation_count = 0
    if validation_output_path and validation_records:
        validation_count = _write_jsonl(validation_output_path, validation_records)

    return {
        "input_path": input_path,
        "train_output_path": train_output_path,
        "validation_output_path": validation_output_path if validation_count else None,
        "examples_read": len(samples),
        "examples_exported": len(records),
        "train_examples": train_count,
        "validation_examples": validation_count,
        "skipped_empty_prompt": skipped_empty_prompt,
        "skipped_empty_response": skipped_empty_response,
    }


def _adapter_size_name(adapter_size: int) -> str:
    """Convert an integer adapter size to the Vertex REST enum value."""
    if adapter_size not in VERTEX_ADAPTER_SIZES:
        raise ValueError(
            f"adapter_size must be one of {sorted(VERTEX_ADAPTER_SIZES)}"
        )
    names = {
        1: "ADAPTER_SIZE_ONE",
        4: "ADAPTER_SIZE_FOUR",
        8: "ADAPTER_SIZE_EIGHT",
        16: "ADAPTER_SIZE_SIXTEEN",
    }
    return names[adapter_size]


def build_vertex_tuning_request(
    *,
    project_id: str,
    location: str,
    base_model: str,
    training_dataset_uri: str,
    tuned_model_display_name: Optional[str] = None,
    validation_dataset_uri: Optional[str] = None,
    adapter_size: Optional[int] = None,
    epoch_count: Optional[int] = None,
    learning_rate_multiplier: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a Vertex AI tuningJobs.create endpoint and request body."""
    if not project_id:
        raise ValueError("project_id is required")
    if not location:
        raise ValueError("location is required")
    if not base_model:
        raise ValueError("base_model is required")
    if not training_dataset_uri.startswith("gs://"):
        raise ValueError("training_dataset_uri must be a gs:// URI")
    if validation_dataset_uri and not validation_dataset_uri.startswith("gs://"):
        raise ValueError("validation_dataset_uri must be a gs:// URI")
    if epoch_count is not None and epoch_count <= 0:
        raise ValueError("epoch_count must be positive")
    if learning_rate_multiplier is not None and learning_rate_multiplier <= 0:
        raise ValueError("learning_rate_multiplier must be positive")

    supervised_spec: Dict[str, Any] = {
        "trainingDatasetUri": training_dataset_uri,
    }
    if validation_dataset_uri:
        supervised_spec["validationDatasetUri"] = validation_dataset_uri

    hyper_parameters: Dict[str, Any] = {}
    if epoch_count is not None:
        hyper_parameters["epochCount"] = epoch_count
    if adapter_size is not None:
        hyper_parameters["adapterSize"] = _adapter_size_name(adapter_size)
    if learning_rate_multiplier is not None:
        hyper_parameters["learningRateMultiplier"] = learning_rate_multiplier
    if hyper_parameters:
        supervised_spec["hyperParameters"] = hyper_parameters

    body: Dict[str, Any] = {
        "baseModel": base_model,
        "supervisedTuningSpec": supervised_spec,
    }
    if tuned_model_display_name:
        body["tunedModelDisplayName"] = tuned_model_display_name

    endpoint = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/"
        f"{project_id}/locations/{location}/tuningJobs"
    )
    return {"endpoint": endpoint, "request": body}
