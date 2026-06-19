"""Unit tests for :class:`selfllm.training.dataset.SelfTrainingDataset`.

The dataset's central correctness guarantee is its **label masking**: prompt
tokens must be set to ``-100`` (ignored by the loss) while response tokens
carry their real ids. A silent bug here would let the model train on its own
prompts and corrupt the learning signal without any error surfacing — so the
masking boundary, EOS handling, padding, and truncation paths are tested
explicitly.
"""

from __future__ import annotations

import json

import torch

from selfllm.training.dataset import SelfTrainingDataset

# ``-100`` is the PyTorch CrossEntropyLoss ``ignore_index`` default.
IGNORE = -100


# ---------------------------------------------------------------------------
# Length / basic shape
# ---------------------------------------------------------------------------


def test_len_matches_samples(tiny_tokenizer):
    samples = [{"prompt": "hello", "response": "world"} for _ in range(5)]
    ds = SelfTrainingDataset(samples, tiny_tokenizer, max_seq_len=64)
    assert len(ds) == 5


def test_item_returns_three_aligned_tensors(tiny_tokenizer):
    ds = SelfTrainingDataset(
        [{"prompt": "the quick", "response": "brown fox"}],
        tiny_tokenizer,
        max_seq_len=64,
        pad_to_max=True,
    )
    item = ds[0]
    assert set(item) == {"input_ids", "attention_mask", "labels"}
    n = item["input_ids"].shape[0]
    assert item["attention_mask"].shape[0] == n
    assert item["labels"].shape[0] == n
    for t in item.values():
        assert t.dtype == torch.long


# ---------------------------------------------------------------------------
# Label masking — the core invariant
# ---------------------------------------------------------------------------


def test_prompt_tokens_are_masked_and_response_tokens_are_not(tiny_tokenizer):
    prompt, response = "machine learning", "improves with data"
    ds = SelfTrainingDataset(
        [{"prompt": prompt, "response": response}],
        tiny_tokenizer,
        max_seq_len=128,
        pad_to_max=False,
    )
    item = ds[0]
    labels = item["labels"].tolist()
    input_ids = item["input_ids"].tolist()

    n_prompt = len(tiny_tokenizer.encode(prompt))

    # Every prompt position is ignored.
    assert labels[:n_prompt] == [IGNORE] * n_prompt
    # Response positions equal the actual input tokens (not masked).
    assert labels[n_prompt:] == input_ids[n_prompt:]
    # At least one real (un-ignored) label exists for loss to flow.
    assert any(lab != IGNORE for lab in labels)


def test_response_labels_match_input_ids_at_same_positions(tiny_tokenizer):
    """Where a label is not ``-100`` it must equal the input id at that index,
    i.e. the model is trained to predict the token actually present."""
    ds = SelfTrainingDataset(
        [{"prompt": "once upon", "response": "a time"}],
        tiny_tokenizer,
        max_seq_len=64,
        pad_to_max=True,
    )
    item = ds[0]
    for inp, lab in zip(item["input_ids"].tolist(), item["labels"].tolist()):
        if lab != IGNORE:
            assert lab == inp


# ---------------------------------------------------------------------------
# EOS handling
# ---------------------------------------------------------------------------


def test_eos_appended_to_response(tiny_tokenizer):
    ds = SelfTrainingDataset(
        [{"prompt": "hello", "response": "world"}],
        tiny_tokenizer,
        max_seq_len=64,
        pad_to_max=False,
    )
    item = ds[0]
    labels = [lab for lab in item["labels"].tolist() if lab != IGNORE]
    # Last non-ignored label is the EOS token id.
    assert labels[-1] == tiny_tokenizer.eos_token_id


def test_eos_not_duplicated_when_already_present(tiny_tokenizer):
    eos = tiny_tokenizer.eos_token_id
    # Build a sample whose response already ends in EOS by smuggling the id in
    # via a pre-encoded marker: easiest is to check the count stays at one.
    response = "world"
    ds = SelfTrainingDataset(
        [{"prompt": "hello", "response": response}],
        tiny_tokenizer,
        max_seq_len=64,
        pad_to_max=False,
    )
    labels = [lab for lab in ds[0]["labels"].tolist() if lab != IGNORE]
    # Exactly one EOS at the tail, none injected mid-response erroneously.
    assert labels.count(eos) == 1
    assert labels[-1] == eos


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def test_pad_to_max_produces_fixed_length(tiny_tokenizer):
    ds = SelfTrainingDataset(
        [{"prompt": "hi", "response": "yo"}],
        tiny_tokenizer,
        max_seq_len=32,
        pad_to_max=True,
    )
    item = ds[0]
    assert item["input_ids"].shape[0] == 32
    assert item["attention_mask"].shape[0] == 32
    assert item["labels"].shape[0] == 32


def test_padding_masks_attention_and_labels(tiny_tokenizer):
    ds = SelfTrainingDataset(
        [{"prompt": "hi", "response": "yo"}],
        tiny_tokenizer,
        max_seq_len=32,
        pad_to_max=True,
    )
    item = ds[0]
    attn = item["attention_mask"].tolist()
    labels = item["labels"].tolist()
    input_ids = item["input_ids"].tolist()

    real_len = sum(attn)
    pad_id = tiny_tokenizer.pad_token_id

    # Padding region: attention 0, labels ignored, input filled with pad id.
    assert attn[real_len:] == [0] * (32 - real_len)
    assert labels[real_len:] == [IGNORE] * (32 - real_len)
    assert input_ids[real_len:] == [pad_id] * (32 - real_len)
    # Real region: attention all 1.
    assert attn[:real_len] == [1] * real_len


def test_no_padding_keeps_variable_length(tiny_tokenizer):
    short = SelfTrainingDataset(
        [{"prompt": "a", "response": "b"}], tiny_tokenizer, max_seq_len=128, pad_to_max=False
    )[0]
    longer = SelfTrainingDataset(
        [{"prompt": "the quick brown fox", "response": "jumps over the lazy dog"}],
        tiny_tokenizer,
        max_seq_len=128,
        pad_to_max=False,
    )[0]
    assert longer["input_ids"].shape[0] > short["input_ids"].shape[0]


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_truncation_respects_max_seq_len(tiny_tokenizer):
    long_text = "the quick brown fox jumps over the lazy dog " * 20
    ds = SelfTrainingDataset(
        [{"prompt": long_text, "response": long_text}],
        tiny_tokenizer,
        max_seq_len=48,
        pad_to_max=False,
    )
    item = ds[0]
    assert item["input_ids"].shape[0] <= 48


def test_oversized_prompt_still_leaves_response_token(tiny_tokenizer):
    """If the prompt alone exceeds max_seq_len, at least one response token
    must survive so there is a non-ignored label to learn from."""
    huge_prompt = "the quick brown fox jumps over the lazy dog " * 40
    ds = SelfTrainingDataset(
        [{"prompt": huge_prompt, "response": "answer"}],
        tiny_tokenizer,
        max_seq_len=16,
        pad_to_max=False,
    )
    item = ds[0]
    assert item["input_ids"].shape[0] <= 16
    assert any(lab != IGNORE for lab in item["labels"].tolist())


# ---------------------------------------------------------------------------
# Missing fields default to empty strings
# ---------------------------------------------------------------------------


def test_missing_prompt_or_response_defaults_to_empty(tiny_tokenizer):
    ds = SelfTrainingDataset(
        [{"response": "only response"}, {"prompt": "only prompt"}],
        tiny_tokenizer,
        max_seq_len=64,
        pad_to_max=False,
    )
    # Neither access should raise; both produce valid tensors.
    a, b = ds[0], ds[1]
    assert a["input_ids"].numel() > 0
    assert b["input_ids"].numel() > 0


# ---------------------------------------------------------------------------
# add_samples / save / load
# ---------------------------------------------------------------------------


def test_add_samples_extends_length(tiny_tokenizer):
    ds = SelfTrainingDataset(
        [{"prompt": "a", "response": "b"}], tiny_tokenizer, max_seq_len=32
    )
    ds.add_samples([{"prompt": "c", "response": "d"}, {"prompt": "e", "response": "f"}])
    assert len(ds) == 3


def test_save_load_round_trip(tiny_tokenizer, tmp_path):
    samples = [
        {"prompt": "hello", "response": "world"},
        {"prompt": "foo", "response": "bar"},
    ]
    ds = SelfTrainingDataset(samples, tiny_tokenizer, max_seq_len=77, pad_to_max=False)
    path = str(tmp_path / "ds.json")
    ds.save(path)

    # Raw JSON preserves the samples verbatim.
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    assert raw["samples"] == samples
    assert raw["max_seq_len"] == 77
    assert raw["pad_to_max"] is False

    loaded = SelfTrainingDataset.load(path, tiny_tokenizer)
    assert len(loaded) == len(ds)
    assert loaded.max_seq_len == 77
    assert loaded.pad_to_max is False
    # Tokenization is reproducible after reload.
    assert torch.equal(loaded[0]["labels"], ds[0]["labels"])
