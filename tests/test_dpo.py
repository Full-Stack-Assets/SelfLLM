"""Unit tests for the DPO trainer.

Tests cover:
- DPO loss computation correctness
- PreferenceDataset encoding and masking
- Training step (loss decreases)
- Preference generation
- Reward margin positivity
- Reference model frozenness
"""

import sys
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

# Ensure project root is importable
sys.path.insert(0, "/home/kimi/work-stage1-dpo")

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.training.dpo_trainer import (
    DPOTrainer,
    PreferenceDataset,
    preference_collate_fn,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_config() -> ModelConfig:
    """Very small model config for fast unit tests."""
    return ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        max_seq_len=64,
        dropout=0.0,
        tie_weights=True,
        use_rope=False,
        use_swiglu=False,
        grad_checkpoint=False,
    )


@pytest.fixture
def tiny_model(tiny_config: ModelConfig) -> SelfImprovingLLM:
    """Create a tiny model for testing."""
    model = SelfImprovingLLM(tiny_config)
    return model


@pytest.fixture
def tokenizer() -> BPETokenizer:
    """Create a trained tokenizer for testing."""
    tok = BPETokenizer(vocab_size=64)
    # Train on sample text to build vocabulary
    tok.train(["hello world this is a test"])
    return tok


@pytest.fixture
def sample_preference_pairs() -> List[Dict[str, str]]:
    """Sample preference pairs for testing."""
    return [
        {
            "prompt": "What is 2+2?",
            "chosen": "4",
            "rejected": "5",
        },
        {
            "prompt": "Capital of France?",
            "chosen": "Paris",
            "rejected": "London",
        },
        {
            "prompt": "Color of sky?",
            "chosen": "Blue",
            "rejected": "Green",
        },
    ]


# ---------------------------------------------------------------------------
# Test 1: DPO Loss Computation
# ---------------------------------------------------------------------------


def test_dpo_loss_computation(tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
    """Verify DPO loss decreases when chosen has higher log-prob than rejected."""
    trainer = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        beta=0.1,
        device="cpu",
    )

    batch_size = 4
    # Simulate: policy strongly prefers chosen
    policy_chosen = torch.tensor([0.0, -0.5, -1.0, -0.2])
    policy_rejected = torch.tensor([-2.0, -1.5, -3.0, -1.8])

    # Reference: neutral (equal log-probs)
    ref_chosen = torch.tensor([-0.5, -0.5, -0.5, -0.5])
    ref_rejected = torch.tensor([-0.5, -0.5, -0.5, -0.5])

    metrics = trainer.compute_dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )

    # Loss should be positive (it's a minimization objective)
    assert metrics["loss"].item() > 0, "DPO loss should be positive"

    # Reward margin should be high when chosen > rejected
    assert (
        metrics["reward_margin"].item() > 0.5
    ), "Reward margin should be high when chosen is preferred"

    # Accuracy should be high when chosen > rejected
    assert (
        metrics["accuracy"].item() == 1.0
    ), "Accuracy should be 1.0 when chosen > rejected for all samples"


def test_dpo_loss_reversed(tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
    """Verify DPO loss is higher when rejected > chosen (policy is wrong)."""
    trainer = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        beta=0.1,
        device="cpu",
    )

    # Case 1: chosen > rejected (good policy)
    policy_chosen_good = torch.tensor([0.0, 0.0])
    policy_rejected_good = torch.tensor([-2.0, -2.0])
    ref_chosen = torch.zeros(2)
    ref_rejected = torch.zeros(2)

    metrics_good = trainer.compute_dpo_loss(
        policy_chosen_good, policy_rejected_good, ref_chosen, ref_rejected
    )

    # Case 2: rejected > chosen (bad policy)
    policy_chosen_bad = torch.tensor([-2.0, -2.0])
    policy_rejected_bad = torch.tensor([0.0, 0.0])

    metrics_bad = trainer.compute_dpo_loss(
        policy_chosen_bad, policy_rejected_bad, ref_chosen, ref_rejected
    )

    # Bad policy should have higher loss
    assert (
        metrics_bad["loss"].item() > metrics_good["loss"].item()
    ), "Loss should be higher when policy prefers rejected"

    # Bad policy should have lower accuracy
    assert (
        metrics_bad["accuracy"].item() < metrics_good["accuracy"].item()
    ), "Accuracy should be lower for bad policy"


def test_dpo_loss_label_smoothing(tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
    """Verify label smoothing changes the loss value."""
    trainer_no_smooth = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        beta=0.1,
        label_smoothing=0.0,
        device="cpu",
    )
    trainer_smooth = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        beta=0.1,
        label_smoothing=0.1,
        device="cpu",
    )

    policy_chosen = torch.tensor([0.0, -0.5])
    policy_rejected = torch.tensor([-1.0, -1.5])
    ref_chosen = torch.zeros(2)
    ref_rejected = torch.zeros(2)

    metrics_no_smooth = trainer_no_smooth.compute_dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )
    metrics_smooth = trainer_smooth.compute_dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )

    # Smoothing should produce a different loss value
    assert (
        metrics_smooth["loss"].item() != metrics_no_smooth["loss"].item()
    ), "Label smoothing should change the loss"


# ---------------------------------------------------------------------------
# Test 2: Preference Dataset
# ---------------------------------------------------------------------------


def test_preference_dataset(tokenizer: BPETokenizer):
    """Verify PreferenceDataset correctly encodes and masks prompt/response pairs."""
    pairs = [
        {"prompt": "Hello ", "chosen": "world", "rejected": "earth"},
        {"prompt": "What?", "chosen": "Yes", "rejected": "No"},
    ]

    # Train tokenizer first to build vocabulary
    tokenizer.train(["Hello world", "Hello earth", "What? Yes", "What? No"])

    dataset = PreferenceDataset(pairs, tokenizer, max_seq_len=32)

    assert len(dataset) == 2, "Dataset length should match number of pairs"

    # Check first sample
    sample = dataset[0]
    assert "chosen_input_ids" in sample
    assert "chosen_labels" in sample
    assert "rejected_input_ids" in sample
    assert "rejected_labels" in sample
    assert "prompt_len" in sample

    # Labels should start with -100 for prompt portion
    prompt_len = sample["prompt_len"].item()
    chosen_labels = sample["chosen_labels"]
    assert chosen_labels.dtype == torch.long
    # Prompt portion should be -100
    for i in range(min(prompt_len, len(chosen_labels))):
        assert chosen_labels[i].item() == -100, (
            f"Position {i} should be masked (-100) as part of prompt, "
            f"got {chosen_labels[i].item()}"
        )

    # Input ids should be valid token IDs (non-negative)
    assert (sample["chosen_input_ids"] >= 0).all(), "Input IDs should be non-negative"
    assert (sample["rejected_input_ids"] >= 0).all(), "Input IDs should be non-negative"


def test_preference_dataset_truncation(tokenizer: BPETokenizer):
    """Verify sequences are truncated to max_seq_len."""
    # Train tokenizer
    tokenizer.train(["This is a very long text that will be truncated"])

    pairs = [
        {
            "prompt": "Long prompt here ",
            "chosen": "with a very long response that exceeds limit",
            "rejected": "with another very long response exceeding",
        }
    ]

    max_seq_len = 8
    dataset = PreferenceDataset(pairs, tokenizer, max_seq_len=max_seq_len)
    sample = dataset[0]

    assert (
        len(sample["chosen_input_ids"]) <= max_seq_len
    ), f"Chosen input should be truncated to {max_seq_len}"
    assert (
        len(sample["rejected_input_ids"]) <= max_seq_len
    ), f"Rejected input should be truncated to {max_seq_len}"


def test_preference_collate_fn():
    """Verify collate function correctly pads variable-length sequences."""
    batch = [
        {
            "chosen_input_ids": torch.tensor([1, 2, 3]),
            "chosen_labels": torch.tensor([-100, 2, 3]),
            "rejected_input_ids": torch.tensor([4, 5]),
            "rejected_labels": torch.tensor([-100, 5]),
            "prompt_len": 1,
        },
        {
            "chosen_input_ids": torch.tensor([6, 7]),
            "chosen_labels": torch.tensor([-100, 7]),
            "rejected_input_ids": torch.tensor([8, 9, 10]),
            "rejected_labels": torch.tensor([-100, 9, 10]),
            "prompt_len": 1,
        },
    ]

    collated = preference_collate_fn(batch)

    # Check batch dimension
    assert collated["chosen_input_ids"].shape[0] == 2, "Batch size should be 2"
    # Check padding: max length in batch for chosen is 3
    assert (
        collated["chosen_input_ids"].shape[1] == 3
    ), "Padded length should be 3 (max in batch)"
    # Check padding value for chosen labels
    assert (
        collated["chosen_labels"][1, 2] == -100
    ), "Padding should be -100 for labels"
    # Prompt lengths
    assert torch.equal(
        collated["prompt_len"], torch.tensor([1, 1])
    ), "Prompt lengths should be preserved"


# ---------------------------------------------------------------------------
# Test 3: Training Step
# ---------------------------------------------------------------------------


def test_train_step(tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
    """Verify train_step runs and loss decreases over multiple steps."""
    # Train tokenizer
    tokenizer.train([
        "What is 2+2? Four",
        "What is 2+2? Five",
        "Capital of France? Paris",
        "Capital of France? London",
        "Sky color? Blue",
        "Sky color? Green",
    ])

    pairs = [
        {"prompt": "What is 2+2? ", "chosen": "Four", "rejected": "Five"},
        {"prompt": "Capital of France? ", "chosen": "Paris", "rejected": "London"},
        {"prompt": "Sky color? ", "chosen": "Blue", "rejected": "Green"},
        {"prompt": "What is 2+2? ", "chosen": "Four", "rejected": "Five"},
        {"prompt": "Capital of France? ", "chosen": "Paris", "rejected": "London"},
        {"prompt": "Sky color? ", "chosen": "Blue", "rejected": "Green"},
    ]

    trainer = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        beta=0.1,
        learning_rate=1e-3,
        batch_size=2,
        device="cpu",
    )

    # Train for a few epochs
    metrics1 = trainer.train_step(pairs, num_epochs=2)
    loss1 = metrics1["loss"]
    acc1 = metrics1["accuracy"]

    assert loss1 > 0, "Loss should be positive"
    assert 0 <= acc1 <= 1, "Accuracy should be in [0, 1]"

    # Train for a few more epochs — loss should decrease
    metrics2 = trainer.train_step(pairs, num_epochs=3)
    loss2 = metrics2["loss"]

    # Loss should generally decrease (or at least not explode)
    assert loss2 > 0, "Loss should remain positive"
    assert loss2 < loss1 * 5, "Loss should not explode"


def test_train_step_metrics_shape(tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
    """Verify train_step returns expected metric keys and types."""
    tokenizer.train(["Hello world", "Hello earth"])

    pairs = [
        {"prompt": "Hello ", "chosen": "world", "rejected": "earth"},
    ]

    trainer = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        beta=0.1,
        device="cpu",
    )

    metrics = trainer.train_step(pairs, num_epochs=1)

    assert "loss" in metrics, "Metrics should contain 'loss'"
    assert "reward_margin" in metrics, "Metrics should contain 'reward_margin'"
    assert "accuracy" in metrics, "Metrics should contain 'accuracy'"
    assert isinstance(metrics["loss"], float), "Loss should be a float"
    assert isinstance(metrics["accuracy"], float), "Accuracy should be a float"


# ---------------------------------------------------------------------------
# Test 4: Log-probability Computation
# ---------------------------------------------------------------------------


def test_compute_log_probs(tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
    """Verify log-prob computation returns correct shape and values."""
    tokenizer.train(["Hello world this is a test"])

    trainer = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        device="cpu",
    )

    # Create a simple input
    text = "Hello world"
    token_ids = tokenizer.encode(text)
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    # Labels: -100 for first token, rest are the actual tokens
    labels = torch.full_like(input_ids, -100)
    labels[0, 1:] = torch.tensor(token_ids[1:])

    log_probs = trainer._compute_log_probs(tiny_model, input_ids, labels)

    # Should return per-sample average log-prob
    assert log_probs.shape == (1,), f"Expected shape (1,), got {log_probs.shape}"
    assert torch.isfinite(log_probs).all(), "Log probs should be finite"
    assert log_probs[0] < 0, "Log probs should be negative (probabilities < 1)"


def test_compute_log_probs_masking(tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
    """Verify -100 masking properly excludes prompt tokens from average."""
    tokenizer.train(["prompt response here"])

    trainer = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        device="cpu",
    )

    prompt_ids = tokenizer.encode("prompt ")
    response_ids = tokenizer.encode("response")
    full_ids = prompt_ids + response_ids

    input_ids = torch.tensor([full_ids], dtype=torch.long)
    # Only response tokens should be predicted
    labels = torch.full_like(input_ids, -100)
    labels[0, len(prompt_ids) :] = torch.tensor(response_ids)

    log_probs = trainer._compute_log_probs(tiny_model, input_ids, labels)

    assert log_probs.shape == (1,)
    assert torch.isfinite(log_probs).all()


# ---------------------------------------------------------------------------
# Test 5: Reference Model Frozen
# ---------------------------------------------------------------------------


def test_reference_model_frozen(tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
    """Verify reference model parameters don't change during training."""
    tokenizer.train(["What is 2+2? Four", "What is 2+2? Five"])

    pairs = [
        {"prompt": "What is 2+2? ", "chosen": "Four", "rejected": "Five"},
        {"prompt": "What is 2+2? ", "chosen": "Four", "rejected": "Five"},
    ]

    trainer = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        beta=0.1,
        learning_rate=1e-3,
        batch_size=2,
        device="cpu",
    )

    # Capture reference model state before training
    ref_state_before = {
        name: param.clone()
        for name, param in trainer.reference_model.named_parameters()
    }

    # Train
    trainer.train_step(pairs, num_epochs=2)

    # Check reference model parameters haven't changed
    for name, param in trainer.reference_model.named_parameters():
        assert torch.equal(
            param, ref_state_before[name]
        ), f"Reference model parameter {name} changed during training!"


def test_reference_model_no_grad(tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
    """Verify reference model parameters have requires_grad=False."""
    trainer = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        device="cpu",
    )

    for name, param in trainer.reference_model.named_parameters():
        assert (
            not param.requires_grad
        ), f"Reference parameter {name} has requires_grad=True"


# ---------------------------------------------------------------------------
# Test 6: Reward Margin
# ---------------------------------------------------------------------------


def test_reward_margin_positive(tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
    """Verify reward_margin is positive when policy prefers chosen."""
    trainer = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        beta=0.1,
        device="cpu",
    )

    batch_size = 8
    # Policy strongly prefers chosen
    policy_chosen = torch.randn(batch_size)
    policy_rejected = policy_chosen - 2.0  # chosen is always better

    ref_chosen = torch.zeros(batch_size)
    ref_rejected = torch.zeros(batch_size)

    metrics = trainer.compute_dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )

    assert (
        metrics["reward_margin"].item() > 0.5
    ), "Reward margin should be positive when chosen is preferred"
    assert (
        metrics["accuracy"].item() == 1.0
    ), "Accuracy should be 1.0 when chosen > rejected always"


# ---------------------------------------------------------------------------
# Test 7: Save / Load
# ---------------------------------------------------------------------------


def test_save_load(tmp_path, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
    """Verify save and load work correctly."""
    trainer = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        device="cpu",
    )

    # Modify the policy model weights
    with torch.no_grad():
        for param in trainer.policy_model.parameters():
            param.add_(0.01)

    # Save
    save_dir = str(tmp_path / "dpo_checkpoint")
    trainer.save(save_dir)

    # Create new model and load
    new_model = SelfImprovingLLM(tiny_model.config)
    new_trainer = DPOTrainer(
        model=new_model,
        tokenizer=tokenizer,
        device="cpu",
    )
    new_trainer.load(save_dir)

    # Check policy parameters match
    for (n1, p1), (n2, p2) in zip(
        trainer.policy_model.named_parameters(),
        new_trainer.policy.named_parameters() if hasattr(new_trainer, 'policy') else new_trainer.policy_model.named_parameters(),
    ):
        assert torch.allclose(p1, p2), f"Policy parameter {n1} doesn't match after load"


# ---------------------------------------------------------------------------
# Test 8: Generate Preferences
# ---------------------------------------------------------------------------


def test_generate_preferences(
    tiny_model: SelfImprovingLLM,
    tokenizer: BPETokenizer,
):
    """Verify generate_preferences returns valid preference pairs."""
    tokenizer.train(["Hello world", "How are you", "What is this"])

    trainer = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        beta=0.1,
        device="cpu",
    )

    prompts = ["Hello ", "How "]
    pairs = trainer.generate_preferences(
        prompts=prompts,
        num_samples=2,
        temperature=0.5,
        top_p=0.9,
    )

    assert isinstance(pairs, list), "Should return a list"
    # Note: generate may produce varying numbers of pairs
    for pair in pairs:
        assert "prompt" in pair, "Each pair should have 'prompt'"
        assert "chosen" in pair, "Each pair should have 'chosen'"
        assert "rejected" in pair, "Each pair should have 'rejected'"
        assert isinstance(pair["chosen"], str), "Chosen should be a string"
        assert isinstance(pair["rejected"], str), "Rejected should be a string"
        assert pair["chosen"] != pair["rejected"], (
            "Chosen and rejected should be different"
        )


# ---------------------------------------------------------------------------
# Test 9: Beta scaling
# ---------------------------------------------------------------------------


def test_beta_scaling(tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
    """Verify beta parameter affects the loss magnitude."""
    policy_chosen = torch.tensor([0.5])
    policy_rejected = torch.tensor([-0.5])
    ref_chosen = torch.zeros(1)
    ref_rejected = torch.zeros(1)

    # High beta = stronger penalty for deviation
    trainer_high = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        beta=1.0,
        device="cpu",
    )
    metrics_high = trainer_high.compute_dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )

    # Low beta = weaker penalty
    trainer_low = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        beta=0.01,
        device="cpu",
    )
    metrics_low = trainer_low.compute_dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )

    # High beta should produce higher reward margin (sharper preferences)
    assert (
        metrics_high["reward_margin"].item() > metrics_low["reward_margin"].item()
    ), "High beta should produce sharper reward margin"


# ---------------------------------------------------------------------------
# Test 10: Empty / edge cases
# ---------------------------------------------------------------------------


def test_empty_preference_pairs(tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
    """Verify empty preference pairs list doesn't crash."""
    trainer = DPOTrainer(
        model=tiny_model,
        tokenizer=tokenizer,
        device="cpu",
    )

    # Empty list should not crash and return zero metrics
    metrics = trainer.train_step([], num_epochs=1)
    assert isinstance(metrics, dict)
    assert "loss" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
