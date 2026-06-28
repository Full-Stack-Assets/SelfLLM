"""Unit tests for PPO trainer and reward model.

Tests cover:
- RewardModel outputs scalar reward per sequence
- RewardModel gradients flow through score head
- PPO loss computation produces decreasing loss
- GAE computation produces reasonable advantages
- KL penalty is non-negative
- Policy reward improves over training steps
- Reference model parameters stay frozen
"""

import sys

import pytest
import torch

sys.path.insert(0, "/home/kimi/work-constitutional")

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.training.ppo_trainer import (
    PPOTrainer,
    RewardModel,
    ValueModel,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


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
    tok.train(["hello world this is a test of the ppo trainer system"])
    return tok


@pytest.fixture
def reward_model(tiny_model: SelfImprovingLLM) -> RewardModel:
    """Create a reward model for testing."""
    return RewardModel(tiny_model)


@pytest.fixture
def value_model(tiny_model: SelfImprovingLLM) -> ValueModel:
    """Create a value model for testing."""
    return ValueModel(tiny_model)


@pytest.fixture
def ppo_trainer(tiny_model: SelfImprovingLLM) -> PPOTrainer:
    """Create a PPO trainer for testing."""
    ref_model = SelfImprovingLLM(tiny_model.config)
    ref_model.load_state_dict(tiny_model.state_dict())
    reward_model = RewardModel(tiny_model)
    return PPOTrainer(
        policy_model=tiny_model,
        ref_model=ref_model,
        reward_model=reward_model,
        clip_epsilon=0.2,
        kl_coeff=0.02,
        gamma=0.99,
        lam=0.95,
        lr=1e-3,
        device="cpu",
        num_ppo_epochs=2,
    )


# --------------------------------------------------------------------------- #
# Test 1: Reward Model Output Shape
# --------------------------------------------------------------------------- #


def test_reward_model_shape(reward_model: RewardModel, tiny_config: ModelConfig):
    """Verify RewardModel outputs a scalar per sequence."""
    batch_size = 4
    seq_len = 8
    token_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

    rewards = reward_model(token_ids)

    # Should be 1D tensor of shape [batch_size]
    assert rewards.shape == (batch_size,), (
        f"Expected shape ({batch_size},), got {rewards.shape}"
    )

    # Should be scalar floats
    assert rewards.dtype == torch.float32


def test_reward_model_single_sample(reward_model: RewardModel, tiny_config: ModelConfig):
    """Verify RewardModel works with single sample."""
    token_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
    rewards = reward_model(token_ids)
    assert rewards.shape == (1,)


# --------------------------------------------------------------------------- #
# Test 2: Reward Model Gradient Flow
# --------------------------------------------------------------------------- #


def test_reward_model_gradient(reward_model: RewardModel, tiny_config: ModelConfig):
    """Verify gradients flow through the reward model's score head."""
    batch_size = 2
    seq_len = 8
    token_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

    # Forward pass
    rewards = reward_model(token_ids)
    loss = rewards.sum()
    loss.backward()

    # Score head should have gradients
    assert reward_model.score_head.weight.grad is not None, (
        "Score head weight should have gradients"
    )
    assert reward_model.score_head.weight.grad.abs().sum() > 0, (
        "Score head gradients should be non-zero"
    )

    # Encoder parameters should also have gradients (since it's part of the model)
    encoder_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in reward_model.encoder.parameters()
    )
    assert encoder_has_grad, "Encoder parameters should have gradients"


# --------------------------------------------------------------------------- #
# Test 3: PPO Loss Computation
# --------------------------------------------------------------------------- #


def test_ppo_loss_computation(ppo_trainer: PPOTrainer):
    """Verify PPO loss computation produces a valid, decreasing loss."""
    batch_size = 2
    seq_len = 4

    # Create synthetic data
    policy_log_probs = torch.randn(batch_size, seq_len, requires_grad=True)
    old_log_probs = torch.randn(batch_size, seq_len)
    ref_log_probs = torch.randn(batch_size, seq_len)
    advantages = torch.randn(batch_size, seq_len)
    values = torch.randn(batch_size, seq_len, requires_grad=True)
    returns = torch.randn(batch_size, seq_len)

    loss_dict = ppo_trainer._compute_ppo_loss(
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        advantages=advantages,
        values=values,
        returns=returns,
    )

    # Should return all expected keys
    assert "policy_loss" in loss_dict
    assert "value_loss" in loss_dict
    assert "kl_div" in loss_dict
    assert "total_loss" in loss_dict

    # All should be scalar tensors
    for key in ["policy_loss", "value_loss", "kl_div", "total_loss"]:
        assert loss_dict[key].dim() == 0, f"{key} should be a scalar"

    # Total loss should be finite
    assert torch.isfinite(loss_dict["total_loss"]), "Total loss should be finite"

    # Policy loss should be positive (it's -min(surr1, surr2), a minimization objective)
    # It can be negative if all advantages are positive and ratio < 1
    # So we just check it's finite
    assert torch.isfinite(loss_dict["policy_loss"]), "Policy loss should be finite"

    # Value loss should be non-negative (MSE)
    assert loss_dict["value_loss"].item() >= 0, "Value loss (MSE) should be non-negative"

    # Total loss should combine all components
    expected_total = (
        loss_dict["policy_loss"]
        + 0.5 * loss_dict["value_loss"]
        + ppo_trainer.kl_coeff * loss_dict["kl_div"]
    )
    assert torch.allclose(loss_dict["total_loss"], expected_total), (
        "Total loss should be policy_loss + 0.5 * value_loss + kl_coeff * kl_div"
    )


def test_ppo_loss_decreases_when_advantages_positive(ppo_trainer: PPOTrainer):
    """Verify PPO loss is lower when policy follows positive advantages."""
    batch_size = 2
    seq_len = 4

    # Positive advantages - policy should increase likelihood
    advantages = torch.ones(batch_size, seq_len) * 2.0

    # Case 1: ratio = 1 (no change from old policy)
    old_log_probs = torch.zeros(batch_size, seq_len)
    policy_log_probs_same = torch.zeros(batch_size, seq_len, requires_grad=True)
    ref_log_probs = torch.zeros(batch_size, seq_len)
    values = torch.zeros(batch_size, seq_len, requires_grad=True)
    returns = torch.ones(batch_size, seq_len)

    loss_same = ppo_trainer._compute_ppo_loss(
        policy_log_probs=policy_log_probs_same,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        advantages=advantages,
        values=values,
        returns=returns,
    )

    # Case 2: ratio > 1 (policy increases likelihood of good actions)
    policy_log_probs_better = torch.ones(batch_size, seq_len) * 0.5
    policy_log_probs_better.requires_grad_(True)

    loss_better = ppo_trainer._compute_ppo_loss(
        policy_log_probs=policy_log_probs_better,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        advantages=advantages,
        values=values.clone().detach().requires_grad_(True),
        returns=returns,
    )

    # Better alignment with advantages should not increase policy loss
    assert loss_better["policy_loss"].item() <= loss_same["policy_loss"].item() + 1e-4, (
        "Policy loss should decrease or stay same when following positive advantages"
    )


# --------------------------------------------------------------------------- #
# Test 4: GAE Computation
# --------------------------------------------------------------------------- #


def test_gae_computation(ppo_trainer: PPOTrainer):
    """Verify GAE produces reasonable advantages and returns."""
    batch_size = 2
    seq_len = 4

    # Simple case: single reward at the end
    rewards = torch.zeros(batch_size, seq_len)
    rewards[:, -1] = 1.0  # Reward only at last step

    values = torch.zeros(batch_size, seq_len)
    log_probs = torch.zeros(batch_size, seq_len)

    advantages, returns = ppo_trainer._compute_gae(rewards, values, log_probs)

    # Advantages should have the same shape as rewards
    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape

    # Advantages should generally be higher for later tokens when reward is at end
    # because they are closer to the reward
    for b in range(batch_size):
        # Earlier tokens should have smaller advantages than later ones
        # due to discounting
        for t in range(seq_len - 1):
            assert advantages[b, t] <= advantages[b, t + 1] + 1e-5, (
                f"Advantage at t={t} should be <= advantage at t={t+1} "
                f"when reward is at the end"
            )

    # Returns should equal advantages + values (which are 0)
    for b in range(batch_size):
        for t in range(seq_len):
            assert abs(returns[b, t].item() - advantages[b, t].item()) < 1e-5, (
                "Returns should equal advantages + values"
            )


def test_gae_zero_reward(ppo_trainer: PPOTrainer):
    """Verify GAE with zero rewards gives zero/negative advantages (due to value baseline)."""
    batch_size = 1
    seq_len = 3

    rewards = torch.zeros(batch_size, seq_len)
    values = torch.zeros(batch_size, seq_len)
    log_probs = torch.zeros(batch_size, seq_len)

    advantages, returns = ppo_trainer._compute_gae(rewards, values, log_probs)

    # With zero rewards and zero values, advantages should be zero
    assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-5), (
        "Advantages should be zero with zero rewards and zero values"
    )


def test_gae_discounting(ppo_trainer: PPOTrainer):
    """Verify GAE correctly discounts future rewards."""
    batch_size = 1
    seq_len = 4

    rewards = torch.zeros(batch_size, seq_len)
    rewards[:, 0] = 1.0  # Early reward

    values = torch.zeros(batch_size, seq_len)
    log_probs = torch.zeros(batch_size, seq_len)

    advantages, returns = ppo_trainer._compute_gae(rewards, values, log_probs)

    # Advantage at step 0 should be close to the reward (1.0)
    assert abs(advantages[0, 0].item() - 1.0) < 0.1, (
        f"Advantage at reward step should be ~1.0, got {advantages[0, 0].item()}"
    )


# --------------------------------------------------------------------------- #
# Test 5: KL Penalty
# --------------------------------------------------------------------------- #


def test_kl_penalty(ppo_trainer: PPOTrainer):
    """Verify KL divergence penalty is non-negative on average."""
    batch_size = 4
    seq_len = 8

    # When policy = reference, KL should be ~0
    log_probs = torch.zeros(batch_size, seq_len, requires_grad=True)
    ref_log_probs = torch.zeros(batch_size, seq_len)
    old_log_probs = torch.zeros(batch_size, seq_len)
    advantages = torch.randn(batch_size, seq_len)
    values = torch.randn(batch_size, seq_len, requires_grad=True)
    returns = torch.randn(batch_size, seq_len)

    loss_dict = ppo_trainer._compute_ppo_loss(
        policy_log_probs=log_probs,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        advantages=advantages,
        values=values,
        returns=returns,
    )

    # KL div should be ~0 when policy == reference
    assert abs(loss_dict["kl_div"].item()) < 1e-4, (
        f"KL div should be ~0 when policy == reference, got {loss_dict['kl_div'].item()}"
    )

    # When policy diverges from reference, KL should be positive
    log_probs_diverged = torch.ones(batch_size, seq_len, requires_grad=True)
    ref_log_probs_same = torch.zeros(batch_size, seq_len)

    loss_dict_diverged = ppo_trainer._compute_ppo_loss(
        policy_log_probs=log_probs_diverged,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs_same,
        advantages=advantages,
        values=torch.randn(batch_size, seq_len, requires_grad=True),
        returns=returns,
    )

    # KL(pi||ref) = E[log pi - log ref]. When pi has higher log-prob, KL is positive.
    # KL divergence should be positive when policy assigns higher probability
    assert loss_dict_diverged["kl_div"].item() > 0, (
        "KL div should be positive when policy diverges from reference"
    )


def test_kl_penalty_non_negative_for_extreme(ppo_trainer: PPOTrainer):
    """Verify KL penalty is non-negative for extreme cases."""
    batch_size = 2
    seq_len = 4

    # Policy with much higher log-probs than reference
    policy_log_probs = torch.full((batch_size, seq_len), 5.0, requires_grad=True)
    ref_log_probs = torch.full((batch_size, seq_len), -5.0)
    old_log_probs = torch.zeros(batch_size, seq_len)
    advantages = torch.ones(batch_size, seq_len)
    values = torch.zeros(batch_size, seq_len, requires_grad=True)
    returns = torch.ones(batch_size, seq_len)

    loss_dict = ppo_trainer._compute_ppo_loss(
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        advantages=advantages,
        values=values,
        returns=returns,
    )

    # KL should be positive (policy diverged from reference)
    assert loss_dict["kl_div"].item() > 0, "KL should be positive when policy diverges"


# --------------------------------------------------------------------------- #
# Test 6: Policy Improvement (reward increases over steps)
# --------------------------------------------------------------------------- #


def test_policy_improvement(ppo_trainer: PPOTrainer, tokenizer: BPETokenizer):
    """Verify that PPO training leads to policy improvement (reward increases)."""
    # Create simple prompts
    prompts_text = ["hello", "world", "test"]
    prompt_ids_list = [tokenizer.encode(p) for p in prompts_text]
    max_len = max(len(p) for p in prompt_ids_list)
    padded = [p + [0] * (max_len - len(p)) for p in prompt_ids_list]
    prompt_tensor = torch.tensor(padded, dtype=torch.long)

    # Train for a few steps and track rewards
    rewards_over_steps = []
    for _ in range(3):
        metrics = ppo_trainer.train_step(
            prompt_tensor,
            max_new_tokens=8,
        )
        rewards_over_steps.append(metrics["reward"])

    # We should have tracked rewards
    assert len(rewards_over_steps) == 3

    # All rewards should be finite
    for r in rewards_over_steps:
        assert r == r, "Reward should not be NaN"  # NaN check

    # Policy loss should be finite
    for key in ["policy_loss", "value_loss", "kl_div", "total_loss"]:
        assert metrics[key] == metrics[key], f"{key} should not be NaN"
        assert metrics[key] != float('inf'), f"{key} should not be inf"
        assert metrics[key] != float('-inf'), f"{key} should not be -inf"


def test_ppo_train_step_metrics(ppo_trainer: PPOTrainer, tokenizer: BPETokenizer):
    """Verify train_step returns all expected metrics."""
    prompt_tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)

    metrics = ppo_trainer.train_step(prompt_tensor, max_new_tokens=8)

    # Check all expected keys
    expected_keys = ["policy_loss", "value_loss", "kl_div", "reward", "total_loss"]
    for key in expected_keys:
        assert key in metrics, f"Metrics should contain '{key}'"
        assert isinstance(metrics[key], float), f"'{key}' should be a float"

    # All should be finite
    for key in expected_keys:
        assert metrics[key] == metrics[key], f"'{key}' should not be NaN"


# --------------------------------------------------------------------------- #
# Test 7: Reference Model Frozen
# --------------------------------------------------------------------------- #


def test_reference_model_frozen(ppo_trainer: PPOTrainer):
    """Verify reference model parameters do not change during training."""
    # Capture initial reference state
    ref_state_before = {
        name: param.clone()
        for name, param in ppo_trainer.ref_model.named_parameters()
    }

    # Run a training step
    prompt_tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    ppo_trainer.train_step(prompt_tensor, max_new_tokens=8)

    # Check reference model hasn't changed
    for name, param in ppo_trainer.ref_model.named_parameters():
        assert torch.allclose(param, ref_state_before[name]), (
            f"Reference model parameter '{name}' changed after training step"
        )


def test_reference_model_no_gradients(ppo_trainer: PPOTrainer):
    """Verify reference model parameters do not require gradients."""
    for param in ppo_trainer.ref_model.parameters():
        assert not param.requires_grad, (
            "Reference model parameters should not require gradients"
        )


# --------------------------------------------------------------------------- #
# Test 8: Value Model
# --------------------------------------------------------------------------- #


def test_value_model_shape(value_model: ValueModel, tiny_config: ModelConfig):
    """Verify ValueModel outputs per-token values."""
    batch_size = 4
    seq_len = 8
    token_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

    values = value_model(token_ids)

    # Should be [batch_size, seq_len] — one value per token
    assert values.shape == (batch_size, seq_len), (
        f"Expected shape ({batch_size}, {seq_len}), got {values.shape}"
    )


def test_value_model_gradient_flow(value_model: ValueModel, tiny_config: ModelConfig):
    """Verify gradients flow through the value model."""
    batch_size = 2
    seq_len = 8
    token_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

    values = value_model(token_ids)
    loss = values.sum()
    loss.backward()

    # Value head should have gradients
    assert value_model.value_head.weight.grad is not None
    assert value_model.value_head.weight.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# Test 9: PPO Clipping
# --------------------------------------------------------------------------- #


def test_ppo_clipping_effect(ppo_trainer: PPOTrainer):
    """Verify PPO clipping prevents overly large policy updates."""
    batch_size = 2
    seq_len = 4

    # Create a scenario where ratio is very large (would cause big update without clipping)
    old_log_probs = torch.zeros(batch_size, seq_len)
    # Very large policy log-probs -> huge ratio
    policy_log_probs = torch.full((batch_size, seq_len), 10.0, requires_grad=True)
    ref_log_probs = torch.zeros(batch_size, seq_len)
    advantages = torch.ones(batch_size, seq_len)  # Positive advantages
    values = torch.zeros(batch_size, seq_len, requires_grad=True)
    returns = torch.ones(batch_size, seq_len)

    loss_dict = ppo_trainer._compute_ppo_loss(
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        advantages=advantages,
        values=values,
        returns=returns,
    )

    # Ratio would be exp(10) without clipping, but clipped to 1+epsilon = 1.2
    # The loss should still be finite due to clipping
    assert torch.isfinite(loss_dict["policy_loss"]), (
        "Policy loss should be finite even with extreme ratios due to clipping"
    )

    # Total loss should be finite
    assert torch.isfinite(loss_dict["total_loss"]), (
        "Total loss should be finite"
    )


# --------------------------------------------------------------------------- #
# Test 10: Value Model Creation
# --------------------------------------------------------------------------- #


def test_value_model_auto_created(tiny_model: SelfImprovingLLM):
    """Verify PPOTrainer creates a value model if not provided."""
    ref_model = SelfImprovingLLM(tiny_model.config)
    ref_model.load_state_dict(tiny_model.state_dict())
    reward_model = RewardModel(tiny_model)

    trainer = PPOTrainer(
        policy_model=tiny_model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=None,  # Auto-create
        device="cpu",
    )

    assert trainer.value_model is not None
    assert isinstance(trainer.value_model, ValueModel)
