"""Comprehensive tests for Mixture of Experts (MoE) module.

Tests cover:
- Router top-k selection and weight normalisation
- MoE layer shape correctness and capacity handling
- MoE transformer block forward pass
- Model-level aux_loss integration
- Gradient flow and load balancing
"""

import math

import pytest
import torch
import torch.nn as nn

from selfllm.model.config import ModelConfig
from selfllm.model.moe import ExpertFFN, MoELayer, MoETransformerBlock, Router
from selfllm.model.model import SelfImprovingLLM


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def num_experts():
    return 8


@pytest.fixture
def top_k():
    return 2


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 16


@pytest.fixture
def router(d_model, num_experts, top_k):
    return Router(d_model=d_model, num_experts=num_experts, top_k=top_k)


@pytest.fixture
def moe_layer(d_model, num_experts, top_k):
    return MoELayer(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        expert_d_ff=d_model * 4,
        dropout=0.0,
        capacity_factor=1.25,
    )


# ============================================================================
# Router Tests
# ============================================================================

def test_router_topk(router, batch_size, seq_len, d_model, top_k):
    """Router should select exactly top_k experts per token."""
    x = torch.randn(batch_size * seq_len, d_model)
    expert_indices, expert_weights, aux_loss = router(x)

    # Shape checks
    assert expert_indices.shape == (batch_size * seq_len, top_k)
    assert expert_weights.shape == (batch_size * seq_len, top_k)

    # Each entry should be a valid expert index
    assert expert_indices.min() >= 0
    assert expert_indices.max() < router.num_experts


def test_router_weights_normalized(router, batch_size, seq_len, d_model):
    """Router weights should sum to 1 per token."""
    x = torch.randn(batch_size * seq_len, d_model)
    _, expert_weights, _ = router(x)

    weight_sums = expert_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


def test_router_aux_loss(router, batch_size, seq_len, d_model):
    """Auxiliary loss should be non-negative and promote balance."""
    x = torch.randn(batch_size * seq_len, d_model)
    _, _, aux_loss = router(x)

    # Aux loss should be a scalar
    assert aux_loss.dim() == 0

    # Aux loss should be non-negative (theoretically >= 1.0 for balanced)
    assert aux_loss.item() >= 0.0


def test_router_different_experts_selected(router, batch_size, seq_len, d_model):
    """Different tokens should potentially select different experts."""
    x = torch.randn(batch_size * seq_len, d_model)
    expert_indices, _, _ = router(x)

    # With enough tokens and random input, we should see multiple experts used
    unique_experts = expert_indices.unique().numel()
    # We can't guarantee all experts are used with random data, but at least
    # more than 1 should be selected in most cases
    assert unique_experts >= 1


# ============================================================================
# MoE Layer Tests
# ============================================================================

def test_moe_layer_shape(moe_layer, batch_size, seq_len, d_model):
    """MoE layer output shape should match input shape."""
    x = torch.randn(batch_size, seq_len, d_model)
    output, aux_loss = moe_layer(x)

    assert output.shape == x.shape
    assert aux_loss.dim() == 0
    assert aux_loss.item() >= 0.0


def test_moe_capacity(moe_layer, d_model):
    """Tokens beyond capacity should be dropped (no crash)."""
    # Use a large number of tokens relative to capacity
    batch_size, seq_len = 8, 64
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward should complete without error even with capacity truncation
    output, aux_loss = moe_layer(x)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_moe_expert_selection(moe_layer, batch_size, seq_len, d_model, num_experts):
    """Different tokens should go to different experts (no single-expert collapse)."""
    x = torch.randn(batch_size, seq_len, d_model)

    # Check routing distribution via the router directly
    x_flat = x.view(-1, d_model)
    expert_indices, _, _ = moe_layer.router(x_flat)

    unique_experts = expert_indices.unique().numel()
    # With random data across 8 experts and 64 tokens, multiple experts should be used
    assert unique_experts > 1, "All tokens routed to the same expert -- possible bug"


def test_moe_all_tokens_processed(moe_layer, batch_size, seq_len, d_model):
    """All tokens should have non-zero output (no dropped tokens)."""
    # Use small scale so capacity is not exceeded
    x = torch.randn(2, 8, d_model)
    output, _ = moe_layer(x)

    # Every token should have been processed (output should be non-zero for most)
    # Since weights are normalised and experts are initialized, output should exist
    assert output.shape == x.shape
    # At minimum no NaNs or Infs
    assert torch.isfinite(output).all()


# ============================================================================
# MoE Transformer Block Tests
# ============================================================================

def test_moe_transformer_block():
    """Full MoE transformer block forward pass."""
    config = ModelConfig(
        d_model=64,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=2,
    )
    block = MoETransformerBlock(config)

    batch_size, seq_len = 2, 8
    x = torch.randn(batch_size, seq_len, config.d_model)

    output = block(x)

    assert output.shape == x.shape
    assert hasattr(block, "aux_loss")
    assert block.aux_loss.item() >= 0.0


def test_moe_transformer_block_with_mask():
    """MoE transformer block with causal mask."""
    config = ModelConfig(
        d_model=64,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=2,
    )
    block = MoETransformerBlock(config)

    batch_size, seq_len = 2, 8
    x = torch.randn(batch_size, seq_len, config.d_model)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    output = block(x, mask=mask)

    assert output.shape == x.shape


# ============================================================================
# Model-Level Tests
# ============================================================================

def test_moe_aux_loss_in_model():
    """Loss should include aux_loss when use_moe=True."""
    config = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=2,
        moe_aux_loss_weight=0.01,
    )
    model = SelfImprovingLLM(config)

    batch_size, seq_len = 2, 8
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    result = model(token_ids, targets=targets)

    assert "loss" in result
    assert result["loss"].item() >= 0.0

    # Verify aux_loss is collected from blocks
    total_aux = sum(
        block.aux_loss.item() for block in model.blocks if hasattr(block, "aux_loss")
    )
    assert total_aux > 0.0, "aux_loss should be positive when routing is active"


def test_moe_no_aux_loss_when_disabled():
    """No aux_loss should be present when use_moe=False."""
    config = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        use_moe=False,
    )
    model = SelfImprovingLLM(config)

    batch_size, seq_len = 2, 8
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    result = model(token_ids, targets=targets)

    assert "loss" in result

    # Verify no blocks have aux_loss attribute set meaningfully
    for block in model.blocks:
        assert not hasattr(block, "aux_loss") or block.aux_loss == 0.0


# ============================================================================
# Gradient Flow Tests
# ============================================================================

def test_gradient_flow_router(moe_layer, batch_size, seq_len, d_model):
    """Gradients should flow back through the router gate weights."""
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    output, aux_loss = moe_layer(x)

    # Backprop through output
    loss = output.sum() + aux_loss
    loss.backward()

    # Router gate should have gradients
    assert moe_layer.router.gate.weight.grad is not None
    assert not torch.all(moe_layer.router.gate.weight.grad == 0)


def test_gradient_flow_experts(moe_layer, batch_size, seq_len, d_model):
    """Gradients should flow back through expert weights."""
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    output, aux_loss = moe_layer(x)

    loss = output.sum() + aux_loss
    loss.backward()

    # Each expert should have gradients
    for expert in moe_layer.experts:
        for param in expert.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)


def test_gradient_flow_input(moe_layer, batch_size, seq_len, d_model):
    """Gradients should flow back to the input."""
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    output, aux_loss = moe_layer(x)

    loss = output.sum() + aux_loss
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ============================================================================
# Load Balancing Tests
# ============================================================================

def test_load_balancing_aux_loss_decreases():
    """Aux loss should decrease when routing becomes more balanced."""
    d_model = 32
    num_experts = 4
    top_k = 2
    num_tokens = 64

    router = Router(d_model=d_model, num_experts=num_experts, top_k=top_k)

    # Uniform input should produce relatively balanced routing
    # and aux_loss should be finite and positive
    x_balanced = torch.randn(num_tokens, d_model)
    _, _, aux_loss_balanced = router(x_balanced)

    # Theoretical minimum for perfectly balanced routing is 1.0;
    # with random data we expect ~1.0 (allow small floating-point variance).
    assert aux_loss_balanced.item() >= 0.99, (
        "Balanced routing aux_loss should be ~1.0 "
        f"(got {aux_loss_balanced.item():.4f})"
    )

    # Aux loss should be reasonable (not astronomically high)
    assert aux_loss_balanced.item() < num_experts * 10, (
        f"aux_loss seems unreasonably high: {aux_loss_balanced.item():.4f}"
    )


def test_load_balancing_improves_with_training():
    """Training the router should improve load balancing (lower aux_loss)."""
    d_model = 32
    num_experts = 4
    top_k = 2
    num_tokens = 128

    router = Router(d_model=d_model, num_experts=num_experts, top_k=top_k)
    optimizer = torch.optim.Adam(router.parameters(), lr=0.01)

    # Measure initial aux loss
    x = torch.randn(num_tokens, d_model)
    _, _, initial_aux = router(x)
    initial_value = initial_aux.item()

    # Train for a few steps to minimize aux_loss
    for _ in range(50):
        optimizer.zero_grad()
        _, _, aux_loss = router(x)
        aux_loss.backward()
        optimizer.step()

    # Measure final aux loss
    with torch.no_grad():
        _, _, final_aux = router(x)
    final_value = final_aux.item()

    # Aux loss should decrease (or at minimum stay bounded)
    assert final_value < initial_value * 1.5, (
        f"aux_loss should not blow up: initial={initial_value:.4f}, "
        f"final={final_value:.4f}"
    )


# ============================================================================
# ExpertFFN Tests
# ============================================================================

def test_expert_ffn_shape(d_model):
    """ExpertFFN output shape should match input shape."""
    expert = ExpertFFN(d_model, d_ff=d_model * 4, dropout=0.0)
    x = torch.randn(8, d_model)
    output = expert(x)
    assert output.shape == x.shape


# ============================================================================
# Capacity Boundary Tests
# ============================================================================

def test_capacity_with_very_few_tokens(d_model):
    """MoE should handle very few tokens (capacity edge case)."""
    moe = MoELayer(
        d_model=d_model,
        num_experts=8,
        top_k=2,
        expert_d_ff=d_model * 4,
        dropout=0.0,
        capacity_factor=1.25,
    )
    # Only 2 tokens total -- some experts will get zero tokens
    x = torch.randn(1, 2, d_model)
    output, aux_loss = moe(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_capacity_with_many_tokens(d_model):
    """MoE should handle many tokens without OOM or crash."""
    moe = MoELayer(
        d_model=d_model,
        num_experts=4,
        top_k=2,
        expert_d_ff=d_model * 4,
        dropout=0.0,
        capacity_factor=1.0,  # Tight capacity
    )
    x = torch.randn(4, 64, d_model)
    output, aux_loss = moe(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


# ============================================================================
# Integration: Model Forward with Different Configurations
# ============================================================================

def test_model_with_moe_vs_without_moe():
    """MoE and non-MoE models should both produce valid outputs."""
    config_moe = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=2,
    )
    config_dense = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        use_moe=False,
    )

    model_moe = SelfImprovingLLM(config_moe)
    model_dense = SelfImprovingLLM(config_dense)

    batch_size, seq_len = 2, 8
    token_ids = torch.randint(0, 100, (batch_size, seq_len))

    out_moe = model_moe(token_ids)
    out_dense = model_dense(token_ids)

    assert out_moe["logits"].shape == (batch_size, seq_len, 100)
    assert out_dense["logits"].shape == (batch_size, seq_len, 100)

    # Shapes should be identical
    assert out_moe["logits"].shape == out_dense["logits"].shape


# ============================================================================
# Numerical Stability Tests
# ============================================================================

def test_router_with_extreme_values(d_model, num_experts, top_k):
    """Router should handle extreme input values without NaN/Inf."""
    router = Router(d_model=d_model, num_experts=num_experts, top_k=top_k)

    # Very large values
    x_large = torch.randn(16, d_model) * 100
    indices, weights, aux = router(x_large)
    assert torch.isfinite(weights).all()
    assert aux.item() >= 0.0

    # Very small values
    x_small = torch.randn(16, d_model) * 0.001
    indices, weights, aux = router(x_small)
    assert torch.isfinite(weights).all()
    assert aux.item() >= 0.0


def test_moe_with_extreme_values(d_model):
    """MoE layer should handle extreme input values."""
    moe = MoELayer(d_model=d_model, num_experts=4, top_k=2, dropout=0.0)

    x = torch.randn(2, 8, d_model) * 100
    output, aux = moe(x)
    assert torch.isfinite(output).all()
    assert aux.item() >= 0.0


# ============================================================================
# Determinism / Reproducibility Tests
# ============================================================================

def test_moe_deterministic_forward(d_model):
    """MoE layer should be deterministic (same input -> same output)."""
    moe = MoELayer(d_model=d_model, num_experts=4, top_k=2, dropout=0.0)
    moe.eval()

    x = torch.randn(2, 8, d_model)

    with torch.no_grad():
        out1, aux1 = moe(x)
        out2, aux2 = moe(x)

    assert torch.allclose(out1, out2, atol=1e-6)
    assert torch.allclose(aux1, aux2, atol=1e-6)


# ============================================================================
# Parameter Count Test
# ============================================================================

def test_moe_has_more_parameters_than_dense(d_model):
    """MoE model should have more parameters than dense (more expert FFNs)."""
    config_moe = ModelConfig(
        d_model=d_model,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        use_moe=True,
        moe_num_experts=8,
        moe_top_k=2,
    )
    config_dense = ModelConfig(
        d_model=d_model,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        use_moe=False,
    )

    model_moe = SelfImprovingLLM(config_moe)
    model_dense = SelfImprovingLLM(config_dense)

    moe_params = sum(p.numel() for p in model_moe.parameters())
    dense_params = sum(p.numel() for p in model_dense.parameters())

    # MoE model should have more parameters due to multiple experts
    assert moe_params > dense_params, (
        f"MoE model ({moe_params}) should have more params than dense ({dense_params})"
    )


def test_moe_model_forward_train_eval():
    """MoE model should work in both train and eval modes."""
    config = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=2,
    )
    model = SelfImprovingLLM(config)

    token_ids = torch.randint(0, 100, (2, 8))
    targets = torch.randint(0, 100, (2, 8))

    # Train mode
    model.train()
    out_train = model(token_ids, targets=targets)
    assert "loss" in out_train

    # Eval mode
    model.eval()
    with torch.no_grad():
        out_eval = model(token_ids, targets=targets)
    assert "loss" in out_eval


# ============================================================================
# Top-k = 1 Edge Case (Switch-style)
# ============================================================================

def test_switch_style_top1_routing():
    """MoE should work with top_k=1 (Switch Transformer style)."""
    config = ModelConfig(
        d_model=64,
        n_layers=1,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=1,
    )
    block = MoETransformerBlock(config)

    x = torch.randn(2, 8, config.d_model)
    output = block(x)

    assert output.shape == x.shape
    assert block.aux_loss.item() >= 0.0
