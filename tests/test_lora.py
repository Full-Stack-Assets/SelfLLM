"""Unit tests for the LoRA (Low-Rank Adaptation) module."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

# We need to import from the package under test.  The conftest (or the
# environment) should have arranged ``sys.path`` so that ``selfllm`` is
# importable.  If not, these imports will fail loudly and early.
from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.lora import (
    LoRALayer,
    LinearWithLoRA,
    inject_lora,
    merge_lora_weights,
    unmerge_lora_weights,
    get_lora_parameters,
    count_lora_parameters,
    save_lora_weights,
    load_lora_weights,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def tiny_config() -> ModelConfig:
    """Return a tiny model config for fast tests."""
    return ModelConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
        tie_weights=False,
    )


@pytest.fixture
def tiny_model(tiny_config: ModelConfig) -> SelfImprovingLLM:
    """Return a tiny SelfImprovingLLM instance."""
    return SelfImprovingLLM(tiny_config)


# --------------------------------------------------------------------------- #
# LoRALayer tests
# --------------------------------------------------------------------------- #

class TestLoRALayer:
    """Tests for the standalone :class:`LoRALayer`."""

    def test_forward_shape(self):
        """forward should preserve batch dimensions and produce the
        correct output feature dimension."""
        layer = LoRALayer(in_features=32, out_features=64, rank=4, alpha=8.0)
        x = torch.randn(2, 10, 32)  # [batch, seq, in_features]
        out = layer(x)
        assert out.shape == (2, 10, 64)

    def test_forward_1d_input(self):
        """A 1-D input vector should work too."""
        layer = LoRALayer(in_features=16, out_features=8, rank=2, alpha=4.0)
        x = torch.randn(16)
        out = layer(x)
        assert out.shape == (8,)

    def test_forward_3d_input(self):
        """A 3-D input (batch, seq, features) should work."""
        layer = LoRALayer(in_features=16, out_features=8, rank=2, alpha=4.0)
        x = torch.randn(2, 5, 16)
        out = layer(x)
        assert out.shape == (2, 5, 8)

    def test_scaling_computed_correctly(self):
        """The internal ``scaling`` attribute should equal ``alpha / rank``."""
        layer = LoRALayer(in_features=16, out_features=8, rank=4, alpha=12.0)
        assert layer.scaling == pytest.approx(3.0)

    def test_b_initialised_to_zero(self):
        """lora_B should be all zeros so the initial output is zero."""
        layer = LoRALayer(in_features=16, out_features=8, rank=4)
        assert torch.allclose(layer.lora_B, torch.zeros_like(layer.lora_B))

    def test_a_not_all_zero(self):
        """lora_A should not be all zeros (Kaiming init)."""
        layer = LoRALayer(in_features=16, out_features=8, rank=4)
        assert not torch.allclose(
            layer.lora_A, torch.zeros_like(layer.lora_A)
        )

    def test_zero_output_at_init(self):
        """Because B=0, the LoRA path should contribute nothing at init."""
        layer = LoRALayer(in_features=16, out_features=8, rank=4, alpha=4.0)
        x = torch.randn(2, 16)
        out = layer(x)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_dropout_training_vs_eval(self):
        """In eval mode dropout should be a no-op."""
        layer = LoRALayer(
            in_features=16, out_features=8, rank=4, dropout=0.5
        )
        x = torch.randn(2, 16)
        layer.train()

        layer.eval()
        # With dropout=0.5 and B=0, even in train mode the output is zero
        # because B is zero.  Let's set B to non-zero for a meaningful test.
        nn.init.ones_(layer.lora_B)
        out_eval = layer(x)

        # In eval mode dropout is identity, so output should be deterministic
        out_eval_2 = layer(x)
        assert torch.allclose(out_eval, out_eval_2)


# --------------------------------------------------------------------------- #
# LinearWithLoRA tests
# --------------------------------------------------------------------------- #

class TestLinearWithLoRA:
    """Tests for :class:`LinearWithLoRA`."""

    def test_forward_shape(self):
        """Output shape should match the base linear layer."""
        base = nn.Linear(32, 64, bias=False)
        layer = LinearWithLoRA(base, rank=4, alpha=8.0)
        x = torch.randn(2, 10, 32)
        out = layer(x)
        assert out.shape == (2, 10, 64)

    def test_base_frozen(self):
        """Base layer parameters should have ``requires_grad=False``."""
        base = nn.Linear(16, 8, bias=True)
        layer = LinearWithLoRA(base, rank=2, alpha=4.0)
        for param in layer.base_layer.parameters():
            assert not param.requires_grad

    def test_lora_trainable(self):
        """LoRA parameters should have ``requires_grad=True``."""
        base = nn.Linear(16, 8, bias=False)
        layer = LinearWithLoRA(base, rank=2, alpha=4.0)
        assert layer.lora.lora_A.requires_grad
        assert layer.lora.lora_B.requires_grad

    def test_equivalent_to_base_at_init(self):
        """At init (B=0), LinearWithLoRA should behave exactly like base."""
        torch.manual_seed(42)
        base = nn.Linear(16, 8, bias=False)
        # Copy base weights before wrapping
        base_copy = nn.Linear(16, 8, bias=False)
        base_copy.weight.data = base.weight.data.clone()

        layer = LinearWithLoRA(base, rank=2, alpha=4.0)
        x = torch.randn(2, 16)

        out_base = base_copy(x)
        out_lora = layer(x)
        assert torch.allclose(out_base, out_lora)


# --------------------------------------------------------------------------- #
# inject_lora tests
# --------------------------------------------------------------------------- #

class TestInjectLoRA:
    """Tests for :func:`inject_lora`."""

    def test_injects_default_targets(self, tiny_model: SelfImprovingLLM):
        """By default q_proj and v_proj should be wrapped."""
        inject_lora(tiny_model, rank=4, alpha=8.0)

        for block in tiny_model.blocks:
            assert isinstance(block.attn.attn.q_proj, LinearWithLoRA)
            assert isinstance(block.attn.attn.v_proj, LinearWithLoRA)

    def test_does_not_inject_non_targets(self, tiny_model: SelfImprovingLLM):
        """k_proj and out_proj should remain plain nn.Linear by default."""
        inject_lora(tiny_model, rank=4, alpha=8.0)

        for block in tiny_model.blocks:
            assert isinstance(block.attn.attn.k_proj, nn.Linear)
            assert not isinstance(block.attn.attn.k_proj, LinearWithLoRA)
            assert isinstance(block.attn.attn.out_proj, nn.Linear)
            assert not isinstance(block.attn.attn.out_proj, LinearWithLoRA)

    def test_custom_targets(self, tiny_model: SelfImprovingLLM):
        """Custom target list should be respected."""
        inject_lora(
            tiny_model, target_modules=["q_proj", "k_proj"], rank=4, alpha=8.0
        )

        for block in tiny_model.blocks:
            assert isinstance(block.attn.attn.q_proj, LinearWithLoRA)
            assert isinstance(block.attn.attn.k_proj, LinearWithLoRA)
            # v_proj was NOT targeted
            assert not isinstance(block.attn.attn.v_proj, LinearWithLoRA)

    def test_no_double_injection(self, tiny_model: SelfImprovingLLM):
        """Calling inject_lora twice should not double-wrap layers."""
        inject_lora(tiny_model, rank=4, alpha=8.0)
        # Second injection should be a no-op because of the
        # ``not isinstance(child, LinearWithLoRA)`` guard
        inject_lora(tiny_model, rank=4, alpha=8.0)

        for block in tiny_model.blocks:
            assert isinstance(block.attn.attn.q_proj, LinearWithLoRA)
            # Should still be the first wrapper, not nested
            assert isinstance(
                block.attn.attn.q_proj.base_layer, nn.Linear
            )
            assert not isinstance(
                block.attn.attn.q_proj.base_layer, LinearWithLoRA
            )

    def test_returns_same_model(self, tiny_model: SelfImprovingLLM):
        """inject_lora should return the same model object."""
        result = inject_lora(tiny_model, rank=4, alpha=8.0)
        assert result is tiny_model


# --------------------------------------------------------------------------- #
# Parameter freezing / trainability tests
# --------------------------------------------------------------------------- #

class TestParameterFreezing:
    """Tests that verify base params are frozen and LoRA params are trainable."""

    def test_base_frozen_after_inject(self, tiny_model: SelfImprovingLLM):
        """Base model parameters should be frozen after inject_lora."""
        inject_lora(tiny_model, rank=4, alpha=8.0)

        # Count frozen base params (non-LoRA)
        for name, param in tiny_model.named_parameters():
            if "lora_" not in name:
                assert not param.requires_grad, (
                    f"Base parameter {name} should be frozen"
                )

    def test_lora_trainable_after_inject(self, tiny_model: SelfImprovingLLM):
        """LoRA parameters should be trainable after inject_lora."""
        inject_lora(tiny_model, rank=4, alpha=8.0)

        lora_params = get_lora_parameters(tiny_model)
        assert len(lora_params) > 0
        for param in lora_params:
            assert param.requires_grad

    def test_only_lora_trainable(self, tiny_model: SelfImprovingLLM):
        """Only LoRA params should require gradients."""
        inject_lora(tiny_model, rank=4, alpha=8.0)

        trainable_names = [
            name
            for name, param in tiny_model.named_parameters()
            if param.requires_grad
        ]
        assert all("lora_" in name for name in trainable_names), (
            f"Non-LoRA param trainable: "
            f"{[n for n in trainable_names if 'lora_' not in n]}"
        )


# --------------------------------------------------------------------------- #
# count_lora_parameters tests
# --------------------------------------------------------------------------- #

class TestCountLoRAParameters:
    """Tests for :func:`count_lora_parameters`."""

    def test_small_lora_percentage(self, tiny_model: SelfImprovingLLM):
        """LoRA params should be a small fraction of total params."""
        inject_lora(tiny_model, rank=4, alpha=8.0)
        counts = count_lora_parameters(tiny_model)

        assert counts["lora"] > 0
        assert counts["total"] > 0
        assert counts["lora_pct"] < 10.0, (
            f"LoRA should be <10% of total, got {counts['lora_pct']:.2f}%"
        )

    def test_trainable_equals_lora(self, tiny_model: SelfImprovingLLM):
        """All trainable params should be LoRA params."""
        inject_lora(tiny_model, rank=4, alpha=8.0)
        counts = count_lora_parameters(tiny_model)

        assert counts["trainable"] == counts["lora"]

    def test_frozen_equals_total_minus_lora(self, tiny_model: SelfImprovingLLM):
        """Frozen params = total - LoRA params."""
        inject_lora(tiny_model, rank=4, alpha=8.0)
        counts = count_lora_parameters(tiny_model)

        assert counts["frozen"] == counts["total"] - counts["lora"]


# --------------------------------------------------------------------------- #
# save / load LoRA weights tests
# --------------------------------------------------------------------------- #

class TestSaveLoadLoRA:
    """Tests for :func:`save_lora_weights` and :func:`load_lora_weights`."""

    def test_save_only_lora_params(self, tiny_model: SelfImprovingLLM):
        """save_lora_weights should write a file containing only LoRA keys."""
        inject_lora(tiny_model, rank=4, alpha=8.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lora_weights.pt")
            save_lora_weights(tiny_model, path)

            assert os.path.exists(path)
            state = torch.load(path, map_location="cpu", weights_only=True)
            # Every key should contain "lora_"
            assert all("lora_" in k for k in state.keys())
            # Should have some keys
            assert len(state) > 0

    def test_checkpoint_small(self, tiny_model: SelfImprovingLLM):
        """The LoRA checkpoint should be much smaller than full model."""
        inject_lora(tiny_model, rank=4, alpha=8.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = os.path.join(tmpdir, "lora.pt")
            full_path = os.path.join(tmpdir, "full.pt")

            save_lora_weights(tiny_model, lora_path)
            torch.save(tiny_model.state_dict(), full_path)

            lora_size = os.path.getsize(lora_path)
            full_size = os.path.getsize(full_path)
            # LoRA checkpoint should be < 20% of full model
            assert lora_size < full_size * 0.2, (
                f"LoRA checkpoint ({lora_size}) not much smaller "
                f"than full ({full_size})"
            )

    def test_load_roundtrip(self, tiny_model: SelfImprovingLLM):
        """Saving and loading should restore the same LoRA weights."""
        inject_lora(tiny_model, rank=4, alpha=8.0)

        # Mutate LoRA weights to non-zero values
        for param in get_lora_parameters(tiny_model):
            param.data = torch.randn_like(param.data)

        # Capture LoRA state before saving
        lora_state_before = {
            k: v.clone()
            for k, v in tiny_model.state_dict().items()
            if "lora_" in k
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lora.pt")
            save_lora_weights(tiny_model, path)

            # Reset LoRA weights to zero
            for param in get_lora_parameters(tiny_model):
                param.data.zero_()

            # Load back
            load_lora_weights(tiny_model, path)

            lora_state_after = {
                k: v.clone()
                for k, v in tiny_model.state_dict().items()
                if "lora_" in k
            }

            for key in lora_state_before:
                assert torch.allclose(
                    lora_state_before[key], lora_state_after[key]
                ), f"Mismatch for {key} after load"


# --------------------------------------------------------------------------- #
# merge_lora_weights tests
# --------------------------------------------------------------------------- #

class TestMergeLoRA:
    """Tests for :func:`merge_lora_weights`."""

    def test_lora_removed_after_merge(self, tiny_model: SelfImprovingLLM):
        """After merging, no LinearWithLoRA wrappers should remain."""
        inject_lora(tiny_model, rank=4, alpha=8.0)
        merge_lora_weights(tiny_model)

        def _check_no_lora(module: nn.Module) -> None:
            for child in module.children():
                assert not isinstance(child, LinearWithLoRA)
                _check_no_lora(child)

        _check_no_lora(tiny_model)

    def test_base_unfrozen_after_merge(self, tiny_model: SelfImprovingLLM):
        """After merging, base parameters should be trainable again."""
        inject_lora(tiny_model, rank=4, alpha=8.0)
        merge_lora_weights(tiny_model)

        # At least some parameters should now be trainable (the merged ones)
        trainable_count = sum(
            1 for p in tiny_model.parameters() if p.requires_grad
        )
        assert trainable_count > 0

    def test_forward_equivalent_before_after_merge(
        self, tiny_model: SelfImprovingLLM, tiny_config: ModelConfig
    ):
        """Output should be (nearly) identical before and after merging."""
        torch.manual_seed(123)
        inject_lora(tiny_model, rank=4, alpha=8.0)

        # Set LoRA weights to non-zero so there's something to merge
        for block in tiny_model.blocks:
            if isinstance(block.attn.attn.q_proj, LinearWithLoRA):
                nn.init.normal_(block.attn.attn.q_proj.lora.lora_A, mean=0.0, std=0.1)
                nn.init.normal_(block.attn.attn.q_proj.lora.lora_B, mean=0.0, std=0.1)
            if isinstance(block.attn.attn.v_proj, LinearWithLoRA):
                nn.init.normal_(block.attn.attn.v_proj.lora.lora_A, mean=0.0, std=0.1)
                nn.init.normal_(block.attn.attn.v_proj.lora.lora_B, mean=0.0, std=0.1)

        tiny_model.eval()
        x = torch.randint(0, tiny_config.vocab_size, (1, 8))

        with torch.no_grad():
            out_before = tiny_model(x)["logits"]

        merge_lora_weights(tiny_model)
        tiny_model.eval()

        with torch.no_grad():
            out_after = tiny_model(x)["logits"]

        assert torch.allclose(out_before, out_after, atol=1e-5), (
            "Outputs differ before/after merge beyond tolerance"
        )


# --------------------------------------------------------------------------- #
# unmerge_lora_weights tests
# --------------------------------------------------------------------------- #

class TestUnmergeLoRA:
    """Tests for :func:`unmerge_lora_weights` (currently a no-op)."""

    def test_returns_same_model(self, tiny_model: SelfImprovingLLM):
        """unmerge_lora_weights should return the same model object."""
        inject_lora(tiny_model, rank=4, alpha=8.0)
        result = unmerge_lora_weights(tiny_model)
        assert result is tiny_model


# --------------------------------------------------------------------------- #
# Integration tests
# --------------------------------------------------------------------------- #

class TestLoRAIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self, tiny_model: SelfImprovingLLM, tiny_config: ModelConfig):
        """Inject -> forward -> save -> load -> merge pipeline."""
        torch.manual_seed(456)

        # 1. Save base weights BEFORE injecting LoRA
        base_state = {k: v.clone() for k, v in tiny_model.state_dict().items()}

        # 2. Inject LoRA and set non-zero LoRA weights
        inject_lora(tiny_model, rank=4, alpha=8.0)
        for block in tiny_model.blocks:
            if isinstance(block.attn.attn.q_proj, LinearWithLoRA):
                nn.init.normal_(block.attn.attn.q_proj.lora.lora_A, mean=0.0, std=0.1)
                nn.init.normal_(block.attn.attn.q_proj.lora.lora_B, mean=0.0, std=0.1)
            if isinstance(block.attn.attn.v_proj, LinearWithLoRA):
                nn.init.normal_(block.attn.attn.v_proj.lora.lora_A, mean=0.0, std=0.1)
                nn.init.normal_(block.attn.attn.v_proj.lora.lora_B, mean=0.0, std=0.1)

        # 3. Forward pass
        tiny_model.eval()
        x = torch.randint(0, tiny_config.vocab_size, (1, 8))
        with torch.no_grad():
            out1 = tiny_model(x)["logits"]

        # 4. Save / load roundtrip
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lora.pt")
            save_lora_weights(tiny_model, path)

            # Create fresh model with SAME base weights, inject LoRA, load weights
            fresh_model = SelfImprovingLLM(tiny_config)
            fresh_model.load_state_dict(base_state, strict=True)
            inject_lora(fresh_model, rank=4, alpha=8.0)
            load_lora_weights(fresh_model, path)

            fresh_model.eval()
            with torch.no_grad():
                out2 = fresh_model(x)["logits"]

            assert torch.allclose(out1, out2, atol=1e-6), (
                "Outputs differ after save/load roundtrip"
            )

            # 5. Merge and verify equivalent output
            merge_lora_weights(fresh_model)
            fresh_model.eval()
            with torch.no_grad():
                out3 = fresh_model(x)["logits"]

            assert torch.allclose(out1, out3, atol=1e-5), (
                "Outputs differ after merge"
            )

    def test_gradient_flow(self, tiny_model: SelfImprovingLLM, tiny_config: ModelConfig):
        """Gradients should only flow through LoRA parameters."""
        inject_lora(tiny_model, rank=4, alpha=8.0)
        tiny_model.train()

        # Create non-zero LoRA weights so gradients are non-trivial
        for block in tiny_model.blocks:
            if isinstance(block.attn.attn.q_proj, LinearWithLoRA):
                nn.init.normal_(block.attn.attn.q_proj.lora.lora_A, mean=0.0, std=0.1)
                nn.init.normal_(block.attn.attn.q_proj.lora.lora_B, mean=0.0, std=0.1)
            if isinstance(block.attn.attn.v_proj, LinearWithLoRA):
                nn.init.normal_(block.attn.attn.v_proj.lora.lora_A, mean=0.0, std=0.1)
                nn.init.normal_(block.attn.attn.v_proj.lora.lora_B, mean=0.0, std=0.1)

        x = torch.randint(0, tiny_config.vocab_size, (1, 8))
        targets = torch.randint(0, tiny_config.vocab_size, (1, 8))

        out = tiny_model(x, targets=targets)
        loss = out["loss"]
        loss.backward()

        # LoRA parameters should have non-None gradients
        for param in get_lora_parameters(tiny_model):
            assert param.grad is not None, (
                f"LoRA param {param.shape} has no gradient"
            )

        # Base parameters should NOT have gradients
        for name, param in tiny_model.named_parameters():
            if "lora_" not in name:
                assert param.grad is None, (
                    f"Base parameter {name} should not have gradient"
                )
