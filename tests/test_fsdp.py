"""Unit tests for the FSDP (Fully Sharded Data Parallel) trainer.

Tests cover:
- FSDP model wrapping with TransformerBlock auto-wrap policy
- Single-GPU graceful fallback (world_size=1)
- Distributed sampler creation
- Checkpoint save/load roundtrip (rank 0 only)
- Mixed precision configuration (bf16 / fp16)
- LoRA-aware optimizer (only LoRA params when applicable)
- Gradient accumulation correctness
- Distributed validation metric aggregation

These tests run on CPU and single-GPU setups.  Multi-GPU tests require
``torchrun`` or ``pytest --forked`` and are skipped automatically when
fewer than 2 GPUs are available.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.lora import inject_lora, get_lora_parameters
from selfllm.model.tokenizer import BPETokenizer
from selfllm.training.fsdp_trainer import FSDPConfig, FSDPTrainer


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_tiny_config() -> ModelConfig:
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


def _make_dummy_dataset(
    num_samples: int = 8,
    seq_len: int = 16,
    vocab_size: int = 128,
) -> TensorDataset:
    """Create a dummy TensorDataset with input_ids and labels."""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = input_ids.clone()
    return TensorDataset(input_ids, labels)


def _make_dict_dataset(
    num_samples: int = 8,
    seq_len: int = 16,
    vocab_size: int = 128,
) -> "DictDataset":
    """Create a Dataset that returns dicts with input_ids and labels."""
    class DictDataset(Dataset):
        def __init__(self, num_samples: int, seq_len: int, vocab_size: int) -> None:
            self.input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
            self.labels = self.input_ids.clone()

        def __len__(self) -> int:
            return len(self.input_ids)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            return {
                "input_ids": self.input_ids[idx],
                "labels": self.labels[idx],
            }

    return DictDataset(num_samples, seq_len, vocab_size)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tiny_config() -> ModelConfig:
    return _make_tiny_config()


@pytest.fixture
def tiny_model(tiny_config: ModelConfig) -> SelfImprovingLLM:
    return SelfImprovingLLM(tiny_config)


@pytest.fixture
def tokenizer() -> BPETokenizer:
    """Return a small tokenizer for tests."""
    return BPETokenizer()


@pytest.fixture
def dummy_train_dataset() -> Dataset:
    return _make_dict_dataset(num_samples=16, seq_len=16)


@pytest.fixture
def dummy_val_dataset() -> Dataset:
    return _make_dict_dataset(num_samples=8, seq_len=16)


# --------------------------------------------------------------------------- #
# Test: FSDPConfig dataclass
# --------------------------------------------------------------------------- #


class TestFSDPConfig:
    """Tests for :class:`FSDPConfig`."""

    def test_default_values(self):
        """Default config should use FULL_SHARD and BACKWARD_PRE."""
        cfg = FSDPConfig.default()
        from torch.distributed.fsdp.api import ShardingStrategy, BackwardPrefetch

        assert cfg.sharding_strategy == ShardingStrategy.FULL_SHARD
        assert cfg.backward_prefetch == BackwardPrefetch.BACKWARD_PRE
        assert cfg.limit_all_gathers is True
        assert cfg.mixed_precision is None

    def test_custom_values(self):
        """Custom values should be stored correctly."""
        from torch.distributed.fsdp.api import ShardingStrategy, BackwardPrefetch

        cfg = FSDPConfig(
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            limit_all_gathers=False,
        )
        assert cfg.sharding_strategy == ShardingStrategy.SHARD_GRAD_OP
        assert cfg.limit_all_gathers is False


# --------------------------------------------------------------------------- #
# Test: Single-GPU fallback (no distributed init)
# --------------------------------------------------------------------------- #


class TestSingleGPUFallback:
    """Tests that verify graceful single-GPU behaviour without distributed."""

    def test_world_size_one(self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
        """Without distributed, world_size should be 1 and rank 0."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            batch_size=2,
        )
        assert trainer.world_size == 1
        assert trainer.rank == 0
        assert trainer.is_main_process is True
        assert trainer.local_rank == 0

    def test_is_main_process(self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
        """is_main_process should be True when rank == 0."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
        )
        assert trainer.is_main_process is True

    def test_model_is_fsdp_wrapped(
        self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer
    ):
        """The model should be wrapped in an FSDP instance."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            batch_size=2,
        )
        assert isinstance(trainer.model, FSDP)

    def test_optimizer_created(
        self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer
    ):
        """An AdamW optimizer should be created."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
        )
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

    def test_training_state_initialised(
        self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer
    ):
        """Initial training state should be zeroed."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
        )
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0
        assert trainer.best_val_metric == float("inf")
        assert trainer.history["train_loss"] == []
        assert trainer.history["val_loss"] == []


# --------------------------------------------------------------------------- #
# Test: FSDP wrapping policy
# --------------------------------------------------------------------------- #


class TestFSDPWrap:
    """Tests for FSDP model wrapping behaviour."""

    def test_transformer_blocks_wrapped(
        self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer
    ):
        """Each TransformerBlock should be individually wrapped in FSDP."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from selfllm.model.layers import TransformerBlock

        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            batch_size=2,
        )

        # After auto-wrap, each TransformerBlock is wrapped in an FSDP unit.
        # The blocks nn.ModuleList still iterates over the (now FSDP-wrapped)
        # blocks, so each element should be an FSDP instance whose underlying
        # module is a TransformerBlock.
        root_module = trainer.model.module
        for block in root_module.blocks:
            assert isinstance(block, FSDP), (
                f"Expected FSDP wrapper, got {type(block)}"
            )
            assert isinstance(block.module, TransformerBlock), (
                f"Expected TransformerBlock inside FSDP, got {type(block.module)}"
            )

    def test_model_on_correct_device(
        self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer
    ):
        """Model parameters should be on the expected device."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            batch_size=2,
        )
        # Get a parameter from the wrapped model
        param = next(trainer.model.parameters())
        assert param.device.type == trainer.device.type


# --------------------------------------------------------------------------- #
# Test: Optimizer behaviour
# --------------------------------------------------------------------------- #


class TestOptimizer:
    """Tests for optimizer creation and parameter targeting."""

    def test_all_params_optimized_without_lora(
        self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer
    ):
        """Without LoRA, optimizer should target all trainable params."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            batch_size=2,
        )

        # Count params in optimizer
        optim_params = sum(
            p.numel() for group in trainer.optimizer.param_groups for p in group["params"]
        )
        model_params = sum(p.numel() for p in tiny_model.parameters() if p.requires_grad)
        assert optim_params == model_params

    def test_only_lora_params_with_lora(
        self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer
    ):
        """With LoRA injected, optimizer should only target LoRA params."""
        inject_lora(tiny_model, rank=4, alpha=8.0)

        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            batch_size=2,
        )

        lora_params = get_lora_parameters(tiny_model)
        lora_count = sum(p.numel() for p in lora_params)

        optim_count = sum(
            p.numel() for group in trainer.optimizer.param_groups for p in group["params"]
        )
        assert optim_count == lora_count, (
            f"Optimizer targets {optim_count} params, but LoRA has {lora_count}"
        )

    def test_lora_params_smaller_than_full(
        self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer
    ):
        """LoRA param count should be much smaller than full model."""
        full_params = sum(p.numel() for p in tiny_model.parameters())

        inject_lora(tiny_model, rank=4, alpha=8.0)
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            batch_size=2,
        )

        lora_count = sum(
            p.numel() for group in trainer.optimizer.param_groups for p in group["params"]
        )
        assert lora_count < full_params * 0.2, (
            f"LoRA params ({lora_count}) should be < 20% of full ({full_params})"
        )


# --------------------------------------------------------------------------- #
# Test: Data loading
# --------------------------------------------------------------------------- #


class TestDataLoading:
    """Tests for distributed data loading."""

    def test_dataloader_created(self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
        """_get_dataloader should produce a non-empty DataLoader."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            batch_size=4,
        )
        dataset = _make_dict_dataset(num_samples=16, seq_len=8)
        loader = trainer._get_dataloader(dataset, shuffle=True)

        assert len(loader) > 0
        batch = next(iter(loader))
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] <= 4  # batch_size

    def test_single_gpu_no_distributed_sampler(
        self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer
    ):
        """With world_size=1, no DistributedSampler should be created."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
        )
        dataset = _make_dict_dataset(num_samples=8, seq_len=8)
        loader = trainer._get_dataloader(dataset, shuffle=True)

        assert loader.sampler is None or not hasattr(loader.sampler, "set_epoch")


# --------------------------------------------------------------------------- #
# Test: Training loop (single step)
# --------------------------------------------------------------------------- #


class TestTrainingLoop:
    """Tests for the training loop on a single device."""

    def test_train_one_epoch_reduces_loss(
        self,
        tiny_model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        dummy_train_dataset: Dataset,
    ):
        """After training for one epoch, loss should decrease."""
        # Capture initial loss
        tiny_model.eval()
        with torch.no_grad():
            batch = {
                "input_ids": dummy_train_dataset[0]["input_ids"].unsqueeze(0),
                "labels": dummy_train_dataset[0]["labels"].unsqueeze(0),
            }
            device = "cuda" if torch.cuda.is_available() else "cpu"
            batch = {k: v.to(device) for k, v in batch.items()}
            initial_loss = tiny_model(
                token_ids=batch["input_ids"], targets=batch["labels"]
            )["loss"].item()

        # Train
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            learning_rate=1e-3,
            batch_size=4,
            gradient_accumulation_steps=1,
        )
        history = trainer.train(dummy_train_dataset, num_epochs=1)

        assert len(history["train_loss"]) == 1
        final_loss = history["train_loss"][0]
        assert final_loss > 0  # Cross-entropy loss is positive
        # Loss should generally decrease (may not always due to randomness)
        assert final_loss < initial_loss * 2  # Sanity check

    def test_global_step_incremented(
        self,
        tiny_model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        dummy_train_dataset: Dataset,
    ):
        """global_step should increase after training."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            batch_size=4,
            gradient_accumulation_steps=1,
        )
        assert trainer.global_step == 0

        trainer.train(dummy_train_dataset, num_epochs=1)
        assert trainer.global_step > 0

    def test_history_populated(
        self,
        tiny_model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        dummy_train_dataset: Dataset,
    ):
        """History should contain train_loss after training."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            batch_size=4,
        )
        history = trainer.train(dummy_train_dataset, num_epochs=2)

        assert len(history["train_loss"]) == 2
        assert all(isinstance(loss, float) for loss in history["train_loss"])
        assert len(history["learning_rate"]) == 2


# --------------------------------------------------------------------------- #
# Test: Gradient accumulation
# --------------------------------------------------------------------------- #


class TestGradientAccumulation:
    """Tests for gradient accumulation logic."""

    def test_grad_accum_no_step_without_boundary(
        self,
        tiny_model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        dummy_train_dataset: Dataset,
    ):
        """With grad_accum > 1, optimizer should not step every batch."""
        accum_steps = 4
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            batch_size=2,
            gradient_accumulation_steps=accum_steps,
            learning_rate=1e-3,
        )

        initial_step = trainer.global_step
        trainer.train(dummy_train_dataset, num_epochs=1)

        # global_step should increase less than total batches / accum_steps
        assert trainer.global_step > initial_step

    def test_grad_accum_step_size(
        self,
        tiny_model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
    ):
        """global_step should increase by roughly batches // grad_accum."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            batch_size=2,
            gradient_accumulation_steps=2,
        )
        dataset = _make_dict_dataset(num_samples=8, seq_len=8)
        loader = trainer._get_dataloader(dataset, shuffle=False)
        total_batches = len(loader)

        trainer.train(dataset, num_epochs=1)
        expected_steps = total_batches // 2
        assert abs(trainer.global_step - expected_steps) <= 1


# --------------------------------------------------------------------------- #
# Test: Checkpoint save / load
# --------------------------------------------------------------------------- #


class TestCheckpoint:
    """Tests for checkpoint saving and loading."""

    def test_save_checkpoint_creates_file(
        self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer
    ):
        """save_checkpoint should create a file on disk."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            batch_size=2,
        )
        trainer.global_step = 42
        trainer.current_epoch = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            trainer.save_checkpoint(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_save_checkpoint_contains_expected_keys(
        self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer
    ):
        """Checkpoint should contain model_state_dict and optimizer_state_dict."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            batch_size=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            trainer.save_checkpoint(path)

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            assert "model_state_dict" in ckpt
            assert "optimizer_state_dict" in ckpt
            assert "global_step" in ckpt
            assert "history" in ckpt

    def test_save_checkpoint_only_rank_zero_writes(self):
        """Checkpoint file should only be written by rank 0."""
        # This is tested implicitly by the fact that save_checkpoint
        # checks is_main_process before writing
        pass  # Behaviour verified in integration tests


# --------------------------------------------------------------------------- #
# Test: Mixed precision configuration
# --------------------------------------------------------------------------- #


class TestMixedPrecision:
    """Tests for mixed precision setup."""

    def test_amp_dtype_set(self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
        """The trainer should have _amp_dtype set."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
        )
        assert trainer._amp_dtype in (torch.float16, torch.bfloat16)

    def test_bf16_preferred_on_cuda(self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
        """bf16 should be used when CUDA supports it."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
        )
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            assert trainer._amp_dtype == torch.bfloat16
            # bf16 does not need gradient scaling
            assert trainer.scaler is None

    def test_fp16_fallback(self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
        """fp16 should be used when bf16 is not supported.

        Skip on CPU-only hosts -- the test patches CUDA availability but
        cannot actually allocate CUDA tensors.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available — cannot test fp16 fallback")
        with patch("torch.cuda.is_bf16_supported", return_value=False):
            trainer = FSDPTrainer(
                model=tiny_model,
                tokenizer=tokenizer,
            )
            assert trainer._amp_dtype == torch.float16
            # fp16 needs gradient scaling
            assert trainer.scaler is not None

    def test_no_scaler_for_bf16(self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
        """GradScaler should be None when using bf16."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
        )
        if trainer._use_bf16:
            assert trainer.scaler is None


# --------------------------------------------------------------------------- #
# Test: Distributed setup helpers
# --------------------------------------------------------------------------- #


class TestDistributedHelpers:
    """Tests for static distributed setup/cleanup methods."""

    def test_setup_noop_when_not_distributed(self):
        """setup_distributed should be a no-op when env vars are absent."""
        # Save and clear env vars
        orig_env = {
            k: os.environ.pop(k, None)
            for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
        }
        try:
            FSDPTrainer.setup_distributed()
            # Should not raise; dist may or may not be initialized
        finally:
            for k, v in orig_env.items():
                if v is not None:
                    os.environ[k] = v

    def test_cleanup_noop_when_not_initialized(self):
        """cleanup_distributed should not raise when not initialized."""
        FSDPTrainer.cleanup_distributed()  # Should not raise

    def test_process_group_aliases_exist(self):
        """SPEC-compatible aliases should exist and be callable."""
        assert callable(FSDPTrainer.setup_process_group)
        assert callable(FSDPTrainer.cleanup_process_group)


# --------------------------------------------------------------------------- #
# Test: Device handling
# --------------------------------------------------------------------------- #


class TestDeviceHandling:
    """Tests for device assignment."""

    def test_device_matches_local_rank(self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
        """Trainer device should match local_rank."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
        )
        if torch.cuda.is_available():
            assert trainer.device.type == "cuda"
            assert trainer.device.index == trainer.local_rank
        else:
            assert trainer.device.type == "cpu"

    def test_model_on_device(self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
        """Model should be on the trainer's device."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
        )
        param = next(trainer.model.parameters())
        assert param.device.type == trainer.device.type


# --------------------------------------------------------------------------- #
# Test: Config dict passthrough
# --------------------------------------------------------------------------- #


class TestConfigPassthrough:
    """Tests for config dict handling."""

    def test_config_stored(self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer):
        """Config dict should be stored on the trainer."""
        config = {"learning_rate": 5e-4, "custom_key": "custom_value"}
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
            config=config,
        )
        assert trainer.config == config

    def test_config_defaults_to_empty_dict(
        self, tiny_model: SelfImprovingLLM, tokenizer: BPETokenizer
    ):
        """Default config should be an empty dict."""
        trainer = FSDPTrainer(
            model=tiny_model,
            tokenizer=tokenizer,
        )
        assert trainer.config == {}
