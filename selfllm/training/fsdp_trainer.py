"""Fully Sharded Data Parallel (FSDP) training for large models.

FSDP shards model parameters, gradients, and optimizer states across GPUs.
Each GPU holds only a fraction of the model at any time.  Communication
for sharding is overlapped with computation via BACKWARD_PRE prefetch.

Usage (single-node multi-GPU)::

    python -m torch.distributed.run --nproc_per_node=4 train_fsdp.py

Usage (multi-node)::

    torchrun --nnodes=2 --nproc_per_node=8 --master_addr=... train_fsdp.py

References:
    - https://pytorch.org/docs/stable/fsdp.html
    - https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
"""

import logging
import math
import os
import time
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from selfllm.model.layers import TransformerBlock
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.utils import AverageMeter, get_lr_scheduler

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# FSDP configuration dataclass
# --------------------------------------------------------------------------- #


class FSDPConfig:
    """Immutable configuration knobs for FSDP wrapping.

    Attributes:
        sharding_strategy: FULL_SHARD (default), SHARD_GRAD_OP, NO_SHARD, etc.
        mixed_precision: Custom MixedPrecision policy or None.
        backward_prefetch: BACKWARD_PRE (default) or BACKWARD_POST.
        cpu_offload: CPUOffload instance for offloading params to CPU.
        limit_all_gathers: Rate-limit communication to reduce contention.
        device_id: The CUDA device to bind this rank to.
    """

    def __init__(
        self,
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
        mixed_precision: Optional[MixedPrecision] = None,
        backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE,
        cpu_offload: Optional[CPUOffload] = None,
        limit_all_gathers: bool = True,
        device_id: Optional[torch.device] = None,
    ) -> None:
        self.sharding_strategy = sharding_strategy
        self.mixed_precision = mixed_precision
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload if cpu_offload is not None else CPUOffload(offload_params=False)
        self.limit_all_gathers = limit_all_gathers
        self.device_id = device_id

    @staticmethod
    def default() -> "FSDPConfig":
        """Return sensible defaults for training on modern GPUs."""
        return FSDPConfig()


# --------------------------------------------------------------------------- #
# FSDPTrainer
# --------------------------------------------------------------------------- #


class FSDPTrainer:
    """FSDP training loop for SelfLLM.

    This trainer mirrors :class:`LLMTrainer` but distributes the model across
    all available GPUs via PyTorch FSDP.  Only the process with ``rank == 0``
    logs metrics, runs validation, and writes checkpoints to disk.

    Features:
        - FSDP with TransformerBlock auto-wrap policy
        - bf16 / fp16 mixed precision (bf16 preferred on Ampere+)
        - Gradient accumulation across distributed workers
        - Cosine LR schedule with linear warmup
        - Gradient clipping
        - LoRA-aware optimizer (only trains LoRA params when present)
        - Full state-dict checkpoint saving (rank 0 only)
        - Graceful single-GPU fallback (world_size=1)
    """

    def __init__(
        self,
        model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        config: Optional[Dict[str, Any]] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.1,
        device: str = "cuda",
        fsdp_config: Optional[FSDPConfig] = None,
    ) -> None:
        """Initialise the FSDP trainer.

        Args:
            model: The SelfImprovingLLM instance to distribute and train.
            tokenizer: Tokeniser for encoding/decoding text.
            config: Optional training configuration dict.  Keys that overlap
                with explicit arguments are ignored.
            learning_rate: Peak learning rate after warmup.
            weight_decay: Weight decay coefficient for AdamW.
            batch_size: Per-device micro-batch size.
            gradient_accumulation_steps: Number of forward/backward passes
                before an optimizer step.
            max_grad_norm: Maximum gradient norm for clipping.
            warmup_ratio: Fraction of total steps used for linear warmup.
            device: Device string (``"cuda"`` or ``"cpu"``).
            fsdp_config: Custom FSDP configuration.  ``None`` uses defaults.
        """
        self.tokenizer = tokenizer
        self.config = config or {}

        # ------------------------------------------------------------------
        # Distributed setup
        # ------------------------------------------------------------------
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0

        self.is_main_process = self.rank == 0

        # ------------------------------------------------------------------
        # Training hyperparameters
        # ------------------------------------------------------------------
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio

        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        # Mixed precision: bf16 when available (no gradient scaling needed)
        self._use_amp = torch.cuda.is_available()
        self._use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        self._amp_dtype = torch.bfloat16 if self._use_bf16 else torch.float16

        self.fsdp_config = fsdp_config or FSDPConfig.default()
        # Override device_id in the config
        self.fsdp_config.device_id = self.device

        # ------------------------------------------------------------------
        # Setup FSDP model
        # ------------------------------------------------------------------
        self.model = self._setup_fsdp(model)

        # ------------------------------------------------------------------
        # Optimiser (LoRA-aware)
        # ------------------------------------------------------------------
        self.optimizer = self._create_optimizer()

        # GradScaler only needed for fp16; bf16 does not require scaling
        self.scaler: Optional[GradScaler] = GradScaler() if (torch.cuda.is_available() and not self._use_bf16) else None

        # ------------------------------------------------------------------
        # Training state
        # ------------------------------------------------------------------
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_metric = float("inf")
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_perplexity": [],
            "learning_rate": [],
            "tokens_per_sec": [],
        }

    # ------------------------------------------------------------------ #
    # FSDP setup
    # ------------------------------------------------------------------ #

    def _setup_fsdp(self, model: SelfImprovingLLM) -> FSDP:
        """Wrap *model* with FSDP using the TransformerBlock policy.

        If torch.distributed has not been initialised and this is a single
        process, a dummy "gloo" process group is created automatically so
        that FSDP can be used for local testing.

        Returns:
            The FSDP-wrapped model already placed on the correct device.
        """
        # Ensure a process group exists -- create a dummy one for single-process
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            # Use a random available port to avoid collisions in test suites
            if "MASTER_PORT" not in os.environ:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", 0))
                    _, port = s.getsockname()
                    os.environ["MASTER_PORT"] = str(port)
            dist.init_process_group(
                backend="gloo",
                rank=0,
                world_size=1,
            )
            self.world_size = 1
            self.rank = 0
            self.is_main_process = True

        # Build mixed-precision policy.
        # When the user supplies a custom policy via fsdp_config we honour it.
        # Otherwise mixed precision is only enabled on CUDA (CPU lacks fp16/bf16
        # compute kernels) and we default to bf16 on Ampere+ and fp16 elsewhere.
        if self.fsdp_config.mixed_precision is not None:
            mp_policy = self.fsdp_config.mixed_precision
        elif self._use_bf16:
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif self._use_amp:
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:
            mp_policy = None  # full precision on CPU

        model = model.to(self.device)

        fsdp_model = FSDP(
            model,
            device_id=self.device,
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={TransformerBlock},
            ),
            mixed_precision=mp_policy,
            sharding_strategy=self.fsdp_config.sharding_strategy,
            backward_prefetch=self.fsdp_config.backward_prefetch,
            cpu_offload=self.fsdp_config.cpu_offload,
            limit_all_gathers=self.fsdp_config.limit_all_gathers,
            use_orig_params=True,
        )

        if self.is_main_process:
            logger.info(
                f"FSDP setup complete -- world_size={self.world_size}, "
                f"rank={self.rank}, local_rank={self.local_rank}"
            )
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            logger.info(
                f"Model: {total_params:,} parameters "
                f"({total_params / 1e6:.1f}M) | "
                f"trainable: {trainable_params:,} ({trainable_params / 1e6:.1f}M)"
            )
            logger.info(f"Mixed precision: {mp_policy.param_dtype if mp_policy else 'full (fp32)'}")
            logger.info(f"Sharding strategy: {self.fsdp_config.sharding_strategy.name}")

        return fsdp_model

    # ------------------------------------------------------------------ #
    # Optimiser
    # ------------------------------------------------------------------ #

    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimiser, optimising only LoRA params when present."""
        lora_params: List[torch.nn.Parameter] = []
        try:
            from selfllm.model.lora import get_lora_parameters

            lora_params = get_lora_parameters(self.model)
            if lora_params and self.is_main_process:
                logger.info(
                    f"Optimising {len(lora_params)} LoRA parameter groups"
                )
        except ImportError:
            pass  # LoRA module not available -- fall back to all trainable params

        if lora_params:
            params_to_optimize = lora_params
        else:
            # Separate decay / no-decay groups (mirror LLMTrainer logic)
            decay_params = []
            no_decay_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "bias" in name or "norm" in name or "ln" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            params_to_optimize = [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]

        return AdamW(
            params_to_optimize,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def _get_dataloader(
        self, dataset: Dataset, shuffle: bool = True
    ) -> DataLoader:
        """Create a :class:`DataLoader` with an optional distributed sampler."""
        sampler: Optional[DistributedSampler] = None
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            shuffle=(shuffle and sampler is None),
        )

    # ------------------------------------------------------------------ #
    # Loss computation
    # ------------------------------------------------------------------ #

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute cross-entropy loss for a single batch."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        outputs = self.model(token_ids=input_ids, targets=labels)
        return outputs["loss"]

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate(self, val_dataset: Dataset) -> Dict[str, float]:
        """Run validation and return aggregated metrics.

        Only rank 0 returns meaningful values; other ranks return zeroed
        dicts so that all processes stay in sync.
        """
        val_loader = self._get_dataloader(val_dataset, shuffle=False)

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        pbar = tqdm(
            val_loader,
            desc="Validating",
            leave=False,
            disable=not self.is_main_process,
        )

        with torch.no_grad():
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(token_ids=input_ids, targets=labels)
                loss = outputs["loss"]

                # Weight by non-masked token count for accurate perplexity
                mask = labels != -100
                num_tokens = mask.sum().item()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                num_batches += 1

                if self.is_main_process:
                    pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        # Synchronise across ranks
        if self.world_size > 1:
            local_stats = torch.tensor(
                [total_loss, total_tokens, float(num_batches)],
                device=self.device,
            )
            dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
            total_loss = local_stats[0].item()
            total_tokens = local_stats[1].item()
            num_batches = int(local_stats[2].item())

        self.model.train()

        if total_tokens == 0:
            return {"loss": float("inf"), "perplexity": float("inf")}

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        return {"loss": avg_loss, "perplexity": perplexity}

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        num_epochs: int = 10,
    ) -> Dict[str, List[float]]:
        """Run the FSDP distributed training loop.

        Args:
            train_dataset: PyTorch Dataset for training.
            val_dataset: Optional PyTorch Dataset for validation.
            num_epochs: Number of epochs to train.

        Returns:
            History dict with training metrics per epoch.
        """
        train_loader = self._get_dataloader(train_dataset, shuffle=True)

        # Scheduler
        total_steps = (
            len(train_loader)
            // self.gradient_accumulation_steps
            * num_epochs
        )
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_lr_scheduler(self.optimizer, warmup_steps, total_steps)

        if self.is_main_process:
            logger.info(f"Training for {num_epochs} epochs")
            logger.info(f"Steps per epoch: {len(train_loader)}")
            logger.info(f"Total training steps: {total_steps}")
            logger.info(f"Warmup steps: {warmup_steps}")
            logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps}")
            logger.info(f"Global batch size: {self.batch_size * self.gradient_accumulation_steps * self.world_size}")
            logger.info(f"Mixed precision: {self._amp_dtype}")
            logger.info(f"Gradient scaler: {'enabled' if self.scaler is not None else 'disabled (bf16)'}")

        self.global_step = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            epoch_loss = self._train_epoch(
                train_loader, scheduler, epoch, num_epochs
            )
            self.history["train_loss"].append(epoch_loss)

            # Validation
            if val_dataset is not None:
                val_metrics = self._validate(val_dataset)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_perplexity"].append(val_metrics["perplexity"])

                if self.is_main_process:
                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs} -- "
                        f"train_loss: {epoch_loss:.4f}, "
                        f"val_loss: {val_metrics['loss']:.4f}, "
                        f"val_ppl: {val_metrics['perplexity']:.2f}"
                    )

                    # Save best checkpoint
                    if val_metrics["perplexity"] < self.best_val_metric:
                        self.best_val_metric = val_metrics["perplexity"]
                        logger.info(
                            f"New best model! Val perplexity: {self.best_val_metric:.2f}"
                        )
            else:
                if self.is_main_process:
                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs} -- train_loss: {epoch_loss:.4f}"
                    )

        if self.is_main_process:
            logger.info("Training complete!")

        return self.history

    def _train_epoch(
        self,
        dataloader: DataLoader,
        scheduler: "torch.optim.lr_scheduler.LambdaLR",
        epoch: int,
        num_epochs: int,
    ) -> float:
        """Train for a single epoch.

        Args:
            dataloader: Training DataLoader.
            scheduler: LR scheduler (stepped every gradient accumulation boundary).
            epoch: Current epoch index (0-based).
            num_epochs: Total number of epochs.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()

        loss_meter = AverageMeter()
        lr_meter = AverageMeter()
        tok_meter = AverageMeter()

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
            disable=not self.is_main_process,
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            step_start = time.time()

            # Forward + backward with optional AMP
            if self.scaler is not None:
                with autocast(dtype=self._amp_dtype):
                    loss = self._compute_loss(batch)
                    loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            elif self._use_amp:
                # bf16 autocast (no gradient scaling needed)
                with autocast(dtype=self._amp_dtype):
                    loss = self._compute_loss(batch)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            else:
                # Full precision (CPU fallback)
                loss = self._compute_loss(batch)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

            loss_meter.update(loss.item() * self.gradient_accumulation_steps)

            # Gradient accumulation step
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                scheduler.step()

                self.global_step += 1
                lr_meter.update(scheduler.get_last_lr()[0])

            # Throughput
            batch_tokens = batch["input_ids"].numel()
            step_time = time.time() - step_start
            tokens_per_sec = batch_tokens / step_time if step_time > 0 else 0
            tok_meter.update(tokens_per_sec)

            # Update progress bar
            if self.is_main_process:
                pbar.set_postfix(
                    {
                        "loss": f"{loss_meter.avg:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        "tok/s": f"{tok_meter.avg:.0f}",
                    }
                )

        self.history["learning_rate"].append(lr_meter.avg)
        self.history["tokens_per_sec"].append(tok_meter.avg)

        return loss_meter.avg

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path: str) -> None:
        """Save a full-model checkpoint.  Only rank 0 writes to disk.

        The state dict is gathered from all shards using
        ``FullStateDictConfig(offload_to_cpu=True, rank0_only=True)`` so
        that only rank 0 materialises the full tensor in CPU memory.

        Args:
            path: Destination file path.  Parent directories are created
                automatically.
        """
        FSDP.set_state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            ),
        )

        state_dict = self.model.state_dict()

        if self.is_main_process:
            os.makedirs(
                os.path.dirname(path) if os.path.dirname(path) else ".",
                exist_ok=True,
            )
            checkpoint = {
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "best_val_metric": self.best_val_metric,
                "history": self.history,
            }
            if self.scaler is not None:
                checkpoint["scaler_state_dict"] = self.scaler.state_dict()

            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint into the FSDP model (rank 0 loads, broadcast).

        Args:
            path: Path to the checkpoint file created by :meth:`save_checkpoint`.
        """
        FSDP.set_state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            ),
        )

        if self.is_main_process:
            checkpoint = torch.load(path, map_location="cpu")
            logger.info(
                f"Loading checkpoint from {path} "
                f"(epoch {checkpoint.get('current_epoch', '?')}, "
                f"step {checkpoint.get('global_step', '?')})"
            )
        else:
            checkpoint = {}

        # Load model state -- FSDP handles the broadcast
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Optimiser and other state are only meaningful on rank 0
        if self.is_main_process:
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.global_step = checkpoint.get("global_step", 0)
            self.current_epoch = checkpoint.get("current_epoch", 0)
            self.best_val_metric = checkpoint.get("best_val_metric", float("inf"))
            self.history = checkpoint.get(
                "history",
                {
                    "train_loss": [],
                    "val_loss": [],
                    "val_perplexity": [],
                    "learning_rate": [],
                    "tokens_per_sec": [],
                },
            )
            if self.scaler is not None and "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # ------------------------------------------------------------------ #
    # Distributed process management (static helpers)
    # ------------------------------------------------------------------ #

    @staticmethod
    def setup_distributed(
        backend: Optional[str] = None,
        timeout_seconds: int = 1800,
    ) -> None:
        """Initialise the torch.distributed process group.

        Should be called **before** constructing the trainer.  Reads
        environment variables set by ``torchrun`` or ``torch.distributed.run``:

        - ``RANK``
        - ``WORLD_SIZE``
        - ``LOCAL_RANK``
        - ``MASTER_ADDR``
        - ``MASTER_PORT``

        Args:
            backend: ``"nccl"`` for GPU, ``"gloo"`` for CPU.  Auto-detected
                when *None*.
            timeout_seconds: Process group timeout in seconds.
        """
        if dist.is_initialized():
            return  # Already initialised

        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            rank = 0
            world_size = 1
            local_rank = 0

        if world_size <= 1:
            return  # Single-process -- no need for distributed setup

        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        logger.info(
            f"Distributed initialised -- rank={rank}, world_size={world_size}, "
            f"backend={backend}"
        )

    @staticmethod
    def cleanup_distributed() -> None:
        """Destroy the torch.distributed process group.

        Should be called **after** training completes.
        """
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")

    @staticmethod
    def setup_process_group() -> None:
        """Alias for :meth:`setup_distributed` (SPEC compatibility)."""
        FSDPTrainer.setup_distributed()

    @staticmethod
    def cleanup_process_group() -> None:
        """Alias for :meth:`cleanup_distributed` (SPEC compatibility)."""
        FSDPTrainer.cleanup_distributed()
