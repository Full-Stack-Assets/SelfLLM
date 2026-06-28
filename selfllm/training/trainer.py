"""Standard training loop for initial model training on seed corpus."""
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of steps for linear warmup.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine cycles (default: 0.5 for half cosine).
        min_lr_ratio: Minimum LR as a ratio of base LR at the end of training.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (
            1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
        )
        decayed = (1 - min_lr_ratio) * cosine_decay + min_lr_ratio
        return decayed

    return LambdaLR(optimizer, lr_lambda)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


class LLMTrainer:
    """Standard training loop for initial model training on seed corpus.

    Features:
        - AdamW optimizer with weight decay
        - Cosine learning rate schedule with linear warmup
        - Gradient accumulation for large effective batch sizes
        - Gradient clipping for training stability
        - Mixed precision training (torch.cuda.amp)
        - Validation loop with perplexity calculation
        - Checkpoint saving (best + periodic)
        - Progress bars and structured logging
    """

    def __init__(
        self,
        model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the trainer.

        Args:
            model: The SelfImprovingLLM to train.
            tokenizer: BPETokenizer for encoding/decoding.
            config: Training configuration dict with keys:
                - learning_rate (float): Default 5e-4
                - weight_decay (float): Default 0.01
                - warmup_ratio (float): Default 0.1
                - max_grad_norm (float): Default 1.0
                - gradient_accumulation_steps (int): Default 1
                - batch_size (int): Default 16
                - use_amp (bool): Default True
                - save_every (int): Save checkpoint every N epochs. Default 1
                - checkpoint_dir (str): Default "./checkpoints"
            device: Device to train on ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or {}

        # Extract config with defaults
        self.learning_rate = self.config.get("learning_rate", 5e-4)
        self.weight_decay = self.config.get("weight_decay", 0.01)
        self.warmup_ratio = self.config.get("warmup_ratio", 0.1)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps = self.config.get(
            "gradient_accumulation_steps", 1
        )
        self.batch_size = self.config.get("batch_size", 16)
        self.use_amp = self.config.get("use_amp", True) and device == "cuda"
        self.save_every = self.config.get("save_every", 1)
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Mixed precision scaler
        self.scaler: Optional[GradScaler] = GradScaler() if self.use_amp else None

        # Training state
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

    def _create_optimizer(self) -> AdamW:
        """Create the AdamW optimizer with weight decay.

        Returns:
            AdamW optimizer instance.
        """
        # Separate parameters that should/shouldn't get weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "ln" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(param_groups, lr=self.learning_rate)

    def _create_scheduler(
        self, optimizer: AdamW, num_training_steps: int
    ) -> LambdaLR:
        """Create the learning rate scheduler.

        Args:
            optimizer: The optimizer to schedule.
            num_training_steps: Total number of training steps.

        Returns:
            LambdaLR scheduler with warmup and cosine decay.
        """
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the loss for a batch.

        Args:
            batch: Dict with 'input_ids', 'attention_mask', 'labels'.

        Returns:
            Scalar loss tensor.
        """
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        outputs = self.model(token_ids=input_ids, targets=labels)
        return outputs["loss"]

    def _compute_perplexity(self, dataloader: DataLoader) -> float:
        """Compute average perplexity on a dataset.

        Args:
            dataloader: DataLoader for the evaluation dataset.

        Returns:
            Average perplexity (float).
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(token_ids=input_ids, targets=labels)
                loss = outputs["loss"]

                # Count non-ignored tokens
                mask = labels != -100
                num_tokens = mask.sum().item()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        if total_tokens == 0:
            return float("inf")

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        num_epochs: int = 10,
    ) -> Dict[str, List[float]]:
        """Run the standard training loop with validation.

        Args:
            train_dataset: PyTorch Dataset for training.
            val_dataset: Optional PyTorch Dataset for validation.
            num_epochs: Number of training epochs.

        Returns:
            History dict with training metrics per epoch.
        """
        import math  # Ensure math is available

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device == "cuda",
        )

        val_loader: Optional[DataLoader] = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=self.device == "cuda",
            )

        optimizer = self._create_optimizer()

        # Use ceil so the trailing (incomplete) accumulation window -- which
        # now produces an optimizer step -- is counted, and never let the
        # scheduler horizon collapse to 0 on tiny datasets.
        total_steps = max(
            1,
            math.ceil(len(train_loader) / self.gradient_accumulation_steps)
            * num_epochs,
        )
        scheduler = self._create_scheduler(optimizer, total_steps)

        logger.info(f"Training for {num_epochs} epochs")
        logger.info(f"Steps per epoch: {len(train_loader)}")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"AMP enabled: {self.use_amp}")
        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

        self.global_step = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch(
                train_loader, optimizer, scheduler, epoch, num_epochs
            )
            self.history["train_loss"].append(epoch_loss)

            # Validation
            if val_loader is not None:
                val_metrics = self._validate(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_perplexity"].append(val_metrics["perplexity"])

                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"train_loss: {epoch_loss:.4f}, "
                    f"val_loss: {val_metrics['loss']:.4f}, "
                    f"val_ppl: {val_metrics['perplexity']:.2f}"
                )

                # Save best checkpoint
                if val_metrics["perplexity"] < self.best_val_metric:
                    self.best_val_metric = val_metrics["perplexity"]
                    self._save_checkpoint(
                        os.path.join(self.checkpoint_dir, "best_model.pt"),
                        optimizer,
                        scheduler,
                        is_best=True,
                    )
                    logger.info(
                        f"New best model! Val perplexity: {self.best_val_metric:.2f}"
                    )
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - train_loss: {epoch_loss:.4f}"
                )

            # Periodic checkpoint
            if (epoch + 1) % self.save_every == 0:
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"
                )
                self._save_checkpoint(ckpt_path, optimizer, scheduler)
                logger.info(f"Saved checkpoint: {ckpt_path}")

        logger.info("Training complete!")
        return self.history

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: AdamW,
        scheduler: LambdaLR,
        epoch: int,
        num_epochs: int,
    ) -> float:
        """Train for a single epoch.

        Args:
            dataloader: Training data loader.
            optimizer: AdamW optimizer.
            scheduler: LR scheduler.
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
        )

        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            step_start = time.time()

            # Forward pass with optional AMP
            if self.use_amp and self.scaler is not None:
                with autocast():
                    loss = self._compute_loss(batch)
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
            else:
                loss = self._compute_loss(batch)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

            loss_meter.update(loss.item() * self.gradient_accumulation_steps)

            # Gradient accumulation step
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                if self.use_amp and self.scaler is not None:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()

                self.global_step += 1
                lr_meter.update(scheduler.get_last_lr()[0])

            # Throughput
            batch_tokens = batch["input_ids"].numel()
            step_time = time.time() - step_start
            tokens_per_sec = batch_tokens / step_time if step_time > 0 else 0
            tok_meter.update(tokens_per_sec)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss_meter.avg:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "tok/s": f"{tok_meter.avg:.0f}",
                }
            )

        # Flush a trailing incomplete accumulation window so the epoch's final
        # batches contribute an optimizer step instead of being discarded.
        if len(dataloader) > 0 and (step + 1) % self.gradient_accumulation_steps != 0:
            if self.use_amp and self.scaler is not None:
                self.scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            if self.use_amp and self.scaler is not None:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            self.global_step += 1
            lr_meter.update(scheduler.get_last_lr()[0])

        self.history["learning_rate"].append(lr_meter.avg)
        self.history["tokens_per_sec"].append(tok_meter.avg)

        return loss_meter.avg

    def _validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation and compute metrics.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dict with 'loss' and 'perplexity' keys.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Validating", leave=False)

        with torch.no_grad():
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(token_ids=input_ids, targets=labels)
                loss = outputs["loss"]

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = math.exp(avg_loss)

        return {"loss": avg_loss, "perplexity": perplexity}

    def _save_checkpoint(
        self,
        path: str,
        optimizer: AdamW,
        scheduler: LambdaLR,
        is_best: bool = False,
    ) -> None:
        """Save a training checkpoint.

        Args:
            path: File path to save to.
            optimizer: Optimizer state to save.
            scheduler: Scheduler state to save.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_metric": self.best_val_metric,
            "history": self.history,
            "config": self.config,
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(
        self, path: str, optimizer: Optional[AdamW] = None
    ) -> None:
        """Load a training checkpoint.

        Args:
            path: Path to the checkpoint file.
            optimizer: Optional optimizer to load state into.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
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

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from {path} "
                     f"(epoch {self.current_epoch}, step {self.global_step})")
