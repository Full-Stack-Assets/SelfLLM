"""Utility functions and classes for SelfLLM.

Provides:
- Random seed management for reproducibility
- Device detection (CUDA > CPU)
- Model parameter counting and number formatting
- JSON serialization helpers
- Learning rate scheduler factory
- Early stopping, metric averaging, and structured logging
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Random seed & device
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make cuDNN deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Return the best available compute device.

    Prefers CUDA, then Apple Silicon (MPS), then CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Model / number utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: A PyTorch module.

    Returns:
        Total count of parameters that require gradients.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_num(num: int) -> str:
    """Format a large integer into a human-readable string.

    Examples:
        >>> format_num(1_200_000)
        '1.2M'
        >>> format_num(350_000)
        '350K'
        >>> format_num(12_345_678)
        '12.3M'

    Args:
        num: Integer to format.

    Returns:
        Formatted string with K/M/B/T suffix.
    """
    if num < 0:
        return "-" + format_num(-num)
    for threshold, suffix in [
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ]:
        if num >= threshold:
            val = num / threshold
            # Show integer if whole number, else one decimal
            return f"{val:.1f}{suffix}" if val != int(val) else f"{int(val)}{suffix}"
    return str(num)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def save_json(data: Any, path: str) -> None:
    """Save an object to a JSON file.

    Args:
        data: Any JSON-serializable object.
        path: Destination file path (parent directories are created).
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    """Load a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The deserialized Python object.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Learning-rate scheduler
# ---------------------------------------------------------------------------

def get_lr_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Build a cosine learning-rate scheduler with linear warmup.

    The schedule follows the HuggingFace *get_cosine_schedule_with_warmup*
    pattern:

    1. **Warmup** phase: LR linearly increases from 0 to the initial LR
       over ``num_warmup_steps``.
    2. **Cosine decay** phase: LR decays following a cosine curve from the
       initial LR down to 0 over the remaining steps.

    Args:
        optimizer: The PyTorch optimizer whose LR will be scheduled.
        num_warmup_steps: Number of warmup steps (must be >= 0).
        num_training_steps: Total number of training steps.

    Returns:
        A :class:`torch.optim.lr_scheduler.LambdaLR` instance.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay after warmup
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Early stopping with patience.

    Monitors a validation metric and signals training to stop when the
    metric has not improved for a given number of evaluations.

    Args:
        patience: Number of evaluations with no improvement after which
            training will be stopped.
        min_delta: Minimum change in the monitored quantity to qualify as
            an improvement (must be positive).

    Example:
        >>> es = EarlyStopping(patience=3, min_delta=0.001)
        >>> for epoch in range(100):
        ...     score = evaluate(model)
        ...     if es(score):
        ...         print("Early stopping triggered!")
        ...         break
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0) -> None:
        self.patience: int = patience
        self.min_delta: float = min_delta
        self._best_score: Optional[float] = None
        self._counter: int = 0
        self._should_stop: bool = False

    def __call__(self, score: float) -> bool:
        """Evaluate a new score and decide whether to stop.

        Args:
            score: The validation metric to monitor. Higher is assumed
                to be better.

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        if self._best_score is None:
            self._best_score = score
            return False

        if score > self._best_score + self.min_delta:
            # Improvement seen
            self._best_score = score
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self._should_stop = True

        return self._should_stop

    @property
    def best_score(self) -> Optional[float]:
        """The best score observed so far."""
        return self._best_score


# ---------------------------------------------------------------------------
# Average meter
# ---------------------------------------------------------------------------

class AverageMeter:
    """Track a running average of a scalar metric.

    Useful for computing epoch- or batch-level averages on the fly during
    training loops.

    Example:
        >>> loss_meter = AverageMeter()
        >>> for batch in dataloader:
        ...     loss = compute_loss(batch)
        ...     loss_meter.update(loss.item(), n=len(batch))
        ... print(f"Average loss: {loss_meter.avg:.4f}")
    """

    def __init__(self) -> None:
        self._val: float = 0.0
        self._sum: float = 0.0
        self._count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update the meter with a new value.

        Args:
            val: The value to incorporate.
            n: The weight (number of samples) associated with *val*.
        """
        self._val = val
        self._sum += val * n
        self._count += n

    @property
    def avg(self) -> float:
        """The current average value."""
        return self._sum / max(self._count, 1)


# ---------------------------------------------------------------------------
# Structured logger
# ---------------------------------------------------------------------------

class Logger:
    """Structured logging with file and console output.

    Creates a standard-library :class:`logging.Logger` that writes INFO-level
    (and above) messages to both the console and a timestamped log file.

    Args:
        log_dir: Directory where log files will be stored.
        name: Logger name (also used as the log filename prefix).

    Example:
        >>> logger = Logger("./logs", name="training")
        >>> logger.info("Starting training...")
        >>> logger.log_metrics({"loss": 0.42, "acc": 0.91}, step=100)
    """

    def __init__(self, log_dir: str, name: str = "training") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.name = name

        # Create a uniquely named log file with a timestamp
        timestamp = _timestamp()
        self.log_file = self.log_dir / f"{name}_{timestamp}.log"

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()  # Avoid duplicate handlers on re-init

        # Console handler (INFO+)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(console_fmt)
        self._logger.addHandler(console_handler)

        # File handler (INFO+)
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        self._logger.addHandler(file_handler)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def info(self, msg: str) -> None:
        """Log an INFO-level message."""
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log a WARNING-level message."""
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log an ERROR-level message."""
        self._logger.error(msg)

    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: int) -> None:
        """Log a dictionary of metrics at a given training step.

        The metrics are formatted as a single JSON line for easy downstream
        parsing.

        Args:
            metrics: Mapping of metric name to value.
            step: Current training step (e.g. global step or epoch).
        """
        metrics_line = json.dumps({"step": step, **metrics})
        self._logger.info(f"METRICS {metrics_line}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    """Return a filesystem-safe timestamp string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
