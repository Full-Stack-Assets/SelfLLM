"""Unit tests for :mod:`selfllm.training.trainer`.

These tests exercise the *real* training loop end-to-end on a tiny CPU model,
rather than mocking ``LLMTrainer.train`` (as the integration tests do). They
cover the pieces most likely to harbour silent bugs:

- the cosine LR schedule shape (warmup ramp, decay, ``min_lr_ratio`` floor)
- the ``AverageMeter`` accounting helper
- a genuine optimizer step actually updates parameters and reduces loss on a
  trivially memorisable batch
- gradient accumulation still produces optimizer steps (including the trailing
  partial window)
- checkpoint save → load round-trips trainer state and model weights

All training runs on CPU with ``use_amp=False`` (AMP is CUDA-only in the
trainer) and a 2-layer d_model=32 model, so the whole file runs in seconds.
"""

from __future__ import annotations

import copy
import math
import os

import pytest
import torch
from torch.optim import SGD

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.training.dataset import SelfTrainingDataset
from selfllm.training.trainer import (
    AverageMeter,
    LLMTrainer,
    get_cosine_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# get_cosine_schedule_with_warmup — pure function, no model needed
# ---------------------------------------------------------------------------


def _lrs(num_warmup, num_training, base_lr=1.0, **kw):
    """Return the LR multiplier at every step of a schedule (base_lr=1.0)."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = SGD([param], lr=base_lr)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup, num_training, **kw)
    lrs = []
    for _ in range(num_training):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs


def test_warmup_ramps_linearly_from_zero():
    lrs = _lrs(num_warmup=4, num_training=20)
    # Step 0 LR is 0, then increases linearly to the base LR at the warmup end.
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[1] == pytest.approx(0.25)
    assert lrs[2] == pytest.approx(0.50)
    assert lrs[3] == pytest.approx(0.75)
    # Monotonically increasing through warmup.
    assert lrs[0] < lrs[1] < lrs[2] < lrs[3]


def test_peak_lr_reached_at_end_of_warmup():
    lrs = _lrs(num_warmup=5, num_training=50)
    assert lrs[5] == pytest.approx(1.0, abs=1e-6)


def test_cosine_decays_after_warmup():
    lrs = _lrs(num_warmup=5, num_training=50)
    post_warmup = lrs[5:]
    # Non-increasing after the peak (cosine half-cycle decay).
    for a, b in zip(post_warmup, post_warmup[1:]):
        assert b <= a + 1e-9


def test_min_lr_ratio_is_the_floor():
    ratio = 0.1
    warmup = 2
    lrs = _lrs(num_warmup=warmup, num_training=30, min_lr_ratio=ratio)
    # The floor governs the cosine decay phase (warmup legitimately starts at 0).
    decay_phase = lrs[warmup:]
    assert min(decay_phase) >= ratio - 1e-9
    assert lrs[-1] == pytest.approx(ratio, abs=0.05)


def test_zero_warmup_does_not_divide_by_zero():
    lrs = _lrs(num_warmup=0, num_training=10)
    assert lrs[0] == pytest.approx(1.0)  # starts at full LR, no warmup


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------


def test_average_meter_running_mean():
    m = AverageMeter()
    m.update(2.0)
    m.update(4.0)
    assert m.val == pytest.approx(4.0)
    assert m.avg == pytest.approx(3.0)
    assert m.count == 2


def test_average_meter_weighted_updates():
    m = AverageMeter()
    m.update(1.0, n=3)   # sum=3, count=3
    m.update(5.0, n=1)   # sum=8, count=4
    assert m.avg == pytest.approx(2.0)
    assert m.sum == pytest.approx(8.0)


def test_average_meter_reset():
    m = AverageMeter()
    m.update(10.0, n=5)
    m.reset()
    assert (m.val, m.avg, m.sum, m.count) == (0.0, 0.0, 0.0, 0)


# ---------------------------------------------------------------------------
# LLMTrainer — real training on a tiny CPU model
# ---------------------------------------------------------------------------


@pytest.fixture()
def cpu_model(tiny_model_config: ModelConfig) -> SelfImprovingLLM:
    torch.manual_seed(0)
    return SelfImprovingLLM(tiny_model_config)


def _dataset(tokenizer, n=8):
    samples = [
        {"prompt": "the quick brown", "response": "fox jumps over"}
        for _ in range(n)
    ]
    return SelfTrainingDataset(samples, tokenizer, max_seq_len=24, pad_to_max=True)


def _trainer(model, tokenizer, tmp_path, **overrides):
    cfg = {
        "learning_rate": 1e-2,
        "batch_size": 4,
        "use_amp": False,
        "warmup_ratio": 0.0,
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "save_every": 1,
    }
    cfg.update(overrides)
    return LLMTrainer(model, tokenizer, config=cfg, device="cpu")


def test_amp_disabled_on_cpu(cpu_model, tiny_tokenizer, tmp_path):
    """use_amp must be forced off on CPU even when requested in config."""
    t = _trainer(cpu_model, tiny_tokenizer, tmp_path, use_amp=True)
    assert t.use_amp is False
    assert t.scaler is None


def test_train_runs_and_records_history(cpu_model, tiny_tokenizer, tmp_path):
    t = _trainer(cpu_model, tiny_tokenizer, tmp_path)
    ds = _dataset(tiny_tokenizer)
    history = t.train(ds, num_epochs=2)

    assert len(history["train_loss"]) == 2
    assert len(history["learning_rate"]) == 2
    assert all(math.isfinite(x) for x in history["train_loss"])
    # An optimizer step actually happened.
    assert t.global_step > 0


def test_training_updates_parameters(cpu_model, tiny_tokenizer, tmp_path):
    before = copy.deepcopy({k: v.clone() for k, v in cpu_model.state_dict().items()})
    t = _trainer(cpu_model, tiny_tokenizer, tmp_path)
    t.train(_dataset(tiny_tokenizer), num_epochs=1)

    after = cpu_model.state_dict()
    changed = any(
        not torch.equal(before[k], after[k])
        for k in before
        if after[k].dtype.is_floating_point
    )
    assert changed, "Expected at least one parameter to change after training"


def test_loss_decreases_on_memorisable_batch(cpu_model, tiny_tokenizer, tmp_path):
    """On a single repeated sample with a high LR, the loss should fall."""
    t = _trainer(cpu_model, tiny_tokenizer, tmp_path, learning_rate=3e-2)
    ds = _dataset(tiny_tokenizer, n=4)
    history = t.train(ds, num_epochs=6)
    losses = history["train_loss"]
    # Later loss should be below the first epoch (allow noise tolerance).
    assert losses[-1] < losses[0]


def test_gradient_accumulation_steps_count(cpu_model, tiny_tokenizer, tmp_path):
    """With grad accumulation, optimizer steps = ceil(batches/accum) per epoch,
    including the trailing partial window."""
    t = _trainer(
        cpu_model, tiny_tokenizer, tmp_path,
        batch_size=2, gradient_accumulation_steps=2,
    )
    # 6 samples / batch_size 2 = 3 batches; accum 2 -> ceil(3/2)=2 steps/epoch.
    ds = _dataset(tiny_tokenizer, n=6)
    t.train(ds, num_epochs=1)
    assert t.global_step == 2


def test_validation_populates_perplexity(cpu_model, tiny_tokenizer, tmp_path):
    t = _trainer(cpu_model, tiny_tokenizer, tmp_path)
    train_ds = _dataset(tiny_tokenizer, n=8)
    val_ds = _dataset(tiny_tokenizer, n=4)
    history = t.train(train_ds, val_dataset=val_ds, num_epochs=1)
    assert len(history["val_loss"]) == 1
    assert len(history["val_perplexity"]) == 1
    assert history["val_perplexity"][0] > 0
    assert math.isfinite(history["val_perplexity"][0])


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def test_checkpoint_written_each_epoch(cpu_model, tiny_tokenizer, tmp_path):
    t = _trainer(cpu_model, tiny_tokenizer, tmp_path, save_every=1)
    t.train(_dataset(tiny_tokenizer), num_epochs=2)
    ckpt_dir = tmp_path / "ckpt"
    assert (ckpt_dir / "checkpoint_epoch_1.pt").exists()
    assert (ckpt_dir / "checkpoint_epoch_2.pt").exists()


def test_best_model_saved_with_validation(cpu_model, tiny_tokenizer, tmp_path):
    t = _trainer(cpu_model, tiny_tokenizer, tmp_path)
    t.train(_dataset(tiny_tokenizer, 8), val_dataset=_dataset(tiny_tokenizer, 4), num_epochs=1)
    assert (tmp_path / "ckpt" / "best_model.pt").exists()
    assert t.best_val_metric != float("inf")


def test_checkpoint_round_trip_restores_state(tiny_model_config, tiny_tokenizer, tmp_path):
    # Train one model and checkpoint it.
    torch.manual_seed(0)
    model_a = SelfImprovingLLM(tiny_model_config)
    t_a = _trainer(model_a, tiny_tokenizer, tmp_path)
    t_a.train(_dataset(tiny_tokenizer), num_epochs=2)
    ckpt = str(tmp_path / "ckpt" / "checkpoint_epoch_2.pt")
    assert os.path.exists(ckpt)

    # Fresh, differently-initialised model loads the checkpoint.
    torch.manual_seed(123)
    model_b = SelfImprovingLLM(tiny_model_config)
    t_b = LLMTrainer(model_b, tiny_tokenizer, config={"use_amp": False}, device="cpu")
    t_b.load_checkpoint(ckpt)

    # Trainer bookkeeping restored.
    assert t_b.global_step == t_a.global_step
    assert t_b.current_epoch == t_a.current_epoch
    assert t_b.history["train_loss"] == t_a.history["train_loss"]

    # Model weights now match the trained model exactly.
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"weight mismatch at {k}"
