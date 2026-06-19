"""Unit tests for :mod:`selfllm.utils`.

These helpers underpin reproducibility (``set_seed``), checkpointing
(``save_json`` / ``load_json``), the training loop (``get_lr_scheduler``,
``EarlyStopping``, ``AverageMeter``) and run logging (``Logger``). They are
pure and fast, so they were cheap, high-value coverage to add.

Note: this ``AverageMeter`` is distinct from the one in
``selfllm.training.trainer`` — it exposes ``avg`` as a property and has no
``reset()``. The tests pin that specific contract.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from selfllm.utils import (
    AverageMeter,
    EarlyStopping,
    Logger,
    count_parameters,
    format_num,
    get_device,
    get_lr_scheduler,
    load_json,
    save_json,
    set_seed,
)


# ---------------------------------------------------------------------------
# set_seed / get_device
# ---------------------------------------------------------------------------


def test_set_seed_makes_torch_deterministic():
    set_seed(1234)
    a = torch.rand(5)
    set_seed(1234)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_set_seed_different_seeds_differ():
    set_seed(1)
    a = torch.rand(5)
    set_seed(2)
    b = torch.rand(5)
    assert not torch.equal(a, b)


def test_get_device_returns_valid_string():
    assert get_device() in {"cuda", "mps", "cpu"}


# ---------------------------------------------------------------------------
# count_parameters / format_num
# ---------------------------------------------------------------------------


def test_count_parameters_counts_trainable_only():
    model = nn.Linear(4, 3)  # 4*3 weights + 3 bias = 15
    assert count_parameters(model) == 15
    for p in model.parameters():
        p.requires_grad_(False)
    assert count_parameters(model) == 0


@pytest.mark.parametrize(
    "num,expected",
    [
        (0, "0"),
        (999, "999"),
        (1_000, "1K"),
        (1_500, "1.5K"),
        (350_000, "350K"),
        (1_200_000, "1.2M"),
        (1_000_000_000, "1B"),
        (1_500_000_000_000, "1.5T"),
    ],
)
def test_format_num(num, expected):
    assert format_num(num) == expected


def test_format_num_negative_is_mirrored():
    assert format_num(-1_200_000) == "-1.2M"
    assert format_num(-500) == "-500"


# ---------------------------------------------------------------------------
# save_json / load_json
# ---------------------------------------------------------------------------


def test_json_round_trip(tmp_path):
    data = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
    path = tmp_path / "out.json"
    save_json(data, str(path))
    assert load_json(str(path)) == data


def test_save_json_creates_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "nested" / "dir" / "out.json"
    save_json({"ok": 1}, str(path))
    assert path.exists()
    assert load_json(str(path)) == {"ok": 1}


def test_save_json_preserves_unicode(tmp_path):
    path = tmp_path / "u.json"
    save_json({"text": "héllo — wörld"}, str(path))
    # ensure_ascii=False keeps the raw characters in the file.
    raw = path.read_text(encoding="utf-8")
    assert "héllo — wörld" in raw


# ---------------------------------------------------------------------------
# get_lr_scheduler
# ---------------------------------------------------------------------------


def _schedule(num_warmup, num_training, base_lr=1.0):
    param = nn.Parameter(torch.zeros(1))
    opt = SGD([param], lr=base_lr)
    sched = get_lr_scheduler(opt, num_warmup, num_training)
    lrs = []
    for _ in range(num_training):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs


def test_lr_scheduler_warmup_then_decay_to_zero():
    lrs = _schedule(num_warmup=5, num_training=50)
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[5] == pytest.approx(1.0, abs=1e-6)   # peak at end of warmup
    # Cosine decay heads toward 0 by the final step.
    assert lrs[-1] == pytest.approx(0.0, abs=1e-2)
    # Monotonic warmup.
    assert lrs[0] < lrs[1] < lrs[2]


def test_lr_scheduler_zero_warmup_no_div_by_zero():
    lrs = _schedule(num_warmup=0, num_training=10)
    assert lrs[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------


def test_early_stopping_first_call_records_baseline():
    es = EarlyStopping(patience=2)
    assert es(0.5) is False
    assert es.best_score == pytest.approx(0.5)


def test_early_stopping_triggers_after_patience():
    es = EarlyStopping(patience=2, min_delta=0.0)
    assert es(0.5) is False   # baseline
    assert es(0.4) is False   # no improvement, counter=1
    assert es(0.45) is True   # no improvement, counter=2 -> stop


def test_early_stopping_resets_on_improvement():
    es = EarlyStopping(patience=2, min_delta=0.0)
    es(0.5)              # baseline
    es(0.4)              # counter=1
    assert es(0.7) is False  # improvement resets counter
    assert es.best_score == pytest.approx(0.7)
    assert es(0.6) is False  # counter=1 again, not yet at patience


def test_early_stopping_respects_min_delta():
    es = EarlyStopping(patience=1, min_delta=0.1)
    es(0.5)                   # baseline
    # 0.55 is an increase but below min_delta -> not an improvement -> stop.
    assert es(0.55) is True


def test_early_stopping_stays_stopped():
    es = EarlyStopping(patience=1, min_delta=0.0)
    es(0.5)
    assert es(0.4) is True
    # Even a later improvement does not un-stop the already-latched flag.
    assert es(0.9) is True


# ---------------------------------------------------------------------------
# AverageMeter (utils variant)
# ---------------------------------------------------------------------------


def test_average_meter_empty_avg_is_zero():
    assert AverageMeter().avg == 0.0   # guarded against div-by-zero


def test_average_meter_unweighted_mean():
    m = AverageMeter()
    for v in (2.0, 4.0, 6.0):
        m.update(v)
    assert m.avg == pytest.approx(4.0)


def test_average_meter_weighted_mean():
    m = AverageMeter()
    m.update(1.0, n=3)
    m.update(5.0, n=1)
    assert m.avg == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


def test_logger_creates_log_file(tmp_path):
    logger = Logger(str(tmp_path / "logs"), name="unittest")
    logger.info("hello")
    assert logger.log_file.exists()
    content = logger.log_file.read_text(encoding="utf-8")
    assert "hello" in content


def test_logger_log_metrics_emits_json_line(tmp_path):
    logger = Logger(str(tmp_path / "logs"), name="metrics_test")
    logger.log_metrics({"loss": 0.42, "acc": 0.9}, step=10)
    content = logger.log_file.read_text(encoding="utf-8")
    assert "METRICS" in content
    # The JSON payload is parseable and contains the step + metrics.
    payload = content.split("METRICS", 1)[1].strip().splitlines()[0]
    parsed = json.loads(payload)
    assert parsed["step"] == 10
    assert parsed["loss"] == pytest.approx(0.42)


def test_logger_reinit_does_not_duplicate_handlers(tmp_path):
    Logger(str(tmp_path / "logs"), name="dup")
    second = Logger(str(tmp_path / "logs"), name="dup")
    # Handlers are cleared on re-init: console + file == 2.
    assert len(second._logger.handlers) == 2
