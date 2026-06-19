"""Unit tests for the :mod:`selfllm.train` command-line interface.

The CLI is the largest module in the repo and was previously untested. The
risk concentrates in three pure, torch-free functions:

- ``load_yaml_config`` — YAML file → dict (empty file tolerated)
- ``merge_config``     — layering CLI flags over YAML with documented precedence
- ``build_parser``     — argparse wiring (subcommands, dest names, required
                         flags, store_false toggles, defaults)

plus ``main`` dispatch. None of these need a GPU or a real model, so the whole
file runs in milliseconds. Where the merge logic has a sharp edge (the
``X or default`` idiom mishandling falsy values like ``seed=0``), the current
behaviour is pinned with an explicit test so a future fix is a deliberate,
visible change.
"""

from __future__ import annotations

import argparse

import pytest

from selfllm import train as cli


# ---------------------------------------------------------------------------
# load_yaml_config
# ---------------------------------------------------------------------------


def test_load_yaml_config_reads_nested_dict(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("model:\n  d_model: 256\ntraining:\n  batch_size: 4\n")
    cfg = cli.load_yaml_config(str(p))
    assert cfg["model"]["d_model"] == 256
    assert cfg["training"]["batch_size"] == 4


def test_load_yaml_config_empty_file_returns_empty_dict(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert cli.load_yaml_config(str(p)) == {}


# ---------------------------------------------------------------------------
# merge_config — precedence
# ---------------------------------------------------------------------------


def _ns(**kw) -> argparse.Namespace:
    """A Namespace with arbitrary attributes; merge_config uses getattr/None."""
    return argparse.Namespace(**kw)


def test_cli_overrides_yaml_for_model():
    yaml_cfg = {"model": {"d_model": 128, "n_layers": 4}}
    merged = cli.merge_config(yaml_cfg, _ns(d_model=512))
    assert merged["model"]["d_model"] == 512   # CLI wins
    assert merged["model"]["n_layers"] == 4     # untouched YAML preserved


def test_yaml_used_when_cli_value_is_none():
    yaml_cfg = {"training": {"batch_size": 8, "learning_rate": 1e-3}}
    merged = cli.merge_config(yaml_cfg, _ns(batch_size=None, learning_rate=None))
    assert merged["training"]["batch_size"] == 8
    assert merged["training"]["learning_rate"] == 1e-3


def test_missing_cli_attrs_fall_back_to_yaml():
    """merge_config uses getattr(..., None), so a Namespace lacking an
    attribute entirely must behave the same as the attribute being None."""
    yaml_cfg = {"recursive": {"max_iterations": 7}}
    merged = cli.merge_config(yaml_cfg, _ns())  # no attributes at all
    assert merged["recursive"]["max_iterations"] == 7


def test_all_sections_present_with_empty_inputs():
    merged = cli.merge_config({}, _ns())
    for section in ("model", "training", "recursive", "generation", "real_training"):
        assert section in merged
        assert isinstance(merged[section], dict)


def test_path_defaults_when_absent():
    merged = cli.merge_config({}, _ns())
    assert merged["checkpoint_dir"] == "./checkpoints"
    assert merged["seed"] == 42
    assert merged["model_path"] == ""
    assert merged["interactive"] is False


def test_generation_arg_is_renamed_to_max_new_tokens():
    """--max-tokens maps onto the generation config's max_new_tokens key."""
    merged = cli.merge_config({}, _ns(max_tokens=64))
    assert merged["generation"]["max_new_tokens"] == 64


def test_checkpoint_dir_cli_overrides_yaml():
    merged = cli.merge_config(
        {"checkpoint_dir": "/from/yaml"}, _ns(checkpoint_dir="/from/cli")
    )
    assert merged["checkpoint_dir"] == "/from/cli"


def test_seed_zero_is_swallowed_by_or_idiom():
    """KNOWN EDGE: merged['seed'] uses ``cli or yaml or 42``. Because 0 is
    falsy, passing --seed 0 on the CLI is silently replaced by the YAML value
    (or the 42 default) rather than honoured. Pinned so a future fix to use an
    explicit ``is not None`` check is a visible, intentional change.
    """
    merged = cli.merge_config({"seed": 99}, _ns(seed=0))
    assert merged["seed"] == 99  # the 0 was dropped, YAML's 99 used instead


def test_real_training_lora_flag_passthrough():
    merged = cli.merge_config({}, _ns(use_lora=False, lora_rank=16))
    assert merged["real_training"]["use_lora"] is False
    assert merged["real_training"]["lora_rank"] == 16


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------


def test_no_command_leaves_command_none():
    args = cli.build_parser().parse_args([])
    assert args.command is None


def test_init_subcommand_dest_names():
    args = cli.build_parser().parse_args(
        ["init", "--vocab-size", "1000", "--d-model", "64", "--n-layers", "2"]
    )
    assert args.command == "init"
    assert args.vocab_size == 1000
    assert args.d_model == 64       # hyphen flag -> underscore dest
    assert args.n_layers == 2


def test_self_improve_dest_names():
    args = cli.build_parser().parse_args(
        ["self-improve", "--model-path", "m", "--tokenizer-path", "t",
         "--max-iterations", "10", "--samples-per-iteration", "20"]
    )
    assert args.command == "self-improve"
    assert args.model_path == "m"
    assert args.max_iterations == 10
    assert args.samples_per_iteration == 20


def test_global_config_flag_parsed_before_subcommand():
    args = cli.build_parser().parse_args(["--config", "cfg.yaml", "init"])
    assert args.config == "cfg.yaml"
    assert args.command == "init"


@pytest.mark.parametrize("cmd", ["pretrain", "self-improve", "generate", "evaluate", "serve", "ppo"])
def test_required_args_missing_exits(cmd):
    """Each of these subcommands has at least one required flag; omitting it
    must cause argparse to exit (rather than silently proceed)."""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([cmd])


def test_real_training_defaults():
    args = cli.build_parser().parse_args(["real-training"])
    assert args.scale == "small"
    assert args.use_lora is True
    assert args.use_dpo is True
    assert args.num_books == 100


def test_no_lora_and_no_dpo_toggles():
    args = cli.build_parser().parse_args(["real-training", "--no-lora", "--no-dpo"])
    assert args.use_lora is False
    assert args.use_dpo is False


def test_serve_defaults():
    args = cli.build_parser().parse_args(
        ["serve", "--model-path", "m", "--tokenizer-path", "t"]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 8000
    assert args.max_batch_size == 32


def test_invalid_device_choice_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--device", "tpu", "init"])


# ---------------------------------------------------------------------------
# main — dispatch
# ---------------------------------------------------------------------------


def test_main_no_command_prints_help_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 0


def test_main_missing_config_file_exits_one(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--config", str(tmp_path / "nope.yaml"), "init"])
    assert exc.value.code == 1


def test_main_dispatches_to_handler(monkeypatch, tmp_path):
    """main() should route to the correct command handler with the merged
    config + a logger, without us needing a real model."""
    received = {}

    def fake_init(cfg, logger):
        received["cfg"] = cfg
        received["logger"] = logger

    monkeypatch.setattr(cli, "cmd_init", fake_init)
    cli.main(["--checkpoint-dir", str(tmp_path / "ck"), "--seed", "7", "init", "--d-model", "32"])

    assert received, "handler was never invoked"
    assert received["cfg"]["model"]["d_model"] == 32
    assert received["cfg"]["seed"] == 7
    assert received["logger"] is not None
