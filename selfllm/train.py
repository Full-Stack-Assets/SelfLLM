"""Command-line interface for SelfLLM.

Provides subcommands for the full model lifecycle:

    python -m selfllm init           # Initialize model & tokenizer
    python -m selfllm pretrain       # Pre-train on seed corpus
    python -m selfllm self-improve   # Run recursive self-improvement
    python -m selfllm generate       # Generate text from trained model
    python -m selfllm evaluate       # Evaluate model quality
    python -m selfllm serve          # Launch OpenAI-compatible API server

Configuration can be provided via ``--config`` (YAML) and/or individual CLI
arguments.  CLI arguments always override YAML values.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from selfllm.utils import get_device, set_seed, Logger, format_num

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Nested dictionary of configuration values.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def merge_config(yaml_cfg: Dict[str, Any], cli_args: argparse.Namespace) -> Dict[str, Any]:
    """Merge YAML configuration with CLI overrides.

    CLI values that are not *None* take precedence over YAML values.

    Args:
        yaml_cfg: Configuration loaded from YAML.
        cli_args: Parsed CLI arguments.

    Returns:
        Merged configuration dictionary.
    """
    merged: Dict[str, Any] = {}

    # Model config
    model_cfg = dict(yaml_cfg.get("model", {}))
    if getattr(cli_args, "vocab_size", None) is not None:
        model_cfg["vocab_size"] = cli_args.vocab_size
    if getattr(cli_args, "d_model", None) is not None:
        model_cfg["d_model"] = cli_args.d_model
    if getattr(cli_args, "n_layers", None) is not None:
        model_cfg["n_layers"] = cli_args.n_layers
    if getattr(cli_args, "n_heads", None) is not None:
        model_cfg["n_heads"] = cli_args.n_heads
    merged["model"] = model_cfg

    # Training config
    training_cfg = dict(yaml_cfg.get("training", {}))
    if getattr(cli_args, "batch_size", None) is not None:
        training_cfg["batch_size"] = cli_args.batch_size
    if getattr(cli_args, "learning_rate", None) is not None:
        training_cfg["learning_rate"] = cli_args.learning_rate
    if getattr(cli_args, "num_epochs", None) is not None:
        training_cfg["num_epochs"] = cli_args.num_epochs
    if getattr(cli_args, "weight_decay", None) is not None:
        training_cfg["weight_decay"] = cli_args.weight_decay
    if getattr(cli_args, "warmup_ratio", None) is not None:
        training_cfg["warmup_ratio"] = cli_args.warmup_ratio
    if getattr(cli_args, "max_grad_norm", None) is not None:
        training_cfg["max_grad_norm"] = cli_args.max_grad_norm
    if getattr(cli_args, "save_every", None) is not None:
        training_cfg["save_every"] = cli_args.save_every
    merged["training"] = training_cfg

    # Recursive config
    recursive_cfg = dict(yaml_cfg.get("recursive", {}))
    if getattr(cli_args, "max_iterations", None) is not None:
        recursive_cfg["max_iterations"] = cli_args.max_iterations
    if getattr(cli_args, "samples_per_iteration", None) is not None:
        recursive_cfg["samples_per_iteration"] = cli_args.samples_per_iteration
    if getattr(cli_args, "target_improvement", None) is not None:
        recursive_cfg["target_improvement"] = cli_args.target_improvement
    if getattr(cli_args, "patience", None) is not None:
        recursive_cfg["patience"] = cli_args.patience
    merged["recursive"] = recursive_cfg

    # Generation config
    gen_cfg = dict(yaml_cfg.get("generation", {}))
    if getattr(cli_args, "max_tokens", None) is not None:
        gen_cfg["max_new_tokens"] = cli_args.max_tokens
    if getattr(cli_args, "temperature", None) is not None:
        gen_cfg["temperature"] = cli_args.temperature
    if getattr(cli_args, "top_p", None) is not None:
        gen_cfg["top_p"] = cli_args.top_p
    if getattr(cli_args, "top_k", None) is not None:
        gen_cfg["top_k"] = cli_args.top_k
    merged["generation"] = gen_cfg

    # Real-training specific args
    real_cfg = dict(yaml_cfg.get("real_training", {}))
    if getattr(cli_args, "scale", None) is not None:
        real_cfg["scale"] = cli_args.scale
    if getattr(cli_args, "data_dir", None) is not None:
        real_cfg["data_dir"] = cli_args.data_dir
    if getattr(cli_args, "output_dir", None) is not None:
        real_cfg["output_dir"] = cli_args.output_dir
    if getattr(cli_args, "num_books", None) is not None:
        real_cfg["num_books"] = cli_args.num_books
    if getattr(cli_args, "pretrain_epochs", None) is not None:
        real_cfg["pretrain_epochs"] = cli_args.pretrain_epochs
    if getattr(cli_args, "pretrain_batch_size", None) is not None:
        real_cfg["pretrain_batch_size"] = cli_args.pretrain_batch_size
    if getattr(cli_args, "pretrain_lr", None) is not None:
        real_cfg["pretrain_lr"] = cli_args.pretrain_lr
    if getattr(cli_args, "self_improve_iterations", None) is not None:
        real_cfg["self_improve_iterations"] = cli_args.self_improve_iterations
    if getattr(cli_args, "use_lora", None) is not None:
        real_cfg["use_lora"] = cli_args.use_lora
    if getattr(cli_args, "lora_rank", None) is not None:
        real_cfg["lora_rank"] = cli_args.lora_rank
    if getattr(cli_args, "use_dpo", None) is not None:
        real_cfg["use_dpo"] = cli_args.use_dpo
    merged["real_training"] = real_cfg

    # Paths & runtime
    merged["checkpoint_dir"] = getattr(cli_args, "checkpoint_dir", None) or yaml_cfg.get("checkpoint_dir", "./checkpoints")
    merged["device"] = getattr(cli_args, "device", None) or yaml_cfg.get("device", get_device())
    merged["seed"] = getattr(cli_args, "seed", None) or yaml_cfg.get("seed", 42)
    merged["data_path"] = getattr(cli_args, "data_path", None) or yaml_cfg.get("data_path", "")
    for _k in ("host", "port", "max_batch_size"):
        _v = getattr(cli_args, _k, None)
        if _v is not None:
            merged[_k] = _v
    merged["model_path"] = getattr(cli_args, "model_path", None) or yaml_cfg.get("model_path", "")
    merged["tokenizer_path"] = getattr(cli_args, "tokenizer_path", None) or yaml_cfg.get("tokenizer_path", "")
    if getattr(cli_args, "num_iterations", None) is not None:
        merged["num_iterations"] = cli_args.num_iterations
    merged["save_path"] = getattr(cli_args, "save_path", None) or yaml_cfg.get("save_path", "")
    merged["eval_data"] = getattr(cli_args, "eval_data", None) or yaml_cfg.get("eval_data", "")
    merged["prompt"] = getattr(cli_args, "prompt", None) or yaml_cfg.get("prompt", "")
    merged["output_path"] = getattr(cli_args, "output_path", None) or yaml_cfg.get("output_path", "")
    merged["interactive"] = getattr(cli_args, "interactive", False)

    return merged


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="selfllm",
        description="SelfLLM — A recursively self-improving foundation language model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s init --vocab-size 32000 --d-model 512 --save-path ./checkpoints/base
              %(prog)s pretrain --config config.yaml --data-path data/corpus.txt
              %(prog)s self-improve --model-path ./checkpoints/base --max-iterations 10
              %(prog)s generate --model-path ./checkpoints/best --prompt "Explain recursion"
              %(prog)s evaluate --model-path ./checkpoints/best --eval-data data/eval.txt
              %(prog)s serve --model-path ./checkpoints/best --tokenizer-path ./checkpoints/tokenizer --port 8000
        """),
    )

    # Global arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (overridden by explicit CLI flags).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for saving/loading checkpoints (default: ./checkpoints).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Compute device: cuda or cpu (auto-detected if omitted).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: 42).",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new model and tokenizer from configuration.",
        description="Create a fresh ModelConfig, instantiate the model and tokenizer, and save them.",
    )
    init_parser.add_argument("--vocab-size", type=int, default=None, help="Tokenizer vocabulary size.")
    init_parser.add_argument("--d-model", type=int, default=None, help="Model embedding dimension.")
    init_parser.add_argument("--n-layers", type=int, default=None, help="Number of transformer layers.")
    init_parser.add_argument("--n-heads", type=int, default=None, help="Number of attention heads.")
    init_parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Directory path where the initialized model and tokenizer will be saved.",
    )

    # ------------------------------------------------------------------
    # pretrain
    # ------------------------------------------------------------------
    pretrain_parser = subparsers.add_parser(
        "pretrain",
        help="Pre-train the model on a seed text corpus.",
        description="Load a model, train it on the provided text data, and save checkpoints.",
    )
    pretrain_parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the training corpus (text file)."
    )
    pretrain_parser.add_argument(
        "--num-epochs", type=int, default=None, help="Number of training epochs."
    )
    pretrain_parser.add_argument(
        "--batch-size", type=int, default=None, help="Training batch size."
    )
    pretrain_parser.add_argument(
        "--learning-rate", type=float, default=None, help="Peak learning rate."
    )
    pretrain_parser.add_argument(
        "--save-every", type=int, default=None, help="Save a checkpoint every N epochs."
    )

    # ------------------------------------------------------------------
    # self-improve
    # ------------------------------------------------------------------
    improve_parser = subparsers.add_parser(
        "self-improve",
        help="Run recursive self-improvement loop.",
        description=(
            "Load a trained model and run the recursive self-improvement pipeline: "
            "generate data, filter by quality, fine-tune, evaluate, and repeat."
        ),
    )
    improve_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the starting model checkpoint."
    )
    improve_parser.add_argument(
        "--tokenizer-path", type=str, required=True, help="Path to the trained tokenizer."
    )
    improve_parser.add_argument(
        "--max-iterations", type=int, default=None, help="Maximum number of self-improvement iterations."
    )
    improve_parser.add_argument(
        "--samples-per-iteration",
        type=int,
        default=None,
        help="Number of training samples to generate per iteration.",
    )
    improve_parser.add_argument(
        "--target-improvement",
        type=float,
        default=None,
        help="Minimum quality improvement to continue (stop if below for `patience` iterations).",
    )
    improve_parser.add_argument(
        "--patience", type=int, default=None, help="Iterations without improvement before stopping."
    )

    # ------------------------------------------------------------------
    # real-training
    # ------------------------------------------------------------------
    real_train_parser = subparsers.add_parser(
        "real-training",
        help="End-to-end training pipeline for a real model (data + pretrain + self-improve).",
        description=(
            "Download Gutenberg books, train a tokenizer, pre-train a model, "
            "and optionally run recursive self-improvement with LoRA + DPO."
        ),
    )
    real_train_parser.add_argument(
        "--scale",
        choices=["small", "medium", "full"],
        default="small",
        help="Model size: small=5M, medium=50M, full=350M params (default: small).",
    )
    real_train_parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory for downloaded data (default: ./data).",
    )
    real_train_parser.add_argument(
        "--output-dir", type=str, default="./real_model",
        help="Directory for saved models (default: ./real_model).",
    )
    real_train_parser.add_argument(
        "--num-books", type=int, default=100,
        help="Number of Gutenberg books to download (default: 100).",
    )
    real_train_parser.add_argument(
        "--pretrain-epochs", type=int, default=3,
        help="Pre-training epochs (default: 3).",
    )
    real_train_parser.add_argument(
        "--pretrain-batch-size", type=int, default=8,
        help="Training batch size (default: 8).",
    )
    real_train_parser.add_argument(
        "--pretrain-lr", type=float, default=1e-3,
        help="Learning rate for pre-training (default: 1e-3).",
    )
    real_train_parser.add_argument(
        "--self-improve-iterations", type=int, default=5,
        help="Recursive self-improvement iterations (default: 5, 0 to skip).",
    )
    real_train_parser.add_argument(
        "--use-lora", action="store_true", default=True,
        help="Use LoRA for self-improvement (default: True).",
    )
    real_train_parser.add_argument(
        "--no-lora", dest="use_lora", action="store_false",
        help="Disable LoRA for self-improvement.",
    )
    real_train_parser.add_argument(
        "--lora-rank", type=int, default=8,
        help="LoRA rank (default: 8).",
    )
    real_train_parser.add_argument(
        "--use-dpo", action="store_true", default=True,
        help="Use DPO for preference learning (default: True).",
    )
    real_train_parser.add_argument(
        "--no-dpo", dest="use_dpo", action="store_false",
        help="Disable DPO for preference learning.",
    )

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate text from a trained model.",
        description="Load a model and generate text given a prompt.",
    )
    gen_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model checkpoint."
    )
    gen_parser.add_argument(
        "--tokenizer-path", type=str, required=True, help="Path to the tokenizer."
    )
    gen_parser.add_argument(
        "--prompt", type=str, default="", help="Input prompt for generation."
    )
    gen_parser.add_argument(
        "--max-tokens", type=int, default=None, help="Maximum number of new tokens to generate."
    )
    gen_parser.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature."
    )
    gen_parser.add_argument(
        "--top-p", type=float, default=None, help="Nucleus sampling probability threshold."
    )
    gen_parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Run in interactive mode (read prompts from stdin).",
    )

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model quality.",
        description="Run the full evaluation suite on a trained model and save results.",
    )
    eval_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model checkpoint."
    )
    eval_parser.add_argument(
        "--tokenizer-path", type=str, required=True, help="Path to the tokenizer."
    )
    eval_parser.add_argument(
        "--eval-data", type=str, default=None, help="Path to evaluation text data."
    )
    eval_parser.add_argument(
        "--output-path", type=str, default=None, help="Path to save evaluation results (JSON)."
    )

    # ------------------------------------------------------------------
    # serve
    # ------------------------------------------------------------------
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch OpenAI-compatible API server with PagedAttention.",
        description="Serve the model via a FastAPI server implementing the OpenAI API protocol.",
    )
    serve_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model checkpoint."
    )
    serve_parser.add_argument(
        "--tokenizer-path", type=str, required=True, help="Path to the tokenizer."
    )
    serve_parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to (default: 0.0.0.0)."
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on (default: 8000)."
    )
    serve_parser.add_argument(
        "--max-batch-size", type=int, default=32, help="Maximum concurrent batch size (default: 32)."
    )

    # ------------------------------------------------------------------
    # fsdp-pretrain (multi-GPU)
    # ------------------------------------------------------------------
    fsdp_parser = subparsers.add_parser(
        "fsdp-pretrain",
        help="Pre-train with multi-GPU FSDP (launch via torchrun).",
        description=(
            "Fully Sharded Data Parallel pre-training. Launch with "
            "`torchrun --nproc_per_node=N -m selfllm fsdp-pretrain ...`; "
            "also runs single-process for local testing."
        ),
    )
    fsdp_parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the training corpus (text file)."
    )
    fsdp_parser.add_argument(
        "--num-epochs", type=int, default=None, help="Number of training epochs."
    )
    fsdp_parser.add_argument(
        "--batch-size", type=int, default=None, help="Per-device micro-batch size."
    )
    fsdp_parser.add_argument(
        "--learning-rate", type=float, default=None, help="Peak learning rate."
    )
    fsdp_parser.add_argument(
        "--save-every", type=int, default=None, help="Save a checkpoint every N epochs."
    )

    # ------------------------------------------------------------------
    # ppo (RLHF)
    # ------------------------------------------------------------------
    ppo_parser = subparsers.add_parser(
        "ppo",
        help="Run PPO (RLHF) policy updates on a trained model.",
        description="Optimize a loaded model with PPO using a reward model and GAE.",
    )
    ppo_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model checkpoint."
    )
    ppo_parser.add_argument(
        "--tokenizer-path", type=str, required=True, help="Path to the tokenizer."
    )
    ppo_parser.add_argument(
        "--num-iterations", type=int, default=None, help="Number of PPO update iterations."
    )

    return parser


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_init(cfg: Dict[str, Any], logger: Logger) -> None:
    """Initialize model and tokenizer from configuration."""
    from selfllm.model.config import ModelConfig
    from selfllm.model.tokenizer import BPETokenizer
    from selfllm.model.model import SelfImprovingLLM

    model_cfg = cfg["model"]
    config = ModelConfig(
        vocab_size=model_cfg.get("vocab_size", 32000),
        d_model=model_cfg.get("d_model", 512),
        n_layers=model_cfg.get("n_layers", 8),
        n_heads=model_cfg.get("n_heads", 8),
        d_ff=model_cfg.get("d_ff", 2048),
        max_seq_len=model_cfg.get("max_seq_len", 512),
        dropout=model_cfg.get("dropout", 0.1),
        use_rope=model_cfg.get("use_rope", True),
        use_swiglu=model_cfg.get("use_swiglu", True),
    )

    logger.info(f"Initializing model with config: {config}")

    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    model = SelfImprovingLLM(config)
    model.to(cfg["device"])

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized: {format_num(param_count)} parameters")

    save_path = cfg.get("save_path") or str(Path(cfg["checkpoint_dir"]) / "init")
    Path(save_path).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_path)
    tokenizer.save(save_path)
    logger.info(f"Model and tokenizer saved to {save_path}")


def cmd_pretrain(cfg: Dict[str, Any], logger: Logger) -> None:
    """Pre-train model on seed corpus."""
    from selfllm.model.model import SelfImprovingLLM
    from selfllm.model.tokenizer import BPETokenizer
    from selfllm.training.dataset import SelfTrainingDataset
    from selfllm.training.trainer import LLMTrainer

    data_path = cfg.get("data_path", "")
    if not data_path or not Path(data_path).exists():
        logger.error(f"Data path not found: {data_path}")
        sys.exit(1)

    logger.info(f"Loading training data from {data_path}")
    with open(data_path, "r", encoding="utf-8") as fh:
        text = fh.read()

    # Train tokenizer on seed corpus
    tokenizer = BPETokenizer(vocab_size=cfg["model"].get("vocab_size", 32000))
    logger.info("Training tokenizer on seed corpus...")
    tokenizer.train([text])

    # Save tokenizer
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(checkpoint_dir / "tokenizer"))
    logger.info(f"Tokenizer saved to {checkpoint_dir / 'tokenizer'}")

    # Build dataset
    dataset = SelfTrainingDataset.from_text(
        text,
        tokenizer,
        max_seq_len=cfg["model"].get("max_seq_len", 512),
    )
    logger.info(f"Dataset created: {len(dataset)} sequences")

    # Initialize or load model
    from selfllm.model.config import ModelConfig
    model_cfg = ModelConfig(
        vocab_size=cfg["model"].get("vocab_size", 32000),
        d_model=cfg["model"].get("d_model", 512),
        n_layers=cfg["model"].get("n_layers", 8),
        n_heads=cfg["model"].get("n_heads", 8),
        d_ff=cfg["model"].get("d_ff", 2048),
        max_seq_len=cfg["model"].get("max_seq_len", 512),
        dropout=cfg["model"].get("dropout", 0.1),
        use_rope=cfg["model"].get("use_rope", True),
        use_swiglu=cfg["model"].get("use_swiglu", True),
    )
    model = SelfImprovingLLM(model_cfg).to(cfg["device"])
    logger.info(f"Model: {format_num(sum(p.numel() for p in model.parameters() if p.requires_grad))} params")

    # Training configuration
    from selfllm.training.trainer import TrainingConfig
    training_cfg = TrainingConfig(
        batch_size=cfg["training"].get("batch_size", 16),
        learning_rate=cfg["training"].get("learning_rate", 5e-4),
        weight_decay=cfg["training"].get("weight_decay", 0.01),
        warmup_ratio=cfg["training"].get("warmup_ratio", 0.1),
        max_grad_norm=cfg["training"].get("max_grad_norm", 1.0),
        num_epochs=cfg["training"].get("num_epochs", 10),
        save_every=cfg["training"].get("save_every", 1),
        checkpoint_dir=str(checkpoint_dir),
    )

    trainer = LLMTrainer(model, tokenizer, training_cfg, device=cfg["device"])
    logger.info("Starting pre-training...")
    history = trainer.train(dataset, num_epochs=training_cfg.num_epochs)

    # Save final model
    final_path = checkpoint_dir / "pretrained"
    model.save_pretrained(str(final_path))
    logger.info(f"Pre-training complete. Final model saved to {final_path}")

    # Save training history
    from selfllm.utils import save_json
    save_json(history, str(checkpoint_dir / "training_history.json"))


def cmd_fsdp_pretrain(cfg: Dict[str, Any], logger: Logger) -> None:
    """Pre-train the model with multi-GPU FSDP.

    Intended for ``torchrun --nproc_per_node=N -m selfllm fsdp-pretrain``; the
    FSDP trainer also creates a single-process group automatically for local
    testing.
    """
    from selfllm.model.config import ModelConfig
    from selfllm.model.model import SelfImprovingLLM
    from selfllm.model.tokenizer import BPETokenizer
    from selfllm.training.dataset import SelfTrainingDataset
    from selfllm.training.fsdp_trainer import FSDPTrainer

    data_path = cfg.get("data_path", "")
    if not data_path or not Path(data_path).exists():
        logger.error(f"Data path not found: {data_path}")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as fh:
        text = fh.read()

    tokenizer = BPETokenizer(vocab_size=cfg["model"].get("vocab_size", 32000))
    logger.info("Training tokenizer on seed corpus...")
    tokenizer.train([text])

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(checkpoint_dir / "tokenizer"))

    # Build pre-training samples by chunking the corpus into fixed-size pieces.
    max_seq_len = cfg["model"].get("max_seq_len", 512)
    chunk_chars = max(1, max_seq_len * 4)  # rough chars-per-token heuristic
    samples = [
        {"prompt": "", "response": text[i : i + chunk_chars]}
        for i in range(0, len(text), chunk_chars)
        if text[i : i + chunk_chars].strip()
    ]
    dataset = SelfTrainingDataset(
        samples=samples, tokenizer=tokenizer, max_seq_len=max_seq_len
    )
    logger.info(f"Dataset created: {len(dataset)} sequences")

    model_cfg = ModelConfig(
        vocab_size=cfg["model"].get("vocab_size", 32000),
        d_model=cfg["model"].get("d_model", 512),
        n_layers=cfg["model"].get("n_layers", 8),
        n_heads=cfg["model"].get("n_heads", 8),
        d_ff=cfg["model"].get("d_ff", 2048),
        max_seq_len=cfg["model"].get("max_seq_len", 512),
        dropout=cfg["model"].get("dropout", 0.1),
        use_rope=cfg["model"].get("use_rope", True),
        use_swiglu=cfg["model"].get("use_swiglu", True),
    )
    model = SelfImprovingLLM(model_cfg)

    training = cfg.get("training", {})
    trainer = FSDPTrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=training.get("learning_rate", 1e-4),
        weight_decay=training.get("weight_decay", 0.01),
        batch_size=training.get("batch_size", 4),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 4),
        max_grad_norm=training.get("max_grad_norm", 1.0),
        warmup_ratio=training.get("warmup_ratio", 0.1),
        device=cfg["device"],
    )

    num_epochs = training.get("num_epochs", 10)
    logger.info("Starting FSDP pre-training...")
    try:
        trainer.train(dataset, num_epochs=num_epochs)
        ckpt_path = str(checkpoint_dir / "fsdp_pretrained.pt")
        trainer.save_checkpoint(ckpt_path)
        if trainer.is_main_process:
            logger.info(f"FSDP pre-training complete. Checkpoint: {ckpt_path}")
    finally:
        FSDPTrainer.cleanup_process_group()


def cmd_ppo(cfg: Dict[str, Any], logger: Logger) -> None:
    """Run PPO (RLHF) policy updates on a trained model."""
    import copy

    from selfllm.model.model import SelfImprovingLLM
    from selfllm.model.tokenizer import BPETokenizer
    from selfllm.training.ppo_trainer import PPOTrainer, RewardModel

    model_path = cfg.get("model_path", "")
    tokenizer_path = cfg.get("tokenizer_path", "")
    if not model_path or not Path(model_path).exists():
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)
    if not tokenizer_path or not Path(tokenizer_path).exists():
        logger.error(f"Tokenizer path not found: {tokenizer_path}")
        sys.exit(1)

    logger.info("Loading model and tokenizer for PPO...")
    model = SelfImprovingLLM.from_pretrained(model_path).to(cfg["device"])
    tokenizer = BPETokenizer(vocab_size=cfg["model"].get("vocab_size", 32000))
    tokenizer.load(tokenizer_path)

    logger.warning(
        "PPO is using a freshly-initialized RewardModel score head that has not "
        "been trained. The reward signal is therefore not meaningful -- this "
        "subcommand is a working scaffold for RLHF, not a useful reward model. "
        "Train/load a real reward model before relying on PPO results."
    )

    trainer = PPOTrainer(
        policy_model=model,
        ref_model=copy.deepcopy(model),
        reward_model=RewardModel(copy.deepcopy(model)),
        device=cfg["device"],
    )

    prompts = cfg.get("recursive", {}).get("eval_prompts") or [
        "The future of artificial intelligence is",
        "Once upon a time, in a distant land,",
        "The most important scientific discovery was",
    ]
    num_iterations = cfg.get("num_iterations")
    if num_iterations is None:
        num_iterations = 5

    logger.info(f"Running {num_iterations} PPO iterations on {len(prompts)} prompts...")
    for it in range(num_iterations):
        metrics = trainer.train_step_from_text(prompts, tokenizer, max_new_tokens=32)
        logger.info(
            f"PPO iter {it + 1}/{num_iterations}: "
            + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

    out_dir = Path(cfg["checkpoint_dir"]) / "ppo_model"
    model.save_pretrained(str(out_dir))
    logger.info(f"PPO complete. Policy model saved to {out_dir}")


def cmd_self_improve(cfg: Dict[str, Any], logger: Logger) -> None:
    """Run recursive self-improvement loop."""
    from selfllm.model.model import SelfImprovingLLM
    from selfllm.model.tokenizer import BPETokenizer
    from selfllm.recursive.recursive_config import RecursiveConfig
    from selfllm.recursive.recursive_trainer import RecursiveSelfTrainer

    model_path = cfg.get("model_path", "")
    tokenizer_path = cfg.get("tokenizer_path", "")

    if not model_path or not Path(model_path).exists():
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)
    if not tokenizer_path or not Path(tokenizer_path).exists():
        logger.error(f"Tokenizer path not found: {tokenizer_path}")
        sys.exit(1)

    logger.info("Loading model and tokenizer...")
    model = SelfImprovingLLM.from_pretrained(model_path).to(cfg["device"])
    tokenizer = BPETokenizer(vocab_size=cfg["model"].get("vocab_size", 32000))
    tokenizer.load(tokenizer_path)

    recursive_cfg = RecursiveConfig(
        samples_per_iteration=cfg["recursive"].get("samples_per_iteration", 2000),
        responses_per_prompt=cfg["recursive"].get("responses_per_prompt", 3),
        generation_temperature=cfg["generation"].get("temperature", 0.8),
        generation_top_p=cfg["generation"].get("top_p", 0.92),
        generation_max_tokens=cfg["generation"].get("max_new_tokens", 256),
        min_quality_score=cfg["recursive"].get("min_quality_score", 0.6),
        keep_ratio=cfg["recursive"].get("keep_ratio", 0.4),
        max_perplexity=cfg["recursive"].get("max_perplexity", 80.0),
        min_diversity=cfg["recursive"].get("min_diversity", 0.35),
        training_epochs=cfg["recursive"].get("training_epochs", 2),
        learning_rate=cfg["recursive"].get("learning_rate", 5e-5),
        batch_size=cfg["recursive"].get("batch_size", 16),
        gradient_accumulation_steps=cfg["recursive"].get("gradient_accumulation_steps", 4),
        warmup_ratio=cfg["recursive"].get("warmup_ratio", 0.1),
        weight_decay=cfg["recursive"].get("weight_decay", 0.01),
        max_grad_norm=cfg["recursive"].get("max_grad_norm", 1.0),
        replay_buffer_size=cfg["recursive"].get("replay_buffer_size", 10000),
        replay_ratio=cfg["recursive"].get("replay_ratio", 0.3),
        max_iterations=cfg["recursive"].get("max_iterations", 10),
        target_improvement=cfg["recursive"].get("target_improvement", 0.01),
        patience=cfg["recursive"].get("patience", 3),
        rollback_on_degradation=cfg["recursive"].get("rollback_on_degradation", True),
        checkpoint_dir=cfg.get("checkpoint_dir", "./checkpoints"),
        save_every=cfg["recursive"].get("save_every", 1),
        keep_last_n=cfg["recursive"].get("keep_last_n", 5),
    )

    trainer = RecursiveSelfTrainer(model, tokenizer, recursive_cfg, device=cfg["device"])

    max_iterations = cfg["recursive"].get("max_iterations", 10)
    target_improvement = cfg["recursive"].get("target_improvement", 0.01)
    patience = cfg["recursive"].get("patience", 3)

    logger.info(
        f"Starting recursive self-improvement: "
        f"max_iterations={max_iterations}, "
        f"target_improvement={target_improvement}, "
        f"patience={patience}"
    )

    iteration_results = trainer.run(
        max_iterations=max_iterations,
        target_improvement=target_improvement,
        patience=patience,
    )

    logger.info(f"Self-improvement complete. {len(iteration_results)} iterations executed.")
    for result in iteration_results:
        logger.info(
            f"  Iteration {result['iteration']}: "
            f"loss={result['train_loss']:.4f}, "
            f"ppl={result['eval_perplexity']:.2f}, "
            f"quality={result['quality_score']:.4f}, "
            f"delta={result['improvement_delta']:.4f}"
        )

    # Save results
    from selfllm.utils import save_json
    save_json(iteration_results, str(Path(cfg["checkpoint_dir"]) / "iteration_results.json"))


def cmd_real_training(cfg: Dict[str, Any], logger: Logger) -> None:
    """Run the end-to-end real model training pipeline."""
    from selfllm.real_training import train_real_model

    # Read from real_training config section or fall back to top-level keys
    real_cfg = cfg.get("real_training", {})
    scale = real_cfg.get("scale", cfg.get("scale", "small"))
    data_dir = real_cfg.get("data_dir", cfg.get("data_dir", "./data"))
    output_dir = real_cfg.get("output_dir", cfg.get("output_dir", "./real_model"))
    num_books = real_cfg.get("num_books", cfg.get("num_books", 100))
    pretrain_epochs = real_cfg.get("pretrain_epochs", cfg.get("pretrain_epochs", 3))
    pretrain_batch_size = real_cfg.get("pretrain_batch_size", cfg.get("pretrain_batch_size", 8))
    pretrain_lr = real_cfg.get("pretrain_lr", cfg.get("pretrain_lr", 1e-3))
    self_improve_iterations = real_cfg.get("self_improve_iterations", cfg.get("self_improve_iterations", 5))
    use_lora = real_cfg.get("use_lora", cfg.get("use_lora", True))
    lora_rank = real_cfg.get("lora_rank", cfg.get("lora_rank", 8))
    use_dpo = real_cfg.get("use_dpo", cfg.get("use_dpo", True))
    device = cfg.get("device", get_device())
    seed = cfg.get("seed", 42)

    logger.info(f"Starting real training pipeline: scale={scale}")
    logger.info(f"  Data dir: {data_dir} | Output dir: {output_dir}")
    logger.info(f"  Books: {num_books} | Epochs: {pretrain_epochs}")
    logger.info(f"  LoRA: {use_lora} (rank={lora_rank}) | DPO: {use_dpo}")

    results = train_real_model(
        scale=scale,
        data_dir=data_dir,
        output_dir=output_dir,
        num_books=num_books,
        pretrain_epochs=pretrain_epochs,
        pretrain_batch_size=pretrain_batch_size,
        pretrain_lr=pretrain_lr,
        self_improve_iterations=self_improve_iterations,
        use_lora=use_lora,
        lora_rank=lora_rank,
        use_dpo=use_dpo,
        device=device,
        seed=seed,
    )

    logger.info(f"Real training complete!")
    logger.info(f"  Total parameters: {format_num(results['total_parameters'])}")
    logger.info(f"  Pre-train loss: {results['pretrain_loss']:.4f}")
    logger.info(f"  Books downloaded: {results['num_books']}")
    logger.info(f"  Dataset size: {results['dataset_size']}")
    logger.info(f"  Self-improve iterations: {results['self_improve_iterations']}")
    logger.info(f"  Output directory: {results['output_dir']}")

    # Save results
    from selfllm.utils import save_json
    results_path = os.path.join(output_dir, "training_results.json")
    # Filter out non-serializable items
    serializable_results = {
        k: v for k, v in results.items()
        if k != "iteration_metrics" or isinstance(v, list)
    }
    save_json(serializable_results, results_path)
    logger.info(f"Results saved to {results_path}")


def cmd_generate(cfg: Dict[str, Any], logger: Logger) -> None:
    """Generate text from a trained model."""
    from selfllm.model.model import SelfImprovingLLM
    from selfllm.model.tokenizer import BPETokenizer

    model_path = cfg.get("model_path", "")
    tokenizer_path = cfg.get("tokenizer_path", "")

    if not model_path or not Path(model_path).exists():
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)
    if not tokenizer_path or not Path(tokenizer_path).exists():
        logger.error(f"Tokenizer path not found: {tokenizer_path}")
        sys.exit(1)

    logger.info("Loading model and tokenizer...")
    model = SelfImprovingLLM.from_pretrained(model_path).to(cfg["device"])
    model.eval()
    tokenizer = BPETokenizer(vocab_size=cfg["model"].get("vocab_size", 32000))
    tokenizer.load(tokenizer_path)

    max_tokens = cfg["generation"].get("max_new_tokens", 256)
    temperature = cfg["generation"].get("temperature", 0.8)
    top_p = cfg["generation"].get("top_p", 0.92)
    top_k = cfg["generation"].get("top_k", 50)

    if cfg.get("interactive"):
        logger.info("Interactive mode. Type 'quit' or 'exit' to stop.")
        while True:
            try:
                prompt = input("\nPrompt> ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip().lower() in ("quit", "exit", "q"):
                break
            if not prompt.strip():
                continue
            _generate_and_print(model, tokenizer, prompt, max_tokens, temperature, top_p, top_k, cfg["device"])
    else:
        prompt = cfg.get("prompt", "")
        if not prompt:
            logger.error("No prompt provided. Use --prompt or --interactive.")
            sys.exit(1)
        _generate_and_print(model, tokenizer, prompt, max_tokens, temperature, top_p, top_k, cfg["device"])


def _generate_and_print(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
) -> None:
    """Helper to generate text and print the result."""
    import torch

    with torch.no_grad():
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        result = model.generate(
            prompt_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        output_ids = result["sequences"][0].cpu().tolist()
        output_text = tokenizer.decode(output_ids)

    print(f"\n{'=' * 60}")
    print(f"Prompt:  {prompt}")
    print(f"{'=' * 60}")
    print(f"Output:\n{output_text}")
    print(f"{'=' * 60}")


def cmd_evaluate(cfg: Dict[str, Any], logger: Logger) -> None:
    """Evaluate model quality."""
    from selfllm.model.model import SelfImprovingLLM
    from selfllm.model.tokenizer import BPETokenizer
    from selfllm.recursive.evaluator import SelfImprovementEvaluator

    model_path = cfg.get("model_path", "")
    tokenizer_path = cfg.get("tokenizer_path", "")

    if not model_path or not Path(model_path).exists():
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)
    if not tokenizer_path or not Path(tokenizer_path).exists():
        logger.error(f"Tokenizer path not found: {tokenizer_path}")
        sys.exit(1)

    logger.info("Loading model and tokenizer for evaluation...")
    model = SelfImprovingLLM.from_pretrained(model_path).to(cfg["device"])
    model.eval()
    tokenizer = BPETokenizer(vocab_size=cfg["model"].get("vocab_size", 32000))
    tokenizer.load(tokenizer_path)

    # Load eval prompts
    eval_prompts: List[str] = []
    eval_data_path = cfg.get("eval_data", "")
    if eval_data_path and Path(eval_data_path).exists():
        with open(eval_data_path, "r", encoding="utf-8") as fh:
            eval_prompts = [line.strip() for line in fh if line.strip()]
    else:
        eval_prompts = [
            "Explain the concept of recursion in programming:",
            "Write a short story about a robot learning to feel emotions:",
            "Compare and contrast machine learning and traditional programming:",
            "Describe the process of photosynthesis:",
            "What are the implications of artificial general intelligence?",
        ]
        logger.info("Using default evaluation prompts")

    evaluator = SelfImprovementEvaluator(model, tokenizer, eval_prompts, device=cfg["device"])
    logger.info("Running evaluation suite...")
    results = evaluator.full_evaluation()

    logger.info("Evaluation results:")
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    # Save results
    output_path = cfg.get("output_path")
    if output_path:
        from selfllm.utils import save_json
        save_json(results, output_path)
        logger.info(f"Results saved to {output_path}")


def cmd_serve(cfg: Dict[str, Any], logger: Logger) -> None:
    """Launch the OpenAI-compatible API server with PagedAttention backend."""
    from selfllm.serving.server import serve

    model_path = cfg.get("model_path", "")
    tokenizer_path = cfg.get("tokenizer_path", "")

    if not model_path or not Path(model_path).exists():
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)
    if not tokenizer_path or not Path(tokenizer_path).exists():
        logger.error(f"Tokenizer path not found: {tokenizer_path}")
        sys.exit(1)

    host = cfg.get("host") or "0.0.0.0"
    # Honor the PORT env var when no explicit port is given -- container
    # platforms (Cloud Run, etc.) inject the port the service must listen on.
    port = cfg.get("port") or int(os.environ.get("PORT") or 8000)
    max_batch_size = cfg.get("max_batch_size", 32)

    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Tokenizer: {tokenizer_path}")

    serve(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        host=host,
        port=port,
        max_batch_size=max_batch_size,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entry point.

    Args:
        argv: Optional list of command-line arguments (defaults to
            ``sys.argv[1:]``).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Load YAML config if provided
    yaml_cfg: Dict[str, Any] = {}
    if args.config:
        if not Path(args.config).exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        yaml_cfg = load_yaml_config(args.config)

    # Merge YAML + CLI
    cfg = merge_config(yaml_cfg, args)

    # Set seed and device
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = cfg.get("device", get_device())
    cfg["device"] = device

    # Setup logger
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = checkpoint_dir.parent / "logs"
    logger = Logger(str(logs_dir), name=f"selfllm_{args.command}")
    logger.info(f"Command: {args.command}")
    logger.info(f"Seed: {seed} | Device: {device}")
    if args.config:
        logger.info(f"Config file: {args.config}")

    # Dispatch
    command_map = {
        "init": cmd_init,
        "pretrain": cmd_pretrain,
        "fsdp-pretrain": cmd_fsdp_pretrain,
        "ppo": cmd_ppo,
        "self-improve": cmd_self_improve,
        "real-training": cmd_real_training,
        "generate": cmd_generate,
        "evaluate": cmd_evaluate,
        "serve": cmd_serve,
    }

    handler = command_map[args.command]
    try:
        handler(cfg, logger)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.error(f"Command failed: {exc}")
        raise


if __name__ == "__main__":
    main()
