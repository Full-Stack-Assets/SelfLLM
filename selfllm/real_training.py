"""
End-to-end training pipeline for a real 350M parameter SelfLLM.

Steps:
1. Download 1,000+ Gutenberg books
2. Train BPE tokenizer on corpus
3. Create chunked training dataset
4. Initialize 350M model
5. Pre-train with FSDP (multi-GPU) or single GPU
6. Run recursive self-improvement with LoRA + DPO
7. Save final model

Usage:
    # Single GPU (small model)
    python -m selfllm.real_training --scale small --device cpu

    # Multi-GPU (full 350M model)
    torchrun --nproc_per_node=8 -m selfllm.real_training --scale full

    # Medium model (laptop-friendly)
    python -m selfllm.real_training --scale medium --device cuda
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

from selfllm.data.pipeline import DataPipeline
from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.recursive.recursive_config import RecursiveConfig
from selfllm.recursive.recursive_trainer import RecursiveSelfTrainer
from selfllm.training.dataset import SelfTrainingDataset
from selfllm.training.trainer import LLMTrainer
from selfllm.utils import count_parameters, format_num, get_device, set_seed

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model configs for three scales
# --------------------------------------------------------------------------- #


def get_small_config() -> ModelConfig:
    """Small config for CPU/testing: ~5M parameters."""
    return ModelConfig(
        vocab_size=8000,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1,
        use_rope=True,
        use_swiglu=True,
    )


def get_medium_config() -> ModelConfig:
    """Medium config for single GPU: ~50M parameters."""
    return ModelConfig(
        vocab_size=10000,
        d_model=512,
        n_layers=12,
        n_heads=8,
        d_ff=2048,
        max_seq_len=1024,
        dropout=0.1,
        use_rope=True,
        use_swiglu=True,
        grad_checkpoint=True,
    )


def get_full_config() -> ModelConfig:
    """Full 350M parameter config for multi-GPU."""
    return ModelConfig(
        vocab_size=32000,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        d_ff=4096,
        max_seq_len=2048,
        dropout=0.1,
        use_rope=True,
        use_swiglu=True,
        grad_checkpoint=True,
    )


# --------------------------------------------------------------------------- #
# Training pipeline
# --------------------------------------------------------------------------- #


def train_real_model(
    scale: str = "small",
    data_dir: str = "./data",
    output_dir: str = "./real_model",
    num_books: int = 100,
    pretrain_epochs: int = 3,
    pretrain_batch_size: int = 8,
    pretrain_lr: float = 1e-3,
    self_improve_iterations: int = 5,
    use_lora: bool = True,
    lora_rank: int = 8,
    use_dpo: bool = True,
    device: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full training pipeline for a real model.

    Args:
        scale: ``"small"`` (~5M), ``"medium"`` (~50M), or ``"full"`` (~350M).
        data_dir: Directory for downloaded data.
        output_dir: Directory for saved models.
        num_books: Number of Gutenberg books to download.
        pretrain_epochs: Pre-training epochs.
        pretrain_batch_size: Training batch size.
        pretrain_lr: Learning rate for pre-training.
        self_improve_iterations: Recursive self-improvement iterations.
        use_lora: Use LoRA for self-improvement.
        lora_rank: LoRA rank.
        use_dpo: Use DPO for preference learning.
        device: ``"cuda"`` or ``"cpu"``.  Auto-detected when *None*.
        seed: Random seed.

    Returns:
        Dict with training results and paths.
    """
    set_seed(seed)
    device = device or get_device()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"  SelfLLM Real Model Training — Scale: {scale.upper()}")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Step 1: Select config
    # ------------------------------------------------------------------ #
    config_fn = {
        "small": get_small_config,
        "medium": get_medium_config,
        "full": get_full_config,
    }[scale]
    config = config_fn()

    print(f"\n[Step 1] Model Config:")
    print(
        f"  Layers: {config.n_layers} | Heads: {config.n_heads} | "
        f"Dim: {config.d_model}"
    )
    print(
        f"  Max Seq: {config.max_seq_len} | Vocab: {config.vocab_size}"
    )

    # ------------------------------------------------------------------ #
    # Step 2: Download books
    # ------------------------------------------------------------------ #
    print(f"\n[Step 2] Downloading {num_books} Gutenberg books...")
    pipeline = DataPipeline(tokenizer=None, max_seq_len=config.max_seq_len)
    book_paths = pipeline.download_gutenberg_books(
        output_dir=os.path.join(data_dir, "books"),
        num_books=num_books,
    )
    print(f"  Downloaded {len(book_paths)} books")

    # ------------------------------------------------------------------ #
    # Step 3: Train tokenizer
    # ------------------------------------------------------------------ #
    print(f"\n[Step 3] Training BPE tokenizer...")
    tokenizer = pipeline.train_tokenizer_on_corpus(
        corpus_dir=os.path.join(data_dir, "books"),
        vocab_size=config.vocab_size,
    )
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"  Tokenizer saved to {tokenizer_path}")
    pipeline.tokenizer = tokenizer

    # ------------------------------------------------------------------ #
    # Step 4: Create dataset
    # ------------------------------------------------------------------ #
    print(f"\n[Step 4] Creating training dataset...")
    dataset = pipeline.create_training_dataset(
        corpus_dir=os.path.join(data_dir, "books"),
        output_path=os.path.join(data_dir, "train_dataset.pt"),
    )
    print(f"  Dataset: {len(dataset)} samples")

    # ------------------------------------------------------------------ #
    # Step 5: Initialize model
    # ------------------------------------------------------------------ #
    print(f"\n[Step 5] Initializing model...")
    model = SelfImprovingLLM(config)
    total_params = count_parameters(model)
    print(f"  Model: {format_num(total_params)} parameters ({total_params / 1e6:.1f}M)")
    print(f"  Device: {device}")
    model = model.to(device)

    # ------------------------------------------------------------------ #
    # Step 6: Pre-train
    # ------------------------------------------------------------------ #
    print(
        f"\n[Step 6] Pre-training "
        f"({pretrain_epochs} epochs, lr={pretrain_lr})..."
    )
    trainer = LLMTrainer(
        model=model,
        tokenizer=tokenizer,
        # LLMTrainer reads flat config keys (config.get("learning_rate"), ...),
        # so the hyperparameters must be passed at the top level -- a nested
        # {"training": {...}} dict is silently ignored and every value falls
        # back to its default (lr=5e-4, batch=16, ...).
        config={
            "batch_size": pretrain_batch_size,
            "learning_rate": pretrain_lr,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
        },
        device=device,
    )

    history = trainer.train(dataset, num_epochs=pretrain_epochs)
    print(f"  Pre-training complete!")
    print(f"  Final loss: {history['train_loss'][-1]:.4f}")

    # Save pre-trained model
    pretrain_path = os.path.join(output_dir, "pretrained")
    model.save_pretrained(pretrain_path)
    print(f"  Saved pre-trained model to {pretrain_path}")

    # ------------------------------------------------------------------ #
    # Step 7: Test generation
    # ------------------------------------------------------------------ #
    print(f"\n[Step 7] Testing generation...")
    test_prompts = [
        "Once upon a time",
        "The science of",
        "In the year 1900",
    ]
    model.eval()
    for prompt in test_prompts:
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        with torch.no_grad():
            output = model.generate(
                prompt_ids, max_new_tokens=20, temperature=0.8, top_p=0.9
            )
        full_text = tokenizer.decode(output["sequences"][0].tolist())
        generated = full_text[len(prompt) :].strip()[:80]
        print(f"  '{prompt}' -> '{generated}'")

    # ------------------------------------------------------------------ #
    # Step 8: Recursive self-improvement
    # ------------------------------------------------------------------ #
    iteration_metrics: List[Dict[str, Any]] = []
    if self_improve_iterations > 0:
        print(
            f"\n[Step 8] Recursive Self-Improvement "
            f"({self_improve_iterations} iterations)..."
        )
        print(f"  LoRA: {use_lora} (rank={lora_rank}) | DPO: {use_dpo}")

        # Inject LoRA if requested
        if use_lora:
            from selfllm.model.lora import inject_lora, count_lora_parameters

            model = inject_lora(model, rank=lora_rank, alpha=lora_rank * 2)
            lora_info = count_lora_parameters(model)
            print(
                f"  LoRA: {format_num(lora_info['lora'])} trainable "
                f"({lora_info['lora_pct']:.2f}%)"
            )

        rec_config = RecursiveConfig(
            samples_per_iteration=200,
            responses_per_prompt=3,
            keep_ratio=0.4,
            min_quality_score=0.4,
            training_epochs=2,
            learning_rate=5e-5 if use_lora else 1e-5,
            batch_size=pretrain_batch_size,
            gradient_accumulation_steps=4,
            max_iterations=self_improve_iterations,
            patience=3,
            target_improvement=0.01,
            checkpoint_dir=os.path.join(output_dir, "recursive"),
            eval_prompts=test_prompts,
            replay_buffer_size=2000,
            replay_ratio=0.3,
            use_dpo=use_dpo,
        )

        recursive_trainer = RecursiveSelfTrainer(
            model=model,
            tokenizer=tokenizer,
            config=rec_config,
            device=device,
        )

        iteration_metrics = recursive_trainer.run(
            max_iterations=self_improve_iterations,
            target_improvement=rec_config.target_improvement,
            patience=rec_config.patience,
        )

        print(f"\n  Self-improvement complete!")
        for m in iteration_metrics:
            print(
                f"  Iter {m['iteration']}: gen={m['samples_generated']}, "
                f"kept={m['samples_kept']}, "
                f"loss={m['train_loss']:.4f}, "
                f"quality={m['quality_score']:.4f}"
            )

    # ------------------------------------------------------------------ #
    # Step 9: Save final model
    # ------------------------------------------------------------------ #
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    print(f"\n[Step 9] Final model saved to {final_path}")

    # When LoRA was used, also save just the adapter weights so the small
    # adapter can be distributed/loaded separately from the full checkpoint.
    if use_lora:
        from selfllm.model.lora import save_lora_weights

        lora_path = os.path.join(output_dir, "lora_adapter.pt")
        save_lora_weights(model, lora_path)
        print(f"  LoRA adapter saved to {lora_path}")

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2)
    print(f"  Config saved to {config_path}")

    print("\n" + "=" * 70)
    print("  Training Complete!")
    print(f"  Model: {format_num(total_params)} parameters")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    return {
        "output_dir": output_dir,
        "total_parameters": total_params,
        "pretrain_loss": history["train_loss"][-1],
        "num_books": len(book_paths),
        "dataset_size": len(dataset),
        "self_improve_iterations": (
            self_improve_iterations if self_improve_iterations > 0 else 0
        ),
        "iteration_metrics": iteration_metrics,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the real training script."""
    parser = argparse.ArgumentParser(
        description="Train a real SelfLLM model end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Small model (CPU-friendly, ~5M params)
  python -m selfllm.real_training --scale small --device cpu

  # Medium model (single GPU, ~50M params)
  python -m selfllm.real_training --scale medium --device cuda

  # Full model (multi-GPU, ~350M params)
  torchrun --nproc_per_node=4 -m selfllm.real_training --scale full
        """,
    )
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "full"],
        default="small",
        help="Model size: small=5M, medium=50M, full=350M params (default: small)",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory for downloaded data (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        default="./real_model",
        help="Directory for saved models (default: ./real_model)",
    )
    parser.add_argument(
        "--num-books",
        type=int,
        default=100,
        help="Number of Gutenberg books to download (default: 100)",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=3,
        help="Pre-training epochs (default: 3)",
    )
    parser.add_argument(
        "--pretrain-batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--pretrain-lr",
        type=float,
        default=1e-3,
        help="Learning rate for pre-training (default: 1e-3)",
    )
    parser.add_argument(
        "--self-improve-iterations",
        type=int,
        default=5,
        help="Recursive self-improvement iterations (default: 5, 0 to skip)",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for self-improvement (default: True)",
    )
    parser.add_argument(
        "--no-lora",
        dest="use_lora",
        action="store_false",
        help="Disable LoRA for self-improvement",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--use-dpo",
        action="store_true",
        default=True,
        help="Use DPO for preference learning (default: True)",
    )
    parser.add_argument(
        "--no-dpo",
        dest="use_dpo",
        action="store_false",
        help="Disable DPO for preference learning",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Device: cuda or cpu (auto-detected if omitted)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for real model training."""
    parser = build_parser()
    args = parser.parse_args(argv)

    train_real_model(
        scale=args.scale,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_books=args.num_books,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_batch_size=args.pretrain_batch_size,
        pretrain_lr=args.pretrain_lr,
        self_improve_iterations=args.self_improve_iterations,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        use_dpo=args.use_dpo,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
