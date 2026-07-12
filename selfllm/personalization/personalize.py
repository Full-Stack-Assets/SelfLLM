"""End-to-end personalization pipeline on frontier SelfLLM architectures."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from selfllm.data.pipeline import DataPipeline
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.personalization.frontier_config import get_frontier_config
from selfllm.personalization.profile import UserProfile, load_profile
from selfllm.recursive.recursive_config import RecursiveConfig
from selfllm.recursive.recursive_trainer import RecursiveSelfTrainer
from selfllm.training.trainer import LLMTrainer
from selfllm.utils import count_parameters, format_num, get_device, set_seed

logger = logging.getLogger(__name__)


def _prepare_corpus_dir(corpus_path: str, work_dir: str) -> str:
    """Normalize *corpus_path* to a directory of ``.txt`` files."""
    source = Path(corpus_path)
    if not source.exists():
        raise FileNotFoundError(f"Corpus path not found: {corpus_path}")

    corpus_dir = os.path.join(work_dir, "corpus")
    os.makedirs(corpus_dir, exist_ok=True)

    if source.is_file():
        if source.suffix.lower() != ".txt":
            raise ValueError("Single-file corpus must be a .txt file")
        shutil.copy2(source, os.path.join(corpus_dir, source.name))
        return corpus_dir

    if source.is_dir():
        txt_files = list(source.rglob("*.txt"))
        if not txt_files:
            raise ValueError(f"No .txt files found under {corpus_path}")
        for txt_file in txt_files:
            rel = txt_file.relative_to(source)
            dest = Path(corpus_dir) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(txt_file, dest)
        return corpus_dir

    raise ValueError(f"Corpus path must be a file or directory: {corpus_path}")


def _build_profile_manifest(
    profile: UserProfile,
    scale: str,
    total_params: int,
    corpus_path: str,
    pretrain_loss: float,
    iteration_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a JSON manifest describing the personalized model."""
    return {
        "model_name": f"selfllm-{profile.name.lower().replace(' ', '-')}",
        "profile": profile.to_dict(),
        "architecture": {
            "type": "frontier",
            "scale": scale,
            "parameters": total_params,
            "features": ["moe", "lora", "dpo", "recursive_self_improvement"],
        },
        "training": {
            "corpus_path": corpus_path,
            "pretrain_loss": pretrain_loss,
            "self_improve_iterations": len(iteration_metrics),
            "iteration_metrics": iteration_metrics,
        },
    }


def personalize_model(
    corpus_path: str,
    output_dir: str = "./personalized_model",
    profile_path: Optional[str] = None,
    base_model_path: Optional[str] = None,
    base_tokenizer_path: Optional[str] = None,
    scale: str = "small",
    pretrain_epochs: int = 3,
    pretrain_batch_size: int = 8,
    pretrain_lr: float = 1e-3,
    self_improve_iterations: int = 5,
    use_lora: bool = True,
    lora_rank: int = 8,
    use_dpo: bool = True,
    use_constitutional: bool = False,
    max_chunks: Optional[int] = None,
    device: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build a personalized frontier LLM from user text and profile.

    The pipeline:
      1. Ingest user corpus (``.txt`` file or directory)
      2. Train or load tokenizer
      3. Initialize frontier base model (MoE) or load existing checkpoint
      4. Pre-train on user corpus
      5. Inject LoRA adapters and run recursive self-improvement with DPO

    Args:
        corpus_path: Path to user text corpus (``.txt`` file or directory).
        output_dir: Where to save the personalized model and manifest.
        profile_path: Optional YAML profile with name, topics, style, eval prompts.
        base_model_path: Optional existing model to personalize instead of a fresh init.
        base_tokenizer_path: Tokenizer for *base_model_path* (required when base is set).
        scale: Frontier scale when no base model is provided (``small``, ``medium``, ``full``).
        pretrain_epochs: Epochs of corpus pre-training.
        pretrain_batch_size: Training batch size.
        pretrain_lr: Pre-training learning rate.
        self_improve_iterations: Recursive self-improvement iterations (0 to skip).
        use_lora: Inject LoRA before self-improvement.
        lora_rank: LoRA rank.
        use_dpo: Enable DPO preference alignment during self-improvement.
        use_constitutional: Enable constitutional AI revision pass.
        max_chunks: Cap training chunks for faster runs.
        device: ``cuda`` or ``cpu`` (auto-detected when omitted).
        seed: Random seed.

    Returns:
        Dict with output paths, parameter count, and training metrics.
    """
    set_seed(seed)
    device = device or get_device()
    profile = load_profile(profile_path)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"  SelfLLM Personalization — {profile.name}")
    print("=" * 70)

    # Step 1: Corpus
    print("\n[Step 1] Ingesting user corpus...")
    corpus_dir = _prepare_corpus_dir(corpus_path, output_dir)
    print(f"  Corpus directory: {corpus_dir}")

    # Step 2: Model config / load base
    if base_model_path:
        print(f"\n[Step 2] Loading base model from {base_model_path}...")
        if not base_tokenizer_path:
            raise ValueError("base_tokenizer_path is required when base_model_path is set")
        model = SelfImprovingLLM.from_pretrained(base_model_path).to(device)
        config = model.config
        scale = "base"
        tokenizer = BPETokenizer(vocab_size=config.vocab_size)
        tokenizer.load(base_tokenizer_path)
        pipeline = DataPipeline(tokenizer=tokenizer, max_seq_len=config.max_seq_len)
    else:
        print(f"\n[Step 2] Initializing frontier model (scale={scale})...")
        config = get_frontier_config(scale)
        pipeline = DataPipeline(tokenizer=None, max_seq_len=config.max_seq_len)

        print("\n[Step 3] Training tokenizer on user corpus...")
        tokenizer = pipeline.train_tokenizer_on_corpus(
            corpus_dir=corpus_dir,
            vocab_size=config.vocab_size,
        )
        model = SelfImprovingLLM(config).to(device)

    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    pipeline.tokenizer = tokenizer
    print(f"  Tokenizer saved to {tokenizer_path}")

    total_params = count_parameters(model)
    print(f"  Model: {format_num(total_params)} parameters ({total_params / 1e6:.1f}M)")

    # Step 4: Dataset
    print("\n[Step 4] Building personalized training dataset...")
    dataset_path = os.path.join(output_dir, "train_dataset.pt")
    dataset = pipeline.create_training_dataset(
        corpus_dir=corpus_dir,
        output_path=dataset_path,
        max_chunks=max_chunks,
    )
    if len(dataset) == 0:
        raise ValueError("No training samples created from corpus — add more text content")
    print(f"  Dataset: {len(dataset)} samples")

    # Step 5: Pre-train
    print(f"\n[Step 5] Pre-training on user corpus ({pretrain_epochs} epochs)...")
    trainer = LLMTrainer(
        model=model,
        tokenizer=tokenizer,
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
    pretrain_loss = history["train_loss"][-1]
    print(f"  Pre-training complete — final loss: {pretrain_loss:.4f}")

    pretrain_path = os.path.join(output_dir, "pretrained")
    model.save_pretrained(pretrain_path)
    print(f"  Saved pretrained checkpoint to {pretrain_path}")

    # Step 6: LoRA + self-improvement
    iteration_metrics: List[Dict[str, Any]] = []
    if self_improve_iterations > 0:
        print(
            f"\n[Step 6] Personalizing via self-improvement "
            f"({self_improve_iterations} iterations)..."
        )
        print(f"  LoRA: {use_lora} (rank={lora_rank}) | DPO: {use_dpo}")

        if use_lora:
            from selfllm.model.lora import count_lora_parameters, inject_lora

            model = inject_lora(model, rank=lora_rank, alpha=lora_rank * 2)
            lora_info = count_lora_parameters(model)
            print(
                f"  LoRA trainable: {format_num(lora_info['lora'])} "
                f"({lora_info['lora_pct']:.2f}%)"
            )

        eval_prompts = profile.default_eval_prompts()
        rec_config = RecursiveConfig(
            samples_per_iteration=min(200, max(20, len(dataset) // 2)),
            responses_per_prompt=3,
            keep_ratio=0.4,
            min_quality_score=0.35,
            training_epochs=2,
            learning_rate=5e-5 if use_lora else 1e-5,
            batch_size=pretrain_batch_size,
            gradient_accumulation_steps=2,
            max_iterations=self_improve_iterations,
            patience=3,
            target_improvement=0.01,
            checkpoint_dir=os.path.join(output_dir, "recursive"),
            eval_prompts=eval_prompts,
            replay_buffer_size=1000,
            replay_ratio=0.3,
            use_dpo=use_dpo,
            use_constitutional=use_constitutional,
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

        print("\n  Self-improvement complete!")
        for m in iteration_metrics:
            print(
                f"  Iter {m['iteration']}: kept={m['samples_kept']}, "
                f"loss={m['train_loss']:.4f}, quality={m['quality_score']:.4f}"
            )

    # Step 7: Save final artifacts
    print("\n[Step 7] Saving personalized model...")
    final_path = os.path.join(output_dir, "final")

    if use_lora:
        from selfllm.model.lora import merge_lora_weights, save_lora_weights

        lora_path = os.path.join(output_dir, "lora_adapter.pt")
        save_lora_weights(model, lora_path)
        print(f"  LoRA adapter saved to {lora_path}")
        model = merge_lora_weights(model)
        print("  LoRA weights merged into final checkpoint for serving")

    model.save_pretrained(final_path)

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config.__dict__, fh, indent=2)

    manifest = _build_profile_manifest(
        profile=profile,
        scale=scale,
        total_params=total_params,
        corpus_path=corpus_path,
        pretrain_loss=pretrain_loss,
        iteration_metrics=iteration_metrics,
    )
    manifest_path = os.path.join(output_dir, "profile_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"  Final model: {final_path}")
    print(f"  Profile manifest: {manifest_path}")
    print("\n" + "=" * 70)
    print(f"  Personalization complete for {profile.name}!")
    print("=" * 70)

    return {
        "output_dir": output_dir,
        "final_model_path": final_path,
        "tokenizer_path": tokenizer_path,
        "manifest_path": manifest_path,
        "total_parameters": total_params,
        "pretrain_loss": pretrain_loss,
        "dataset_size": len(dataset),
        "self_improve_iterations": len(iteration_metrics),
        "iteration_metrics": iteration_metrics,
        "profile": profile.to_dict(),
    }
