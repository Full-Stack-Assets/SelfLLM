"""
SelfLLM V5 — End-to-End Training Runner
========================================
Calibrated for CPU execution in the available sandbox:

  Pre-training  : 3 epochs, 500 samples, seq_len=128, batch=8
                  → ~55 seconds total

  Self-improvement: 10 iterations, LoRA + DPO
                  60 samples/iter, 32-token generation, seq_len=128
                  → ~6 minutes total

All artefacts written to  ./run_output/
Progress logged to stdout + ./run_output/training_log.jsonl

Usage:
    python run_training.py
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "project"))

import torch
from torch.utils.data import DataLoader

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.recursive.recursive_config import RecursiveConfig
from selfllm.recursive.recursive_trainer import RecursiveSelfTrainer
from selfllm.serving.optimized import compile_model, is_compiled
from selfllm.training.dataset import SelfTrainingDataset
from selfllm.training.trainer import LLMTrainer
from selfllm.utils import count_parameters, format_num, set_seed

# ---------------------------------------------------------------------------
# Config — calibrated for CPU (0.3s/step at seq_len=128, batch=8)
# ---------------------------------------------------------------------------

SEED           = 42
DEVICE         = "cpu"
DATA_DIR       = "./real_model/data"
OUTPUT_DIR     = "./run_output"
TOKENIZER_PATH = "./real_model/tokenizer.json"
DATASET_PATH   = os.path.join(DATA_DIR, "train_dataset.pt")

# Pre-training
PRETRAIN_SEQ_LEN   = 128   # fits in seq_len≤512 model; ~16× faster attention
PRETRAIN_SAMPLES   = 500   # 63 steps/epoch at batch=8 → ~18s/epoch
PRETRAIN_EPOCHS    = 3     # total ≈ 55s
PRETRAIN_BATCH     = 8
PRETRAIN_LR        = 3e-4

# Self-improvement
SELF_IMPROVE_ITERS = 10
SAMPLES_PER_ITER   = 60    # generate 60 → keep ~24 (40%)
GEN_MAX_TOKENS     = 32    # short sequences for speed
SEQ_LEN_SI         = 128   # keep same seq_len for consistency
LORA_RANK          = 8
USE_DPO            = True

TEST_PROMPTS = [
    "The science of",
    "Once upon a time",
    "It was a dark and stormy night",
    "In the beginning",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
log_path = os.path.join(OUTPUT_DIR, "training_log.jsonl")

# Wipe stale logs
for _f in [log_path, os.path.join(OUTPUT_DIR, "run.log")]:
    if os.path.exists(_f):
        os.remove(_f)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "run.log")),
    ],
)
logger = logging.getLogger(__name__)

# Suppress tqdm / internal trainer noise on CPU
for _n in ("selfllm.training.trainer", "selfllm.recursive.recursive_trainer",
           "selfllm.training.data_generator", "selfllm.training.quality_filter"):
    logging.getLogger(_n).setLevel(logging.WARNING)


def log_event(evt: Dict[str, Any]) -> None:
    evt["ts"] = time.time()
    with open(log_path, "a") as fh:
        fh.write(json.dumps(evt) + "\n")


def banner(title: str) -> None:
    bar = "─" * 68
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_pretrain_dataset(tokenizer: BPETokenizer) -> SelfTrainingDataset:
    """Load and subset the corpus, rebuild with seq_len=128."""
    full = SelfTrainingDataset.load(DATASET_PATH, tokenizer=tokenizer)
    # Evenly spaced sample across the full corpus
    step = max(1, len(full) // PRETRAIN_SAMPLES)
    selected = [full.samples[i] for i in range(0, len(full), step)][:PRETRAIN_SAMPLES]
    ds = SelfTrainingDataset(
        samples=selected,
        tokenizer=tokenizer,
        max_seq_len=PRETRAIN_SEQ_LEN,
        pad_to_max=True,
    )
    logger.info(f"  Training split  : {len(ds)} samples  (seq_len={PRETRAIN_SEQ_LEN})")
    logger.info(f"  Full corpus size: {len(full):,} samples  (46 books)")
    return ds


def test_generation(model: SelfImprovingLLM, tokenizer: BPETokenizer, tag: str) -> None:
    model.eval()
    logger.info(f"\n  ── Generation after {tag} ──")
    for prompt in TEST_PROMPTS:
        ids = tokenizer.encode(prompt)[:30]
        input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=25,
                temperature=0.9,
                top_p=0.9,
            )
        seq = out["sequences"][0].tolist() if isinstance(out, dict) else out[0].tolist()
        text = tokenizer.decode(seq)
        generated = text[len(prompt):].strip().replace("\n", " ")[:72]
        logger.info(f"    {prompt!r:30s} → {generated!r}")
    model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(SEED)
    t_total = time.time()

    # ------------------------------------------------------------------ #
    # Step 1 — Model + tokenizer
    # ------------------------------------------------------------------ #
    banner("Step 1 — Model & Tokenizer")
    config = ModelConfig(
        vocab_size=8000,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_seq_len=512,    # full capacity; actual sequences are 128
        dropout=0.1,
        use_rope=True,
        use_swiglu=True,
    )
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    tokenizer.load(TOKENIZER_PATH)
    logger.info(f"  Tokenizer  vocab={tokenizer.vocab_size}  merges={len(tokenizer._merges)}")

    model = SelfImprovingLLM(config).to(DEVICE)
    n_params = count_parameters(model)
    logger.info(f"  Model      {format_num(n_params)} params ({n_params/1e6:.1f}M)  device={DEVICE}")
    log_event({"phase": "init", "params": n_params, "seq_len": PRETRAIN_SEQ_LEN})

    # ------------------------------------------------------------------ #
    # Step 2 — Dataset
    # ------------------------------------------------------------------ #
    banner("Step 2 — Build Training Dataset")
    dataset = build_pretrain_dataset(tokenizer)
    log_event({"phase": "dataset", "num_samples": len(dataset), "seq_len": PRETRAIN_SEQ_LEN})

    # ------------------------------------------------------------------ #
    # Step 3 — Pre-training
    # ------------------------------------------------------------------ #
    banner(f"Step 3 — Pre-Training  ({PRETRAIN_EPOCHS} epochs, lr={PRETRAIN_LR})")

    pretrain_ckpt_dir = os.path.join(OUTPUT_DIR, "pretrain_checkpoints")
    trainer = LLMTrainer(
        model=model,
        tokenizer=tokenizer,
        config={
            "batch_size": PRETRAIN_BATCH,
            "learning_rate": PRETRAIN_LR,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "max_grad_norm": 1.0,
            "gradient_accumulation_steps": 1,
            "use_amp": False,
            "save_every": 1,
            "checkpoint_dir": pretrain_ckpt_dir,
        },
        device=DEVICE,
    )

    t0 = time.time()
    history = trainer.train(dataset, num_epochs=PRETRAIN_EPOCHS)
    pretrain_time = time.time() - t0

    final_pretrain_loss = history["train_loss"][-1]
    logger.info(f"\n  Pre-training complete  {pretrain_time:.1f}s  ({pretrain_time/60:.1f}min)")
    logger.info(f"  Loss:  {' → '.join(f'{x:.4f}' for x in history['train_loss'])}")

    log_event({
        "phase": "pretrain_done",
        "epochs": PRETRAIN_EPOCHS,
        "final_loss": final_pretrain_loss,
        "loss_history": history["train_loss"],
        "duration_s": pretrain_time,
    })

    pretrained_path = os.path.join(OUTPUT_DIR, "pretrained")
    model.save_pretrained(pretrained_path)
    logger.info(f"  Saved  → {pretrained_path}")

    test_generation(model, tokenizer, tag="pre-training")

    # ------------------------------------------------------------------ #
    # Step 4 — Inject LoRA
    # ------------------------------------------------------------------ #
    banner(f"Step 4 — Inject LoRA  (rank={LORA_RANK}, alpha={LORA_RANK*2})")
    from selfllm.model.lora import inject_lora, count_lora_parameters

    model = inject_lora(model, rank=LORA_RANK, alpha=LORA_RANK * 2)
    lora_info = count_lora_parameters(model)
    logger.info(f"  Trainable : {format_num(lora_info['lora'])}  ({lora_info['lora_pct']:.2f}%)")
    logger.info(f"  Frozen    : {format_num(lora_info['frozen'])}")
    log_event({"phase": "lora_injected", **lora_info})

    # ------------------------------------------------------------------ #
    # Step 4b — Wrap with compile_model() for future-proof acceleration
    # ------------------------------------------------------------------ #
    # compile_model() falls back to eager on Python 3.12 / torch 2.2 (torch.dynamo
    # unsupported), so this is zero-risk.  On Python ≤3.11 or a GPU environment
    # it activates torch.compile(mode="reduce-overhead") for 2–6× speedup.
    # LoRA injection must complete BEFORE compilation so LoRA params are included.
    model = compile_model(model, mode="reduce-overhead")
    active = is_compiled(model)
    logger.info(f"  compile_model()  active={active}  "
                f"({'torch.compile ACTIVE' if active else 'eager fallback — Python 3.12 / torch.dynamo unsupported'})")
    log_event({"phase": "compile_model", "compile_active": active})

    # ------------------------------------------------------------------ #
    # Step 5 — Recursive Self-Improvement
    # ------------------------------------------------------------------ #
    banner(f"Step 5 — Recursive Self-Improvement  ({SELF_IMPROVE_ITERS} iters, DPO={USE_DPO})")

    rec_config = RecursiveConfig(
        samples_per_iteration=SAMPLES_PER_ITER,
        responses_per_prompt=2,
        generation_temperature=0.85,
        generation_top_p=0.92,
        generation_max_tokens=GEN_MAX_TOKENS,
        keep_ratio=0.4,
        min_quality_score=0.3,
        max_perplexity=1000.0,
        min_diversity=0.1,
        training_epochs=1,
        learning_rate=5e-5,
        batch_size=8,
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        use_dpo=USE_DPO,
        dpo_beta=0.1,
        replay_buffer_size=500,
        replay_ratio=0.3,
        max_iterations=SELF_IMPROVE_ITERS,
        target_improvement=0.003,
        patience=6,            # high patience — let it run all 10 iters
        rollback_on_degradation=True,
        checkpoint_dir=os.path.join(OUTPUT_DIR, "recursive_checkpoints"),
        save_every=1,
        keep_last_n=3,
        eval_every=1,
    )

    recursive_trainer = RecursiveSelfTrainer(
        model=model,
        tokenizer=tokenizer,
        config=rec_config,
        device=DEVICE,
    )

    t0 = time.time()
    iteration_metrics: List[Dict[str, Any]] = recursive_trainer.run(
        max_iterations=SELF_IMPROVE_ITERS,
        target_improvement=rec_config.target_improvement,
        patience=rec_config.patience,
    )
    recursive_time = time.time() - t0

    # Print results table
    logger.info(f"\n  Self-improvement complete  {recursive_time:.1f}s  ({recursive_time/60:.1f}min)")
    logger.info(f"\n  {'Iter':>4}  {'Gen':>5}  {'Kept':>4}  {'Loss':>8}  "
                f"{'PPL':>8}  {'Quality':>7}  {'Δ':>8}")
    logger.info(f"  {'─'*58}")
    for m in iteration_metrics:
        logger.info(
            f"  {m['iteration']:>4}  "
            f"{m['samples_generated']:>5}  "
            f"{m['samples_kept']:>4}  "
            f"{m['train_loss']:>8.4f}  "
            f"{m.get('eval_perplexity', 0.0):>8.2f}  "
            f"{m['quality_score']:>7.4f}  "
            f"{m.get('improvement_delta', 0.0):>+8.4f}"
        )

    log_event({
        "phase": "recursive_done",
        "iterations": len(iteration_metrics),
        "duration_s": recursive_time,
        "metrics": iteration_metrics,
    })

    test_generation(model, tokenizer, tag=f"{len(iteration_metrics)} self-improvement iterations")

    # ------------------------------------------------------------------ #
    # Step 6 — Save final artefacts
    # ------------------------------------------------------------------ #
    banner("Step 6 — Save Final Artefacts")

    final_path = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save(os.path.join(OUTPUT_DIR, "tokenizer.json"))
    logger.info(f"  Final model  → {final_path}")
    logger.info(f"  Tokenizer    → {OUTPUT_DIR}/tokenizer.json")

    # Loss trajectory across self-improvement
    si_losses = [m["train_loss"] for m in iteration_metrics]
    loss_delta = (si_losses[-1] - si_losses[0]) if len(si_losses) >= 2 else 0.0
    loss_delta_pct = (loss_delta / si_losses[0] * 100) if si_losses else 0.0

    summary = {
        "model_params": n_params,
        "model_config": config.__dict__,
        "tokenizer": {"vocab_size": tokenizer.vocab_size, "num_merges": len(tokenizer._merges)},
        "corpus": {"num_books": 46, "total_samples": 33594,
                   "training_samples_used": len(dataset), "seq_len": PRETRAIN_SEQ_LEN},
        "pretrain": {
            "epochs": PRETRAIN_EPOCHS,
            "initial_loss": history["train_loss"][0],
            "final_loss": final_pretrain_loss,
            "loss_history": history["train_loss"],
            "duration_s": round(pretrain_time, 2),
        },
        "lora": lora_info,
        "self_improvement": {
            "iterations_run": len(iteration_metrics),
            "samples_per_iter": SAMPLES_PER_ITER,
            "gen_max_tokens": GEN_MAX_TOKENS,
            "use_dpo": USE_DPO,
            "lora_rank": LORA_RANK,
            "first_iter_loss": si_losses[0] if si_losses else None,
            "last_iter_loss": si_losses[-1] if si_losses else None,
            "loss_improvement_pct": round(loss_delta_pct, 2),
            "duration_s": round(recursive_time, 2),
            "metrics": iteration_metrics,
        },
        "total_duration_s": round(time.time() - t_total, 2),
    }

    summary_path = os.path.join(OUTPUT_DIR, "training_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"  Summary      → {summary_path}")

    total_time = time.time() - t_total
    banner("Training Complete!")
    logger.info(f"  Model         : {format_num(n_params)} parameters ({n_params/1e6:.1f}M)")
    logger.info(f"  Corpus        : 46 books · 33,594 total chunks · {len(dataset)} used for pre-training")
    logger.info(f"  Pre-training  : {pretrain_time:.1f}s  |  loss {history['train_loss'][0]:.4f} → {final_pretrain_loss:.4f}")
    if si_losses:
        logger.info(f"  Self-improve  : {recursive_time:.1f}s  |  {len(iteration_metrics)} iters  |  "
                    f"loss {si_losses[0]:.4f} → {si_losses[-1]:.4f}  ({loss_delta_pct:+.1f}%)")
    logger.info(f"  Total time    : {total_time:.1f}s  ({total_time/60:.1f}min)")
    logger.info(f"  Outputs       : {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
