"""Core recursive self-improvement trainer.

Orchestrates the full recursive self-improvement loop:
    generate -> filter -> curate (replay) -> train -> evaluate -> decide
"""

import json
import math
import os
import random
import shutil
import time
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .recursive_config import RecursiveConfig
from .evaluator import SelfImprovementEvaluator

# Imports from sibling modules (model, training pipeline).
# These modules live on branches that are merged into the final project.
try:
    from ..model.model import SelfImprovingLLM
    from ..model.tokenizer import BPETokenizer
    from ..training.data_generator import SelfTrainingDataGenerator
    from ..training.quality_filter import QualityFilter
    from ..training.dataset import SelfTrainingDataset
except ImportError:
    # Graceful fallback for isolated branch development / type-checking.
    SelfImprovingLLM = Any  # type: ignore[misc, assignment]
    BPETokenizer = Any  # type: ignore[misc, assignment]
    SelfTrainingDataGenerator = Any  # type: ignore[misc, assignment]
    QualityFilter = Any  # type: ignore[misc, assignment]
    SelfTrainingDataset = Any  # type: ignore[misc, assignment]


class RecursiveSelfTrainer:
    """Orchestrates the recursive self-improvement loop.

    The loop consists of:
        1. Generate training data using the current model.
        2. Filter by quality using the quality filter.
        3. Combine with previous high-quality data (experience replay).
        4. Fine-tune model on the curated dataset.
        5. Evaluate improvement.
        6. If improved: save checkpoint and repeat.
        7. If degraded: rollback to previous best.

    Attributes:
        model: The SelfImprovingLLM being trained.
        tokenizer: The BPETokenizer for encoding/decoding.
        config: RecursiveConfig with all loop hyperparameters.
        device: Device string ("cuda" or "cpu").
        evaluator: SelfImprovementEvaluator for quality assessment.
        replay_buffer: List of high-quality samples from past iterations.
        iteration: Current iteration counter.
        best_metrics: Best evaluation metrics seen so far.
        best_checkpoint_path: Path to the best checkpoint.
        metrics_history: List of metrics dicts from all iterations.
    """

    def __init__(
        self,
        model: "SelfImprovingLLM",
        tokenizer: "BPETokenizer",
        config: RecursiveConfig,
        data_generator: Optional["SelfTrainingDataGenerator"] = None,
        quality_filter: Optional["QualityFilter"] = None,
        device: str = "cuda",
    ):
        """Initialize the recursive self-trainer.

        Args:
            model: The language model to improve recursively.
            tokenizer: The tokenizer for encoding/decoding.
            config: Configuration for the recursive loop.
            data_generator: Optional pre-built data generator.
                If None, one is created internally.
            quality_filter: Optional pre-built quality filter.
                If None, one is created internally.
            device: Device to run training on ("cuda" or "cpu").
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.data_generator = data_generator
        self.quality_filter = quality_filter

        # Move model to device
        self.model.to(self.device)

        # Initialize evaluator
        self.evaluator = SelfImprovementEvaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_prompts=self.config.eval_prompts,
            device=self.device,
        )

        # Experience replay buffer
        self.replay_buffer: List[Dict[str, Any]] = []

        # Iteration tracking
        self.iteration: int = 0
        self.best_metrics: Optional[Dict[str, float]] = None
        self.best_checkpoint_path: Optional[str] = None
        self.metrics_history: List[Dict[str, float]] = []
        # Persistent reward model (lazily created by _run_reward_model_step when
        # use_reward_model is enabled), kept across iterations.
        self.reward_trainer: Optional[Any] = None

        # Ensure checkpoint directory exists
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Ensure data generator and quality filter are available
        if self.data_generator is None:
            from ..training.data_generator import SelfTrainingDataGenerator

            self.data_generator = SelfTrainingDataGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )

        if self.quality_filter is None:
            from ..training.quality_filter import QualityFilter

            self.quality_filter = QualityFilter(
                base_model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )

    def run_iteration(self) -> Dict[str, float]:
        """Execute one full recursive improvement iteration.

        Steps:
            1. Generate training data via SelfTrainingDataGenerator.
            2. Filter via QualityFilter.
            3. Combine with experience replay buffer.
            4. Fine-tune model (AdamW, cosine schedule, gradient accumulation).
            5. Evaluate via SelfImprovementEvaluator.
            6. Compare to previous metrics.
            7. If improved: save checkpoint, update best.
            8. If degraded and rollback_on_degradation: load previous best.

        Returns:
            Metrics dict for this iteration including:
                iteration, samples_generated, samples_kept, train_loss,
                eval_perplexity, quality_score, improvement_delta, checkpoint_path.
        """
        self.iteration += 1
        iter_start_time = time.time()

        print(f"\n{'=' * 60}")
        print(f"  Recursive Self-Improvement Iteration {self.iteration}")
        print(f"{'=' * 60}")

        # Step 1: Generate training data
        print(f"[Step 1/7] Generating training data "
              f"({self.config.samples_per_iteration} samples, "
              f"{self.config.responses_per_prompt} responses/prompt) ...")
        raw_samples = self.data_generator.generate_training_batch(
            num_samples=self.config.samples_per_iteration,
            num_critic_samples=self.config.responses_per_prompt,
        )
        samples_generated = len(raw_samples)
        print(f"  Generated {samples_generated} raw samples.")

        # Step 2: Filter by quality
        print(f"[Step 2/7] Filtering by quality "
              f"(min_score={self.config.min_quality_score}, "
              f"keep_ratio={self.config.keep_ratio}) ...")
        filtered_samples = self.quality_filter.filter_batch(
            samples=raw_samples,
            min_quality_score=self.config.min_quality_score,
            max_perplexity=self.config.max_perplexity,
            min_diversity=self.config.min_diversity,
            keep_ratio=self.config.keep_ratio,
        )
        samples_kept = len(filtered_samples)
        print(f"  Kept {samples_kept}/{samples_generated} samples "
              f"({100 * samples_kept / max(samples_generated, 1):.1f}%).")

        # Adaptive fallback: if filtering produces too few samples, lower threshold
        min_samples_needed = max(1, int(self.config.samples_per_iteration * self.config.keep_ratio * 0.25))
        if samples_kept < min_samples_needed:
            print(f"  ⚠️  Too few samples passed filtering. "
                  f"Adaptive fallback: lowering threshold ...")
            # Retry with more lenient parameters
            filtered_samples = self.quality_filter.filter_batch(
                samples=raw_samples,
                min_quality_score=max(0.1, self.config.min_quality_score - 0.3),
                max_perplexity=self.config.max_perplexity * 2,
                min_diversity=max(0.05, self.config.min_diversity - 0.15),
                keep_ratio=min(0.8, self.config.keep_ratio * 2),
            )
            samples_kept = len(filtered_samples)
            print(f"  Kept {samples_kept}/{samples_generated} samples with fallback "
                  f"({100 * samples_kept / max(samples_generated, 1):.1f}%).")

        # Step 3: Combine with experience replay buffer
        print(f"[Step 3/7] Managing experience replay "
              f"(buffer_size={self.config.replay_buffer_size}, "
              f"replay_ratio={self.config.replay_ratio}) ...")
        training_samples = self._prepare_training_batch(filtered_samples)
        print(f"  Training batch size: {len(training_samples)} "
              f"({len(filtered_samples)} new + "
              f"{len(training_samples) - len(filtered_samples)} replay).")

        # Step 3b: optional reward-guided selection -- keep the highest-reward
        # samples (per the reward model trained in prior iterations).
        if (
            getattr(self.config, "use_reward_guided_selection", False)
            and self.reward_trainer is not None
            and len(training_samples) > 1
        ):
            training_samples = self._reward_select(training_samples)

        # Guard: skip training if no samples available
        if len(training_samples) == 0:
            print("  ⚠️  No training samples available. Skipping training step.")
            avg_loss = float('inf')
            eval_metrics = self.evaluator.full_evaluation() if self.evaluator else {"perplexity": float('inf'), "overall_quality": 0.0, "self_consistency": 0.0}
            improvement_delta = -1.0
            is_improved = False

            # Save metrics and return
            metrics = {
                "iteration": self.iteration,
                "samples_generated": samples_generated,
                "samples_kept": 0,
                "train_loss": avg_loss,
                "eval_perplexity": eval_metrics.get("perplexity", float("inf")),
                "quality_score": eval_metrics.get("overall_quality", 0.0),
                "improvement_delta": improvement_delta,
                "checkpoint_path": self.best_checkpoint_path or "",
            }
            self.metrics_history.append(metrics)
            self._save_metrics_history()
            return metrics

        # Step 4: Fine-tune model
        print(f"[Step 4/7] Fine-tuning model "
              f"(epochs={self.config.training_epochs}, "
              f"lr={self.config.learning_rate}, "
              f"batch={self.config.batch_size}, "
              f"grad_accum={self.config.gradient_accumulation_steps}) ...")
        from ..training.dataset import SelfTrainingDataset

        dataset = SelfTrainingDataset(
            samples=training_samples,
            tokenizer=self.tokenizer,
            max_seq_len=getattr(
                self.model, "config", None
            ) and getattr(self.model.config, "max_seq_len", 512) or 512,
        )
        avg_loss = self.train_step(
            dataset=dataset,
            num_epochs=self.config.training_epochs,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )
        print(f"  Average training loss: {avg_loss:.4f}")

        # Optional DPO preference-alignment pass. Self-critic generates several
        # responses per eval prompt, ranks them, and optimizes the policy toward
        # the preferred ones. Guarded so a failure here never aborts the loop.
        if getattr(self.config, "use_dpo", False):
            self._run_dpo_step()

        # Optional Constitutional AI pass: self-critique + revise to build
        # (initial, revised) preference pairs, then align toward the revised
        # responses with DPO. Guarded so a failure never aborts the loop.
        if getattr(self.config, "use_constitutional", False):
            self._run_constitutional_step()

        # Optional reasoning self-distillation: fine-tune on the model's own
        # high-confidence self-consistent reasoning traces. Guarded.
        if getattr(self.config, "use_reasoning_distillation", False):
            self._run_reasoning_distillation_step()

        # Step 5: Evaluate
        print(f"[Step 5/7] Running evaluation suite ...")
        eval_metrics = self.evaluator.full_evaluation()
        print(f"  Perplexity: {eval_metrics['perplexity']:.2f}")
        print(f"  Quality:    {eval_metrics['overall_quality']:.4f}")
        print(f"  Consistency: {eval_metrics['self_consistency']:.4f}")

        # Step 6: Compute improvement delta
        print(f"[Step 6/7] Computing improvement delta ...")
        improvement_delta = self.evaluator.compute_improvement_delta(
            current_metrics=eval_metrics,
            previous_metrics=self.best_metrics,
        )
        is_improved = improvement_delta > 0
        print(f"  Improvement delta: {improvement_delta:+.4f} "
              f"({'IMPROVED' if is_improved else 'DEGRADED' if self.best_metrics else 'BASELINE'})")

        # Step 7: Checkpoint / rollback decision
        checkpoint_path = ""
        if is_improved or self.best_metrics is None:
            # Save checkpoint
            if self.iteration % self.config.save_every == 0 or is_improved:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"iteration_{self.iteration:03d}",
                )
                print(f"[Step 7/7] Saving checkpoint to {checkpoint_path} ...")
                self.model.save_pretrained(checkpoint_path)
                self.best_checkpoint_path = checkpoint_path

                # Enforce the checkpoint retention policy.
                self._cleanup_old_checkpoints()

            # Update best metrics
            self.best_metrics = eval_metrics.copy()
        elif (
            not is_improved
            and self.config.rollback_on_degradation
            and self.best_checkpoint_path is not None
        ):
            print(f"[Step 7/7] Degradation detected. "
                  f"Rolling back to best checkpoint: {self.best_checkpoint_path} ...")
            from ..model.model import SelfImprovingLLM

            self.model = SelfImprovingLLM.from_pretrained(self.best_checkpoint_path)
            self.model.to(self.device)

            # Re-create evaluator with rolled-back model
            self.evaluator = SelfImprovementEvaluator(
                model=self.model,
                tokenizer=self.tokenizer,
                eval_prompts=self.config.eval_prompts,
                device=self.device,
            )
            checkpoint_path = self.best_checkpoint_path
        else:
            print(f"[Step 7/7] Degradation detected, rollback disabled or no best checkpoint.")
            checkpoint_path = self.best_checkpoint_path or ""

        # Step 6b: optional eval-suite benchmark hook (MMLU/GSM8K/HumanEval).
        # Runs on the model that is actually retained this iteration (post
        # rollback decision) so the recorded scores match the kept checkpoint.
        benchmark_scores: Dict[str, float] = {}
        if getattr(self.config, "run_benchmarks", False) and (
            self.iteration % max(1, getattr(self.config, "benchmark_every", 1)) == 0
        ):
            benchmark_scores = self._run_benchmarks()

        # Step 6c: optional reward-model update on self-critic preferences.
        reward_scores: Dict[str, float] = {}
        if getattr(self.config, "use_reward_model", False):
            reward_scores = self._run_reward_model_step()

        iter_duration = time.time() - iter_start_time

        # Build and store metrics dict
        iteration_metrics = {
            "iteration": self.iteration,
            "samples_generated": samples_generated,
            "samples_kept": samples_kept,
            "train_loss": avg_loss,
            "eval_perplexity": eval_metrics["perplexity"],
            "quality_score": eval_metrics["overall_quality"],
            "self_consistency": eval_metrics["self_consistency"],
            "diversity_score": eval_metrics["diversity_score"],
            "coherence_score": eval_metrics["coherence_score"],
            "repetition_rate": eval_metrics["repetition_rate"],
            "improvement_delta": improvement_delta,
            "checkpoint_path": checkpoint_path,
            "duration_seconds": iter_duration,
        }
        # Merge benchmark + reward scores into the iteration record.
        iteration_metrics.update(benchmark_scores)
        iteration_metrics.update(reward_scores)
        self.metrics_history.append(iteration_metrics)

        # Save metrics to JSON
        self._save_metrics_history()
        print(f"  Iteration duration: {iter_duration:.1f}s")

        return iteration_metrics

    def train_step(
        self,
        dataset: "SelfTrainingDataset",
        num_epochs: int = 2,
        learning_rate: float = 5e-5,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 4,
    ) -> float:
        """Fine-tune model on curated dataset.

        Uses AdamW optimizer with cosine learning rate schedule and
        gradient accumulation for effective training.

        Args:
            dataset: The SelfTrainingDataset to train on.
            num_epochs: Number of training epochs.
            learning_rate: Peak learning rate.
            batch_size: Per-device batch size.
            gradient_accumulation_steps: Steps to accumulate before updating.

        Returns:
            Average loss over the training run.
        """
        self.model.train()

        # Create data loader. Keep the final partial batch -- on the small
        # datasets produced per iteration, dropping it wastes data and biases
        # gradients toward whatever happens to fill complete batches.
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Compute total optimizer steps for the scheduler. Use ceil so the
        # final (incomplete) accumulation window, which still produces an
        # optimizer step below, is counted.
        steps_per_epoch = max(1, math.ceil(len(dataloader) / gradient_accumulation_steps))
        total_steps = max(1, steps_per_epoch * num_epochs)
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler: linear warmup + cosine decay
        from ..utils import get_lr_scheduler

        try:
            scheduler = get_lr_scheduler(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        except Exception:
            # Fallback: use torch's built-in LambdaLR with cosine schedule
            scheduler = self._build_cosine_scheduler(
                optimizer, warmup_steps, total_steps, learning_rate
            )

        total_loss = 0.0
        num_batches = 0
        global_step = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids, targets=labels)
                loss = outputs["loss"]

                # Scale loss for gradient accumulation
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Track loss (use unscaled loss for reporting)
                batch_loss = loss.item() * gradient_accumulation_steps
                epoch_loss += batch_loss
                total_loss += batch_loss
                num_batches += 1

                # Optimizer step after accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Flush gradients from a trailing incomplete accumulation window so
            # the epoch's last batches actually contribute an update instead of
            # being silently discarded.
            if len(dataloader) > 0 and (batch_idx + 1) % gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            avg_epoch_loss = epoch_loss / max(len(dataloader), 1)
            current_lr = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch + 1}/{num_epochs}: "
                  f"loss={avg_epoch_loss:.4f}, lr={current_lr:.2e}")

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def run(
        self,
        max_iterations: int = 10,
        target_improvement: float = 0.01,
        patience: int = 3,
    ) -> List[Dict[str, float]]:
        """Run the full recursive improvement loop.

        Stopping criteria:
            - max_iterations reached.
            - improvement < target_improvement for ``patience`` consecutive
              iterations.

        Args:
            max_iterations: Maximum number of improvement iterations.
            target_improvement: Minimum improvement delta to count as progress.
            patience: Number of consecutive iterations below target before stopping.

        Returns:
            List of iteration metrics dicts.
        """
        print(f"\n{'#' * 60}")
        print(f"#  Starting Recursive Self-Improvement Loop")
        print(f"#  max_iterations={max_iterations}")
        print(f"#  target_improvement={target_improvement}")
        print(f"#  patience={patience}")
        print(f"#  device={self.device}")
        print(f"{'#' * 60}\n")

        patience_counter = 0

        for iteration_num in range(1, max_iterations + 1):
            iteration_metrics = self.run_iteration()

            # Print progress summary table
            self._print_progress_table()

            # Check stopping criteria
            improvement = iteration_metrics["improvement_delta"]

            if iteration_num == 1 and self.best_metrics is not None:
                # First iteration: don't check patience
                continue

            if improvement < target_improvement:
                patience_counter += 1
                print(f"\n  Improvement ({improvement:+.4f}) below target "
                      f"({target_improvement}). Patience: {patience_counter}/{patience}")
            else:
                patience_counter = 0
                print(f"\n  Improvement ({improvement:+.4f}) meets target.")

            if patience_counter >= patience:
                print(f"\n{'=' * 60}")
                print(f"  Stopping: no improvement for {patience} consecutive iterations.")
                print(f"{'=' * 60}")
                break

        # Final summary
        self._print_final_summary()

        return self.metrics_history

    # --- Internal helpers ---

    def _prepare_training_batch(
        self, new_samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine new samples with experience replay buffer.

        Updates the replay buffer with new samples (maintaining max size)
        and samples from the buffer according to replay_ratio.

        Args:
            new_samples: Newly generated and filtered samples.

        Returns:
            Combined list of new + replay samples.
        """
        # Add new samples to replay buffer
        self.replay_buffer.extend(new_samples)

        # Trim buffer to max size (FIFO eviction)
        if len(self.replay_buffer) > self.config.replay_buffer_size:
            excess = len(self.replay_buffer) - self.config.replay_buffer_size
            self.replay_buffer = self.replay_buffer[excess:]

        # Sample from replay buffer
        if self.replay_buffer and self.config.replay_ratio > 0:
            num_replay = int(
                len(new_samples) * self.config.replay_ratio
            )
            if num_replay > 0:
                replay_samples = random.sample(
                    self.replay_buffer,
                    min(num_replay, len(self.replay_buffer)),
                )
                return new_samples + replay_samples

        return new_samples

    def _build_cosine_scheduler(
        self,
        optimizer: AdamW,
        num_warmup_steps: int,
        num_training_steps: int,
        base_lr: float,
    ):
        """Build a cosine learning rate scheduler with linear warmup.

        Args:
            optimizer: The optimizer to schedule.
            num_warmup_steps: Number of warmup steps.
            num_training_steps: Total number of training steps.
            base_lr: Peak learning rate.

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
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _print_progress_table(self) -> None:
        """Print a formatted progress table of all iterations."""
        if not self.metrics_history:
            return

        print(f"\n  {'Iter':>4} | {'Gen':>5} | {'Kept':>5} | {'Loss':>7} "
              f"| {'PPL':>8} | {'Quality':>7} | {'Delta':>7} | {'Status':>8}")
        print(f"  {'-' * 4}-+-{'-' * 5}-+-{'-' * 5}-+-{'-' * 7}"
              f"-+-{'-' * 8}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 8}")

        for m in self.metrics_history:
            delta = m["improvement_delta"]
            status = "BEST" if delta > 0 else "BASE" if m["iteration"] == 1 else "ROLL"
            print(
                f"  {m['iteration']:>4} | "
                f"{m['samples_generated']:>5} | "
                f"{m['samples_kept']:>5} | "
                f"{m['train_loss']:>7.4f} | "
                f"{m['eval_perplexity']:>8.2f} | "
                f"{m['quality_score']:>7.4f} | "
                f"{delta:>+7.4f} | "
                f"{status:>8}"
            )

    def _print_final_summary(self) -> None:
        """Print a final summary of the recursive improvement loop."""
        print(f"\n{'#' * 60}")
        print(f"#  Recursive Self-Improvement Complete")
        print(f"{'#' * 60}")
        print(f"  Total iterations: {len(self.metrics_history)}")
        print(f"  Best checkpoint: {self.best_checkpoint_path or 'N/A'}")

        if self.best_metrics:
            print(f"\n  Best metrics:")
            for key, value in self.best_metrics.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")

        if self.metrics_history:
            first = self.metrics_history[0]
            last = self.metrics_history[-1]
            ppl_change = last["eval_perplexity"] - first["eval_perplexity"]
            quality_change = last["quality_score"] - first["quality_score"]
            print(f"\n  Overall change:")
            print(f"    Perplexity: {ppl_change:+.4f} "
                  f"({first['eval_perplexity']:.2f} -> {last['eval_perplexity']:.2f})")
            print(f"    Quality:    {quality_change:+.4f} "
                  f"({first['quality_score']:.4f} -> {last['quality_score']:.4f})")

            # Surface any eval-suite benchmark trajectories recorded by the hook.
            from .report import detect_benchmark_keys, summarize_metrics

            benchmark_keys = detect_benchmark_keys(self.metrics_history)
            if benchmark_keys:
                summary = summarize_metrics(self.metrics_history)
                print(f"\n  Benchmarks (first -> last):")
                for key in benchmark_keys:
                    if key in summary:
                        s = summary[key]
                        print(f"    {key[len('benchmark_'):]}: "
                              f"{s['first']:.4f} -> {s['last']:.4f} ({s['delta']:+.4f})")

        print(f"\n  Metrics saved to: "
              f"{os.path.join(self.config.checkpoint_dir, 'metrics_history.json')}")
        print(f"{'#' * 60}\n")

    def _run_dpo_step(self) -> None:
        """Run a DPO preference-alignment pass over the eval prompts.

        Builds (prompt, chosen, rejected) pairs via the DPO trainer's
        self-critic generator and optimizes the policy model in place. Wrapped
        in a guard so any failure degrades gracefully instead of aborting the
        recursive loop.
        """
        try:
            from ..training.dpo_trainer import DPOTrainer

            print("[Step 4b] DPO preference alignment ...")
            dpo = DPOTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                beta=self.config.dpo_beta,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                device=self.device,
            )
            pairs = dpo.generate_preferences(
                self.config.eval_prompts,
                num_samples=max(2, self.config.responses_per_prompt),
                temperature=self.config.generation_temperature,
                top_p=self.config.generation_top_p,
            )
            if not pairs:
                print("  No preference pairs generated; skipping DPO.")
                return
            metrics = dpo.train_step(pairs, num_epochs=self.config.dpo_epochs)
            print(
                f"  DPO: loss={metrics.get('loss', 0.0):.4f}, "
                f"reward_margin={metrics.get('reward_margin', 0.0):.4f}, "
                f"accuracy={metrics.get('accuracy', 0.0):.4f}"
            )
        except Exception as e:  # pragma: no cover - defensive guard
            print(f"  ⚠️  DPO step failed ({type(e).__name__}: {e}); continuing.")

    def _run_constitutional_step(self) -> None:
        """Run a Constitutional AI self-critique + DPO alignment pass.

        Builds (initial, revised) preference pairs via the Constitutional AI
        critique/revise pipeline and optimizes the policy toward the revised
        (preferred) responses. Guarded so any failure degrades gracefully.
        """
        try:
            from ..training.constitutional_ai import ConstitutionalAI
            from ..training.dpo_trainer import DPOTrainer

            print("[Step 4c] Constitutional AI critique + revise ...")
            cai = ConstitutionalAI(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )
            pairs = cai.generate_training_pairs(
                self.config.eval_prompts,
                num_pairs=max(2, len(self.config.eval_prompts)),
            )
            if not pairs:
                print("  No constitutional pairs generated; skipping.")
                return
            dpo = DPOTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                beta=self.config.dpo_beta,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                device=self.device,
            )
            metrics = dpo.train_step(pairs, num_epochs=self.config.dpo_epochs)
            print(
                f"  Constitutional+DPO: loss={metrics.get('loss', 0.0):.4f}, "
                f"accuracy={metrics.get('accuracy', 0.0):.4f}"
            )
        except Exception as e:  # pragma: no cover - defensive guard
            print(f"  ⚠️  Constitutional step failed ({type(e).__name__}: {e}); continuing.")

    def _run_benchmarks(self) -> Dict[str, float]:
        """Run the configured eval-suite benchmarks on the current model.

        Evaluates whichever of MMLU / GSM8K / HumanEval have a source set in the
        config (``mmlu_source`` / ``gsm8k_source`` / ``humaneval_source``), each
        capped at ``benchmark_limit`` examples for speed, and returns a flat
        ``{"benchmark_<name>": score}`` dict to merge into the iteration metrics.

        Guarded so any benchmark failure (or a missing eval package) degrades
        gracefully and never aborts the recursive loop.
        """
        sources = {
            "mmlu_source": getattr(self.config, "mmlu_source", None),
            "gsm8k_source": getattr(self.config, "gsm8k_source", None),
            "humaneval_source": getattr(self.config, "humaneval_source", None),
        }
        if not any(sources.values()):
            return {}

        scores: Dict[str, float] = {}
        try:
            from ..eval import run_all

            print("[Step 6b] Running eval-suite benchmarks ...")
            was_training = self.model.training
            self.model.eval()
            try:
                results, _ = run_all(
                    self.model,
                    self.tokenizer,
                    limit=getattr(self.config, "benchmark_limit", 20),
                    max_new_tokens=getattr(self.config, "benchmark_max_new_tokens", 16),
                    **sources,
                )
            finally:
                if was_training:
                    self.model.train()

            for r in results:
                scores[f"benchmark_{r.name}"] = float(r.score)
                print(f"  {r.name}: {r.score:.4f} (n={r.n})")
        except Exception as e:  # pragma: no cover - defensive guard
            print(f"  ⚠️  Benchmark hook failed ({type(e).__name__}: {e}); skipping.")

        return scores

    def _run_reasoning_distillation_step(self) -> None:
        """Fine-tune on the model's own high-confidence reasoning traces.

        Runs self-consistency over the eval prompts; for each prompt whose
        consensus confidence clears ``distillation_confidence``, keeps the
        winning reasoning trace and fine-tunes (LM objective) on it. This
        distills the model toward its own best reasoning. Guarded so any
        failure degrades gracefully.
        """
        try:
            from ..reasoning import SelfConsistencyStrategy
            from ..reasoning.extract import normalize
            from ..training.dataset import SelfTrainingDataset

            print("[Step 4d] Reasoning self-distillation ...")
            strategy = SelfConsistencyStrategy(
                self.model, self.tokenizer, answer_type="free", device=self.device,
                num_samples=self.config.distillation_num_samples,
                temperature=self.config.generation_temperature,
                top_p=self.config.generation_top_p,
                max_think_tokens=self.config.distillation_max_new_tokens,
                max_answer_tokens=32,
            )
            samples = []
            for question in self.config.eval_prompts:
                result = strategy.solve(question)
                if (
                    result.answer is not None
                    and result.confidence >= self.config.distillation_confidence
                    and result.traces
                ):
                    # Use a trace that produced the winning (consensus) answer.
                    win = normalize(result.answer)
                    answers = result.details.get("answers", [])
                    idx = next(
                        (i for i, a in enumerate(answers) if normalize(a) == win), 0
                    )
                    # Train (LM objective) on the full reasoning trace.
                    samples.append({"prompt": "", "response": result.traces[idx]})

            if not samples:
                print("  No high-confidence reasoning traces; skipping.")
                return

            max_seq = getattr(getattr(self.model, "config", None), "max_seq_len", 512)
            dataset = SelfTrainingDataset(
                samples=samples, tokenizer=self.tokenizer, max_seq_len=max_seq,
            )
            loss = self.train_step(
                dataset,
                num_epochs=1,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gradient_accumulation_steps=1,
            )
            print(f"  Distilled {len(samples)} reasoning traces, loss={loss:.4f}")
        except Exception as e:  # pragma: no cover - defensive guard
            print(f"  ⚠️  Reasoning distillation failed "
                  f"({type(e).__name__}: {e}); continuing.")

    def _run_reward_model_step(self) -> Dict[str, float]:
        """Update a persistent reward model on self-critic preference pairs.

        Builds preference pairs over the eval prompts (DPO self-critic ranking),
        trains the reward model with the Bradley-Terry objective, and returns its
        ranking accuracy/margin as ``reward_accuracy`` / ``reward_margin`` to
        record in the metrics. The reward model persists across iterations on
        ``self.reward_trainer`` so it improves alongside the policy. Guarded.
        """
        try:
            import copy

            from ..training.dpo_trainer import DPOTrainer
            from ..training.ppo_trainer import RewardModel
            from ..training.reward_trainer import RewardModelTrainer

            print("[Step 6c] Updating reward model on self-critic preferences...")
            # Lazily create the reward model (a copy of the current policy with a
            # scalar head) and keep its trainer across iterations.
            if getattr(self, "reward_trainer", None) is None:
                reward_model = RewardModel(copy.deepcopy(self.model))
                self.reward_trainer = RewardModelTrainer(
                    reward_model, self.tokenizer,
                    learning_rate=self.config.learning_rate,
                    batch_size=self.config.batch_size, device=self.device,
                )

            pairs = DPOTrainer(
                self.model, self.tokenizer, device=self.device
            ).generate_preferences(
                self.config.eval_prompts,
                num_samples=self.config.reward_num_samples,
                temperature=self.config.generation_temperature,
                top_p=self.config.generation_top_p,
            )
            if not pairs:
                print("  No preference pairs generated; skipping reward update.")
                return {}

            self.reward_trainer.train_step(
                pairs, num_epochs=self.config.reward_model_epochs
            )
            metrics = self.reward_trainer.evaluate(pairs)
            print(f"  Reward model: accuracy={metrics['accuracy']:.4f}, "
                  f"margin={metrics['reward_margin']:.4f} ({len(pairs)} pairs)")
            return {
                "reward_accuracy": metrics["accuracy"],
                "reward_margin": metrics["reward_margin"],
            }
        except Exception as e:  # pragma: no cover - defensive guard
            print(f"  ⚠️  Reward-model step failed "
                  f"({type(e).__name__}: {e}); continuing.")
            return {}

    def _reward_select(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep the top ``reward_keep_fraction`` of samples by reward-model score.

        Scores each ``prompt + response`` with the persistent reward model and
        retains the highest-scoring fraction, so the model fine-tunes on its own
        best (highest-reward) generations. Guarded: on any failure all samples
        are kept.
        """
        try:
            keep_fraction = getattr(self.config, "reward_keep_fraction", 0.7)
            scored = [
                (
                    self.reward_trainer.score(
                        s.get("prompt", "") + s.get("response", "")
                    ),
                    s,
                )
                for s in samples
            ]
            scored.sort(key=lambda pair: pair[0], reverse=True)
            n_keep = max(1, int(len(scored) * keep_fraction))
            selected = [s for _, s in scored[:n_keep]]
            print(f"  [Step 3b] Reward-guided selection: kept "
                  f"{len(selected)}/{len(samples)} highest-reward samples.")
            return selected
        except Exception as e:  # pragma: no cover - defensive guard
            print(f"  ⚠️  Reward-guided selection failed "
                  f"({type(e).__name__}: {e}); using all samples.")
            return samples

    def _save_metrics_history(self) -> None:
        """Persist the full metrics history to JSON in the checkpoint dir."""
        metrics_path = os.path.join(
            self.config.checkpoint_dir, "metrics_history.json"
        )
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"  Metrics saved to {metrics_path}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N.

        The current best checkpoint is always preserved so rollback remains
        possible even if it is older than the retention window.
        """
        best_dir = (
            os.path.basename(self.best_checkpoint_path)
            if self.best_checkpoint_path
            else None
        )
        checkpoint_dirs = [
            d
            for d in os.listdir(self.config.checkpoint_dir)
            if d.startswith("iteration_") and os.path.isdir(
                os.path.join(self.config.checkpoint_dir, d)
            )
        ]
        checkpoint_dirs.sort()

        if len(checkpoint_dirs) > self.config.keep_last_n:
            to_remove = checkpoint_dirs[: -self.config.keep_last_n]
            for d in to_remove:
                if d == best_dir:
                    continue  # never delete the best checkpoint
                path = os.path.join(self.config.checkpoint_dir, d)
                shutil.rmtree(path, ignore_errors=True)
                print(f"  Cleaned up old checkpoint: {path}")
