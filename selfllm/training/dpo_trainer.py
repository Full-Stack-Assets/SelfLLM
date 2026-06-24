"""Direct Preference Optimization (DPO) trainer for self-improving LLMs.

DPO directly optimizes a language model policy from pairwise preference data
without an explicit reward model. The key idea is to express the RLHF
objective purely in terms of the policy and a frozen reference model.

Reference: Rafailov et al., "Direct Preference Optimization: Your Language Model
is Secretly a Reward Model", NeurIPS 2023.
"""

import copy
import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """Dataset for DPO training: (prompt, chosen, rejected) triples.

    Encodes prompt+response pairs and prepares labels where prompt tokens are
    masked with -100 so the model only learns to predict response tokens.
    """

    def __init__(
        self,
        pairs: List[Dict[str, str]],
        tokenizer: BPETokenizer,
        max_seq_len: int = 512,
    ) -> None:
        """Initialize the preference dataset.

        Args:
            pairs: List of dicts with keys "prompt", "chosen", "rejected".
            tokenizer: BPETokenizer for encoding text.
            max_seq_len: Maximum sequence length (truncate longer sequences).
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single preference pair as tensors.

        Returns:
            Dict with chosen/rejected input_ids, attention_mask, labels,
            and prompt_len for masking during log-prob computation.
        """
        pair = self.pairs[idx]
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]

        # Encode: prompt + response
        prompt_ids = self.tokenizer.encode(prompt)
        chosen_ids = self.tokenizer.encode(chosen)
        rejected_ids = self.tokenizer.encode(rejected)

        # Full sequences: prompt + response, truncated to max_seq_len
        chosen_full = (prompt_ids + chosen_ids)[: self.max_seq_len]
        rejected_full = (prompt_ids + rejected_ids)[: self.max_seq_len]

        # Attention masks (all real tokens = 1)
        chosen_mask = [1] * len(chosen_full)
        rejected_mask = [1] * len(rejected_full)

        # Labels: -100 for prompt portion (to be ignored), actual tokens for response
        # We keep the same length as chosen_full / rejected_full
        chosen_labels = [-100] * len(prompt_ids) + chosen_ids
        chosen_labels = chosen_labels[: self.max_seq_len]

        rejected_labels = [-100] * len(prompt_ids) + rejected_ids
        rejected_labels = rejected_labels[: self.max_seq_len]

        return {
            "chosen_input_ids": torch.tensor(chosen_full, dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_mask, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_full, dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected_mask, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
            "prompt_len": torch.tensor(len(prompt_ids), dtype=torch.long),
        }


def preference_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Collate preference samples with padding to max length in batch.

    Args:
        batch: List of samples from PreferenceDataset.
        pad_token_id: Token ID to use for padding input_ids and attention_mask.

    Returns:
        Batched tensors with appropriate padding.
    """
    keys = batch[0].keys()
    collated: Dict[str, torch.Tensor] = {}

    for key in keys:
        if key == "prompt_len":
            # Stack scalar prompt lengths
            collated[key] = torch.tensor(
                [item[key] for item in batch], dtype=torch.long
            )
        elif "labels" in key:
            # Labels: pad with -100 (ignore index)
            sequences = [item[key] for item in batch]
            max_len = max(seq.size(0) for seq in sequences)
            padded = torch.full(
                (len(sequences), max_len), -100, dtype=sequences[0].dtype
            )
            for i, seq in enumerate(sequences):
                padded[i, : seq.size(0)] = seq
            collated[key] = padded
        else:
            # Input_ids and attention_mask: pad with pad_token_id / 0
            sequences = [item[key] for item in batch]
            max_len = max(seq.size(0) for seq in sequences)
            padded = torch.full(
                (len(sequences), max_len), pad_token_id, dtype=sequences[0].dtype
            )
            for i, seq in enumerate(sequences):
                padded[i, : seq.size(0)] = seq
            collated[key] = padded

    return collated


class DPOTrainer:
    """Direct Preference Optimization trainer.

    Given pairs of (prompt, chosen_response, rejected_response), optimizes:
        L_DPO = -log sigmoid(beta * (log_pi(chosen|prompt) - log_pi(rejected|prompt)
                                    - (log_ref(chosen|prompt) - log_ref(rejected|prompt))))

    No reward model is needed — the policy is optimized directly from
    preferences using a frozen reference model.

    Args:
        model: The policy model to optimize (SelfImprovingLLM).
        tokenizer: BPETokenizer for encoding/decoding text.
        reference_model: Frozen reference model. Defaults to a copy of ``model``.
        beta: DPO temperature parameter controlling deviation from reference.
        label_smoothing: Label smoothing for the DPO loss (0 = no smoothing).
        learning_rate: Learning rate for the AdamW optimizer.
        weight_decay: Weight decay coefficient.
        batch_size: Training batch size.
        device: Device to run training on ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        reference_model: Optional[SelfImprovingLLM] = None,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        batch_size: int = 4,
        device: str = "cuda",
    ) -> None:
        self.policy_model = model.to(device)
        self.tokenizer = tokenizer
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = device

        # Reference model: frozen copy of initial model
        if reference_model is not None:
            self.reference_model = reference_model.to(device)
        else:
            logger.info("Creating reference model as frozen copy of policy model")
            self.reference_model = copy.deepcopy(model)
            self.reference_model = self.reference_model.to(device)

        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()

        # Optimizer (policy model parameters only)
        self.optimizer = self._create_optimizer()

        # Training state
        self.global_step = 0
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "reward_margin": [],
            "accuracy": [],
        }

        logger.info(
            f"DPOTrainer initialized: beta={beta}, lr={learning_rate}, "
            f"label_smoothing={label_smoothing}, batch_size={batch_size}"
        )
        policy_params = sum(
            p.numel() for p in self.policy_model.parameters() if p.requires_grad
        )
        logger.info(f"Policy trainable parameters: {policy_params:,}")

    # ------------------------------------------------------------------ #
    # Optimizer
    # ------------------------------------------------------------------ #

    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer for policy model parameters.

        Returns:
            AdamW optimizer instance.
        """
        # Separate parameters that should/shouldn't get weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.policy_model.named_parameters():
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

    # ------------------------------------------------------------------ #
    # Log-probability computation
    # ------------------------------------------------------------------ #

    def _compute_log_probs(
        self,
        model: SelfImprovingLLM,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample average log-probability of the response portion.

        The response portion is determined by ``labels != -100`` — prompt tokens
        are marked with -100 and are excluded from the average.

        Args:
            model: The model to compute log-probs through.
            input_ids: Token IDs of shape ``[batch, seq_len]``.
            labels: Labels of shape ``[batch, seq_len]`` where -100 marks
                    prompt tokens to be ignored.

        Returns:
            Per-sample average log-probability of shape ``[batch]``.
        """
        outputs = model(input_ids)
        logits = outputs["logits"]  # [B, T, V]

        # Shift for next-token prediction: logits at position t predict token t+1
        shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
        shift_labels = labels[:, 1:].contiguous()  # [B, T-1]

        # Mask: 1 for response tokens, 0 for prompt tokens (-100)
        mask = (shift_labels != -100).float()  # [B, T-1]

        # Replace -100 with 0 for gather (will be masked out anyway)
        safe_labels = shift_labels.clone()
        safe_labels[safe_labels == -100] = 0

        # Compute log probs over vocabulary
        log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]

        # Gather log probs for actual tokens
        safe_labels_unsq = safe_labels.unsqueeze(-1)  # [B, T-1, 1]
        token_log_probs = (
            torch.gather(log_probs, -1, safe_labels_unsq).squeeze(-1)
        )  # [B, T-1]

        # Mask out prompt positions and average over response tokens only
        masked_log_probs = token_log_probs * mask
        avg_log_probs = masked_log_probs.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

        return avg_log_probs

    # ------------------------------------------------------------------ #
    # DPO Loss
    # ------------------------------------------------------------------ #

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute DPO loss and metrics.

        The DPO loss is:
            loss = -log sigmoid(beta * ((chosen_logp - rejected_logp)
                                        - (ref_chosen_logp - ref_rejected_logp)))

        With label smoothing, we interpolate toward a uniform target:
            loss = -log sigmoid(beta * delta) * (1 - smoothing)
                   - log sigmoid(-beta * delta) * smoothing

        Args:
            policy_chosen_logps: Policy log-probs for chosen responses ``[batch]``.
            policy_rejected_logps: Policy log-probs for rejected responses ``[batch]``.
            reference_chosen_logps: Reference log-probs for chosen ``[batch]``.
            reference_rejected_logps: Reference log-probs for rejected ``[batch]``.

        Returns:
            Dict with keys:
                - ``loss``: scalar DPO loss
                - ``reward_margin``: mean(sigmoid(beta * delta))
                - ``accuracy``: fraction of samples where chosen > rejected
                - ``chosen_rewards``: policy_chosen_logps - ref_chosen_logps
                - ``rejected_rewards``: policy_rejected_logps - ref_rejected_logps
        """
        # Implicit rewards: log_pi(y|x) - log_ref(y|x)
        chosen_rewards = policy_chosen_logps - reference_chosen_logps  # [B]
        rejected_rewards = policy_rejected_logps - reference_rejected_logps  # [B]

        # Preference margin: how much better is chosen than rejected
        delta = chosen_rewards - rejected_rewards  # [B]

        # DPO loss with optional label smoothing
        if self.label_smoothing > 0:
            # Mix between standard DPO and reversed DPO
            loss = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )
        else:
            loss = -F.logsigmoid(self.beta * delta)

        # Metrics
        reward_margin = torch.sigmoid(self.beta * delta).mean()
        accuracy = (delta > 0).float().mean()

        return {
            "loss": loss.mean(),
            "reward_margin": reward_margin,
            "accuracy": accuracy,
            "chosen_rewards": chosen_rewards.detach().mean(),
            "rejected_rewards": rejected_rewards.detach().mean(),
            "delta": delta.detach().mean(),
        }

    # ------------------------------------------------------------------ #
    # Training step
    # ------------------------------------------------------------------ #

    def train_step(
        self,
        preference_pairs: List[Dict[str, str]],
        num_epochs: int = 1,
    ) -> Dict[str, float]:
        """Train on preference pairs.

        Args:
            preference_pairs: List of dicts with keys "prompt", "chosen", "rejected".
            num_epochs: Number of epochs to train for.

        Returns:
            Metrics dict with keys: loss, reward_margin, accuracy.
        """
        if not preference_pairs:
            logger.warning("Empty preference_pairs provided, returning zero metrics")
            return {"loss": 0.0, "reward_margin": 0.0, "accuracy": 0.0}

        self.policy_model.train()
        self.reference_model.eval()

        # Create dataset and dataloader. Cap sequence length at the model's
        # context window so prompt+response never exceeds the positional range
        # (which would otherwise raise a shape mismatch in the forward pass).
        model_max = getattr(
            getattr(self.policy_model, "config", None), "max_seq_len", 512
        )
        dataset = PreferenceDataset(
            preference_pairs, self.tokenizer, max_seq_len=min(512, model_max)
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: preference_collate_fn(
                batch, pad_token_id=self.tokenizer.pad_token_id
            ),
            num_workers=0,
        )

        total_loss = 0.0
        total_reward_margin = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_reward_margin = 0.0
            epoch_accuracy = 0.0
            epoch_batches = 0

            pbar = tqdm(
                dataloader,
                desc=f"DPO Epoch {epoch + 1}/{num_epochs}",
                leave=False,
            )

            for batch in pbar:
                self.optimizer.zero_grad()

                # Move to device
                chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                chosen_labels = batch["chosen_labels"].to(self.device)
                rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                rejected_labels = batch["rejected_labels"].to(self.device)

                # Compute policy log-probs
                policy_chosen_logps = self._compute_log_probs(
                    self.policy_model, chosen_input_ids, chosen_labels
                )
                policy_rejected_logps = self._compute_log_probs(
                    self.policy_model, rejected_input_ids, rejected_labels
                )

                # Compute reference log-probs (no gradients)
                with torch.no_grad():
                    reference_chosen_logps = self._compute_log_probs(
                        self.reference_model, chosen_input_ids, chosen_labels
                    )
                    reference_rejected_logps = self._compute_log_probs(
                        self.reference_model, rejected_input_ids, rejected_labels
                    )

                # Compute DPO loss
                metrics = self.compute_dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                )

                loss = metrics["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), max_norm=1.0
                )
                self.optimizer.step()

                self.global_step += 1

                # Track metrics
                epoch_loss += loss.item()
                epoch_reward_margin += metrics["reward_margin"].item()
                epoch_accuracy += metrics["accuracy"].item()
                epoch_batches += 1

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{metrics['accuracy'].item():.4f}",
                        "margin": f"{metrics['reward_margin'].item():.4f}",
                    }
                )

            # Accumulate epoch metrics
            if epoch_batches > 0:
                total_loss += epoch_loss / epoch_batches
                total_reward_margin += epoch_reward_margin / epoch_batches
                total_accuracy += epoch_accuracy / epoch_batches
                num_batches += 1

                logger.info(
                    f"DPO Epoch {epoch + 1}/{num_epochs} - "
                    f"loss: {epoch_loss / epoch_batches:.4f}, "
                    f"accuracy: {epoch_accuracy / epoch_batches:.4f}, "
                    f"reward_margin: {epoch_reward_margin / epoch_batches:.4f}"
                )

        # Average across epochs
        result = {
            "loss": total_loss / max(num_batches, 1),
            "reward_margin": total_reward_margin / max(num_batches, 1),
            "accuracy": total_accuracy / max(num_batches, 1),
        }

        self.history["loss"].append(result["loss"])
        self.history["reward_margin"].append(result["reward_margin"])
        self.history["accuracy"].append(result["accuracy"])

        return result

    # ------------------------------------------------------------------ #
    # Preference generation (self-critic)
    # ------------------------------------------------------------------ #

    def generate_preferences(
        self,
        prompts: List[str],
        num_samples: int = 4,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> List[Dict[str, str]]:
        """Self-critic: generate N responses per prompt and rank them.

        Generates ``num_samples`` responses for each prompt, scores each with
        the quality filter (perplexity + diversity + coherence), and selects
        the best as 'chosen' and worst as 'rejected'.

        Args:
            prompts: List of prompt strings.
            num_samples: Number of responses to generate per prompt.
            temperature: Sampling temperature for generation.
            top_p: Nucleus sampling probability threshold.

        Returns:
            List of preference dicts: ``{"prompt": str, "chosen": str, "rejected": str}``.
        """
        from selfllm.training.quality_filter import QualityFilter

        quality_filter = QualityFilter(
            base_model=self.policy_model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        preference_pairs: List[Dict[str, str]] = []

        for prompt in tqdm(prompts, desc="Generating preferences"):
            # Generate N responses
            prompt_ids = torch.tensor(
                [self.tokenizer.encode(prompt)], device=self.device
            )

            responses: List[str] = []
            for _ in range(num_samples):
                with torch.no_grad():
                    result = self.policy_model.generate(
                        prompt_ids=prompt_ids,
                        max_new_tokens=64,
                        temperature=temperature,
                        top_p=top_p,
                    )
                sequences = result["sequences"][0].tolist()
                # Decode only the response portion (skip prompt)
                prompt_len = prompt_ids.shape[1]
                response_ids = sequences[prompt_len:]
                response_text = self.tokenizer.decode(response_ids)
                responses.append(response_text)

            # Score each response
            scored = []
            for resp in responses:
                perplexity = quality_filter.compute_perplexity(resp)
                diversity = quality_filter.compute_diversity_score(resp)
                coherence = quality_filter.compute_coherence_score(resp)

                # Composite score: lower perplexity, higher diversity/coherence
                normalized_ppl = max(
                    0.0, min(1.0, 1.0 - (perplexity / 100.0))
                )
                composite_score = (
                    0.3 * normalized_ppl + 0.35 * diversity + 0.35 * coherence
                )
                scored.append((composite_score, resp))

            if len(scored) < 2:
                # Not enough samples, skip
                continue

            # Sort by score descending: best = chosen, worst = rejected
            scored.sort(key=lambda x: x[0], reverse=True)
            chosen = scored[0][1]
            rejected = scored[-1][1]

            preference_pairs.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

        logger.info(f"Generated {len(preference_pairs)} preference pairs")
        return preference_pairs

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save the policy model, optimizer state, and training history.

        Args:
            path: Directory path to save to (created if it doesn't exist).
        """
        os.makedirs(path, exist_ok=True)

        # Save policy model
        torch.save(
            self.policy_model.state_dict(),
            os.path.join(path, "policy_model.pt"),
        )

        # Save optimizer state
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(path, "optimizer.pt"),
        )

        # Save training history
        import json

        with open(os.path.join(path, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Saved DPO trainer state to {path}")

    def load(self, path: str) -> None:
        """Load the policy model, optimizer state, and training history.

        Args:
            path: Directory path to load from.
        """
        # Load policy model
        policy_state = torch.load(
            os.path.join(path, "policy_model.pt"),
            map_location=self.device,
            weights_only=True,
        )
        self.policy_model.load_state_dict(policy_state)

        # Load optimizer state
        opt_state = torch.load(
            os.path.join(path, "optimizer.pt"),
            map_location=self.device,
            weights_only=True,
        )
        self.optimizer.load_state_dict(opt_state)

        # Load training history
        import json

        with open(os.path.join(path, "history.json"), "r") as f:
            self.history = json.load(f)

        logger.info(f"Loaded DPO trainer state from {path}")
