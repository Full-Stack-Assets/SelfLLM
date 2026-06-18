"""Proximal Policy Optimization (PPO) trainer for RLHF.

Implements PPO with Generalized Advantage Estimation (GAE) for training
language models from human (or AI) preference feedback.

Components:
    - RewardModel: Scores completions with a scalar reward
    - ValueModel: Estimates state values for advantage computation
    - PPOTrainer: Orchestrates policy, reference, reward, and value models

Reference:
    - Schulman et al., "Proximal Policy Optimization Algorithms", 2017
    - Schulman et al., "High-Dimensional Continuous Control Using Generalized
      Advantage Estimation", 2016
    - Ouyang et al., "Training language models to follow instructions with
      human feedback" (InstructGPT), 2022
    - von Werra et al., "TRL: Transformer Reinforcement Learning", 2020
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from selfllm.model.model import SelfImprovingLLM

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Reward Model
# --------------------------------------------------------------------------- #

class RewardModel(nn.Module):
    """Reward model: scores the quality of a completion.

    Uses the last token's hidden state from a transformer encoder and
    projects it to a scalar reward value. Higher rewards indicate better
    completions according to the learned preference signal.

    Args:
        base_model: The transformer encoder to use for extracting
            hidden states. Can be shared with the policy model.
    """

    def __init__(self, base_model: SelfImprovingLLM) -> None:
        super().__init__()
        self.encoder = base_model
        d_model = base_model.config.d_model
        self.score_head = nn.Linear(d_model, 1, bias=False)

        # Initialize score head with small weights for stability
        nn.init.normal_(self.score_head.weight, mean=0.0, std=0.01)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute scalar reward for each sequence.

        Args:
            token_ids: Token IDs of shape ``[batch, seq_len]``.

        Returns:
            Scalar reward per sequence of shape ``[batch]``.
        """
        # Get hidden states from encoder
        outputs = self.encoder(token_ids)
        hidden_states = outputs["hidden_states"]  # [B, T, D]

        # Use the last token's hidden state (accounts for padding via
        # attention mask inside the transformer)
        last_hidden = hidden_states[:, -1, :]  # [B, D]

        # Score head -> scalar reward
        reward = self.score_head(last_hidden).squeeze(-1)  # [B]

        return reward


# --------------------------------------------------------------------------- #
# Value Model
# --------------------------------------------------------------------------- #

class ValueModel(nn.Module):
    """Value model: estimates the expected cumulative reward (state value).

    Shares the transformer architecture with the policy but has a separate
    value head. Estimates V(s_t) for each token position, used in GAE.

    Args:
        base_model: The transformer encoder to use for extracting
            hidden states.
    """

    def __init__(self, base_model: SelfImprovingLLM) -> None:
        super().__init__()
        self.encoder = base_model
        d_model = base_model.config.d_model
        self.value_head = nn.Linear(d_model, 1, bias=False)

        # Initialize value head with small weights
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.01)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute per-token value estimates for each sequence.

        Args:
            token_ids: Token IDs of shape ``[batch, seq_len]``.

        Returns:
            Per-token values of shape ``[batch, seq_len]``.
            V(s_t) for each position t in the sequence.
        """
        # Get hidden states from encoder
        outputs = self.encoder(token_ids)
        hidden_states = outputs["hidden_states"]  # [B, T, D]

        # Value head: per-token value estimates
        values = self.value_head(hidden_states).squeeze(-1)  # [B, T]

        return values


# --------------------------------------------------------------------------- #
# PPO Trainer
# --------------------------------------------------------------------------- #

class PPOTrainer:
    """PPO trainer for RLHF.

    Trains a policy model using Proximal Policy Optimization with clipped
    objective and Generalized Advantage Estimation. A frozen reference
    model prevents the policy from diverging too far from the initial model.

    Algorithm (per batch):
        1. Generate completions from the policy model.
        2. Score completions with the reward model.
        3. Compute per-token log-probabilities under policy and reference.
        4. Compute advantages using GAE (rewards + value estimates).
        5. Update policy with clipped PPO objective.
        6. Update value function with MSE loss.
        7. Add KL penalty to prevent divergence from reference.

    Args:
        policy_model: The LLM being trained (the policy).
        ref_model: Frozen reference model (pre-trained initialization).
        reward_model: Model that scores completion quality.
        value_model: Optional value model for GAE. If None, a new one
            is created sharing architecture with the policy.
        clip_epsilon: PPO clipping parameter (default: 0.2).
        kl_coeff: KL divergence penalty coefficient (default: 0.02).
        gamma: Discount factor for returns (default: 0.99).
        lam: GAE lambda parameter (default: 0.95).
        lr: Learning rate for AdamW optimizer (default: 1e-5).
        weight_decay: Weight decay coefficient (default: 0.01).
        device: Device to run on (default: "cuda").
        num_ppo_epochs: Number of PPO update epochs per batch (default: 4).
    """

    def __init__(
        self,
        policy_model: SelfImprovingLLM,
        ref_model: SelfImprovingLLM,
        reward_model: RewardModel,
        value_model: Optional[ValueModel] = None,
        clip_epsilon: float = 0.2,
        kl_coeff: float = 0.02,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        device: str = "cuda",
        num_ppo_epochs: int = 4,
    ) -> None:
        self.policy_model = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.reward_model = reward_model.to(device)

        # Value model: create if not provided
        if value_model is not None:
            self.value_model = value_model.to(device)
        else:
            # Create a value model by copying policy architecture
            value_base = copy.deepcopy(policy_model)
            self.value_model = ValueModel(value_base).to(device)

        self.clip_epsilon = clip_epsilon
        self.kl_coeff = kl_coeff
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.num_ppo_epochs = num_ppo_epochs

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        # Optimizers
        self.policy_optimizer = AdamW(
            self.policy_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.value_optimizer = AdamW(
            self.value_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Training state
        self.global_step = 0

        policy_params = sum(
            p.numel() for p in self.policy_model.parameters() if p.requires_grad
        )
        value_params = sum(
            p.numel() for p in self.value_model.parameters() if p.requires_grad
        )
        logger.info(
            f"PPOTrainer initialized: clip={clip_epsilon}, kl_coeff={kl_coeff}, "
            f"gamma={gamma}, lam={lam}, lr={lr}"
        )
        logger.info(f"Policy trainable parameters: {policy_params:,}")
        logger.info(f"Value model trainable parameters: {value_params:,}")

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def _generate_completions(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
    ) -> Dict[str, torch.Tensor]:
        """Generate completions from the policy model.

        Args:
            prompt_ids: Prompt token IDs ``[batch, prompt_len]``.
            max_new_tokens: Max tokens to generate.

        Returns:
            Dict with:
                - ``sequences``: Full sequences ``[batch, total_seq_len]``
                - ``log_probs``: Per-token log-probs under policy
                - ``ref_log_probs``: Per-token log-probs under reference
                - ``values``: Per-token value estimates
                - ``rewards``: Per-sequence rewards
        """
        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]

        self.policy_model.eval()
        self.value_model.eval()

        # Generate with the policy model
        with torch.no_grad():
            gen_result = self.policy_model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
            )

        sequences = gen_result["sequences"]  # [B, prompt_len + gen_len]
        seq_len = sequences.shape[1]

        # Compute per-token log-probs under policy (with grad for later use)
        self.policy_model.train()
        policy_outputs = self.policy_model(sequences)
        policy_logits = policy_outputs["logits"]  # [B, T, V]

        # Compute per-token log probs: log P(a_t | s_{0:t})
        # logits[:, t] predicts token t+1
        policy_log_probs_all = F.log_softmax(policy_logits, dim=-1)  # [B, T, V]

        # Gather log probs of the actual tokens in the sequence
        # For position t, the target token is sequences[:, t+1]
        target_tokens = sequences[:, 1:].unsqueeze(-1)  # [B, T-1, 1]
        policy_log_probs = (
            torch.gather(policy_log_probs_all[:, :-1, :], dim=-1, index=target_tokens)
            .squeeze(-1)
        )  # [B, T-1]

        # Also compute under reference model (no grad)
        self.policy_model.eval()
        with torch.no_grad():
            ref_outputs = self.ref_model(sequences)
            ref_logits = ref_outputs["logits"]
            ref_log_probs_all = F.log_softmax(ref_logits, dim=-1)
            ref_log_probs = (
                torch.gather(ref_log_probs_all[:, :-1, :], dim=-1, index=target_tokens)
                .squeeze(-1)
            )  # [B, T-1]

        # Compute value estimates per token
        with torch.no_grad():
            values = self.value_model(sequences)  # [B, T]
            # We need values for positions up to T-1 (values at each step)
            values = values[:, :-1]  # [B, T-1]

        # Compute rewards from reward model (sequence-level)
        with torch.no_grad():
            rewards_seq = self.reward_model(sequences)  # [B]

        # Create per-token rewards: sparse (reward only at last generated token)
        per_token_rewards = torch.zeros_like(policy_log_probs)  # [B, T-1]
        # Reward at the last generated position for each sequence
        per_token_rewards[:, -1] = rewards_seq

        # Response mask in the [B, T-1] log-prob space.
        # log-prob index t scores the target token sequences[:, t+1]. The first
        # *generated* token is sequences[:, prompt_len], which corresponds to
        # log-prob index prompt_len - 1. So response positions are indices
        # >= prompt_len - 1; all earlier indices score prompt tokens and must
        # be masked out so they never contribute to any loss term.
        num_lp = policy_log_probs.shape[1]  # T - 1
        response_start = max(prompt_len - 1, 0)
        response_mask = torch.zeros_like(policy_log_probs)
        if response_start < num_lp:
            response_mask[:, response_start:] = 1.0

        return {
            "sequences": sequences,
            "prompt_len": prompt_len,
            "policy_log_probs": policy_log_probs,  # [B, T-1]
            "ref_log_probs": ref_log_probs,  # [B, T-1]
            "values": values,  # [B, T-1]
            "rewards": per_token_rewards,  # [B, T-1] (sparse)
            "rewards_seq": rewards_seq,  # [B]
            "response_mask": response_mask,  # [B, T-1]
        }

    # ------------------------------------------------------------------ #
    # GAE computation
    # ------------------------------------------------------------------ #

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE).

        GAE blends bias and variance in advantage estimation:
            A_t^GAE = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}

        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        Fully vectorized over the batch and time dimensions: the reverse
        recurrence is unrolled with a Python loop over time steps only,
        operating on whole ``[batch]`` slices at once (no per-element
        ``.item()`` calls), which is both faster and far more memory
        efficient for long sequences.

        Args:
            rewards: Per-token rewards ``[batch, seq_len]``.
            values: Per-token value estimates ``[batch, seq_len]``.
            log_probs: Per-token log-probs (used for shape only).

        Returns:
            Tuple of:
                - advantages: ``[batch, seq_len]`` — GAE advantages
                - returns: ``[batch, seq_len]`` — TD(lambda) returns
        """
        batch_size, seq_len = rewards.shape

        advantages = torch.zeros_like(rewards)

        # V(s_{t+1}) for each position; terminal state has next value 0.
        next_values = torch.zeros_like(values)
        if seq_len > 1:
            next_values[:, :-1] = values[:, 1:]

        # TD errors for all positions at once: delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
        deltas = rewards + self.gamma * next_values - values  # [B, T]

        # Reverse cumulative GAE: loop over the time dimension only,
        # processing the whole batch as a vector at each step.
        last_gae = torch.zeros(batch_size, dtype=rewards.dtype, device=rewards.device)
        for t in reversed(range(seq_len)):
            last_gae = deltas[:, t] + self.gamma * self.lam * last_gae
            advantages[:, t] = last_gae

        returns = advantages + values

        return advantages, returns

    # ------------------------------------------------------------------ #
    # PPO loss computation
    # ------------------------------------------------------------------ #

    def _compute_ppo_loss(
        self,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        response_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO clipped loss with KL penalty and value loss.

        Only the generated/response tokens contribute to every loss term.
        Prompt positions are excluded via ``response_mask`` (1.0 for response
        tokens, 0.0 for prompt/padding positions). When no mask is supplied,
        all positions are treated as response tokens.

        Args:
            policy_log_probs: Current policy log-probs ``[batch, seq_len]``.
            old_log_probs: Old (snapshot) policy log-probs ``[batch, seq_len]``.
            ref_log_probs: Reference model log-probs ``[batch, seq_len]``.
            advantages: GAE advantages ``[batch, seq_len]``.
            values: Current value estimates ``[batch, seq_len]``.
            returns: TD(lambda) returns ``[batch, seq_len]``.
            response_mask: Optional ``[batch, seq_len]`` mask; 1.0 marks
                response tokens that should contribute to the losses, 0.0
                masks out prompt/padding positions.

        Returns:
            Dict with:
                - ``policy_loss``: Clipped PPO policy loss (scalar)
                - ``value_loss``: MSE value loss (scalar)
                - ``kl_div``: Mean KL divergence over response tokens (scalar)
                - ``total_loss``: Combined loss (scalar)
        """
        if response_mask is None:
            mask = torch.ones_like(policy_log_probs)
        else:
            mask = response_mask.to(dtype=policy_log_probs.dtype)

        # Number of contributing (response) tokens, used for masked means.
        mask_sum = mask.sum().clamp(min=1.0)

        # Log-prob ratio: pi_theta(a_t) / pi_theta_old(a_t)
        log_ratio = policy_log_probs - old_log_probs  # [B, T]
        ratio = torch.exp(log_ratio)  # [B, T]

        # Surrogate objectives
        surr1 = ratio * advantages  # [B, T]
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

        # Clipped policy loss (negative because we maximize), masked to
        # response tokens only.
        policy_loss_per_token = -torch.min(surr1, surr2)  # [B, T]
        policy_loss = (policy_loss_per_token * mask).sum() / mask_sum

        # Value loss: masked MSE between predicted values and returns.
        value_loss_per_token = (values - returns) ** 2  # [B, T]
        value_loss = (value_loss_per_token * mask).sum() / mask_sum

        # KL divergence from reference model (per-token), using the
        # non-negative k3 estimator:
        #   kl = exp(ref_lp - policy_lp) - (ref_lp - policy_lp) - 1  (>= 0)
        # Computed only over response tokens via the mask.
        ref_minus_policy = ref_log_probs - policy_log_probs  # [B, T]
        kl_div_per_token = torch.exp(ref_minus_policy) - ref_minus_policy - 1.0  # [B, T]
        kl_div = (kl_div_per_token * mask).sum() / mask_sum

        # KL penalty: encourage staying close to reference
        kl_penalty = self.kl_coeff * kl_div

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + kl_penalty

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "kl_div": kl_div,
            "total_loss": total_loss,
        }

    # ------------------------------------------------------------------ #
    # Training step
    # ------------------------------------------------------------------ #

    def train_step(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
    ) -> Dict[str, float]:
        """One PPO update step.

        Generates completions, computes rewards and advantages, then
        performs multiple epochs of PPO updates.

        Args:
            prompt_ids: Prompt token IDs ``[batch, prompt_len]``.
            max_new_tokens: Max tokens to generate per completion.

        Returns:
            Metrics dict with keys:
                - ``policy_loss``: PPO policy loss
                - ``value_loss``: Value function MSE loss
                - ``kl_div``: KL divergence from reference
                - ``reward``: Mean reward from reward model
                - ``total_loss``: Combined loss
        """
        prompt_ids = prompt_ids.to(self.device)

        # ---- Phase 1: Generate and collect data ----
        self.policy_model.eval()

        with torch.no_grad():
            rollout = self._generate_completions(prompt_ids, max_new_tokens)

        sequences = rollout["sequences"]
        old_policy_log_probs = rollout["policy_log_probs"].detach()
        ref_log_probs = rollout["ref_log_probs"].detach()
        old_values = rollout["values"].detach()
        rewards = rollout["rewards"].detach()
        rewards_seq = rollout["rewards_seq"].detach()
        response_mask = rollout["response_mask"].detach()

        # ---- Phase 2: Compute GAE advantages ----
        advantages, returns = self._compute_gae(
            rewards, old_values, old_policy_log_probs
        )

        # Zero out prompt positions so they never contribute to the update.
        advantages = advantages * response_mask
        returns = returns * response_mask

        # Normalize advantages for training stability over response tokens only.
        mask_count = response_mask.sum()
        if mask_count > 1:
            adv_mean = (advantages * response_mask).sum() / mask_count
            adv_var = ((advantages - adv_mean) ** 2 * response_mask).sum() / mask_count
            adv_std = adv_var.clamp(min=0.0).sqrt()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            # Re-mask so prompt positions stay exactly zero.
            advantages = advantages * response_mask

        # ---- Phase 3: PPO update epochs ----
        total_metrics: Dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "kl_div": 0.0,
            "total_loss": 0.0,
            "reward": rewards_seq.mean().item(),
        }

        for epoch in range(self.num_ppo_epochs):
            # Compute current policy log probs
            self.policy_model.train()
            policy_outputs = self.policy_model(sequences)
            policy_logits = policy_outputs["logits"]
            policy_log_probs_all = F.log_softmax(policy_logits, dim=-1)

            # Gather log probs of the actual generated tokens
            target_tokens = sequences[:, 1:].unsqueeze(-1)
            policy_log_probs = (
                torch.gather(policy_log_probs_all[:, :-1, :], dim=-1, index=target_tokens)
                .squeeze(-1)
            )  # [B, T-1]

            # Compute current value estimates
            self.value_model.train()
            current_values = self.value_model(sequences)[:, :-1]  # [B, T-1]

            # Compute PPO loss (response tokens only)
            loss_dict = self._compute_ppo_loss(
                policy_log_probs=policy_log_probs,
                old_log_probs=old_policy_log_probs,
                ref_log_probs=ref_log_probs,
                advantages=advantages,
                values=current_values,
                returns=returns,
                response_mask=response_mask,
            )

            # Update policy
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            loss_dict["total_loss"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(), max_norm=1.0
            )
            torch.nn.utils.clip_grad_norm_(
                self.value_model.parameters(), max_norm=1.0
            )

            self.policy_optimizer.step()
            self.value_optimizer.step()

            # Accumulate metrics
            for key in ["policy_loss", "value_loss", "kl_div", "total_loss"]:
                total_metrics[key] += loss_dict[key].item()

        # Average over PPO epochs
        num_epochs = self.num_ppo_epochs
        for key in ["policy_loss", "value_loss", "kl_div", "total_loss"]:
            total_metrics[key] /= num_epochs

        self.global_step += 1

        logger.debug(
            f"PPO step {self.global_step}: policy_loss={total_metrics['policy_loss']:.4f}, "
            f"value_loss={total_metrics['value_loss']:.4f}, "
            f"kl_div={total_metrics['kl_div']:.4f}, "
            f"reward={total_metrics['reward']:.4f}"
        )

        return total_metrics

    # ------------------------------------------------------------------ #
    # Convenience: train from text prompts
    # ------------------------------------------------------------------ #

    def train_step_from_text(
        self,
        prompts: List[str],
        tokenizer: Any,
        max_new_tokens: int = 64,
    ) -> Dict[str, float]:
        """Train step from text prompts.

        Args:
            prompts: List of prompt strings.
            tokenizer: Tokenizer with ``encode`` method.
            max_new_tokens: Max tokens to generate.

        Returns:
            Metrics dict (same as ``train_step``).
        """
        # Encode prompts
        prompt_ids_list = [tokenizer.encode(p) for p in prompts]
        max_len = max(len(p) for p in prompt_ids_list)

        # Pad to max length
        padded = []
        for p in prompt_ids_list:
            padded.append(p + [0] * (max_len - len(p)))

        prompt_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)
        return self.train_step(prompt_tensor, max_new_tokens)
