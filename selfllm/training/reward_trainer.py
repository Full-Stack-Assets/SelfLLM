"""Reward-model training via the Bradley-Terry pairwise ranking loss.

Closes the gap where PPO ran against an untrained, random reward head: given
preference pairs ``(prompt, chosen, rejected)``, this trains a
:class:`~selfllm.training.ppo_trainer.RewardModel` to score ``chosen`` above
``rejected`` by minimizing ``-log sigmoid(r_chosen - r_rejected)``.

Sequences are scored one at a time (no right-padding), so each reward uses the
sequence's *true* last-token hidden state -- the RewardModel takes
``hidden_states[:, -1]`` and the base model has no padding mask, so batching
with padding would otherwise score a pad token.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.optim import AdamW


class RewardModelTrainer:
    """Train a reward model from preference pairs.

    Args:
        reward_model: A ``RewardModel`` exposing ``forward([B, T]) -> [B]``.
        tokenizer: Tokenizer with ``encode``.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        batch_size: Pairs per optimizer step.
        device: Torch device string.
        max_seq_len: Truncate each ``prompt + response`` to this many tokens.
    """

    def __init__(
        self,
        reward_model: Any,
        tokenizer: Any,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        batch_size: int = 8,
        device: str = "cpu",
        max_seq_len: int = 512,
    ) -> None:
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.max_seq_len = max_seq_len
        self.optimizer = AdamW(
            self.reward_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _reward(self, text: str) -> torch.Tensor:
        """Scalar reward for one text (no padding)."""
        ids = self.tokenizer.encode(text)[: self.max_seq_len] or [0]
        seq = torch.tensor([ids], device=self.device)
        return self.reward_model(seq)[0]  # 0-dim tensor

    def train_step(
        self, pairs: List[Dict[str, str]], num_epochs: int = 1
    ) -> Dict[str, float]:
        """Train on ``pairs`` and return averaged metrics.

        Each pair is a dict with ``prompt`` / ``chosen`` / ``rejected``.
        Returns ``{loss, accuracy, reward_margin}`` averaged over update steps.
        """
        if not pairs:
            return {"loss": 0.0, "accuracy": 0.0, "reward_margin": 0.0}

        self.reward_model.train()
        pairs = list(pairs)
        total_loss = total_acc = total_margin = 0.0
        steps = 0

        for _ in range(num_epochs):
            random.shuffle(pairs)
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                r_chosen = torch.stack(
                    [self._reward(p["prompt"] + p["chosen"]) for p in batch]
                )
                r_rejected = torch.stack(
                    [self._reward(p["prompt"] + p["rejected"]) for p in batch]
                )
                loss = -F.logsigmoid(r_chosen - r_rejected).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    total_loss += loss.item()
                    total_acc += (r_chosen > r_rejected).float().mean().item()
                    total_margin += (r_chosen - r_rejected).mean().item()
                steps += 1

        n = max(steps, 1)
        return {
            "loss": total_loss / n,
            "accuracy": total_acc / n,
            "reward_margin": total_margin / n,
        }

    @torch.no_grad()
    def score(self, text: str) -> float:
        """Return the reward model's scalar score for ``text``."""
        self.reward_model.eval()
        return float(self._reward(text).item())

    @torch.no_grad()
    def evaluate(self, pairs: List[Dict[str, str]]) -> Dict[str, float]:
        """Ranking accuracy / mean margin on ``pairs`` (no gradient)."""
        if not pairs:
            return {"accuracy": 0.0, "reward_margin": 0.0}
        self.reward_model.eval()
        correct = 0
        margin = 0.0
        for p in pairs:
            rc = self._reward(p["prompt"] + p["chosen"]).item()
            rr = self._reward(p["prompt"] + p["rejected"]).item()
            correct += int(rc > rr)
            margin += rc - rr
        n = len(pairs)
        return {"accuracy": correct / n, "reward_margin": margin / n}

    def save(self, path: str) -> None:
        """Save the reward model's state dict."""
        torch.save(self.reward_model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load reward model weights (safe load)."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.reward_model.load_state_dict(state)
