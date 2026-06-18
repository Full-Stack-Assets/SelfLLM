"""Speculative decoding for faster generation using a small draft model."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM


class SpeculativeDecoder:
    """
    Speculative decoding: a small draft model generates K candidate tokens,
    the large target model verifies all K in parallel. Accepted tokens are
    kept, a rejected token triggers standard generation.

    Typically 2-3x faster than standard autoregressive generation.
    """

    def __init__(
        self,
        target_model: SelfImprovingLLM,
        draft_model: Optional[SelfImprovingLLM] = None,
        gamma: int = 4,  # Number of draft tokens to generate per step
        device: str = "cuda",
    ):
        self.target_model = target_model
        self.gamma = gamma
        self.device = device

        # Auto-create small draft model if not provided
        if draft_model is None:
            draft_config = self._create_draft_config(target_model.config)
            self.draft_model = SelfImprovingLLM(draft_config).to(device)
            # Copy embeddings from target for compatibility.
            # The draft model may have a different d_model, so we only
            # copy the overlapping vocab & embedding dimensions.
            min_vocab = min(
                draft_config.vocab_size,
                target_model.config.vocab_size,
            )
            copy_d = min(draft_config.d_model, target_model.config.d_model)
            with torch.no_grad():
                self.draft_model.token_embedding.weight.data[
                    :min_vocab, :copy_d
                ] = target_model.token_embedding.weight.data[
                    :min_vocab, :copy_d
                ].clone()
        else:
            self.draft_model = draft_model

        self.draft_model.eval()
        self.target_model.eval()

        # Stats
        self.total_draft_tokens = 0
        self.accepted_tokens = 0

    # ------------------------------------------------------------------ #
    # Draft config
    # ------------------------------------------------------------------ #

    @staticmethod
    def _create_draft_config(target_config: ModelConfig) -> ModelConfig:
        """Create a smaller config for the draft model.

        The draft model is typically 4x smaller in width and depth to
        keep the per-step cost low.

        Args:
            target_config: Configuration of the full target model.

        Returns:
            A ``ModelConfig`` suitable for the draft model.
        """
        return ModelConfig(
            vocab_size=target_config.vocab_size,
            d_model=max(64, target_config.d_model // 4),
            n_layers=max(2, target_config.n_layers // 4),
            n_heads=max(2, target_config.n_heads // 2),
            d_ff=min(target_config.d_ff, max(64, target_config.d_ff // 4)),
            max_seq_len=target_config.max_seq_len,
            dropout=0.0,
            tie_weights=target_config.tie_weights,
        )

    # ------------------------------------------------------------------ #
    # Core speculative generation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Speculative generation.

        Algorithm:
        1. Draft model generates *gamma* candidate tokens autoregressively.
        2. Target model evaluates all *gamma* positions in a **single**
           forward pass.
        3. Accept each draft token with probability
           ``min(1, p_target / p_draft)``.
        4. On the first rejection, resample a corrected token from the
           residual distribution and continue.
        5. If all *gamma* are accepted, keep them and repeat.

        Each sequence in the batch is decoded independently (acceptance counts
        differ per row), then the results are right-padded to a common length.

        Args:
            prompt_ids: Prompt token IDs ``[batch, prompt_len]``.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            eos_token_id: If provided, a sequence stops once it emits this token.

        Returns:
            Dictionary with ``sequences`` and ``acceptance_rate`` keys.
        """
        self.draft_model.eval()
        self.target_model.eval()

        batch = prompt_ids.shape[0]
        rows = [
            self._generate_single(
                prompt_ids[b : b + 1],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=eos_token_id,
            )
            for b in range(batch)
        ]
        max_len = max(r.shape[1] for r in rows)
        pad_id = eos_token_id if eos_token_id is not None else 0
        padded = [
            F.pad(r, (0, max_len - r.shape[1]), value=pad_id) if r.shape[1] < max_len else r
            for r in rows
        ]
        generated = torch.cat(padded, dim=0)

        return {
            "sequences": generated,
            "acceptance_rate": torch.tensor(
                self.acceptance_rate, device=self.device
            ),
        }

    @torch.no_grad()
    def _generate_single(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Speculative-decode a single sequence ``[1, prompt_len]``.

        Returns the full sequence ``[1, prompt_len + generated_len]``.
        """
        generated = prompt_ids.clone()  # [1, seq_len]
        num_generated = 0
        stop = False

        while num_generated < max_new_tokens and not stop:
            # ---- 1. Draft model generates gamma tokens ----
            draft_tokens = self._draft_generate(
                generated, gamma=self.gamma, temperature=temperature, top_p=top_p
            )

            if draft_tokens.numel() == 0:
                break

            self.total_draft_tokens += draft_tokens.shape[1]

            # ---- 2. Target model verifies in parallel ----
            # Concatenate current sequence with draft tokens
            candidate_seq = torch.cat([generated, draft_tokens], dim=1)

            # Single forward pass over the full candidate sequence
            target_out = self.target_model(candidate_seq)
            # Logits at draft positions (last gamma positions)
            target_logits = target_out["logits"][
                :, -draft_tokens.shape[1] - 1 : -1, :
            ]  # [batch, gamma, vocab]
            target_probs = F.softmax(
                target_logits / max(temperature, 1e-6), dim=-1
            )

            # Draft model's own probability distribution at the same positions
            draft_out = self.draft_model(candidate_seq)
            draft_logits = draft_out["logits"][
                :, -draft_tokens.shape[1] - 1 : -1, :
            ]
            draft_probs = F.softmax(
                draft_logits / max(temperature, 1e-6), dim=-1
            )

            # ---- 3. Accept/reject each draft token ----
            accepted = []
            rejected = False

            for k in range(draft_tokens.shape[1]):
                draft_token = draft_tokens[0, k].item()

                if temperature == 0.0:
                    # Greedy: accept if argmax matches
                    target_best = torch.argmax(target_probs[0, k]).item()
                    if target_best == draft_token:
                        accepted.append(draft_token)
                        self.accepted_tokens += 1
                    else:
                        accepted.append(target_best)
                        rejected = True
                        break
                else:
                    # Stochastic: use probability ratio acceptance
                    p_t = target_probs[0, k, draft_token].item()
                    p_d = draft_probs[0, k, draft_token].item()

                    if p_d <= 0:
                        # Draft assigned zero probability — definitely reject
                        accepted.append(
                            torch.argmax(target_probs[0, k]).item()
                        )
                        rejected = True
                        break

                    acceptance_prob = min(1.0, p_t / p_d)
                    u = torch.rand(1).item()

                    if u < acceptance_prob:
                        accepted.append(draft_token)
                        self.accepted_tokens += 1
                    else:
                        # Reject: resample from residual
                        # p_residual ∝ p_target - p_draft (clamped)
                        residual = target_probs[0, k] - draft_probs[0, k]
                        residual = torch.clamp(residual, min=0.0)
                        residual_sum = residual.sum()
                        if residual_sum > 1e-8:
                            residual = residual / residual_sum
                            new_token = torch.multinomial(residual, 1).item()
                        else:
                            new_token = torch.argmax(target_probs[0, k]).item()
                        accepted.append(new_token)
                        rejected = True
                        break

            # Stop at EOS: truncate the accepted run at the first EOS token.
            if eos_token_id is not None and eos_token_id in accepted:
                accepted = accepted[: accepted.index(eos_token_id) + 1]
                stop = True

            # Don't exceed the requested budget.
            if num_generated + len(accepted) > max_new_tokens:
                accepted = accepted[: max_new_tokens - num_generated]

            # Append accepted tokens to the generated sequence
            if accepted:
                accepted_tensor = torch.tensor(
                    [accepted], device=self.device, dtype=torch.long
                )
                generated = torch.cat([generated, accepted_tensor], dim=1)
                num_generated += len(accepted)

            if num_generated >= max_new_tokens:
                break

        return generated

    # ------------------------------------------------------------------ #
    # Draft helper
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _draft_generate(
        self,
        prefix: torch.Tensor,
        gamma: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Generate *gamma* tokens with the draft model.

        Args:
            prefix: Starting token IDs ``[batch, seq_len]``.
            gamma: Number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Draft token IDs ``[batch, gamma]``.
        """
        draft_tokens: List[torch.Tensor] = []

        # Incremental KV cache: prefill the prefix once, then feed a single
        # token per step (O(gamma) instead of re-encoding the prefix each step).
        past: Optional[List] = None
        next_token: Optional[torch.Tensor] = None
        total_len = prefix.shape[1]

        for _ in range(gamma):
            # Forward pass (cached): prefill on the first step, then one token.
            model_input = prefix if past is None else next_token
            out = self.draft_model(
                model_input, past_key_values=past, use_cache=True
            )
            past = out["past_key_values"]
            logits = out["logits"][:, -1, :]  # [batch, vocab]

            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = (
                        sorted_indices_to_remove[:, :-1].clone()
                    )
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            draft_tokens.append(next_token)
            total_len += 1

            # Prevent exceeding max seq len
            if total_len >= self.draft_model.config.max_seq_len:
                break

        if not draft_tokens:
            return torch.empty(
                (prefix.shape[0], 0), device=self.device, dtype=torch.long
            )

        return torch.cat(draft_tokens, dim=1)

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens that were accepted."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_draft_tokens

    def reset_stats(self) -> None:
        """Reset acceptance statistics."""
        self.total_draft_tokens = 0
        self.accepted_tokens = 0
