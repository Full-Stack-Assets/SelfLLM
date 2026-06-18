"""SelfImprovingLLM — Complete transformer language model."""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import RMSNorm, TransformerBlock


def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a causal (lower-triangular) boolean mask.

    Args:
        seq_len: Sequence length.
        device: Target device.

    Returns:
        Boolean mask of shape ``[seq_len, seq_len]`` where ``True``
        indicates positions that should be masked out.
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )


class SelfImprovingLLM(nn.Module):
    """Complete transformer language model for recursive self-improvement.

    Architecture:
        - Token embeddings (optionally tied to output LM head)
        - Stack of ``TransformerBlock`` layers (pre-norm, RoPE, SwiGLU)
        - Final RMSNorm + linear projection to vocabulary logits

    Args:
        config: ``ModelConfig`` instance defining model hyperparameters.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final layer norm
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # LM head — tied to token_embedding when config.tie_weights is True
        if config.tie_weights:
            # Share the embedding weight matrix with the output projection
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        self._reset_parameters()

        # Track for generation
        self._past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

    # ------------------------------------------------------------------ #
    # Parameter initialisation
    # ------------------------------------------------------------------ #

    def _reset_parameters(self) -> None:
        """Initialise weights: Xavier uniform for embeddings, Kaiming for linear."""
        nn.init.xavier_uniform_(self.token_embedding.weight)
        if not self.config.tie_weights:
            nn.init.kaiming_uniform_(self.lm_head.weight, a=math.sqrt(5))

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through the model.

        Args:
            token_ids: Input token IDs ``[batch, seq_len]``.
            targets: Target token IDs for loss computation ``[batch, seq_len]``.
                     If provided, the loss is computed as cross-entropy.
            past_key_values: Per-layer ``(k, v)`` caches from a previous call,
                     used for incremental autoregressive decoding. When given,
                     ``token_ids`` should contain only the new tokens.
            use_cache: If ``True``, return the updated per-layer KV caches under
                     the ``past_key_values`` key.

        Returns:
            Dictionary with keys:
                - ``logits``: ``[batch, seq_len, vocab_size]``
                - ``loss``: scalar cross-entropy loss (if ``targets`` provided)
                - ``hidden_states``: ``[batch, seq_len, d_model]``
                - ``past_key_values``: updated caches (if ``use_cache``)
        """
        batch, seq_len = token_ids.shape
        device = token_ids.device

        # Token embeddings
        x = self.token_embedding(token_ids)  # [B, T, D]
        x = self.dropout(x)

        # Causal mask only for the non-cached (prefill / full) path. During
        # cached decoding the attention layer builds the correct
        # ``[q_len, kv_len]`` mask from the cache length, so leave it ``None``.
        mask = None if past_key_values is not None else _create_causal_mask(seq_len, device)

        # Transformer blocks
        new_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = (
            [] if use_cache else None
        )
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            if use_cache:
                x, layer_cache = block(
                    x, mask=mask, kv_cache=layer_past, use_cache=True
                )
                new_caches.append(layer_cache)
            elif self.config.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, mask)
            else:
                x = block(x, mask=mask)

        # Final norm
        hidden_states = self.norm(x)

        # LM head
        logits = self.lm_head(hidden_states)

        result: Dict[str, Any] = {
            "logits": logits,
            "hidden_states": hidden_states,
        }

        if use_cache:
            result["past_key_values"] = new_caches

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        stop_token_id: Optional[int] = None,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive generation with temperature, top-k, and top-p sampling.

        Args:
            prompt_ids: Prompt token IDs ``[batch, prompt_len]``.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (softmax scaling).
            top_k: Number of highest-probability tokens to consider.
            top_p: Nucleus (cumulative) probability threshold.
            repetition_penalty: Factor to penalise already-seen tokens (<1 = no penalty).
            stop_token_id: Token ID that signals end of generation.
            return_hidden: Whether to return hidden states.

        Returns:
            Dictionary with keys:
                - ``sequences``: ``[batch, prompt_len + generated_len]``
                - ``scores``: Generation confidence scores per step
                - ``hidden_states``: ``[batch, seq_len, d_model]`` (if ``return_hidden``)
        """
        self.eval()
        device = prompt_ids.device
        batch_size = prompt_ids.shape[0]

        generated = prompt_ids.clone()
        scores: List[torch.Tensor] = []
        all_hidden: List[torch.Tensor] = []

        # Track seen tokens for repetition penalty
        seen_tokens: Optional[torch.Tensor] = None
        if repetition_penalty != 1.0:
            seen_tokens = torch.zeros(
                (batch_size, self.config.vocab_size),
                device=device,
            )
            # Mark prompt tokens as seen
            for b in range(batch_size):
                seen_tokens[b].scatter_(0, prompt_ids[b], 1.0)

        # Incremental KV cache: prefill the prompt once, then feed a single
        # token per step. This is O(n) over the generation instead of the
        # O(n^2) full re-encode of the whole prefix each step.
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        next_token: Optional[torch.Tensor] = None
        max_len = self.config.max_seq_len

        for step in range(max_new_tokens):
            # Prepare inputs
            if past_key_values is None:
                # Prefill: feed the prompt (truncated to the context window).
                if generated.shape[1] > max_len:
                    input_ids = generated[:, -max_len:]
                else:
                    input_ids = generated
            else:
                # Decode: feed only the most recently sampled token.
                input_ids = next_token

            # Forward pass (with incremental KV cache)
            out = self.forward(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = out["past_key_values"]
            logits = out["logits"][:, -1, :]  # [batch, vocab_size] — last position
            hidden = out["hidden_states"][:, -1, :]  # [batch, d_model]

            if return_hidden:
                all_hidden.append(hidden)

            # Temperature scaling + sampling
            if temperature == 0.0:
                # Greedy decoding — deterministic
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [batch, 1]
                probs = F.softmax(logits, dim=-1)
                batch_idx = torch.arange(batch_size, device=device)
                scores.append(probs[batch_idx, next_token.squeeze(-1)])
            else:
                logits = logits / temperature

                # Repetition penalty
                if seen_tokens is not None and repetition_penalty != 1.0:
                    penalty_mask = seen_tokens.bool()
                    logits = torch.where(
                        penalty_mask,
                        logits / repetition_penalty,
                        logits,
                    )

                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least the top token
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

                # Record confidence (probability of sampled token)
                batch_idx = torch.arange(batch_size, device=device)
                scores.append(probs[batch_idx, next_token.squeeze(-1)])

            # Update seen tokens
            if seen_tokens is not None:
                for b in range(batch_size):
                    seen_tokens[b, next_token[b, 0]] = 1.0

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for stop token
            if stop_token_id is not None and (next_token == stop_token_id).all():
                break

        result: Dict[str, torch.Tensor] = {
            "sequences": generated,
            "scores": torch.stack(scores, dim=1) if scores else torch.empty(batch_size, 0, device=device),  # [batch, num_steps]
        }

        if return_hidden and all_hidden:
            # Pad hidden states to match sequence length
            result["hidden_states"] = torch.stack(all_hidden, dim=1)  # [batch, steps, D]

        return result

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_pretrained(self, path: str) -> None:
        """Save model weights and config to a directory.

        Args:
            path: Directory path to save to (created if it doesn't exist).
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        # Save config as dict
        import json
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> "SelfImprovingLLM":
        """Load model weights and config from a directory.

        Args:
            path: Directory path containing ``pytorch_model.bin`` and ``config.json``.

        Returns:
            Loaded ``SelfImprovingLLM`` instance.
        """
        import json
        with open(os.path.join(path, "config.json"), "r") as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)
        model = cls(config)
        state_dict = torch.load(
            os.path.join(path, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model
