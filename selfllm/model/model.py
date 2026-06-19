"""SelfImprovingLLM — Complete transformer language model."""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import RMSNorm, TransformerBlock
from .moe import MoETransformerBlock


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


def _create_sliding_window_mask(
    seq_len: int, device: torch.device, window: int, sinks: int = 0
) -> torch.Tensor:
    """Causal mask restricted to a sliding window plus optional attention sinks.

    A query at position ``i`` attends to keys ``j`` where ``j <= i`` and
    ``j > i - window``; additionally the first ``sinks`` tokens are always
    attendable. Returns a boolean mask where ``True`` means "masked out".
    """
    i = torch.arange(seq_len, device=device).unsqueeze(1)  # [S, 1] query
    j = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, S] key
    masked = (j > i) | (j <= i - window)
    if sinks > 0:
        keep_sink = (j < sinks) & (j <= i)  # sink tokens, still causal
        masked = masked & ~keep_sink
    return masked


def _create_sliding_window_decode_mask(
    seq_len: int, cache_len: int, device: torch.device, window: int, sinks: int = 0
) -> torch.Tensor:
    """Sliding-window + sink mask for cached decoding.

    The ``seq_len`` new tokens sit at absolute positions ``cache_len ..
    cache_len + seq_len - 1`` and attend over the full ``kv_len = cache_len +
    seq_len`` keys. A query at absolute position ``p`` attends to keys ``q``
    with ``q <= p`` and ``q > p - window``, plus the first ``sinks`` keys.
    Returns a boolean ``[seq_len, kv_len]`` mask (``True`` = masked out).
    """
    kv_len = cache_len + seq_len
    qpos = torch.arange(cache_len, kv_len, device=device).unsqueeze(1)  # [S, 1]
    kpos = torch.arange(kv_len, device=device).unsqueeze(0)             # [1, kv_len]
    masked = (kpos > qpos) | (kpos <= qpos - window)
    if sinks > 0:
        keep_sink = (kpos < sinks) & (kpos <= qpos)
        masked = masked & ~keep_sink
    return masked


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

        # Transformer blocks (MoE or dense)
        if getattr(config, "use_moe", False):
            self.blocks = nn.ModuleList(
                [MoETransformerBlock(config) for _ in range(config.n_layers)]
            )
        else:
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
        positions: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
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
            positions: Optional per-row absolute positions ``[batch, seq_len]``
                     for batched continuous-batch decoding.
            key_padding_mask: Optional cached-key validity mask ``[batch, cache_len]``
                     for batched decoding over right-padded caches.

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

        # Build the attention mask. For the non-cached (prefill / full) path the
        # attention layer would build a plain causal mask from ``mask=None``;
        # we supply an explicit mask so sliding-window models stay windowed on
        # *both* prefill and cached decode (otherwise decode would silently
        # attend to the full unbounded history).
        window = getattr(self.config, "sliding_window", None)
        sinks = getattr(self.config, "attention_sinks", 0)
        if past_key_values is not None:
            if window and positions is None and key_padding_mask is None:
                cache_len = past_key_values[0][0].shape[2]
                mask = _create_sliding_window_decode_mask(
                    seq_len, cache_len, device, window, sinks
                )
            else:
                # Cached decode: attention builds the correct causal mask from
                # the cache length (or batched serving supplies positions/mask).
                mask = None
        elif window:
            mask = _create_sliding_window_mask(seq_len, device, window, sinks)
        else:
            mask = _create_causal_mask(seq_len, device)

        # Transformer blocks
        new_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = (
            [] if use_cache else None
        )
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            if use_cache:
                x, layer_cache = block(
                    x,
                    mask=mask,
                    kv_cache=layer_past,
                    use_cache=True,
                    positions=positions,
                    key_padding_mask=key_padding_mask,
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

            # Add MoE load-balancing auxiliary loss
            if getattr(self.config, "use_moe", False):
                aux_loss = sum(
                    block.aux_loss for block in self.blocks
                    if hasattr(block, "aux_loss")
                )
                loss = loss + aux_loss * self.config.moe_aux_loss_weight

            result["loss"] = loss

        return result

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _evict_kv_cache(
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        sinks: int,
        window: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Trim each layer's KV cache to ``sinks`` initial + ``window`` recent keys.

        Bounds decode memory to ``sinks + window`` per layer (StreamingLLM-style
        attention sinks + a rolling window). Caches at or under the budget are
        left untouched. Each cache entry has shape ``[B, H, L, head_dim]``.
        """
        budget = sinks + window
        evicted: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for k, v in past_key_values:
            length = k.shape[2]
            if length <= budget:
                evicted.append((k, v))
                continue
            if sinks > 0:
                k = torch.cat([k[:, :, :sinks], k[:, :, length - window:]], dim=2)
                v = torch.cat([v[:, :, :sinks], v[:, :, length - window:]], dim=2)
            else:
                k = k[:, :, length - window:]
                v = v[:, :, length - window:]
            evicted.append((k, v))
        return evicted

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

        # Per-sequence completion tracking so each row in the batch stops at its
        # own stop token rather than waiting for every row to emit one.
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Sliding-window KV-cache eviction: for a windowed model, keep only the
        # first `attention_sinks` keys plus the most recent `sliding_window`,
        # bounding decode memory to O(window) instead of O(n). The new token is
        # given its true absolute position so the kept recent window stays
        # positionally correct.
        window = getattr(self.config, "sliding_window", None)
        sinks = getattr(self.config, "attention_sinks", 0)
        use_evict = bool(window)
        abs_pos = 0  # absolute position of the next token fed to the model

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

            # Forward pass (with incremental KV cache). When evicting, pass the
            # new token's true absolute position so its RoPE matches the kept
            # recent-window keys (which retain their original rotations).
            if use_evict and past_key_values is not None:
                positions = torch.full(
                    (batch_size, input_ids.shape[1]), abs_pos,
                    device=device, dtype=torch.long,
                )
                out = self.forward(
                    input_ids, past_key_values=past_key_values,
                    use_cache=True, positions=positions,
                )
            else:
                out = self.forward(
                    input_ids, past_key_values=past_key_values, use_cache=True
                )
            past_key_values = out["past_key_values"]
            abs_pos += input_ids.shape[1]
            if use_evict:
                past_key_values = self._evict_kv_cache(past_key_values, sinks, window)
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

            # Sequences that already hit the stop token keep emitting it, so a
            # finished row is not extended with fresh content while other rows
            # continue generating.
            if stop_token_id is not None:
                next_token = next_token.masked_fill(
                    finished.unsqueeze(-1), stop_token_id
                )

            # Update seen tokens
            if seen_tokens is not None:
                for b in range(batch_size):
                    seen_tokens[b, next_token[b, 0]] = 1.0

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop only once every sequence in the batch has emitted the stop token.
            if stop_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == stop_token_id)
                if bool(finished.all()):
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
