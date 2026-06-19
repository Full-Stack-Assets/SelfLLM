"""Multi-Head Self-Attention with Rotary Position Embedding (RoPE)."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPEMultiHeadAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding (RoPE).

    Implements causal (autoregressive) masking and supports key-value caching
    for efficient autoregressive generation.

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads. Must divide ``d_model``.
        max_seq_len: Maximum sequence length (for RoPE precomputation).
        dropout: Dropout probability on attention weights and output.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        # Single linear projection for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Precompute RoPE frequency tensor
        self.register_buffer(
            "rope_freqs",
            self._precompute_rope_freqs(max_seq_len, self.head_dim, base=10000.0),
            persistent=False,
        )

        self._reset_parameters()

    # ------------------------------------------------------------------ #
    # Weight initialisation
    # ------------------------------------------------------------------ #

    def _reset_parameters(self) -> None:
        """Kaiming initialisation for linear layers."""
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    # ------------------------------------------------------------------ #
    # RoPE helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _precompute_rope_freqs(
        max_seq_len: int, head_dim: int, base: float = 10000.0
    ) -> torch.Tensor:
        """Precompute RoPE inverse frequency tensor.

        Returns:
            Tensor of shape ``[max_seq_len, head_dim // 2]``.
        """
        dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float64)
        freqs = 1.0 / (base ** (dim_indices / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float64)
        angles = torch.outer(positions, freqs)  # [max_seq_len, head_dim//2]
        return angles.float()  # [max_seq_len, head_dim//2]

    def _apply_rope(
        self,
        x: torch.Tensor,
        seq_len: int,
        start_pos: int = 0,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply rotary position embedding to input tensor.

        Args:
            x: Tensor of shape ``[batch, n_heads, seq_len, head_dim]``.
            seq_len: Sequence length.
            start_pos: Absolute position of the first token (used when
                ``positions`` is ``None``). Non-zero during incremental
                decoding, where ``x`` holds new tokens that follow ``start_pos``
                already-cached positions.
            positions: Optional explicit absolute positions of shape
                ``[batch, seq_len]``. When given, each batch row is rotated by
                its own positions -- required for batched decoding where every
                request sits at a different point in its sequence.

        Returns:
            Rotated tensor of same shape.
        """
        # x shape: [batch, n_heads, seq_len, head_dim]
        max_pos = self.rope_freqs.shape[0]

        if positions is not None:
            # Per-row positions: [batch, seq_len] -> freqs [batch, seq_len, hd//2]
            pos = positions.clamp(min=0, max=max_pos - 1)
            freqs = self.rope_freqs[pos]  # [batch, seq_len, head_dim//2]
            cos_freqs = torch.cos(freqs).unsqueeze(1)  # [batch, 1, seq_len, hd//2]
            sin_freqs = torch.sin(freqs).unsqueeze(1)
        else:
            # Saturating index: positions at/beyond the precomputed table reuse
            # its final entry. This keeps freqs exactly ``seq_len`` long even
            # when the sequence is longer than max_seq_len -- otherwise slicing
            # past the table returns too few rows and the rotation below fails
            # with a shape mismatch (a 500 on any over-long prompt).
            idx = torch.arange(
                start_pos, start_pos + seq_len, device=self.rope_freqs.device
            ).clamp(max=max_pos - 1)
            freqs = self.rope_freqs[idx]  # [seq_len, hd//2]
            cos_freqs = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, hd//2]
            sin_freqs = torch.sin(freqs).unsqueeze(0).unsqueeze(0)

        # Split last dimension into pairs
        x1 = x[..., 0::2]  # [batch, n_heads, seq_len, head_dim//2]
        x2 = x[..., 1::2]  # [batch, n_heads, seq_len, head_dim//2]

        # Apply rotation: [x0, x1] * [cos(m*freq), sin(m*freq)]
        rx1 = x1 * cos_freqs - x2 * sin_freqs
        rx2 = x1 * sin_freqs + x2 * cos_freqs

        # Interleave back
        rotated = torch.stack([rx1, rx2], dim=-1)  # [B, H, T, hd//2, 2]
        return rotated.view_as(x)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        positions: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        streaming: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for multi-head attention with RoPE.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.
            mask: Optional causal mask ``[seq_len, seq_len]`` or broadcastable shape.
                  If ``None``, a causal mask is created automatically.
            kv_cache: Optional tuple of ``(cached_k, cached_v)`` for autoregressive
                      generation. Each has shape ``[batch, n_heads, cache_len, head_dim]``.
            positions: Optional explicit absolute positions ``[batch, seq_len]`` for
                      the new tokens (batched decoding where each request is at a
                      different position). Overrides the cache-length offset.
            key_padding_mask: Optional boolean mask ``[batch, cache_len]`` marking
                      which cached positions are valid (``True``). Used to batch
                      requests whose caches were right-padded to a common length.

        Returns:
            - Output tensor ``[batch, seq_len, d_model]``.
            - Updated ``(k, v)`` cache if ``kv_cache`` was provided, else ``None``.
        """
        batch, seq_len, _ = x.shape

        # Q, K, V projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head: [batch, n_heads, seq_len, head_dim]
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K.
        if streaming:
            # Streaming / attention-time RoPE: the cache stores UN-rotated keys
            # and rotation happens here using cache-relative positions. Combined
            # with sliding-window eviction the cache is bounded, so positions
            # stay in [0, kv_len) ⊂ [0, max_seq_len) and never saturate --
            # enabling correct generation unbounded by the context window.
            if kv_cache is not None:
                cached_k, cached_v = kv_cache  # un-rotated
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)
            new_cache = (k, v)  # store the un-rotated keys
            kv_len = k.shape[2]
            kpos = torch.arange(kv_len, device=x.device).unsqueeze(0).expand(batch, -1)
            k = self._apply_rope(k, kv_len, positions=kpos)
            q = self._apply_rope(q, seq_len, positions=kpos[:, kv_len - seq_len:])
        else:
            # Baked-RoPE path (default). During incremental decoding the new
            # tokens sit at absolute positions just after the cached ones; for
            # batched decode each row has its own position. The cached K/V were
            # already rotated when produced, so only the new tokens are rotated
            # here before concatenation.
            cache_len = kv_cache[0].shape[2] if kv_cache is not None else 0
            q = self._apply_rope(q, seq_len, start_pos=cache_len, positions=positions)
            k = self._apply_rope(k, seq_len, start_pos=cache_len, positions=positions)
            if kv_cache is not None:
                cached_k, cached_v = kv_cache
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)
            new_cache = (k, v)
            kv_len = k.shape[2]

        # Causal mask: auto-create if not provided
        if mask is None:
            # For generation with cache, q_len may be 1; causal mask covers full kv_len
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool),
                diagonal=kv_len - seq_len + 1,
            )
        else:
            causal_mask = mask

        # Whether the explicit masked path is required (SDPA's is_causal shortcut
        # assumes a square q_len == kv_len matrix and cannot express padding).
        use_manual = not (
            mask is None
            and kv_cache is None
            and key_padding_mask is None
            and hasattr(F, "scaled_dot_product_attention")
        )

        if not use_manual:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T_q, T_k]

            if causal_mask is not None:
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Key-padding mask: invalidate padded cached positions per row. The
            # mask covers the cached keys; the new (seq_len) keys are always
            # valid, so extend with a True block before applying.
            if key_padding_mask is not None:
                new_valid = torch.ones(
                    batch, seq_len, device=x.device, dtype=torch.bool
                )
                full_valid = torch.cat([key_padding_mask, new_valid], dim=1)  # [B, kv_len]
                pad = ~full_valid  # True where it should be masked out
                scores = scores.masked_fill(
                    pad.unsqueeze(1).unsqueeze(2), float("-inf")
                )

            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = self.attn_dropout(attn_weights)
            attn_out = torch.matmul(attn_weights, v)  # [B, H, T_q, head_dim]

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.resid_dropout(self.out_proj(attn_out))

        return out, new_cache
