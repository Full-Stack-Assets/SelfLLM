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

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary position embedding to input tensor.

        Args:
            x: Tensor of shape ``[batch, n_heads, seq_len, head_dim]``.
            seq_len: Sequence length.

        Returns:
            Rotated tensor of same shape.
        """
        # x shape: [batch, n_heads, seq_len, head_dim]
        freqs = self.rope_freqs[:seq_len]  # [seq_len, head_dim//2]

        # Split last dimension into pairs
        x1 = x[..., 0::2]  # [batch, n_heads, seq_len, head_dim//2]
        x2 = x[..., 1::2]  # [batch, n_heads, seq_len, head_dim//2]

        cos_freqs = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, hd//2]
        sin_freqs = torch.sin(freqs).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, hd//2]

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
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for multi-head attention with RoPE.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.
            mask: Optional causal mask ``[seq_len, seq_len]`` or broadcastable shape.
                  If ``None``, a causal mask is created automatically.
            kv_cache: Optional tuple of ``(cached_k, cached_v)`` for autoregressive
                      generation. Each has shape ``[batch, n_heads, cache_len, head_dim]``.

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

        # Apply RoPE to Q and K
        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)

        # Merge with KV cache if provided (autoregressive generation)
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

        # Scaled dot-product attention (Flash Attention when available)
        # PyTorch 2.0+ scaled_dot_product_attention handles causal masking internally
        if mask is None and hasattr(F, "scaled_dot_product_attention"):
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

            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = self.attn_dropout(attn_weights)
            attn_out = torch.matmul(attn_weights, v)  # [B, H, T_q, head_dim]

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.resid_dropout(self.out_proj(attn_out))

        return out, new_cache
