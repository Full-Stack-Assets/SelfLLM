"""Transformer layer components: RMSNorm, TransformerBlock, SwiGLU FFN."""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from .attention import RoPEMultiHeadAttention
from .flash_attention import HybridAttention
from .config import ModelConfig


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation.

    Computes ``x * g / sqrt(mean(x^2) + eps)`` where ``g`` is a learned
    per-dimension gain parameter.

    Args:
        dim: Feature dimension to normalise over.
        eps: Numerical stability constant.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.

        Args:
            x: Input tensor of shape ``[..., dim]``.

        Returns:
            Normalised tensor of the same shape.
        """
        # Compute RMS along last dimension
        norm = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight


# --------------------------------------------------------------------------- #
# SwiGLU Feed-Forward
# --------------------------------------------------------------------------- #

class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network: ``Swish(xW) * xV``.

    This is a Gated Linear Unit variant that uses the SiLU (Swish) activation.
    Internally expands to ``(8/3) * d_ff`` to match standard FFN capacity.

    Args:
        d_model: Model embedding dimension.
        d_ff: Feed-forward hidden dimension (before the 8/3 scaling).
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # SwiGLU uses 3 matrices: W, V, and output projection.
        # The hidden dim is scaled by 2/3 for parameter parity with standard FFN.
        hidden_dim = int((2 / 3) * d_ff)

        self.w_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.v_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Kaiming initialisation for linear layers."""
        for m in (self.w_proj, self.v_proj, self.out_proj):
            nn.init.kaiming_uniform_(m.weight, a=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SwiGLU FFN.

        Args:
            x: Input ``[batch, seq_len, d_model]``.

        Returns:
            Output ``[batch, seq_len, d_model]``.
        """
        # SwiGLU: Swish(xW) * (xV)
        return self.dropout(
            self.out_proj(torch.nn.functional.silu(self.w_proj(x)) * self.v_proj(x))
        )


# --------------------------------------------------------------------------- #
# Transformer Block
# --------------------------------------------------------------------------- #

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block.

    Architecture::

        x = x + Attention(Norm(x))
        x = x + FFN(Norm(x))

    Args:
        config: ``ModelConfig`` instance.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = HybridAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.ln2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLUFeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        use_cache: bool = False,
        positions: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.
            mask: Optional causal mask tensor.
            kv_cache: Optional key/value cache for incremental decoding.
            use_cache: If ``True``, also return the updated KV cache.
            positions: Optional per-row absolute positions for batched decode.
            key_padding_mask: Optional cached-key validity mask for batched decode.

        Returns:
            Output tensor ``[batch, seq_len, d_model]``, or a
            ``(output, kv_cache)`` tuple when ``use_cache`` is ``True``.
        """
        # Pre-norm attention with residual
        attn_out, new_cache = self.attn(
            self.ln1(x),
            mask=mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
            positions=positions,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out

        # Pre-norm FFN with residual
        x = x + self.ffn(self.ln2(x))

        if use_cache:
            return x, new_cache
        return x
