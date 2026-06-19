"""Flash Attention 2 with automatic fallback to standard attention.

Flash Attention 2 uses a tiled, IO-aware algorithm that:
- Reduces HBM reads/writes from O(N^2) to O(N)
- Provides exact (not approximate) attention output
- Supports causal masking natively via the ``causal=True`` flag
- Requires bf16/fp16 input on CUDA devices

If ``flash_attn`` is not installed, falls back to ``RoPEMultiHeadAttention``.
"""

import math
import warnings
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import RoPEMultiHeadAttention

# --------------------------------------------------------------------------- #
# Optional flash_attn import
# --------------------------------------------------------------------------- #

try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


# --------------------------------------------------------------------------- #
# Stand-alone RoPE (operates on [B, T, H, D] -- Flash Attention layout)
# --------------------------------------------------------------------------- #

class RoPE(nn.Module):
    """Rotary Position Embedding for Flash-Attention-style tensor layout.

    The public helper ``apply_rotary_emb`` accepts tensors of shape
    ``[batch, seq_len, n_heads, head_dim]`` so that no transpose is required
    before calling Flash Attention.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies: [head_dim // 2]
        dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float64)
        inv_freq = 1.0 / (base ** (dim_indices / head_dim))
        self.register_buffer("inv_freq", inv_freq.float(), persistent=False)

        # Precompute rotation angles for all positions up to max_seq_len
        positions = torch.arange(max_seq_len, dtype=torch.float64)
        angles = torch.outer(positions, self.inv_freq.double())
        self.register_buffer("cos_cached", torch.cos(angles).float(), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles).float(), persistent=False)

    def apply_rotary_emb(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary position embedding.

        Args:
            x: Tensor of shape ``[batch, seq_len, n_heads, head_dim]``.
            positions: 1-D tensor of integer positions, length ``seq_len``.

        Returns:
            Rotated tensor of the same shape as ``x``.
        """
        # Gather cos/sin for the requested positions. Saturate at the final
        # cached entry so positions beyond max_seq_len don't index out of bounds
        # (matches the standard backend's behavior for over-long sequences).
        positions = positions.clamp(max=self.cos_cached.shape[0] - 1)
        cos = self.cos_cached[positions]  # [T, D//2]
        sin = self.sin_cached[positions]  # [T, D//2]

        # Unsqueeze for broadcasting: [1, T, 1, D//2]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # Split into even/odd pairs
        x1 = x[..., 0::2]  # [B, T, H, D//2]
        x2 = x[..., 1::2]  # [B, T, H, D//2]

        # Apply rotation
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos

        # Interleave back
        rotated = torch.stack([rx1, rx2], dim=-1)  # [B, T, H, D//2, 2]
        return rotated.view_as(x)


# --------------------------------------------------------------------------- #
# Flash Attention 2
# --------------------------------------------------------------------------- #

class FlashAttention2(nn.Module):
    """Flash Attention 2 implementation with causal masking and KV-cache support.

    The forward pass follows the exact same API as ``RoPEMultiHeadAttention``
    so that it can be used as a drop-in replacement inside ``TransformerBlock``.

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
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # RoPE -- operates on [B, T, H, D] layout
        self.rope = RoPE(self.head_dim, max_seq_len)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Kaiming initialisation for linear layers."""
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_prefill: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with Flash Attention 2.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.
            mask: Optional explicit attention mask (``True`` = masked out). When
                provided (e.g. a non-causal mask for a vision encoder or a
                block-wise multimodal mask), it is honoured via the manual
                attention path; when ``None`` the fast causal kernel is used.
            kv_cache: Optional tuple of ``(cached_k, cached_v)`` for autoregressive
                generation. Each has shape ``[batch, cache_len, n_heads, head_dim]``.
            is_prefill: ``True`` for full-sequence (prefill) mode,
                ``False`` for single-token (decode) mode.

        Returns:
            - Output tensor ``[batch, seq_len, d_model]``.
            - Updated ``(k, v)`` cache if ``kv_cache`` was provided, else ``None``.
        """
        batch, seq_len, _ = x.shape

        # 1. Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape to [B, T, n_heads, head_dim]  (Flash Attention layout)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim)

        # 3. Apply RoPE to Q and K
        if kv_cache is not None and not is_prefill:
            cache_len = kv_cache[0].shape[1]
            positions = torch.arange(
                cache_len, cache_len + seq_len, device=x.device
            )
        else:
            positions = torch.arange(seq_len, device=x.device)

        q = self.rope.apply_rotary_emb(q, positions)
        k = self.rope.apply_rotary_emb(k, positions)

        # 4. Merge with KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
        new_cache = (k, v)

        # 5. Attention computation. The flash kernel only expresses causal /
        # full attention, so an explicit custom mask routes to the manual path.
        if (
            mask is None
            and HAS_FLASH_ATTN
            and (q.dtype in (torch.float16, torch.bfloat16))
        ):
            # Flash Attention 2 path (requires fp16/bf16)
            if kv_cache is not None and not is_prefill:
                out = flash_attn_with_kvcache(
                    q, k_cache=k, v_cache=v, causal=True,
                )
            else:
                out = flash_attn_func(
                    q, k, v, causal=True, softmax_scale=None,
                )
        else:
            # Fallback: manual scaled dot-product attention (honours `mask`).
            if mask is None and HAS_FLASH_ATTN and q.dtype == torch.float32:
                warnings.warn(
                    "Flash Attention 2 requires fp16/bf16; falling back to manual "
                    "attention. Consider using torch.autocast for mixed-precision "
                    "training/inference.",
                    stacklevel=2,
                )
            out = self._manual_attention(q, k, v, is_prefill, mask=mask)

        # 6. Reshape and output projection
        out = out.reshape(batch, seq_len, self.d_model)
        out = self.resid_dropout(self.out_proj(out))

        return out, new_cache if kv_cache is not None else None

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_prefill: bool,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Manual scaled dot-product attention (fallback).

        Operates on Flash Attention layout ``[B, T, H, D]``; transposes
        internally to ``[B, H, T, D]`` for the matmuls. When ``mask`` is given
        (``True`` = masked out) it is applied verbatim; otherwise a standard
        causal mask is used.
        """
        q_t = q.transpose(1, 2)  # [B, H, T_q, D]
        k_t = k.transpose(1, 2)  # [B, H, T_k, D]
        v_t = v.transpose(1, 2)  # [B, H, T_k, D]

        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.scale

        q_len = q_t.shape[2]
        kv_len = k_t.shape[2]
        if mask is not None:
            # Broadcast the mask to [B, H, q_len, kv_len].
            if mask.dim() == 2:
                m = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                m = mask.unsqueeze(1)
            else:
                m = mask
            scores = scores.masked_fill(m, float("-inf"))
        elif is_prefill or q_len > 1:
            causal_mask = torch.triu(
                torch.ones(q_len, kv_len, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v_t)  # [B, H, T_q, D]
        return out.transpose(1, 2)  # Back to [B, T_q, H, D]


# --------------------------------------------------------------------------- #
# Hybrid Attention -- auto-selects best backend
# --------------------------------------------------------------------------- #

class HybridAttention(nn.Module):
    """Attention layer that auto-selects the best available backend.

    Selection order:
    1. **Flash Attention 2** -- if ``flash_attn`` is installed.
    2. **Standard attention** -- if ``flash_attn`` is unavailable.

    The forward signature is identical to ``RoPEMultiHeadAttention`` so this
    class is a drop-in replacement inside ``TransformerBlock``.

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        if HAS_FLASH_ATTN:
            self.attn: nn.Module = FlashAttention2(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
            self.backend = "flash_attn"
        else:
            self.attn = RoPEMultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
            self.backend = "standard"

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        is_prefill: bool = True,
        use_cache: bool = False,
        positions: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        streaming: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Forward pass -- delegates to the selected backend.

        Normalises the two backends' differing cache conventions so callers
        get a single contract: when ``use_cache`` is ``True`` an updated KV
        cache is always returned (and populated on prefill); when it is
        ``False`` the returned cache is ``None``.

        ``positions`` / ``key_padding_mask`` support batched continuous-batch
        decoding and are only honoured by the standard backend (the batched
        serving path runs on that backend).
        """
        if self.backend == "flash_attn" and key_padding_mask is None and positions is None and not streaming:
            # FlashAttention2 only populates a cache when one is passed in, so
            # seed an empty cache on prefill to capture the prompt's K/V.
            if use_cache and kv_cache is None:
                batch = x.shape[0]
                empty = x.new_zeros(batch, 0, self.n_heads, self.head_dim)
                kv_cache = (empty, empty.clone())
            if kv_cache is not None:
                is_prefill = kv_cache[0].shape[1] == 0
            out, new_cache = self.attn(
                x, mask=mask, kv_cache=kv_cache, is_prefill=is_prefill
            )
        elif self.backend == "flash_attn":
            # Batched decode with padding/positions: use the standard RoPE
            # attention math against the flash module's projections is not
            # supported, so fall back to the flash module's own path without
            # the padding mask is unsafe -- instead require the standard
            # backend for batched serving.
            raise NotImplementedError(
                "Batched continuous-batch decoding (positions/key_padding_mask) "
                "requires the standard attention backend; flash_attn is not "
                "supported for this path."
            )
        else:
            # Standard backend (RoPEMultiHeadAttention).
            out, new_cache = self.attn(
                x,
                mask=mask,
                kv_cache=kv_cache,
                positions=positions,
                key_padding_mask=key_padding_mask,
                streaming=streaming,
            )
        return out, (new_cache if use_cache else None)

    @property
    def backend_name(self) -> str:
        """Return the name of the active backend."""
        return self.backend
