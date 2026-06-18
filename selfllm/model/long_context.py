"""Long Context Memory: Sliding Window Attention, StreamingLLM, and RAG Retrieval.

This module enables processing context far beyond the fixed max_seq_len limit:
- SlidingWindowAttention: Each token only attends to the last W tokens (O(window_size) memory)
- StreamingAttention (StreamingLLM): Attention sinks + sliding window for infinite context
- RAGRetriever: External vector database for long-document retrieval
"""

import math
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Sliding Window Attention
# --------------------------------------------------------------------------- #

class SlidingWindowAttention(nn.Module):
    """Multi-head self-attention with causal sliding window mask.

    Each token can only attend to the most recent ``window_size`` tokens.
    This reduces memory complexity from O(seq_len^2) to O(seq_len * window_size).

    Uses Rotary Position Embedding (RoPE) and supports KV caching for
    autoregressive generation.

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads. Must divide ``d_model``.
        window_size: Number of recent tokens each position can attend to.
        max_seq_len: Maximum sequence length (for RoPE precomputation).
        dropout: Dropout probability on attention weights and output.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 1024,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
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

    def _reset_parameters(self) -> None:
        """Kaiming initialisation for linear layers."""
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

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
        angles = torch.outer(positions, freqs)
        return angles.float()

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary position embedding to input tensor.

        Args:
            x: Tensor of shape ``[batch, n_heads, seq_len, head_dim]``.
            seq_len: Sequence length.

        Returns:
            Rotated tensor of same shape.
        """
        freqs = self.rope_freqs[:seq_len]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        cos_freqs = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
        sin_freqs = torch.sin(freqs).unsqueeze(0).unsqueeze(0)

        rx1 = x1 * cos_freqs - x2 * sin_freqs
        rx2 = x1 * sin_freqs + x2 * cos_freqs

        rotated = torch.stack([rx1, rx2], dim=-1)
        return rotated.view_as(x)

    def _create_sliding_window_mask(self, seq_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
        """Create a causal sliding window mask.

        Each position i can only attend to positions in [max(0, i - window_size + 1), i].

        Args:
            seq_len: Query sequence length.
            kv_len: Key/value sequence length (includes cache).
            device: Target device.

        Returns:
            Boolean mask of shape ``[seq_len, kv_len]`` where ``True``
            indicates positions that should be masked out.
        """
        # Start with a standard causal mask (upper triangular)
        mask = torch.ones(seq_len, kv_len, device=device, dtype=torch.bool)

        for i in range(seq_len):
            # In causal attention, position i can attend to [0, i] (with cache offset)
            query_pos = i
            # The actual position in the full KV sequence
            actual_pos = kv_len - seq_len + i
            # Sliding window: can only attend to last window_size tokens
            window_start = max(0, actual_pos - self.window_size + 1)
            # Allow attention from window_start to actual_pos (inclusive)
            mask[i, window_start:actual_pos + 1] = False

        return mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with sliding window attention.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.
            mask: Optional mask (ignored; sliding window mask is created automatically).
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

        # Merge with KV cache if provided
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_cache = (k, v)

        kv_len = k.shape[2]

        # Sliding window mask: tokens can only attend to recent window_size positions
        if kv_cache is not None and seq_len == 1:
            # Generation mode: single token query
            causal_mask = self._create_sliding_window_mask(seq_len, kv_len, x.device)
        elif kv_cache is not None:
            # Prefill with cache
            causal_mask = self._create_sliding_window_mask(seq_len, kv_len, x.device)
        else:
            # Full prefill (no cache): create sliding window mask
            causal_mask = self._create_sliding_window_mask(seq_len, kv_len, x.device)

        # Use Flash Attention when available and no custom mask needed
        # For sliding window, we always need the custom mask
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.resid_dropout(self.out_proj(attn_out))

        return out, new_cache


# --------------------------------------------------------------------------- #
# Streaming Attention (StreamingLLM)
# --------------------------------------------------------------------------- #

class StreamingAttention(nn.Module):
    """StreamingLLM attention: attention sinks + sliding window for infinite context.

    Key insight from the StreamingLLM paper: the first few tokens (attention sinks)
    must ALWAYS be kept in the KV cache. Without them, attention scores explode
    when the KV cache is truncated because the softmax needs something to "attend to"
    as a sink for excess attention mass.

    Algorithm:
        1. Always keep first ``num_sink_tokens`` tokens (attention sinks) in KV cache.
        2. Plus the most recent ``window_size`` tokens (sliding window).
        3. When the cache exceeds ``num_sink_tokens + window_size``, evict tokens
           from the middle (between sinks and recent window).
        4. Total effective context at any point: ``num_sink_tokens + window_size``.
        5. Can process input of arbitrary/infinite length.

    The KV cache is stored as a tuple ``(k, v, sink_k, sink_v)`` where:
        - ``sink_k, sink_v``: Attention sinks (first num_sink_tokens), never evicted.
        - ``k, v``: Recent tokens (last window_size), subject to eviction.

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads. Must divide ``d_model``.
        window_size: Number of recent tokens to keep in the sliding window.
        num_sink_tokens: Number of initial attention sink tokens to always preserve.
        max_seq_len: Maximum sequence length (for RoPE precomputation).
        dropout: Dropout probability on attention weights and output.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 1024,
        num_sink_tokens: int = 4,
        max_seq_len: int = 32768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if num_sink_tokens < 1:
            raise ValueError(f"num_sink_tokens ({num_sink_tokens}) must be >= 1")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.num_sink_tokens = num_sink_tokens
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        # Single linear projection for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Precompute RoPE frequency tensor (needs to be large for infinite context)
        self.register_buffer(
            "rope_freqs",
            self._precompute_rope_freqs(max_seq_len, self.head_dim, base=10000.0),
            persistent=False,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Kaiming initialisation for linear layers."""
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

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
        angles = torch.outer(positions, freqs)
        return angles.float()

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding with explicit positions.

        Args:
            x: Tensor of shape ``[batch, n_heads, seq_len, head_dim]``.
            positions: 1-D tensor of integer positions, length ``seq_len``.

        Returns:
            Rotated tensor of same shape.
        """
        freqs = self.rope_freqs[positions]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        cos_freqs = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
        sin_freqs = torch.sin(freqs).unsqueeze(0).unsqueeze(0)

        rx1 = x1 * cos_freqs - x2 * sin_freqs
        rx2 = x1 * sin_freqs + x2 * cos_freqs

        rotated = torch.stack([rx1, rx2], dim=-1)
        return rotated.view_as(x)

    def _trim_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        sink_k: torch.Tensor,
        sink_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trim KV cache to keep only sinks + recent window.

        If the total cache (sinks + recent) exceeds the budget, evict from the
        middle (the oldest tokens in the recent section).

        Args:
            k: Recent key tensor ``[batch, n_heads, recent_len, head_dim]``.
            v: Recent value tensor ``[batch, n_heads, recent_len, head_dim]``.
            sink_k: Sink key tensor ``[batch, n_heads, num_sink_tokens, head_dim]``.
            sink_v: Sink value tensor ``[batch, n_heads, num_sink_tokens, head_dim]``.

        Returns:
            Trimmed ``(k, v, sink_k, sink_v)``.
        """
        total_cached = sink_k.shape[2] + k.shape[2]
        max_cache_size = self.num_sink_tokens + self.window_size

        if total_cached > max_cache_size:
            # Calculate how many tokens to evict from the recent section
            tokens_to_evict = total_cached - max_cache_size
            # Evict the oldest tokens from the recent section
            k = k[:, :, tokens_to_evict:, :]
            v = v[:, :, tokens_to_evict:, :]

        return k, v, sink_k, sink_v

    def _create_streaming_mask(
        self,
        seq_len: int,
        kv_len: int,
        sink_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create attention mask for streaming attention.

        The mask ensures:
        1. Causal masking (no look-ahead)
        2. All tokens can attend to sink tokens
        3. Tokens can only attend to recent window_size tokens (besides sinks)

        Args:
            seq_len: Query sequence length.
            kv_len: Total KV length (sinks + recent).
            sink_len: Number of sink tokens.
            device: Target device.

        Returns:
            Boolean mask of shape ``[seq_len, kv_len]`` where ``True``
            indicates positions that should be masked out.
        """
        # Start with everything masked out
        mask = torch.ones(seq_len, kv_len, device=device, dtype=torch.bool)

        for i in range(seq_len):
            # Position of this query in the full KV sequence
            actual_pos = kv_len - seq_len + i

            # 1. Always attend to all sink tokens
            mask[i, :sink_len] = False

            # 2. Attend to recent window_size tokens (excluding sinks)
            recent_kv_start = sink_len  # Start of recent section in KV
            # The sliding window applies to the recent section
            # Position i in query maps to position actual_pos in full sequence
            # Within the recent section, the index is actual_pos (since sinks are separate)
            # Actually, let me reconsider: the KV is [sinks | recent_tokens]
            # The recent section contains the actual non-sink tokens
            # We need to allow attention to the last window_size tokens in the recent section

            # The query at position actual_pos should attend to:
            # - All sink tokens (already unmasked above)
            # - Recent tokens from max(sink_len, actual_pos - window_size + 1) to actual_pos

            # In the concatenated KV [sinks | recent], the recent part starts at sink_len
            # The recent tokens that should be visible:
            recent_start_in_full = sink_len
            # How many recent tokens are there?
            num_recent = kv_len - sink_len
            # Which recent tokens can we attend to?
            # The query at actual_pos should see recent tokens with indices
            # in [max(0, num_recent - window_size), num_recent)
            # But we also need causal: can only see up to actual_pos - sink_len
            recent_index = actual_pos - sink_len  # This query's index in recent space
            if recent_index >= 0:
                window_start_in_recent = max(0, recent_index - self.window_size + 1)
                # Unmask recent tokens in the window
                recent_end = min(num_recent, recent_index + 1)
                if window_start_in_recent < recent_end:
                    mask[i, sink_len + window_start_in_recent:sink_len + recent_end] = False

        return mask

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, ...]] = None,
        is_prefill: bool = True,
        past_sink_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """Forward pass with StreamingLLM attention.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.
            kv_cache: Optional cached KV state. For streaming, this should be
                ``(recent_k, recent_v, sink_k, sink_v)`` or ``None``.
            is_prefill: ``True`` for full-sequence processing, ``False`` for
                single-token generation.
            past_sink_kv: Deprecated; sinks are now part of kv_cache.

        Returns:
            - Output tensor ``[batch, seq_len, d_model]``.
            - Updated KV cache ``(recent_k, recent_v, sink_k, sink_v)``.
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

        # Compute positions for RoPE
        if kv_cache is not None and len(kv_cache) == 4:
            # We have existing cache: (recent_k, recent_v, sink_k, sink_v)
            recent_k, recent_v, sink_k, sink_v = kv_cache
            past_len = sink_k.shape[2] + recent_k.shape[2]
            positions = torch.arange(past_len, past_len + seq_len, device=x.device)
        elif kv_cache is not None and len(kv_cache) == 2:
            # Legacy format: (k, v)
            past_k, past_v = kv_cache
            past_len = past_k.shape[2]
            positions = torch.arange(past_len, past_len + seq_len, device=x.device)
            # Initialize sinks from the beginning of the cache
            if past_k.shape[2] >= self.num_sink_tokens:
                sink_k = past_k[:, :, :self.num_sink_tokens, :]
                sink_v = past_v[:, :, :self.num_sink_tokens, :]
                recent_k = past_k[:, :, self.num_sink_tokens:, :]
                recent_v = past_v[:, :, self.num_sink_tokens:, :]
            else:
                sink_k = past_k
                sink_v = past_v
                recent_k = torch.empty(batch, self.n_heads, 0, self.head_dim, device=x.device, dtype=x.dtype)
                recent_v = torch.empty(batch, self.n_heads, 0, self.head_dim, device=x.device, dtype=x.dtype)
        else:
            # No cache - first call
            positions = torch.arange(seq_len, device=x.device)
            sink_k = torch.empty(batch, self.n_heads, 0, self.head_dim, device=x.device, dtype=x.dtype)
            sink_v = torch.empty(batch, self.n_heads, 0, self.head_dim, device=x.device, dtype=x.dtype)
            recent_k = torch.empty(batch, self.n_heads, 0, self.head_dim, device=x.device, dtype=x.dtype)
            recent_v = torch.empty(batch, self.n_heads, 0, self.head_dim, device=x.device, dtype=x.dtype)

        # Apply RoPE to Q and K
        q = self._apply_rope(q, positions)
        k = self._apply_rope(k, positions)

        # Extract sink tokens from the new input if this is the first call
        if sink_k.shape[2] == 0 and seq_len >= self.num_sink_tokens:
            new_sink_k = k[:, :, :self.num_sink_tokens, :]
            new_sink_v = v[:, :, :self.num_sink_tokens, :]
            sink_k = new_sink_k
            sink_v = new_sink_v
            # Remaining tokens go to recent
            k = k[:, :, self.num_sink_tokens:, :]
            v = v[:, :, self.num_sink_tokens:, :]
        elif sink_k.shape[2] < self.num_sink_tokens and seq_len > 0:
            # Need more sink tokens
            needed = self.num_sink_tokens - sink_k.shape[2]
            if k.shape[2] >= needed:
                new_sinks_k = k[:, :, :needed, :]
                new_sinks_v = v[:, :, :needed, :]
                sink_k = torch.cat([sink_k, new_sinks_k], dim=2) if sink_k.shape[2] > 0 else new_sinks_k
                sink_v = torch.cat([sink_v, new_sinks_v], dim=2) if sink_v.shape[2] > 0 else new_sinks_v
                k = k[:, :, needed:, :]
                v = v[:, :, needed:, :]
            else:
                sink_k = torch.cat([sink_k, k], dim=2) if sink_k.shape[2] > 0 else k
                sink_v = torch.cat([sink_v, v], dim=2) if sink_v.shape[2] > 0 else v
                k = torch.empty(batch, self.n_heads, 0, self.head_dim, device=x.device, dtype=x.dtype)
                v = torch.empty(batch, self.n_heads, 0, self.head_dim, device=x.device, dtype=x.dtype)

        # Append new keys/values to the recent section
        if k.shape[2] > 0:
            recent_k = torch.cat([recent_k, k], dim=2) if recent_k.shape[2] > 0 else k
            recent_v = torch.cat([recent_v, v], dim=2) if recent_v.shape[2] > 0 else v

        # Trim the recent section if it exceeds window_size
        if recent_k.shape[2] > self.window_size:
            recent_k = recent_k[:, :, -self.window_size:, :]
            recent_v = recent_v[:, :, -self.window_size:, :]

        # Concatenate sinks + recent for attention
        full_k = torch.cat([sink_k, recent_k], dim=2) if recent_k.shape[2] > 0 else sink_k
        full_v = torch.cat([sink_v, recent_v], dim=2) if recent_v.shape[2] > 0 else sink_v

        kv_len = full_k.shape[2]

        # Create streaming attention mask
        sink_len = sink_k.shape[2]
        causal_mask = self._create_streaming_mask(seq_len, kv_len, sink_len, x.device)

        # Compute attention
        scores = torch.matmul(q, full_k.transpose(-2, -1)) * self.scale

        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, full_v)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.resid_dropout(self.out_proj(attn_out))

        # Return updated cache
        new_cache = (recent_k, recent_v, sink_k, sink_v)

        return out, new_cache


# --------------------------------------------------------------------------- #
# RAG Retriever
# --------------------------------------------------------------------------- #

class RAGRetriever:
    """Retrieval-Augmented Generation with vector database.

    Uses FAISS for fast approximate nearest neighbor search when available,
    with a brute-force cosine similarity fallback. Supports document chunking
    for long texts.

    Args:
        embedding_dim: Dimension of embeddings.
        index_path: Path to saved FAISS index (optional).
        chunk_size: Maximum chunk size for document splitting.
        chunk_overlap: Overlap between consecutive chunks.
    """

    def __init__(
        self,
        embedding_dim: int,
        index_path: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.documents: List[str] = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Try to import FAISS
        try:
            import faiss
            self._faiss = faiss
            self.use_faiss = True
        except ImportError:
            self._faiss = None
            self.use_faiss = False

        # Initialize index
        if self.use_faiss:
            # Inner product index (for cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            self.index = None
            self._embeddings: Optional[torch.Tensor] = None

        # Load saved index if provided
        if index_path is not None and os.path.exists(index_path):
            self.load_index(index_path)

    def _chunk_document(self, document: str) -> List[str]:
        """Split a long document into overlapping chunks.

        Args:
            document: The document text to split.

        Returns:
            List of document chunks.
        """
        # Simple word-based chunking
        words = document.split()

        if len(words) <= self.chunk_size:
            return [document]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
            if end == len(words):
                break

        return chunks

    def add_documents(
        self,
        documents: List[str],
        embeddings: torch.Tensor,
    ) -> None:
        """Add documents to the retrieval index.

        Documents are automatically chunked if they exceed ``chunk_size``.
        Embeddings should be pre-computed and L2-normalized for cosine similarity.

        Args:
            documents: List of document strings.
            embeddings: Document embeddings ``[num_docs, embedding_dim]``.
                       Should be L2-normalized for cosine similarity with FAISS.
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Number of documents ({len(documents)}) must match "
                f"number of embeddings ({embeddings.shape[0]})"
            )

        # Chunk documents that are too long
        all_chunks: List[str] = []
        all_embeddings: List[torch.Tensor] = []

        for doc, emb in zip(documents, embeddings):
            chunks = self._chunk_document(doc)
            if len(chunks) == 1:
                all_chunks.append(doc)
                all_embeddings.append(emb)
            else:
                # For chunked documents, store each chunk with the same embedding
                # In practice, the embedding should be re-computed per chunk
                # Here we store chunks and use the parent embedding as approximation
                for chunk in chunks:
                    all_chunks.append(chunk)
                    all_embeddings.append(emb)

        self.documents.extend(all_chunks)
        chunk_embeddings = torch.stack(all_embeddings)

        # Ensure embeddings are float32 numpy for FAISS
        embeddings_np = chunk_embeddings.detach().cpu().numpy().astype("float32")

        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
        embeddings_np = embeddings_np / norms

        if self.use_faiss:
            self.index.add(embeddings_np)
        else:
            # Brute-force fallback
            emb_tensor = torch.from_numpy(embeddings_np)
            if self._embeddings is None:
                self._embeddings = emb_tensor
            else:
                self._embeddings = torch.cat([self._embeddings, emb_tensor], dim=0)

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant documents.

        Args:
            query_embedding: Query embedding ``[embedding_dim]`` or ``[1, embedding_dim]``.
            k: Number of documents to retrieve.

        Returns:
            List of dictionaries with ``"document"`` and ``"score"`` keys.
        """
        if len(self.documents) == 0:
            return []

        # Normalize query embedding
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)

        query_np = query_embedding.detach().cpu().numpy().astype("float32")
        query_norm = np.linalg.norm(query_np)
        if query_norm > 0:
            query_np = query_np / query_norm

        k = min(k, len(self.documents))

        if self.use_faiss:
            scores, indices = self.index.search(query_np, k)
            results = []
            for j, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                results.append({
                    "document": self.documents[idx],
                    "score": float(scores[0][j]),
                })
            return results
        else:
            # Brute-force cosine similarity
            query_t = torch.from_numpy(query_np).squeeze()  # [embedding_dim]
            similarities = (self._embeddings @ query_t).squeeze()  # [num_docs]
            top_k = torch.topk(similarities, k)
            return [
                {"document": self.documents[idx], "score": float(top_k.values[j])}
                for j, idx in enumerate(top_k.indices.tolist())
            ]

    def save_index(self, path: str) -> None:
        """Save the retrieval index to disk.

        Args:
            path: Directory path to save to.
        """
        os.makedirs(path, exist_ok=True)

        # Save documents
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

        if self.use_faiss:
            self._faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        else:
            torch.save(self._embeddings, os.path.join(path, "embeddings.pt"))

    def load_index(self, path: str) -> None:
        """Load the retrieval index from disk.

        Args:
            path: Directory path to load from.
        """
        # Load documents
        doc_path = os.path.join(path, "documents.pkl")
        if os.path.exists(doc_path):
            with open(doc_path, "rb") as f:
                self.documents = pickle.load(f)

        if self.use_faiss:
            index_path = os.path.join(path, "faiss.index")
            if os.path.exists(index_path):
                self.index = self._faiss.read_index(index_path)
        else:
            emb_path = os.path.join(path, "embeddings.pt")
            if os.path.exists(emb_path):
                self._embeddings = torch.load(emb_path, weights_only=True)

    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return len(self.documents)


# --------------------------------------------------------------------------- #
# Convenience: StreamingKVCache manager
# --------------------------------------------------------------------------- #

class StreamingKVCache:
    """Manager for KV caches across multiple transformer layers with streaming.

    Wraps per-layer StreamingAttention caches and provides convenient
    methods for initialization, trimming, and reset.

    Args:
        num_layers: Number of transformer layers.
        num_sink_tokens: Number of attention sink tokens.
        window_size: Sliding window size.
    """

    def __init__(
        self,
        num_layers: int,
        num_sink_tokens: int = 4,
        window_size: int = 1024,
    ) -> None:
        self.num_layers = num_layers
        self.num_sink_tokens = num_sink_tokens
        self.window_size = window_size
        self.caches: List[Optional[Tuple]] = [None] * num_layers

    def __getitem__(self, layer_idx: int) -> Optional[Tuple]:
        return self.caches[layer_idx]

    def __setitem__(self, layer_idx: int, value: Tuple) -> None:
        self.caches[layer_idx] = value

    def __len__(self) -> int:
        return self.num_layers

    def get_total_cache_size(self) -> int:
        """Get the total number of tokens stored across all layers."""
        total = 0
        for cache in self.caches:
            if cache is not None and len(cache) == 4:
                recent_k, _, sink_k, _ = cache
                total += sink_k.shape[2] + recent_k.shape[2]
        return total

    def reset(self) -> None:
        """Clear all caches."""
        self.caches = [None] * self.num_layers

    def to_device(self, device: torch.device) -> "StreamingKVCache":
        """Move all cache tensors to the specified device."""
        for i, cache in enumerate(self.caches):
            if cache is not None:
                self.caches[i] = tuple(t.to(device) for t in cache)
        return self
