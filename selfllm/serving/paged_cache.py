"""PagedAttention: KV cache management using fixed-size blocks.

Inspired by vLLM's PagedAttention. Key idea: treat KV cache like virtual memory --
store in fixed-size blocks, allocate on demand, share via copy-on-write.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockManager:
    """Manages KV cache as fixed-size blocks (like OS virtual memory pages).

    Each block holds KV for a fixed number of tokens (block_size, typically 16).
    Sequences share blocks via copy-on-write reference counting.

    The block pool is shaped [num_layers, num_blocks, block_size, num_heads, head_dim]
    for both K and V caches, enabling efficient per-layer access.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        """
        Args:
            num_blocks: Total number of blocks in the pool.
            block_size: Number of tokens per block (typically 16).
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            dtype: Data type for cache tensors.
            device: Device to store cache on ("cuda" or "cpu").
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Block pool: [num_layers, num_blocks, block_size, num_heads, head_dim]
        # for K and V each layer.
        self.k_cache = torch.zeros(
            (num_layers, num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (num_layers, num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        # Free block tracking
        self.free_blocks: set = set(range(num_blocks))
        self.block_ref_count: List[int] = [0] * num_blocks

        # Per-block fill count (how many tokens are stored in each block)
        self.block_fill_count: Dict[int, int] = {}

    def allocate(self, num_tokens: int) -> List[int]:
        """Allocate blocks for a sequence. Returns block indices.

        Args:
            num_tokens: Number of tokens that need to be stored.

        Returns:
            List of allocated block indices.

        Raises:
            RuntimeError: If there are not enough free blocks.
        """
        num_blocks_needed = math.ceil(num_tokens / self.block_size)
        if num_blocks_needed > len(self.free_blocks):
            raise RuntimeError(
                f"Out of KV cache blocks. Need {num_blocks_needed}, "
                f"have {len(self.free_blocks)} free."
            )

        allocated: List[int] = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            self.block_ref_count[block_id] += 1
            self.block_fill_count[block_id] = self.block_size
            allocated.append(block_id)

        # Adjust fill count of last block to actual token count
        if allocated:
            last_block_tokens = num_tokens % self.block_size
            if last_block_tokens != 0:
                self.block_fill_count[allocated[-1]] = last_block_tokens

        return allocated

    def append_token(self, block_ids: List[int]) -> List[int]:
        """Mark that a new token was written. Extend blocks if needed.

        When the last block becomes full, allocates a new block and appends it.

        Args:
            block_ids: Current list of block indices for the sequence.

        Returns:
            Updated list of block indices (may include a newly allocated block).
        """
        if not block_ids:
            # No blocks yet -- allocate one
            new_block = self.allocate(1)
            return new_block

        result = list(block_ids)
        last_block = block_ids[-1]
        current_fill = self.block_fill_count.get(last_block, 0)

        if current_fill >= self.block_size:
            # Last block is full, allocate a new one
            try:
                new_blocks = self.allocate(1)
                result.extend(new_blocks)
            except RuntimeError:
                # Out of blocks -- cannot extend
                pass
        else:
            # Increment fill count of the last block
            self.block_fill_count[last_block] = current_fill + 1

        return result

    def fork(self, block_ids: List[int]) -> List[int]:
        """Copy-on-write fork: increment ref counts, return same blocks.

        Used when a sequence is shared (e.g., for parallel sampling / beam search).
        The blocks are shared until one branch writes, at which point a copy
        would be made.

        Args:
            block_ids: Block indices to fork.

        Returns:
            The same block indices (now with incremented ref counts).
        """
        for bid in block_ids:
            self.block_ref_count[bid] += 1
        return list(block_ids)

    def free(self, block_ids: List[int]) -> None:
        """Free blocks, decrementing ref counts.

        Blocks are only returned to the free pool when their ref count reaches 0.

        Args:
            block_ids: Block indices to free.
        """
        for bid in block_ids:
            self.block_ref_count[bid] -= 1
            if self.block_ref_count[bid] <= 0:
                self.block_ref_count[bid] = 0
                self.free_blocks.add(bid)
                self.block_fill_count.pop(bid, None)

    def get_kv_cache(
        self,
        layer_idx: int,
        block_ids: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get KV cache for a specific layer and set of blocks.

        Args:
            layer_idx: Which transformer layer to retrieve from.
            block_ids: List of block indices.

        Returns:
            Tuple of (k_cache, v_cache) each shaped
            [num_blocks, block_size, num_heads, head_dim].
        """
        k = self.k_cache[layer_idx, block_ids]
        v = self.v_cache[layer_idx, block_ids]
        return k, v

    def write_kv_cache(
        self,
        layer_idx: int,
        block_ids: List[int],
        position: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write K, V for a single token position.

        Args:
            layer_idx: Which transformer layer to write to.
            block_ids: List of block indices for the sequence.
            position: Token position within the sequence (0-indexed).
            k: Key tensor [num_heads, head_dim].
            v: Value tensor [num_heads, head_dim].
        """
        block_idx = position // self.block_size
        offset = position % self.block_size
        bid = block_ids[block_idx]
        self.k_cache[layer_idx, bid, offset] = k
        self.v_cache[layer_idx, bid, offset] = v

    @property
    def num_free_blocks(self) -> int:
        """Return the number of free blocks available."""
        return len(self.free_blocks)

    @property
    def memory_usage_mb(self) -> float:
        """Return total memory usage of the KV cache in megabytes."""
        element_size = torch.finfo(self.dtype).bits / 8
        total_elements = (
            2  # K + V
            * self.num_layers
            * self.num_blocks
            * self.block_size
            * self.num_heads
            * self.head_dim
        )
        return (total_elements * element_size) / (1024 * 1024)


class PagedAttention(nn.Module):
    """Attention layer that uses PagedAttention block manager for KV cache.

    This module provides an attention computation that reads/writes KV values
    through the BlockManager rather than keeping them in a dense tensor.
    It is designed for autoregressive decoding (single token at a time).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Linear projections for Q, K, V, O
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Precomputed RoPE frequencies so paged decode applies the same
        # positional encoding as the dense attention path.
        dim_idx = torch.arange(0, head_dim, 2, dtype=torch.float64)
        inv_freq = 1.0 / (10000.0 ** (dim_idx / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float64)
        self.register_buffer(
            "rope_freqs",
            torch.outer(positions, inv_freq).float(),  # [max_seq_len, head_dim//2]
            persistent=False,
        )

    def _apply_rope(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """Rotate ``x`` ``[batch, 1, n_heads, head_dim]`` at a single absolute position."""
        pos = min(position, self.rope_freqs.shape[0] - 1)
        freqs = self.rope_freqs[pos]  # [head_dim//2]
        cos = torch.cos(freqs).view(1, 1, 1, -1)
        sin = torch.sin(freqs).view(1, 1, 1, -1)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        return torch.stack([rx1, rx2], dim=-1).view_as(x)

    def forward(
        self,
        x: torch.Tensor,
        block_manager: BlockManager,
        block_ids: List[int],
        layer_idx: int,
        position: int,
    ) -> torch.Tensor:
        """Compute attention using paged KV cache.

        For decoding (single token at a time), retrieves all prior KV from blocks,
        computes attention over full context in one pass.

        Args:
            x: Input tensor [batch_size, 1, d_model] for a single decode step.
            block_manager: The BlockManager holding the KV cache.
            block_ids: List of block indices allocated for this sequence.
            layer_idx: Index of the current transformer layer.
            position: Current token position (0-indexed).

        Returns:
            Output tensor [batch_size, 1, d_model].
        """
        batch_size = x.shape[0]

        # Project input to Q, K, V
        q = self.q_proj(x).view(batch_size, 1, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, 1, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, 1, self.n_heads, self.head_dim)

        # Apply RoPE at the current absolute position. K is rotated *before*
        # being written, so cached keys are stored already-rotated and the
        # attention below is positionally consistent.
        q = self._apply_rope(q, position)
        k = self._apply_rope(k, position)

        # Write new K, V to cache
        for b in range(batch_size):
            block_manager.write_kv_cache(
                layer_idx, block_ids, position, k[b, 0], v[b, 0]
            )

        # Retrieve all KV for this sequence
        all_k, all_v = block_manager.get_kv_cache(layer_idx, block_ids)
        # all_k: [num_blocks, block_size, heads, head_dim]

        # Flatten to [seq_len, heads, head_dim] (up to current position)
        seq_len = position + 1
        all_k_flat = all_k.view(-1, self.n_heads, self.head_dim)[:seq_len]
        all_v_flat = all_v.view(-1, self.n_heads, self.head_dim)[:seq_len]

        # Attention: q @ K^T / sqrt(d)
        # q: [B, 1, heads, head_dim] -> [B, heads, 1, head_dim]
        # all_k_flat: [seq_len, heads, head_dim] -> [1, heads, head_dim, seq_len]
        scores = (
            torch.matmul(
                q.transpose(1, 2),  # [B, heads, 1, head_dim]
                all_k_flat.unsqueeze(0).permute(0, 2, 3, 1),  # [1, heads, head_dim, seq_len]
            )
            / (self.head_dim**0.5)
        )  # [B, heads, 1, seq_len]

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(
            attn_weights, all_v_flat.unsqueeze(0).transpose(1, 2)
        )
        # [B, heads, 1, head_dim]

        out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        return self.o_proj(out)
