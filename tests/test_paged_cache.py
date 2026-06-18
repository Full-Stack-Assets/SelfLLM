"""Tests for PagedAttention KV cache management (paged_cache.py)."""

import math

import pytest
import torch

from selfllm.serving.paged_cache import BlockManager, PagedAttention


# ---------------------------------------------------------------------------
# BlockManager tests
# ---------------------------------------------------------------------------


class TestBlockManagerInit:
    """Tests for BlockManager initialization."""

    def test_block_manager_init(self):
        """BlockManager initializes with correct cache shapes."""
        bm = BlockManager(
            num_blocks=10,
            block_size=16,
            num_layers=4,
            num_heads=8,
            head_dim=64,
            dtype=torch.float32,
            device="cpu",
        )
        assert bm.num_blocks == 10
        assert bm.block_size == 16
        assert bm.num_layers == 4
        assert bm.num_heads == 8
        assert bm.head_dim == 64

        # Check cache tensor shapes: [num_layers, num_blocks, block_size, num_heads, head_dim]
        assert bm.k_cache.shape == (4, 10, 16, 8, 64)
        assert bm.v_cache.shape == (4, 10, 16, 8, 64)

    def test_all_blocks_free_at_init(self):
        """All blocks should be free at initialization."""
        bm = BlockManager(
            num_blocks=10, block_size=16, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        assert bm.num_free_blocks == 10
        assert len(bm.free_blocks) == 10

    def test_memory_usage(self):
        """Memory usage should be calculated correctly."""
        bm = BlockManager(
            num_blocks=10, block_size=16, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float16, device="cpu",
        )
        # 2 (K+V) * 10 (blocks) * 16 (block_size) * 2 (layers) * 4 (heads) * 32 (head_dim) * 2 (float16 bytes)
        # = 2 * 10 * 16 * 2 * 4 * 32 * 2 = 163840 bytes
        # / (1024 * 1024) = 0.15625 MB
        expected_mb = (2 * 10 * 16 * 2 * 4 * 32 * 2) / (1024 * 1024)
        assert abs(bm.memory_usage_mb - expected_mb) < 0.01


class TestBlockManagerAllocate:
    """Tests for BlockManager.allocate()."""

    def test_allocate_returns_correct_number_of_blocks(self):
        """allocate() returns the correct number of blocks for given tokens."""
        bm = BlockManager(
            num_blocks=10, block_size=4, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        # 5 tokens with block_size=4 -> ceil(5/4) = 2 blocks
        blocks = bm.allocate(5)
        assert len(blocks) == 2
        assert bm.num_free_blocks == 8

    def test_allocate_exact_block_boundary(self):
        """allocate() with exact block boundary uses correct count."""
        bm = BlockManager(
            num_blocks=10, block_size=4, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        # 8 tokens with block_size=4 -> ceil(8/4) = 2 blocks
        blocks = bm.allocate(8)
        assert len(blocks) == 2

    def test_allocate_one_token(self):
        """allocate() for a single token returns 1 block."""
        bm = BlockManager(
            num_blocks=10, block_size=16, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        blocks = bm.allocate(1)
        assert len(blocks) == 1
        assert bm.num_free_blocks == 9

    def test_allocate_ref_count(self):
        """allocate() increments ref counts."""
        bm = BlockManager(
            num_blocks=10, block_size=4, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        blocks = bm.allocate(5)
        for bid in blocks:
            assert bm.block_ref_count[bid] == 1

    def test_allocate_out_of_blocks(self):
        """allocate() raises RuntimeError when out of blocks."""
        bm = BlockManager(
            num_blocks=2, block_size=4, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        # Allocate all 2 blocks (8 tokens)
        bm.allocate(8)
        assert bm.num_free_blocks == 0

        # Should fail - no more blocks
        with pytest.raises(RuntimeError, match="Out of KV cache blocks"):
            bm.allocate(1)


class TestBlockManagerFree:
    """Tests for BlockManager.free()."""

    def test_free_returns_blocks_to_pool(self):
        """free() returns blocks to the free pool."""
        bm = BlockManager(
            num_blocks=10, block_size=4, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        blocks = bm.allocate(5)
        assert bm.num_free_blocks == 8

        bm.free(blocks)
        assert bm.num_free_blocks == 10

    def test_free_ref_count_decrement(self):
        """free() decrements ref counts correctly."""
        bm = BlockManager(
            num_blocks=10, block_size=4, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        blocks = bm.allocate(5)
        for bid in blocks:
            assert bm.block_ref_count[bid] == 1

        bm.free(blocks)
        for bid in blocks:
            assert bm.block_ref_count[bid] == 0


class TestBlockManagerFork:
    """Tests for BlockManager.fork() (copy-on-write)."""

    def test_fork_increments_ref_count(self):
        """fork() increments ref counts for shared blocks."""
        bm = BlockManager(
            num_blocks=10, block_size=4, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        blocks = bm.allocate(5)
        for bid in blocks:
            assert bm.block_ref_count[bid] == 1

        # Fork returns the same blocks with incremented refs
        forked = bm.fork(blocks)
        assert forked == blocks
        for bid in blocks:
            assert bm.block_ref_count[bid] == 2

    def test_fork_then_free(self):
        """fork() followed by partial free keeps blocks available."""
        bm = BlockManager(
            num_blocks=10, block_size=4, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        blocks = bm.allocate(5)
        forked = bm.fork(blocks)

        # Free original - blocks should NOT return to pool (ref count still 1)
        bm.free(blocks)
        assert bm.num_free_blocks == 8

        # Free forked - NOW blocks return to pool
        bm.free(forked)
        assert bm.num_free_blocks == 10


class TestBlockManagerGetKVCache:
    """Tests for BlockManager.get_kv_cache()."""

    def test_get_kv_cache_shape(self):
        """get_kv_cache() returns tensors with correct shape."""
        bm = BlockManager(
            num_blocks=10, block_size=4, num_layers=2, num_heads=8, head_dim=64,
            dtype=torch.float32, device="cpu",
        )
        blocks = bm.allocate(8)  # 2 blocks
        k, v = bm.get_kv_cache(layer_idx=0, block_ids=blocks)

        # Shape: [num_blocks, block_size, num_heads, head_dim]
        assert k.shape == (2, 4, 8, 64)
        assert v.shape == (2, 4, 8, 64)

    def test_get_kv_cache_per_layer(self):
        """get_kv_cache() returns different caches per layer."""
        bm = BlockManager(
            num_blocks=10, block_size=4, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        blocks = bm.allocate(4)  # 1 block

        # Write different values to layer 0 and layer 1
        bm.k_cache[0, blocks[0], 0] = torch.ones(4, 32)
        bm.k_cache[1, blocks[0], 0] = torch.ones(4, 32) * 2

        k0, _ = bm.get_kv_cache(layer_idx=0, block_ids=blocks)
        k1, _ = bm.get_kv_cache(layer_idx=1, block_ids=blocks)

        assert torch.allclose(k0[0, 0], torch.ones(4, 32))
        assert torch.allclose(k1[0, 0], torch.ones(4, 32) * 2)


class TestBlockManagerWriteKVCache:
    """Tests for BlockManager.write_kv_cache()."""

    def test_write_and_read_back(self):
        """write_kv_cache() stores values that can be read back."""
        bm = BlockManager(
            num_blocks=10, block_size=4, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        blocks = bm.allocate(4)  # 1 block

        k_val = torch.randn(4, 32)
        v_val = torch.randn(4, 32)

        bm.write_kv_cache(layer_idx=0, block_ids=blocks, position=2, k=k_val, v=v_val)

        k, v = bm.get_kv_cache(layer_idx=0, block_ids=blocks)
        assert torch.allclose(k[0, 2], k_val)
        assert torch.allclose(v[0, 2], v_val)

    def test_write_different_positions(self):
        """write_kv_cache() can write to different positions independently."""
        bm = BlockManager(
            num_blocks=10, block_size=4, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        blocks = bm.allocate(4)

        k0 = torch.ones(4, 32)
        k1 = torch.ones(4, 32) * 2

        bm.write_kv_cache(layer_idx=0, block_ids=blocks, position=0, k=k0, v=k0)
        bm.write_kv_cache(layer_idx=0, block_ids=blocks, position=1, k=k1, v=k1)

        k, v = bm.get_kv_cache(layer_idx=0, block_ids=blocks)
        assert torch.allclose(k[0, 0], k0)
        assert torch.allclose(k[0, 1], k1)


class TestBlockManagerAppendToken:
    """Tests for BlockManager.append_token()."""

    def test_append_token_no_new_block(self):
        """append_token() doesn't allocate when block has space."""
        bm = BlockManager(
            num_blocks=10, block_size=4, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        blocks = bm.allocate(2)  # 1 block, 2 tokens used
        assert bm.num_free_blocks == 9

        # Block has space (fill count = 2, block_size = 4)
        result = bm.append_token(blocks)
        assert result == blocks  # Same blocks
        assert bm.num_free_blocks == 9  # No new allocation

    def test_append_token_allocates_new_block(self):
        """append_token() allocates a new block when current is full."""
        bm = BlockManager(
            num_blocks=10, block_size=4, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        blocks = bm.allocate(4)  # 1 block, completely full
        # Manually set fill count to max to simulate full block
        bm.block_fill_count[blocks[-1]] = bm.block_size
        assert bm.num_free_blocks == 9

        result = bm.append_token(blocks)
        assert len(result) == 2  # New block added
        assert bm.num_free_blocks == 8

    def test_append_token_empty_blocks(self):
        """append_token() handles empty block list."""
        bm = BlockManager(
            num_blocks=10, block_size=4, num_layers=2, num_heads=4, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        result = bm.append_token([])
        assert len(result) == 1
        assert bm.num_free_blocks == 9


# ---------------------------------------------------------------------------
# PagedAttention tests
# ---------------------------------------------------------------------------


class TestPagedAttentionForward:
    """Tests for PagedAttention.forward()."""

    def test_forward_output_shape(self):
        """PagedAttention forward produces correct output shape."""
        d_model = 64
        n_heads = 4
        head_dim = 16
        batch_size = 2

        pa = PagedAttention(
            d_model=d_model, n_heads=n_heads, head_dim=head_dim, max_seq_len=128
        )

        bm = BlockManager(
            num_blocks=4,
            block_size=8,
            num_layers=2,
            num_heads=n_heads,
            head_dim=head_dim,
            dtype=torch.float32,
            device="cpu",
        )

        x = torch.randn(batch_size, 1, d_model)
        block_ids = bm.allocate(1)

        out = pa(x, bm, block_ids, layer_idx=0, position=0)
        assert out.shape == (batch_size, 1, d_model)

    def test_forward_batch_size_one(self):
        """PagedAttention forward works with batch size 1."""
        d_model = 32
        n_heads = 2
        head_dim = 16

        pa = PagedAttention(
            d_model=d_model, n_heads=n_heads, head_dim=head_dim, max_seq_len=64
        )

        bm = BlockManager(
            num_blocks=4, block_size=8, num_layers=1, num_heads=n_heads,
            head_dim=head_dim, dtype=torch.float32, device="cpu",
        )

        x = torch.randn(1, 1, d_model)
        block_ids = bm.allocate(1)

        out = pa(x, bm, block_ids, layer_idx=0, position=0)
        assert out.shape == (1, 1, d_model)

    def test_forward_writes_to_cache(self):
        """PagedAttention forward writes K, V to the block manager cache."""
        d_model = 32
        n_heads = 2
        head_dim = 16

        pa = PagedAttention(
            d_model=d_model, n_heads=n_heads, head_dim=head_dim, max_seq_len=64
        )

        bm = BlockManager(
            num_blocks=4, block_size=8, num_layers=1, num_heads=n_heads,
            head_dim=head_dim, dtype=torch.float32, device="cpu",
        )

        x = torch.randn(1, 1, d_model)
        block_ids = bm.allocate(1)

        # Cache should be zeros before
        k_before, v_before = bm.get_kv_cache(0, block_ids)
        assert torch.allclose(k_before[0, 0], torch.zeros(n_heads, head_dim))

        out = pa(x, bm, block_ids, layer_idx=0, position=0)
        assert out.shape == (1, 1, d_model)

        # Cache should now have values written
        k_after, v_after = bm.get_kv_cache(0, block_ids)
        assert not torch.allclose(k_after[0, 0], torch.zeros(n_heads, head_dim))

    def test_forward_multiple_positions(self):
        """PagedAttention forward works for multiple token positions."""
        d_model = 32
        n_heads = 2
        head_dim = 16

        pa = PagedAttention(
            d_model=d_model, n_heads=n_heads, head_dim=head_dim, max_seq_len=64
        )

        bm = BlockManager(
            num_blocks=4, block_size=8, num_layers=1, num_heads=n_heads,
            head_dim=head_dim, dtype=torch.float32, device="cpu",
        )

        block_ids = bm.allocate(4)

        for pos in range(4):
            x = torch.randn(1, 1, d_model)
            out = pa(x, bm, block_ids, layer_idx=0, position=pos)
            assert out.shape == (1, 1, d_model)

    def test_forward_different_layers(self):
        """PagedAttention forward works for different layer indices."""
        d_model = 32
        n_heads = 2
        head_dim = 16

        # Use separate PagedAttention instances per layer so weights differ
        pa0 = PagedAttention(
            d_model=d_model, n_heads=n_heads, head_dim=head_dim, max_seq_len=64
        )
        pa1 = PagedAttention(
            d_model=d_model, n_heads=n_heads, head_dim=head_dim, max_seq_len=64
        )

        bm = BlockManager(
            num_blocks=4, block_size=8, num_layers=2, num_heads=n_heads,
            head_dim=head_dim, dtype=torch.float32, device="cpu",
        )

        x = torch.randn(1, 1, d_model)
        block_ids = bm.allocate(2)

        out0 = pa0(x, bm, block_ids, layer_idx=0, position=0)
        out1 = pa1(x, bm, block_ids, layer_idx=1, position=0)

        assert out0.shape == (1, 1, d_model)
        assert out1.shape == (1, 1, d_model)

        # Different layers should have different cache values
        k0, _ = bm.get_kv_cache(0, block_ids)
        k1, _ = bm.get_kv_cache(1, block_ids)
        assert not torch.allclose(k0, k1)
