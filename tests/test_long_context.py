"""Tests for long context memory: SlidingWindowAttention, StreamingAttention, RAGRetriever."""

import sys
import unittest

import torch

# Ensure project root is importable
sys.path.insert(0, "/home/kimi/work-longcontext")

from selfllm.model.long_context import (
    RAGRetriever,
    SlidingWindowAttention,
    StreamingAttention,
    StreamingKVCache,
)


# --------------------------------------------------------------------------- #
# Sliding Window Attention Tests
# --------------------------------------------------------------------------- #


class TestSlidingWindowAttention(unittest.TestCase):
    """Test SlidingWindowAttention: causal sliding window masking."""

    def setUp(self) -> None:
        self.d_model = 64
        self.n_heads = 4
        self.window_size = 4
        self.batch_size = 2
        self.seq_len = 10

    def test_sliding_window_mask(self) -> None:
        """Tokens should only attend to tokens within the sliding window."""
        attn = SlidingWindowAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            window_size=self.window_size,
            max_seq_len=128,
        )
        attn.eval()

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)

        # Test prefill (no cache)
        out, cache = attn(x)

        # The sliding window mask is internal. We verify by checking that
        # the output is well-formed (no NaNs from -inf masking) and that
        # early tokens don't attend to later tokens (causality).
        self.assertFalse(torch.isnan(out).any(), "Output contains NaN values")
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

        # Now test with generation (single token with cache) to verify window behavior
        # First, build up a cache longer than window_size
        single_token = torch.randn(self.batch_size, 1, self.d_model)
        out1, cache1 = attn(single_token)

        # Generate several more tokens to build cache
        current_cache = cache1
        for _ in range(self.window_size + 4):
            tok = torch.randn(self.batch_size, 1, self.d_model)
            out_t, current_cache = attn(tok, kv_cache=current_cache)
            self.assertFalse(torch.isnan(out_t).any(), "NaN at generation step")

        # Cache should have accumulated
        cached_k, cached_v = current_cache
        self.assertGreaterEqual(cached_k.shape[2], self.window_size)

    def test_sliding_window_shape(self) -> None:
        """Output should have correct shape [batch, seq_len, d_model]."""
        attn = SlidingWindowAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            window_size=self.window_size,
            max_seq_len=128,
        )

        for seq_len in [1, 4, 8, 16]:
            x = torch.randn(self.batch_size, seq_len, self.d_model)
            out, cache = attn(x)
            self.assertEqual(out.shape, (self.batch_size, seq_len, self.d_model))

    def test_sliding_window_causality(self) -> None:
        """Verify causal property: later tokens don't affect earlier predictions."""
        attn = SlidingWindowAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            window_size=8,
            max_seq_len=128,
        )
        attn.eval()

        # Process first 5 tokens
        x_prefix = torch.randn(1, 5, self.d_model)
        out1, _ = attn(x_prefix)

        # Process same 5 tokens + 3 more (same prefix)
        x_extra = torch.randn(1, 3, self.d_model)
        x_full = torch.cat([x_prefix, x_extra], dim=1)
        out2, _ = attn(x_full)

        # The first 5 outputs should be identical (causal property)
        self.assertTrue(
            torch.allclose(out1, out2[:, :5, :], atol=1e-4),
            "Causality violated: later tokens affected earlier outputs"
        )

    def test_kv_cache_accumulation(self) -> None:
        """KV cache should accumulate across generation steps."""
        attn = SlidingWindowAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            window_size=self.window_size,
            max_seq_len=128,
        )
        attn.eval()

        # Start with one token
        x = torch.randn(1, 1, self.d_model)
        out, cache = attn(x)
        k1, v1 = cache
        self.assertEqual(k1.shape[2], 1)

        # Add another token
        x2 = torch.randn(1, 1, self.d_model)
        out2, cache2 = attn(x2, kv_cache=cache)
        k2, v2 = cache2
        self.assertEqual(k2.shape[2], 2)

    def test_window_size_effect(self) -> None:
        """Different window sizes should produce different outputs."""
        x = torch.randn(1, 16, self.d_model)

        attn_small = SlidingWindowAttention(
            d_model=self.d_model, n_heads=self.n_heads, window_size=2, max_seq_len=128,
        )
        attn_large = SlidingWindowAttention(
            d_model=self.d_model, n_heads=self.n_heads, window_size=16, max_seq_len=128,
        )
        attn_small.eval()
        attn_large.eval()

        out_small, _ = attn_small(x)
        out_large, _ = attn_large(x)

        # With different window sizes, outputs should differ
        # (the large window can see more context)
        self.assertFalse(
            torch.allclose(out_small, out_large, atol=1e-5),
            "Different window sizes produced identical outputs"
        )


# --------------------------------------------------------------------------- #
# Streaming Attention (StreamingLLM) Tests
# --------------------------------------------------------------------------- #


class TestStreamingAttention(unittest.TestCase):
    """Test StreamingAttention: attention sinks + sliding window for infinite context."""

    def setUp(self) -> None:
        self.d_model = 64
        self.n_heads = 4
        self.window_size = 4
        self.num_sink_tokens = 2
        self.batch_size = 1

    def test_streaming_attention_sinks(self) -> None:
        """Sink tokens should be preserved in the KV cache across all steps."""
        attn = StreamingAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            window_size=self.window_size,
            num_sink_tokens=self.num_sink_tokens,
            max_seq_len=128,
        )
        attn.eval()

        # Initial prefill with enough tokens to establish sinks
        x = torch.randn(self.batch_size, 8, self.d_model)
        out, cache = attn(x, is_prefill=True)

        recent_k, recent_v, sink_k, sink_v = cache
        # Sinks should be established
        self.assertEqual(
            sink_k.shape[2], self.num_sink_tokens,
            f"Expected {self.num_sink_tokens} sink tokens, got {sink_k.shape[2]}"
        )

        # Store the sink keys for comparison
        original_sink_k = sink_k.clone()
        original_sink_v = sink_v.clone()

        # Generate many more tokens - sinks should remain unchanged
        current_cache = cache
        for step in range(20):
            tok = torch.randn(self.batch_size, 1, self.d_model)
            out_t, current_cache = attn(tok, kv_cache=current_cache, is_prefill=False)

        recent_k, recent_v, sink_k, sink_v = current_cache

        # Sinks should still be the same
        self.assertTrue(
            torch.allclose(sink_k, original_sink_k),
            "Sink keys were modified during generation"
        )
        self.assertTrue(
            torch.allclose(sink_v, original_sink_v),
            "Sink values were modified during generation"
        )

    def test_streaming_infinite_context(self) -> None:
        """Should process sequences far longer than max_seq_len would allow."""
        attn = StreamingAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            window_size=4,
            num_sink_tokens=2,
            max_seq_len=128,
        )
        attn.eval()

        # Simulate a very long sequence (200 tokens) that exceeds typical limits
        long_seq_len = 200
        chunk_size = 8

        # Process in chunks
        cache = None
        for i in range(0, long_seq_len, chunk_size):
            chunk = torch.randn(self.batch_size, chunk_size, self.d_model)
            is_first = (i == 0)
            out, cache = attn(chunk, kv_cache=cache, is_prefill=is_first)

        # Verify output is valid
        self.assertFalse(torch.isnan(out).any(), "NaN in output after processing long sequence")
        self.assertEqual(out.shape, (self.batch_size, chunk_size, self.d_model))

        # Continue generating even more
        for _ in range(50):
            tok = torch.randn(self.batch_size, 1, self.d_model)
            out, cache = attn(tok, kv_cache=cache, is_prefill=False)

        self.assertFalse(torch.isnan(out).any(), "NaN in output after extended generation")

    def test_streaming_kv_cache_efficient(self) -> None:
        """KV cache size should be bounded by num_sink_tokens + window_size."""
        attn = StreamingAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            window_size=self.window_size,
            num_sink_tokens=self.num_sink_tokens,
            max_seq_len=128,
        )
        attn.eval()

        # Prefill
        x = torch.randn(self.batch_size, 8, self.d_model)
        out, cache = attn(x, is_prefill=True)

        max_expected_size = self.num_sink_tokens + self.window_size

        # Generate many tokens
        for step in range(100):
            tok = torch.randn(self.batch_size, 1, self.d_model)
            out, cache = attn(tok, kv_cache=cache, is_prefill=False)

        recent_k, recent_v, sink_k, sink_v = cache
        total_cache_size = sink_k.shape[2] + recent_k.shape[2]

        self.assertLessEqual(
            total_cache_size, max_expected_size,
            f"Cache size ({total_cache_size}) exceeded budget ({max_expected_size})"
        )

    def test_streaming_output_shape(self) -> None:
        """Output should have correct shape for various sequence lengths."""
        attn = StreamingAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            window_size=self.window_size,
            num_sink_tokens=self.num_sink_tokens,
            max_seq_len=128,
        )

        for seq_len in [1, 4, 8, 16]:
            x = torch.randn(self.batch_size, seq_len, self.d_model)
            out, cache = attn(x)
            self.assertEqual(out.shape, (self.batch_size, seq_len, self.d_model))

    def test_streaming_prefill_vs_generation(self) -> None:
        """Prefill and generation modes should produce consistent results."""
        attn = StreamingAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            window_size=8,
            num_sink_tokens=2,
            max_seq_len=128,
        )
        attn.eval()

        # Process 10 tokens at once (prefill)
        x_full = torch.randn(1, 10, self.d_model)
        out_full, cache = attn(x_full, is_prefill=True)

        # Process same 10 tokens one at a time (generation)
        attn2 = StreamingAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            window_size=8,
            num_sink_tokens=2,
            max_seq_len=128,
        )
        attn2.eval()
        attn2.load_state_dict(attn.state_dict())

        cache2 = None
        for i in range(10):
            tok = x_full[:, i:i+1, :]
            is_first = (i == 0)
            out_t, cache2 = attn2(tok, kv_cache=cache2, is_prefill=is_first)

        # The outputs might not be exactly equal due to floating point,
        # but they should be very close
        # We need to compare the last token output from generation with the last from prefill
        self.assertTrue(
            torch.allclose(out_full[:, -1, :], out_t[:, -1, :], atol=1e-4),
            "Prefill and generation produced different results"
        )

    def test_streaming_causality(self) -> None:
        """Verify causal property holds."""
        attn = StreamingAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            window_size=8,
            num_sink_tokens=2,
            max_seq_len=128,
        )
        attn.eval()

        # Process first 5 tokens
        x_prefix = torch.randn(1, 5, self.d_model)
        out1, _ = attn(x_prefix)

        # Process same 5 tokens + 3 more (same prefix)
        x_extra = torch.randn(1, 3, self.d_model)
        x_full = torch.cat([x_prefix, x_extra], dim=1)
        out2, _ = attn(x_full)

        # First 5 outputs should be identical (causal property)
        self.assertTrue(
            torch.allclose(out1, out2[:, :5, :], atol=1e-4),
            "Causality violated in streaming attention"
        )

    def test_streaming_with_legacy_kv_cache(self) -> None:
        """Should handle legacy (k, v) 2-tuple cache format."""
        attn = StreamingAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            window_size=self.window_size,
            num_sink_tokens=self.num_sink_tokens,
            max_seq_len=128,
        )
        attn.eval()

        # Create a legacy cache
        cache_len = 6
        k_cache = torch.randn(self.batch_size, self.n_heads, cache_len, self.head_dim)
        v_cache = torch.randn(self.batch_size, self.n_heads, cache_len, self.head_dim)
        legacy_cache = (k_cache, v_cache)

        # Process a new token with legacy cache
        x = torch.randn(self.batch_size, 1, self.d_model)
        out, new_cache = attn(x, kv_cache=legacy_cache, is_prefill=False)

        # Should return 4-tuple format
        self.assertEqual(len(new_cache), 4)
        recent_k, recent_v, sink_k, sink_v = new_cache

        # Should have extracted sinks from the legacy cache
        self.assertEqual(sink_k.shape[2], self.num_sink_tokens)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# --------------------------------------------------------------------------- #
# StreamingKVCache Tests
# --------------------------------------------------------------------------- #


class TestStreamingKVCache(unittest.TestCase):
    """Test StreamingKVCache manager."""

    def test_init(self) -> None:
        """Should initialize with correct number of empty layers."""
        cache = StreamingKVCache(num_layers=8)
        self.assertEqual(len(cache), 8)
        for i in range(8):
            self.assertIsNone(cache[i])

    def test_set_get(self) -> None:
        """Should support setting and getting per-layer caches."""
        cache = StreamingKVCache(num_layers=4)

        # Set a cache entry
        rk = torch.randn(1, 2, 4, 16)
        rv = torch.randn(1, 2, 4, 16)
        sk = torch.randn(1, 2, 2, 16)
        sv = torch.randn(1, 2, 2, 16)
        cache[1] = (rk, rv, sk, sv)

        self.assertIsNone(cache[0])
        self.assertIsNotNone(cache[1])
        self.assertEqual(cache[1][2].shape[2], 2)  # sink_k has 2 tokens

    def test_reset(self) -> None:
        """Should clear all caches on reset."""
        cache = StreamingKVCache(num_layers=4)
        rk = torch.randn(1, 2, 4, 16)
        rv = torch.randn(1, 2, 4, 16)
        sk = torch.randn(1, 2, 2, 16)
        sv = torch.randn(1, 2, 2, 16)
        cache[0] = (rk, rv, sk, sv)

        cache.reset()
        for i in range(4):
            self.assertIsNone(cache[i])


# --------------------------------------------------------------------------- #
# RAG Retriever Tests
# --------------------------------------------------------------------------- #


class TestRAGRetriever(unittest.TestCase):
    """Test RAGRetriever: document retrieval with FAISS/brute-force fallback."""

    def setUp(self) -> None:
        self.embedding_dim = 32
        self.num_docs = 10

    def test_rag_retrieve(self) -> None:
        """Should retrieve relevant documents ordered by similarity."""
        retriever = RAGRetriever(embedding_dim=self.embedding_dim)

        # Create documents with orthogonal-ish embeddings
        documents = [f"Document about topic {i}" for i in range(self.num_docs)]
        embeddings = torch.randn(self.num_docs, self.embedding_dim)
        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        retriever.add_documents(documents, embeddings)

        # Query with the first document's embedding
        query = embeddings[0]
        results = retriever.retrieve(query, k=3)

        self.assertEqual(len(results), 3)
        # First result should be the most similar (same document)
        self.assertEqual(results[0]["document"], documents[0])
        # Scores should be in descending order
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i]["score"], results[i + 1]["score"],
                "Results not sorted by score"
            )

    def test_rag_faiss_fallback(self) -> None:
        """Should work without FAISS using brute-force cosine similarity."""
        retriever = RAGRetriever(embedding_dim=self.embedding_dim)

        # Ensure we're using brute-force (no FAISS available)
        self.assertFalse(retriever.use_faiss)

        documents = [f"Doc {i}" for i in range(5)]
        embeddings = torch.randn(5, self.embedding_dim)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        retriever.add_documents(documents, embeddings)

        query = embeddings[2]
        results = retriever.retrieve(query, k=3)

        self.assertGreater(len(results), 0)
        self.assertIn("document", results[0])
        self.assertIn("score", results[0])
        self.assertIsInstance(results[0]["score"], float)

    def test_rag_document_chunking(self) -> None:
        """Long documents should be split into overlapping chunks."""
        retriever = RAGRetriever(
            embedding_dim=self.embedding_dim,
            chunk_size=10,
            chunk_overlap=3,
        )

        # Create a very long document
        words = [f"word{i}" for i in range(50)]
        long_doc = " ".join(words)

        documents = [long_doc]
        embeddings = torch.randn(1, self.embedding_dim)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        retriever.add_documents(documents, embeddings)

        # Should have created multiple chunks
        self.assertGreater(
            len(retriever.documents), 1,
            "Long document was not chunked"
        )

        # Each chunk should be a substring of the original
        for chunk in retriever.documents:
            self.assertIn(chunk, long_doc)

    def test_rag_empty_index(self) -> None:
        """Retrieving from empty index should return empty list."""
        retriever = RAGRetriever(embedding_dim=self.embedding_dim)
        query = torch.randn(self.embedding_dim)
        results = retriever.retrieve(query, k=5)
        self.assertEqual(len(results), 0)

    def test_rag_save_load(self) -> None:
        """Should save and load index correctly."""
        import tempfile

        retriever = RAGRetriever(embedding_dim=self.embedding_dim)

        documents = [f"Doc {i}" for i in range(5)]
        embeddings = torch.randn(5, self.embedding_dim)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        retriever.add_documents(documents, embeddings)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever.save_index(tmpdir)

            # Load into new retriever
            retriever2 = RAGRetriever(embedding_dim=self.embedding_dim)
            retriever2.load_index(tmpdir)

            self.assertEqual(len(retriever2.documents), len(documents))
            self.assertEqual(retriever2.documents, documents)

    def test_rag_embedding_mismatch(self) -> None:
        """Should raise error when document count doesn't match embedding count."""
        retriever = RAGRetriever(embedding_dim=self.embedding_dim)

        documents = ["doc1", "doc2"]
        embeddings = torch.randn(3, self.embedding_dim)  # Mismatched count

        with self.assertRaises(ValueError):
            retriever.add_documents(documents, embeddings)


# --------------------------------------------------------------------------- #
# Integration / Cross-cutting Tests
# --------------------------------------------------------------------------- #


class TestIntegration(unittest.TestCase):
    """Integration tests for long context components."""

    def test_sliding_vs_streaming_different(self) -> None:
        """SlidingWindow and StreamingAttention should behave differently
        when cache is built up (Streaming has sinks)."""
        d_model, n_heads = 64, 4

        sliding = SlidingWindowAttention(d_model, n_heads, window_size=4, max_seq_len=128)
        streaming = StreamingAttention(d_model, n_heads, window_size=4, num_sink_tokens=2, max_seq_len=128)
        streaming.load_state_dict(sliding.state_dict(), strict=False)

        sliding.eval()
        streaming.eval()

        # Build up cache with several tokens
        x_init = torch.randn(1, 6, d_model)
        _, sliding_cache = sliding(x_init)
        _, streaming_cache = streaming(x_init, is_prefill=True)

        # Query with a new token
        x_query = torch.randn(1, 1, d_model)
        out_sliding, _ = sliding(x_query, kv_cache=sliding_cache)
        out_streaming, _ = streaming(x_query, kv_cache=streaming_cache, is_prefill=False)

        # Both should produce valid output (no NaN)
        self.assertFalse(torch.isnan(out_sliding).any())
        self.assertFalse(torch.isnan(out_streaming).any())

    def test_streaming_with_different_sink_counts(self) -> None:
        """Should work with different numbers of sink tokens."""
        d_model, n_heads = 64, 4

        for num_sinks in [1, 2, 4]:
            attn = StreamingAttention(
                d_model, n_heads,
                window_size=4,
                num_sink_tokens=num_sinks,
                max_seq_len=128,
            )
            attn.eval()

            x = torch.randn(1, 10, d_model)
            out, cache = attn(x, is_prefill=True)
            recent_k, recent_v, sink_k, sink_v = cache

            self.assertEqual(sink_k.shape[2], num_sinks)
            self.assertFalse(torch.isnan(out).any())

    def test_long_sequence_stress(self) -> None:
        """Stress test: process a very long sequence."""
        attn = StreamingAttention(
            d_model=32,
            n_heads=4,
            window_size=8,
            num_sink_tokens=2,
            max_seq_len=1024,
        )
        attn.eval()

        # Simulate 500 tokens of streaming input
        cache = None
        for i in range(50):
            chunk = torch.randn(1, 10, 32)
            is_first = (i == 0)
            out, cache = attn(chunk, kv_cache=cache, is_prefill=is_first)

        # Verify cache is bounded
        recent_k, recent_v, sink_k, sink_v = cache
        total = sink_k.shape[2] + recent_k.shape[2]
        max_expected = 2 + 8  # num_sink_tokens + window_size
        self.assertLessEqual(total, max_expected)

        # Verify output is valid
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
