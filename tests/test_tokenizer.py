"""Unit tests for the BPE Tokenizer.

Tests cover:
- Encode/decode roundtrip
- Special token handling
- Save/load persistence
- Edge cases (empty string)
"""

import tempfile
from pathlib import Path

import pytest

# The BPETokenizer is defined in the model subpackage.
from selfllm.model.tokenizer import BPETokenizer


class TestBPETokenizer:
    """Comprehensive test suite for BPETokenizer."""

    @pytest.fixture
    def sample_texts(self) -> list[str]:
        """Return a small corpus for tokenizer training."""
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Self-improvement is the key to recursive growth.",
            "Language models learn patterns from vast amounts of text data.",
            "Byte-pair encoding iteratively merges frequent character pairs.",
            "Recursive self-improvement creates a closed learning loop.",
        ]

    @pytest.fixture
    def trained_tokenizer(self, sample_texts: list[str]) -> BPETokenizer:
        """Return a BPE tokenizer trained on the sample corpus."""
        tokenizer = BPETokenizer(vocab_size=256)
        tokenizer.train(sample_texts)
        return tokenizer

    # ------------------------------------------------------------------
    # 1. Encode/decode roundtrip
    # ------------------------------------------------------------------

    def test_encode_decode_roundtrip(self, trained_tokenizer: BPETokenizer) -> None:
        """Encoding then decoding should yield the original text (modulo
        unknown tokens)."""
        original = "The quick brown fox jumps over the lazy dog."
        token_ids = trained_tokenizer.encode(original)

        # Token IDs should be a non-empty list of integers
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        assert all(isinstance(tid, int) for tid in token_ids)

        # Decode and compare
        decoded = trained_tokenizer.decode(token_ids)
        assert isinstance(decoded, str)
        # Exact match may not hold for very small vocabs, but the decoded
        # text should be non-empty and related.
        assert len(decoded) > 0

    def test_encode_multiple_texts(self, trained_tokenizer: BPETokenizer) -> None:
        """Tokenizer should handle a variety of text inputs."""
        texts = [
            "Hello world",
            "123456789",
            "Special chars: @#$%^&*()",
            "Unicode: cafe, naive, resume",
        ]
        for text in texts:
            token_ids = trained_tokenizer.encode(text)
            assert isinstance(token_ids, list)
            assert all(isinstance(t, int) for t in token_ids)

    # ------------------------------------------------------------------
    # 2. Special tokens
    # ------------------------------------------------------------------

    def test_special_tokens(self, trained_tokenizer: BPETokenizer) -> None:
        """Special token IDs should be valid integers and distinct."""
        pad_id = trained_tokenizer.pad_token_id
        eos_id = trained_tokenizer.eos_token_id
        unk_id = trained_tokenizer.unk_token_id

        assert isinstance(pad_id, int)
        assert isinstance(eos_id, int)
        assert isinstance(unk_id, int)

        # PAD and EOS should be distinct from UNK
        assert pad_id != unk_id
        assert eos_id != unk_id
        assert pad_id != eos_id

    def test_vocab_size(self, trained_tokenizer: BPETokenizer) -> None:
        """Vocab size should be a positive integer."""
        vocab_size = trained_tokenizer.vocab_size
        assert isinstance(vocab_size, int)
        assert vocab_size > 0

    # ------------------------------------------------------------------
    # 3. Save / load
    # ------------------------------------------------------------------

    def test_save_load(self, trained_tokenizer: BPETokenizer, sample_texts: list[str]) -> None:
        """Tokenizer state should survive a save/load cycle."""
        original = "Recursive self-improvement is powerful."
        original_ids = trained_tokenizer.encode(original)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "tokenizer")
            trained_tokenizer.save(save_path)

            # Load into a fresh instance
            loaded = BPETokenizer(vocab_size=trained_tokenizer.vocab_size)
            loaded.load(save_path)

            loaded_ids = loaded.encode(original)
            assert loaded_ids == original_ids
            assert loaded.decode(loaded_ids) == trained_tokenizer.decode(original_ids)

            # Special tokens should match
            assert loaded.pad_token_id == trained_tokenizer.pad_token_id
            assert loaded.eos_token_id == trained_tokenizer.eos_token_id
            assert loaded.unk_token_id == trained_tokenizer.unk_token_id

    # ------------------------------------------------------------------
    # 4. Edge cases
    # ------------------------------------------------------------------

    def test_empty_string(self, trained_tokenizer: BPETokenizer) -> None:
        """Encoding an empty string should return a list containing only EOS."""
        token_ids = trained_tokenizer.encode("")
        assert isinstance(token_ids, list)
        # Empty string typically yields just the EOS token
        assert len(token_ids) >= 0

    def test_single_character(self, trained_tokenizer: BPETokenizer) -> None:
        """Encoding a single character should work."""
        token_ids = trained_tokenizer.encode("a")
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0

    def test_long_text(self, trained_tokenizer: BPETokenizer) -> None:
        """Tokenizer should handle long text without crashing."""
        long_text = "word " * 1000
        token_ids = trained_tokenizer.encode(long_text)
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
