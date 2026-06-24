"""Real-world data pipeline: download, preprocess, tokenize, chunk."""

import glob
import json
import logging
import os
import re
import urllib.request
from typing import List, Optional, Set

from selfllm.model.tokenizer import BPETokenizer
from selfllm.training.dataset import SelfTrainingDataset

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    End-to-end data pipeline for training SelfLLM on real corpora.

    Usage:
        pipeline = DataPipeline(tokenizer)
        pipeline.download_gutenberg_books(output_dir="./data/books", num_books=100)
        dataset = pipeline.create_training_dataset(
            "./data/", output_path="./data/train.pt"
        )
    """

    # Gutenberg popular books (text file IDs)
    GUTENBERG_POPULAR = [
        1342,
        84,
        11,
        1661,
        174,
        98,
        2701,
        8438,
        5200,
        2542,
        345,
        43,
        74,
        1400,
        46,
        219,
        2591,
        1184,
        105,
        161,
        1260,
        147,
        55,
        1080,
        35,
        76,
        58585,
        6538,
        205,
        244,
        61236,
        852,
        1497,
        113,
        158,
        30254,
        5740,
        121,
        376,
        145,
        514,
        1259,
        1322,
        10,
        5827,
        2600,
        1212,
        181,
        27827,
        1083,
    ]

    GUTENBERG_SEARCH_URL = (
        "https://www.gutenberg.org/ebooks/search/"
        "?sort_order=downloads&start_index={start_index}"
    )

    def __init__(self, tokenizer: Optional[BPETokenizer], max_seq_len: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    # ------------------------------------------------------------------ #
    # Download
    # ------------------------------------------------------------------ #

    @classmethod
    def _text_urls_for_book(cls, book_id: int) -> List[str]:
        """Return candidate plain-text URLs for a Gutenberg book."""
        return [
            f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
            f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
            f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
            f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt.utf8",
        ]

    @classmethod
    def _discover_gutenberg_book_ids(cls, target_count: int) -> List[int]:
        """Discover popular Gutenberg IDs beyond the curated seed list."""
        seen: Set[int] = set()
        book_ids: List[int] = []
        for book_id in cls.GUTENBERG_POPULAR:
            if book_id not in seen:
                seen.add(book_id)
                book_ids.append(book_id)

        start_index = 1
        while len(book_ids) < target_count:
            url = cls.GUTENBERG_SEARCH_URL.format(start_index=start_index)
            before = len(book_ids)
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to discover Gutenberg IDs from %s: %s", url, exc)
                break

            for raw_id in re.findall(r'href="/ebooks/(\d+)"', html):
                book_id = int(raw_id)
                if book_id not in seen:
                    seen.add(book_id)
                    book_ids.append(book_id)
                    if len(book_ids) >= target_count:
                        break

            if len(book_ids) == before:
                break
            start_index += 25

        return book_ids[:target_count]

    @classmethod
    def _download_book(cls, book_id: int, path: str) -> bool:
        """Download one Gutenberg book to *path* using text URL fallbacks."""
        for url in cls._text_urls_for_book(book_id):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = resp.read()
                if not data.strip():
                    continue
                with open(path, "wb") as fh:
                    fh.write(data)
                return True
            except Exception:  # noqa: BLE001
                continue
        return False

    def download_gutenberg_books(
        self,
        output_dir: str = "./data/books",
        num_books: int = 50,
    ) -> List[str]:
        """
        Download public domain books from Project Gutenberg.

        Args:
            output_dir: Directory to save downloaded books.
            num_books: Maximum number of books to download.

        Returns:
            List of file paths to downloaded books.
        """
        if num_books <= 0:
            raise ValueError("num_books must be positive")

        os.makedirs(output_dir, exist_ok=True)
        downloaded: List[str] = []

        book_ids = self._discover_gutenberg_book_ids(num_books)
        for book_id in book_ids:
            path = os.path.join(output_dir, f"{book_id}.txt")

            if os.path.exists(path):
                downloaded.append(path)
                continue

            if self._download_book(book_id, path):
                downloaded.append(path)
            else:
                logger.warning("Failed to download Gutenberg book %s", book_id)

        logger.info(
            "Downloaded %d/%d requested books to %s",
            len(downloaded),
            num_books,
            output_dir,
        )
        return downloaded

    # ------------------------------------------------------------------ #
    # Preprocess
    # ------------------------------------------------------------------ #

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Clean and normalize text.

        Removes Gutenberg header/footer markers, collapses excessive
        whitespace, and strips non-ASCII characters (keeping common
        punctuation).

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned and normalized text.
        """
        # Remove Gutenberg header/footer
        text = re.sub(
            r"\*\*\* START OF.*?\*\*\*", "", text, flags=re.DOTALL
        )
        text = re.sub(
            r"\*\*\* END OF.*?\*\*\*", "", text, flags=re.DOTALL
        )
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        # Remove non-ASCII chars but keep common punctuation
        text = re.sub(
            r"[^\x00-\x7F\u2013\u2014\u2018\u2019\u201C\u201D]", "", text
        )
        return text.strip()

    # ------------------------------------------------------------------ #
    # Chunk
    # ------------------------------------------------------------------ #

    def chunk_text(self, text: str, overlap: int = 64) -> List[str]:
        """Split text into overlapping token-sized chunks.

        Args:
            text: Text to split into chunks.
            overlap: Number of overlapping tokens between consecutive chunks.

        Returns:
            List of text chunks.
        """
        if self.tokenizer is None:
            raise ValueError("A tokenizer is required before chunking text")

        tokens = self.tokenizer.encode(text)
        chunks: List[str] = []

        start = 0
        while start < len(tokens):
            end = start + self.max_seq_len
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            if len(chunk_text.strip()) > 50:  # Skip very short chunks
                chunks.append(chunk_text)
            start = end - overlap
            if start >= len(tokens) or end >= len(tokens):
                break

        return chunks

    # ------------------------------------------------------------------ #
    # Training dataset creation
    # ------------------------------------------------------------------ #

    def create_training_dataset(
        self,
        corpus_dir: str,
        output_path: str = "./data/training_dataset.pt",
        max_chunks: Optional[int] = None,
    ) -> SelfTrainingDataset:
        """
        Process all text files in *corpus_dir* into a training dataset.

        Walks the directory tree recursively, reads all ``.txt`` files,
        cleans and tokenizes the content, and produces fixed-length
        chunks suitable for language-model training.

        Args:
            corpus_dir: Root directory containing ``.txt`` files.
            output_path: Path where the serialized dataset is written.
            max_chunks: Maximum number of chunks to collect. ``None`` means all chunks.

        Returns:
            A ``SelfTrainingDataset`` instance built from the corpus.
        """
        if self.tokenizer is None:
            raise ValueError("A tokenizer is required before creating a dataset")
        if max_chunks is not None and max_chunks <= 0:
            max_chunks = None

        text_files = glob.glob(
            os.path.join(corpus_dir, "**/*.txt"), recursive=True
        )
        logger.info(
            "Found %d text files in %s", len(text_files), corpus_dir
        )

        all_samples: List[dict] = []
        for filepath in text_files:
            try:
                with open(
                    filepath, "r", encoding="utf-8", errors="ignore"
                ) as fh:
                    text = fh.read()

                text = self.preprocess_text(text)
                if len(text) < 100:
                    continue

                chunks = self.chunk_text(text)
                for chunk in chunks:
                    all_samples.append(
                        {
                            "prompt": "",
                            "response": chunk,
                            "token_ids": self.tokenizer.encode(chunk),
                        }
                    )

                if max_chunks is not None and len(all_samples) >= max_chunks:
                    all_samples = all_samples[:max_chunks]
                    break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error processing %s: %s", filepath, exc)

        logger.info("Created %d training samples", len(all_samples))

        dataset = SelfTrainingDataset(
            samples=all_samples,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        dataset.save(output_path)
        logger.info("Saved dataset to %s", output_path)

        return dataset

    # ------------------------------------------------------------------ #
    # Tokenizer training
    # ------------------------------------------------------------------ #

    def train_tokenizer_on_corpus(
        self,
        corpus_dir: str,
        vocab_size: int = 32000,
        sample_size: Optional[int] = None,
    ) -> BPETokenizer:
        """
        Train a new BPE tokenizer on the corpus.

        Args:
            corpus_dir: Root directory containing ``.txt`` files.
            vocab_size: Target vocabulary size.
            sample_size: Approximate total character budget to read. ``None`` means all text.

        Returns:
            A freshly trained ``BPETokenizer``.
        """
        if sample_size is not None and sample_size <= 0:
            sample_size = None

        text_files = glob.glob(
            os.path.join(corpus_dir, "**/*.txt"), recursive=True
        )

        # Sample text for training
        sample_texts: List[str] = []
        total_chars = 0
        for filepath in text_files:
            with open(
                filepath, "r", encoding="utf-8", errors="ignore"
            ) as fh:
                text = fh.read()
            text = self.preprocess_text(text)
            sample_texts.append(text)
            total_chars += len(text)
            if sample_size is not None and total_chars >= sample_size:
                break

        logger.info(
            "Training tokenizer on %d texts (%d chars)",
            len(sample_texts),
            total_chars,
        )
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.train(sample_texts)
        logger.info("Tokenizer trained: vocab_size=%d", tokenizer.vocab_size)

        return tokenizer
