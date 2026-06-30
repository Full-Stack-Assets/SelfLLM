"""Local retrieval-context packing for managed LLM prompts.

This module provides a lightweight RAG first step before Vertex AI tuning:
load local documents, retrieve relevant chunks with lexical scoring, and pack a
grounded prompt that can be sent to Gemini or saved as a supervised example.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_EXTENSIONS = {".txt", ".md"}
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(frozen=True)
class TextChunk:
    """A source-attributed document chunk."""

    source: str
    index: int
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk with a relevance score."""

    chunk: TextChunk
    score: float


def _tokens(text: str) -> List[str]:
    """Tokenize text for lexical retrieval."""
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def iter_document_paths(
    paths: Sequence[str],
    *,
    recursive: bool = True,
    extensions: Optional[Iterable[str]] = None,
) -> List[Path]:
    """Return sorted document paths from files/directories."""
    allowed = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (extensions or DEFAULT_EXTENSIONS)
    }
    out: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Document path not found: {raw_path}")
        if path.is_file():
            if path.suffix.lower() in allowed:
                out.append(path)
            continue
        pattern = "**/*" if recursive else "*"
        for child in path.glob(pattern):
            if child.is_file() and child.suffix.lower() in allowed:
                out.append(child)
    return sorted(set(out))


def chunk_text(
    text: str,
    *,
    source: str,
    chunk_words: int = 220,
    overlap_words: int = 40,
) -> List[TextChunk]:
    """Split text into overlapping word chunks."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words must be non-negative")
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    words = text.split()
    if not words:
        return []
    chunks: List[TextChunk] = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(TextChunk(source=source, index=len(chunks), text=chunk))
        if end >= len(words):
            break
    return chunks


def load_chunks(
    paths: Sequence[str],
    *,
    recursive: bool = True,
    extensions: Optional[Iterable[str]] = None,
    chunk_words: int = 220,
    overlap_words: int = 40,
) -> List[TextChunk]:
    """Load supported documents and split them into chunks."""
    chunks: List[TextChunk] = []
    for path in iter_document_paths(paths, recursive=recursive, extensions=extensions):
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(
            chunk_text(
                text,
                source=str(path),
                chunk_words=chunk_words,
                overlap_words=overlap_words,
            )
        )
    return chunks


def retrieve_chunks(
    chunks: Sequence[TextChunk],
    query: str,
    *,
    top_k: int = 5,
) -> List[ScoredChunk]:
    """Retrieve the top lexical matches for *query*."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query_terms = set(_tokens(query))
    if not query_terms:
        return []

    chunk_terms = [_tokens(chunk.text) for chunk in chunks]
    doc_freq: Dict[str, int] = {}
    for terms in chunk_terms:
        for term in set(terms):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    n_chunks = max(1, len(chunks))
    scored: List[ScoredChunk] = []
    for chunk, terms in zip(chunks, chunk_terms):
        if not terms:
            continue
        tf: Dict[str, int] = {}
        for term in terms:
            tf[term] = tf.get(term, 0) + 1

        score = 0.0
        for term in query_terms:
            count = tf.get(term, 0)
            if not count:
                continue
            idf = math.log((1 + n_chunks) / (1 + doc_freq.get(term, 0))) + 1.0
            score += (count / len(terms)) * idf
        if score > 0:
            scored.append(ScoredChunk(chunk=chunk, score=score))

    scored.sort(key=lambda item: (-item.score, item.chunk.source, item.chunk.index))
    return scored[: min(top_k, len(scored))]


def pack_grounded_prompt(
    query: str,
    retrieved: Sequence[ScoredChunk],
    *,
    max_context_chars: int = 6000,
    system_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """Pack retrieved chunks into a grounded prompt and manifest."""
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be positive")

    context_blocks: List[str] = []
    sources: List[Dict[str, Any]] = []
    used_chars = 0
    for rank, item in enumerate(retrieved, start=1):
        chunk = item.chunk
        header = f"[Source {rank}: {chunk.source}#chunk-{chunk.index}]"
        block = f"{header}\n{chunk.text}"
        remaining = max_context_chars - used_chars
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip()
        context_blocks.append(block)
        used_chars += len(block)
        sources.append(
            {
                "rank": rank,
                "source": chunk.source,
                "chunk_index": chunk.index,
                "score": item.score,
                "chars": len(block),
            }
        )

    instruction = system_instruction or (
        "Answer the question using only the provided sources. "
        "Cite sources as [Source N]. If the sources are insufficient, say so."
    )
    context = "\n\n".join(context_blocks) if context_blocks else "(no relevant sources found)"
    prompt = (
        f"{instruction}\n\n"
        f"Question:\n{query.strip()}\n\n"
        f"Sources:\n{context}\n\n"
        "Answer:"
    )
    return {
        "query": query,
        "prompt": prompt,
        "sources": sources,
        "context_chars": used_chars,
    }


def build_rag_prompt(
    paths: Sequence[str],
    query: str,
    *,
    top_k: int = 5,
    max_context_chars: int = 6000,
    chunk_words: int = 220,
    overlap_words: int = 40,
    recursive: bool = True,
    extensions: Optional[Iterable[str]] = None,
    system_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """Load docs, retrieve context, and return a grounded prompt manifest."""
    chunks = load_chunks(
        paths,
        recursive=recursive,
        extensions=extensions,
        chunk_words=chunk_words,
        overlap_words=overlap_words,
    )
    retrieved = retrieve_chunks(chunks, query, top_k=top_k)
    packed = pack_grounded_prompt(
        query,
        retrieved,
        max_context_chars=max_context_chars,
        system_instruction=system_instruction,
    )
    packed["chunks_loaded"] = len(chunks)
    return packed
