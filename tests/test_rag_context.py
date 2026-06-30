"""Tests for local RAG context packing helpers."""

from __future__ import annotations

from selfllm.cloud.rag_context import (
    build_rag_prompt,
    chunk_text,
    iter_document_paths,
    load_chunks,
    pack_grounded_prompt,
    retrieve_chunks,
)


def test_iter_document_paths_filters_supported_extensions(tmp_path):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "c.json").write_text("{}", encoding="utf-8")

    paths = iter_document_paths([str(tmp_path)])

    assert [path.name for path in paths] == ["a.md", "b.txt"]


def test_chunk_text_uses_overlap():
    chunks = chunk_text(
        "one two three four five six",
        source="doc.txt",
        chunk_words=4,
        overlap_words=2,
    )

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "three four five six",
    ]
    assert chunks[1].index == 1


def test_retrieve_chunks_ranks_query_terms(tmp_path):
    relevant = tmp_path / "relevant.md"
    irrelevant = tmp_path / "irrelevant.md"
    relevant.write_text("Vertex tuning uses adapter size and Gemini.", encoding="utf-8")
    irrelevant.write_text("Gardening requires soil and water.", encoding="utf-8")
    chunks = load_chunks([str(tmp_path)], chunk_words=20, overlap_words=0)

    results = retrieve_chunks(chunks, "Gemini adapter tuning", top_k=2)

    assert results
    assert results[0].chunk.source.endswith("relevant.md")
    assert results[0].score > 0


def test_pack_grounded_prompt_includes_sources():
    chunks = chunk_text(
        "Gemini adapters are useful for supervised tuning.",
        source="guide.md",
        chunk_words=20,
        overlap_words=0,
    )
    retrieved = retrieve_chunks(chunks, "Which adapters are useful?", top_k=1)

    packed = pack_grounded_prompt("Which adapters are useful?", retrieved)

    assert "Question:" in packed["prompt"]
    assert "[Source 1:" in packed["prompt"]
    assert packed["sources"][0]["rank"] == 1


def test_build_rag_prompt_end_to_end(tmp_path):
    (tmp_path / "vertex.md").write_text(
        "Vertex AI can tune Gemini with supervised adapter jobs.",
        encoding="utf-8",
    )

    result = build_rag_prompt(
        [str(tmp_path)],
        "How can Gemini be tuned?",
        top_k=1,
        chunk_words=20,
        overlap_words=0,
    )

    assert result["chunks_loaded"] == 1
    assert result["sources"][0]["source"].endswith("vertex.md")
    assert "Vertex AI" in result["prompt"]
