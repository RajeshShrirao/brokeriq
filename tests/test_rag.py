"""RAG tests run against an in-memory Qdrant with the MiniLM embedding model,
so the suite stays fast and needs no API keys or docker."""

import pytest

from brokeriq.rag import build_client, hybrid_search, load_corpus, split_markdown, upsert_chunks
from brokeriq.rag.embeddings import embed_sparse, embed_texts
from brokeriq.rag.ingest import Chunk


@pytest.fixture(autouse=True)
def fast_embeddings(monkeypatch):
    """Use the small MiniLM model for tests; clear the lru caches so it applies."""
    monkeypatch.setenv("BROKERIQ_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    from brokeriq.config import get_settings
    from brokeriq.rag import store

    get_settings.cache_clear()
    store.get_client.cache_clear()
    store._memory_client = None
    yield
    get_settings.cache_clear()
    store.get_client.cache_clear()
    store._memory_client = None


def _sample_chunks() -> list[Chunk]:
    return [
        Chunk(doc_id="cyber", section="Security Requirements", text="Most carriers require MFA on email and remote access."),
        Chunk(doc_id="workers-comp", section="Texas", text="Texas allows employers to opt out of workers' compensation."),
        Chunk(doc_id="cgl", section="Pollution Exclusion", text="The standard pollution exclusion excludes most discharges."),
    ]


def test_split_markdown_respects_sections():
    doc = "# Title\n\n## Part One\n\nHello world.\n\n## Part Two\n\nSecond part.\n"
    chunks = split_markdown(doc, "test")
    # headings are preserved as citations; empty sections are dropped
    assert [c.section for c in chunks] == ["Part One", "Part Two"]


def test_load_corpus_finds_markdown(tmp_path):
    (tmp_path / "a.md").write_text("# Doc A\n\nSome text here.\n")
    (tmp_path / "b.md").write_text("# Doc B\n\nMore text.\n")
    chunks = load_corpus(tmp_path)
    assert {c.doc_id for c in chunks} == {"a", "b"}


def test_hybrid_search_roundtrip():
    client = build_client(prefer_memory=True)
    chunks = _sample_chunks()
    texts = [c.text for c in chunks]
    upsert_chunks(client, chunks, embed_texts(texts), embed_sparse(texts))

    hits = hybrid_search(client, embed_texts(["do carriers require mfa"])[0], embed_sparse(["do carriers require mfa"])[0], "mfa requirement", limit=2)
    assert len(hits) >= 1
    assert hits[0]["citation"].startswith("cyber §")


def test_upsert_is_idempotent():
    client = build_client(prefer_memory=True)
    chunks = _sample_chunks()
    upsert_chunks(client, chunks, embed_texts([c.text for c in chunks]), embed_sparse([c.text for c in chunks]))
    count = client.count("compliance").count
    assert count == len(chunks)
