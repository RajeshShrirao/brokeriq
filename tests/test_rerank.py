"""Reranking tests: default RRF pass-through, optional cross-encoder fallback."""

from brokeriq.rag.rerank import rerank


def test_pass_through_keeps_order_and_truncates():
    hits = [{"text": f"hit {i}", "score": 1.0 / (i + 1)} for i in range(6)]
    out = rerank(hits, "query", top_k=3)
    assert [h["text"] for h in out] == ["hit 0", "hit 1", "hit 2"]


def test_empty_hits_is_safe():
    assert rerank([], "query", top_k=5) == []


def test_fallback_when_cross_encoder_missing(monkeypatch):
    monkeypatch.setenv("BROKERIQ_RERANKER_ENABLED", "true")

    from brokeriq.config import get_settings

    get_settings.cache_clear()
    try:
        hits = [{"text": "a", "score": 0.5}, {"text": "b", "score": 0.4}]
        # sentence-transformers is not installed in CI, so rerank falls back
        out = rerank(hits, "query", top_k=2)
        assert out == hits
    finally:
        get_settings.cache_clear()
