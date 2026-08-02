"""Cross-encoder reranking via sentence-transformers (optional extra).

Kept out of the default install: pulls in torch + transformers (~2GB).
Enable with BROKERIQ_RERANKER_ENABLED=true after `pip install sentence-transformers`.
"""

import logging

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        from ..config import get_settings

        _model = CrossEncoder(get_settings().reranker_model)
    return _model


def cross_encoder_rerank(hits: list[dict], query: str, top_k: int = 5) -> list[dict]:
    """Score (query, passage) pairs with a cross-encoder and reorder."""
    model = _get_model()
    pairs = [(query, hit["text"]) for hit in hits]
    scores = model.predict(pairs)
    ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
    return [hit for hit, _score in ranked[:top_k]]
