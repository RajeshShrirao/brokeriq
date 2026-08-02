"""Pluggable reranking on top of hybrid retrieval.

Default: pass-through — hybrid_search already fuses dense + sparse with
reciprocal rank fusion, so the order is good enough for most queries.

Optional: cross-encoder reranking via sentence-transformers, enabled with
BROKERIQ_RERANKER_ENABLED=true (requires `pip install sentence-transformers`,
a ~2GB extra we deliberately keep out of the default install).
"""

import logging

from ..config import get_settings

logger = logging.getLogger(__name__)


def rerank(hits: list[dict], query: str, top_k: int = 5) -> list[dict]:
    """Return the best `top_k` hits for a query, reranked if configured."""
    settings = get_settings()
    if settings.reranker_enabled:
        try:
            from .cross_encoder import cross_encoder_rerank

            return cross_encoder_rerank(hits, query, top_k=top_k)
        except ImportError:
            logger.warning(
                "reranker enabled but sentence-transformers is not installed; "
                "falling back to RRF order"
            )
    return hits[:top_k]
