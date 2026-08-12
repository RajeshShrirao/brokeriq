"""Compliance retrieval tool: hybrid search over the carrier corpus."""

import logging
from pathlib import Path

from ..cache import get_cache
from ..config import get_settings
from ..rag import build_client, hybrid_search, load_corpus, upsert_chunks
from ..rag.embeddings import embed_sparse, embed_texts
from ..rag.rerank import rerank

logger = logging.getLogger(__name__)


async def ensure_indexed() -> None:
    """Idempotent one-shot index of the sample corpus into the local store."""
    settings = get_settings()
    client = build_client()
    if client.collection_exists("compliance"):
        return

    corpus_dir = Path(settings.corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"corpus dir not found: {corpus_dir} (set BROKERIQ_CORPUS_DIR)")

    chunks = load_corpus(corpus_dir)
    texts = [c.text for c in chunks]
    dense = await embed_texts(texts)
    sparse = await embed_sparse(texts)
    upsert_chunks(client, chunks, dense, sparse)


async def compliance_search(query: str, limit: int = 5) -> list[dict]:
    """Return citation-ready compliance/coverage facts for a query.

    Results are served from the two-tier semantic cache when possible, so
    repeat and near-duplicate queries skip embedding + vector search entirely.
    """
    await ensure_indexed()
    client = build_client()

    dense = (await embed_texts([query]))[0]

    cache = get_cache()
    cached = await cache.get(query, dense)
    if cached is not None:
        logger.info("compliance search %r served from cache", query)
        return cached[:limit]

    sparse = (await embed_sparse([query]))[0]
    hits = hybrid_search(client, dense, sparse, query, limit=limit * 3)
    hits = rerank(hits, query, top_k=limit)

    await cache.put(query, dense, hits)
    logger.info("compliance search %r -> %d hits", query, len(hits))
    return hits
