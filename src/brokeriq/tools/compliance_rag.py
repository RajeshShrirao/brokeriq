"""Compliance retrieval tool: hybrid search over the carrier corpus."""

import logging
from pathlib import Path

from ..config import get_settings
from ..rag import build_client, hybrid_search, load_corpus, upsert_chunks
from ..rag.embeddings import embed_sparse, embed_texts

logger = logging.getLogger(__name__)


def ensure_indexed() -> None:
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
    dense = embed_texts(texts)
    sparse = embed_sparse(texts)
    upsert_chunks(client, chunks, dense, sparse)


async def compliance_search(query: str, limit: int = 5) -> list[dict]:
    """Return citation-ready compliance/coverage facts for a query."""
    ensure_indexed()
    client = build_client()

    dense = embed_texts([query])[0]
    sparse = embed_sparse([query])[0]
    hits = hybrid_search(client, dense, sparse, query, limit=limit)

    logger.info("compliance search %r -> %d hits", query, len(hits))
    return hits
