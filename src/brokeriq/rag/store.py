"""Qdrant-backed vector store with hybrid (dense + BM42 sparse) retrieval.

The store auto-detects its backend: an in-memory Qdrant instance is used for
dev/tests, the real server (docker compose) for production.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client import models

from ..config import get_settings
from .ingest import Chunk

logger = logging.getLogger(__name__)

COLLECTION = "compliance"
_DENSE_MODEL_ALIAS = "text-embedding"  # points at the configured fastembed model
_SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"


def build_client(url: str | None = None, prefer_memory: bool = False) -> QdrantClient:
    """Return a Qdrant client; falls back to in-memory when no server is up."""
    settings = get_settings()
    url = url or settings.qdrant_url

    if prefer_memory:
        return QdrantClient(":memory:")

    try:
        client = QdrantClient(url=url, timeout=2)
        client.get_collections()  # raises if unreachable
        return client
    except Exception as exc:
        logger.warning("qdrant server at %s unreachable (%s); using in-memory store", url, exc)
        return QdrantClient(":memory:")


def ensure_collection(client: QdrantClient) -> None:
    """Create the hybrid collection with both dense and sparse vectors."""
    settings = get_settings()
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            _DENSE_MODEL_ALIAS: models.VectorParams(
                size=384 if "mini" in settings.embedding_model.lower() else 1024,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={_SPARSE_MODEL: models.SparseVectorParams()},
    )
    logger.info("created collection %s (dense=%s, sparse=%s)", COLLECTION, _DENSE_MODEL_ALIAS, _SPARSE_MODEL)


def upsert_chunks(client: QdrantClient, chunks: list[Chunk], embeddings: list[list[float]], sparse: list[models.SparseVector]) -> None:
    """Index chunks with dense + sparse vectors in one batch."""
    ensure_collection(client)
    points = [
        models.PointStruct(
            id=idx,
            vector={_DENSE_MODEL_ALIAS: embeddings[idx], _SPARSE_MODEL: sparse[idx]},
            payload={
                "doc_id": chunk.doc_id,
                "section": chunk.section,
                "text": chunk.text,
                "citation": chunk.citation,
            },
        )
        for idx, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    logger.info("indexed %d chunks", len(points))


def hybrid_search(
    client: QdrantClient,
    dense_vector: list[float],
    sparse_vector: models.SparseVector,
    query: str,
    limit: int = 6,
) -> list[dict]:
    """Hybrid retrieval: dense + BM42 sparse fused with reciprocal rank fusion."""
    prefetch = [
        models.Prefetch(query=dense_vector, using=_DENSE_MODEL_ALIAS, limit=limit * 2),
        models.Prefetch(query=sparse_vector, using=_SPARSE_MODEL, limit=limit * 2),
    ]
    response = client.query_points(
        collection_name=COLLECTION,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    hits = []
    for point in response.points:
        payload = point.payload or {}
        hits.append(
            {
                "text": payload.get("text", ""),
                "citation": payload.get("citation", ""),
                "doc_id": payload.get("doc_id", ""),
                "section": payload.get("section", ""),
                "score": point.score,
            }
        )
    return hits


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return build_client()
