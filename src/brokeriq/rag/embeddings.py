"""Embedding + sparse-vector generation via fastembed (local, free, ONNX)."""

from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client import models

from ..config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _text_embedder():
    from fastembed import TextEmbedding

    settings = get_settings()
    logger.info("loading embedding model %s", settings.embedding_model)
    return TextEmbedding(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def _sparse_embedder():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Dense embeddings (batched)."""
    embedder = _text_embedder()
    return [list(vec) for vec in embedder.embed(texts)]


def embed_sparse(texts: list[str]) -> list[models.SparseVector]:
    """BM42 sparse vectors for hybrid retrieval."""
    embedder = _sparse_embedder()
    out = []
    for vec in embedder.embed(texts):
        indices, values = vec.indices, vec.values
        out.append(models.SparseVector(indices=list(indices), values=[float(v) for v in values]))
    return out
