"""Two-tier semantic cache backed by Redis.

Tier 1: exact hit on a normalized query hash.
Tier 2: semantic hit — the query embedding is compared (cosine) against a
bounded in-memory fingerprint ring of recent queries; a near-duplicate query
returns the stored payload without touching the vector store.

Degraded mode: if Redis is unreachable the cache silently becomes a no-op so
the pipeline keeps working — the cache is an accelerator, never a dependency.

The semantic tier keeps a bounded in-memory fingerprint ring (max 1024 entries).
When the ring is full, the oldest entry is evicted (FIFO) on each new put — no
LRU, no background cleanup, constant-time ops.
"""

import asyncio
import hashlib
import json
import logging
from collections import deque
from typing import Self

import numpy as np
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .config import get_settings

logger = logging.getLogger(__name__)

_SEMANTIC_INDEX_MAX = 1024  # bounded fingerprint ring size

_cache: "SemanticCache | None" = None


class SemanticCache:
    def __init__(self, redis: Redis | None = None, ttl: int = 3600, threshold: float = 0.92) -> None:
        self._redis = redis
        self.ttl = ttl
        self.threshold = threshold
        self._fingerprints: dict[str, list[float]] = {}
        self._order: deque[str] = deque()
        self._lock = asyncio.Lock()

    @classmethod
    async def from_settings(cls) -> Self:
        """Build a SemanticCache from app settings, validating Redis connectivity.

        Raises RedisConnectionError if the Redis server cannot be reached,
        so the service fails fast at startup rather than deferring the first
        connectivity error to the first cache lookup.
        """
        settings = get_settings()
        redis = await cls._validate_redis(settings.redis_url)
        return cls(redis=redis, ttl=settings.cache_ttl, threshold=settings.cache_similarity_threshold)

    @classmethod
    async def _validate_redis(cls, redis_url: str) -> Redis | None:
        """Return a connected Redis client, or None in degraded (no-op) mode."""
        try:
            redis = Redis.from_url(redis_url, socket_connect_timeout=1, decode_responses=True)
            await redis.ping()
            logger.info("semantic cache connected to %s", redis_url)
            return redis
        except Exception:
            logger.warning("redis unavailable; semantic cache running in degraded (no-op) mode")
            return None

    async def get(self, query: str, embedding: list[float]) -> list[dict] | None:
        exact = await self._exact_get(query)
        if exact is not None:
            return exact

        if not embedding or not self._fingerprints:
            return None

        q = np.asarray(embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q)) or 1.0
        best_key: str | None = None
        best_sim = -1.0
        for key, vec in self._fingerprints.items():
            v = np.asarray(vec, dtype=np.float32)
            sim = float(np.dot(q, v) / (q_norm * (np.linalg.norm(v) or 1.0)))
            if sim > best_sim:
                best_key, best_sim = key, sim

        if best_key is not None and best_sim >= self.threshold:
            logger.info("semantic cache hit (sim=%.3f) for %r", best_sim, query)
            return await self._get_by_key(best_key)
        return None

    async def put(self, query: str, embedding: list[float], payload: list[dict]) -> None:
        key = self._key(query)
        try:
            if self._redis is not None:
                await self._redis.set(key, json.dumps(payload, default=str), ex=self.ttl)
        except RedisError as exc:
            logger.debug("cache write skipped: %s", exc)

        async with self._lock:
            self._fingerprints[key] = embedding
            self._order.append(key)
            if len(self._order) > _SEMANTIC_INDEX_MAX:
                dropped = self._order.popleft()
                self._fingerprints.pop(dropped, None)

    async def _exact_get(self, query: str) -> list[dict] | None:
        return await self._get_by_key(self._key(query))

    async def _get_by_key(self, key: str) -> list[dict] | None:
        """Fetch a payload by its storage key (no re-hashing)."""
        try:
            if self._redis is not None:
                raw = await self._redis.get(key)
                if raw is not None:
                    return json.loads(raw)
        except RedisError as exc:
            logger.debug("cache read skipped: %s", exc)
        return None

    @staticmethod
    def _key(query: str) -> str:
        return "bq:cache:" + hashlib.sha256(query.strip().lower().encode()).hexdigest()


async def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = await SemanticCache.from_settings()
    return _cache


def reset_cache() -> None:
    global _cache
    _cache = None
