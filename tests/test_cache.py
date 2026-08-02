"""Semantic cache tests using fakeredis (no Redis server needed)."""

import pytest

from brokeriq.cache import SemanticCache, reset_cache

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def fresh_cache():
    reset_cache()
    yield
    reset_cache()


async def test_exact_hit_roundtrip():
    redis = __import__("fakeredis").aioredis.FakeRedis(decode_responses=True)
    cache = SemanticCache(redis=redis, ttl=60)

    payload = [{"text": "Workers comp is compulsory in TX", "score": 0.9}]
    await cache.put("workers comp texas", [0.1, 0.2, 0.3], payload)

    got = await cache.get("workers comp texas", [0.1, 0.2, 0.3])
    assert got == payload


async def test_semantic_hit_on_near_duplicate_query():
    redis = __import__("fakeredis").aioredis.FakeRedis(decode_responses=True)
    cache = SemanticCache(redis=redis, ttl=60, threshold=0.9)

    payload = [{"text": "cyber liability basics", "score": 0.8}]
    stored_embedding = [1.0, 0.0, 0.0]
    await cache.put("cyber liability", stored_embedding, payload)

    # nearly identical query embedding -> semantic hit
    similar = [0.999, 0.02, 0.01]
    got = await cache.get("cyber liability insurance", similar)
    assert got == payload


async def test_miss_below_threshold():
    redis = __import__("fakeredis").aioredis.FakeRedis(decode_responses=True)
    cache = SemanticCache(redis=redis, ttl=60, threshold=0.95)

    await cache.put("workers comp", [1.0, 0.0], [{"text": "x"}])

    unrelated = [0.0, 1.0]
    assert await cache.get("general liability", unrelated) is None


async def test_degraded_mode_without_redis_is_a_noop():
    cache = SemanticCache(redis=None, ttl=60)

    await cache.put("anything", [0.5], [{"text": "y"}])  # must not raise
    assert await cache.get("anything", [0.5]) is None
