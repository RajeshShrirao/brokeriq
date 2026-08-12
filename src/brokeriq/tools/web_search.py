"""Web search with a zero-key default (DuckDuckGo) and optional Tavily."""

import asyncio
import logging

from ddgs import DDGS

from ..config import get_settings

logger = logging.getLogger(__name__)


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Return [{"title", "url", "snippet"}] for a query.

    Defaults to DuckDuckGo (no API key). If TAVILY_API_KEY is configured the
    Tavily endpoint is used instead, which returns slightly richer results.
    """
    settings = get_settings()
    if settings.web_search_provider == "tavily" and settings.tavily_api_key:
        return await _tavily_search(query, max_results)
    return await _ddgs_search(query, max_results)


async def _ddgs_search(query: str, max_results: int) -> list[dict]:
    # ddgs is synchronous under the hood; run it in a thread so we don't block
    # the event loop during a graph run.

    def _run() -> list[dict]:
        try:
            with DDGS() as client:
                return list(client.text(query, max_results=max_results))
        except Exception as exc:  # network hiccups, bot checks, etc.
            logger.warning("ddgs search failed for %r: %s", query, exc)
            return []

    results = await asyncio.to_thread(_run)
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in results
    ]


async def _tavily_search(query: str, max_results: int) -> list[dict]:
    import httpx

    settings = get_settings()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": settings.tavily_api_key, "query": query, "max_results": max_results},
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])
    ]
