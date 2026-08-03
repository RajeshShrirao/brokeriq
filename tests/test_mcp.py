"""MCP server tests: spawn the brokeriq MCP server over stdio and exercise the
compliance_search tool end-to-end through the real MCP protocol.

The server subprocess inherits this test's env (MiniLM embedder, in-memory
Qdrant fallback, sample corpus), so no docker or API keys are needed.
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

REPO_ROOT = Path(__file__).resolve().parent.parent


def _server_params() -> StdioServerParameters:
    """Launch `uv run brokeriq-mcp` from the repo root with test env."""
    env = dict(os.environ)
    env.update(
        {
            "BROKERIQ_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
            "BROKERIQ_CORPUS_DIR": str(REPO_ROOT / "data" / "corpus"),
            "BROKERIQ_QDRANT_URL": "http://127.0.0.1:1",  # unreachable -> in-memory
            "BROKERIQ_REDIS_URL": "redis://127.0.0.1:1",  # unreachable -> degraded cache
            "PYTHONPATH": str(REPO_ROOT / "src"),
        }
    )
    return StdioServerParameters(
        command="uv",
        args=["run", "brokeriq-mcp"],
        env=env,
        cwd=str(REPO_ROOT),
    )


@asynccontextmanager
async def _session():
    """Open a client session to a fresh server subprocess.

    Kept as a plain async context manager (not a pytest fixture) because the
    anyio task group inside stdio_client must be entered and exited in the
    same task — an async-generator fixture crosses task boundaries and blows
    up with "cancel scope in a different task".
    """
    async with stdio_client(_server_params()) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


def _text(result) -> str:
    return "".join(
        block.text for block in result.content if getattr(block, "type", None) == "text"
    )


async def test_server_initializes_and_lists_tools():
    async with _session() as session:
        result = await session.list_tools()
        names = [t.name for t in result.tools]
        assert "compliance_search" in names
        tool = next(t for t in result.tools if t.name == "compliance_search")
        assert "corpus" in (tool.description or "").lower()


async def test_compliance_search_returns_citations():
    async with _session() as session:
        result = await session.call_tool(
            "compliance_search",
            {"query": "do carriers require MFA on email?", "limit": 3},
        )
        text = _text(result)
        assert text, f"no text payload in {result.content!r}"

        payload = json.loads(text)
        assert payload["query"] == "do carriers require MFA on email?"
        assert payload["count"] >= 1
        assert payload["results"][0]["citation"]
        assert payload["results"][0]["text"]


async def test_compliance_search_caps_limit():
    async with _session() as session:
        result = await session.call_tool(
            "compliance_search",
            {"query": "workers compensation", "limit": 99},
        )
        payload = json.loads(_text(result))
        assert payload["count"] <= 10
