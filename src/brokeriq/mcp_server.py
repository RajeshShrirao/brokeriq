"""BrokerIQ MCP server (MCP 2.0).

Exposes the compliance RAG pipeline as a Model Context Protocol tool so any
MCP client (Claude, Hermes, IDEs, agent frameworks) can query the carrier
corpus with citations.

Run:
    uv run brokeriq-mcp          # stdio transport (default for MCP clients)
    uv run python -m brokeriq.mcp_server

Register in a client config, e.g. Hermes `~/.hermes/config.yaml`:

    mcp_servers:
      brokeriq:
        command: "uv"
        args: ["run", "--project", "/abs/path/brokeriq", "brokeriq-mcp"]
"""

from __future__ import annotations

import asyncio
import logging
import sys

from mcp.server.mcpserver import MCPServer

from .config import get_settings
from .logging import setup_logging
from .tools.compliance_rag import compliance_search

logger = logging.getLogger(__name__)


def create_server() -> MCPServer:
    """Build the MCP server with the compliance retrieval tool registered."""
    settings = get_settings()
    server = MCPServer(
        name="brokeriq-compliance",
        version="0.1.0",
        title="BrokerIQ Compliance RAG",
        description=(
            "Hybrid retrieval (dense + BM42 sparse + rerank, semantic-cached) "
            "over the carrier compliance corpus. Returns citation-ready facts "
            "for insurance lead qualification and market research."
        ),
        instructions=(
            "Use compliance_search for coverage, exclusions, and carrier "
            "requirement questions. Always cite returned citations in answers. "
            "If the tool returns no hits, say the corpus has no coverage rather "
            "than guessing."
        ),
    )

    async def compliance_search_tool(query: str, limit: int = 5) -> dict:
        """Search the compliance corpus.

        Args:
            query: Natural-language question or keyword (e.g. "MFA requirements",
                "Texas workers comp opt out").
            limit: Max results to return (1-10).

        Returns:
            dict: {"query": ..., "count": N, "results": [{text, citation,
            doc_id, section, score}]}
        """
        capped = max(1, min(int(limit), 10))
        hits = await compliance_search(query, limit=capped)
        return {"query": query, "count": len(hits), "results": hits}

    server.add_tool(
        compliance_search_tool,
        name="compliance_search",
        title="Compliance corpus search",
        description=(
            "Hybrid semantic + keyword search over the BrokerIQ carrier "
            "compliance corpus. Returns citation-ready facts with doc/section "
            "references. Results are served from a two-tier semantic cache when "
            "possible."
        ),
    )

    logger.info("brokeriq MCP server built (model=%s)", settings.embedding_model)
    return server


async def _run_stdio() -> None:
    server = create_server()
    await server.run_stdio_async()


def main() -> None:
    # stdio transport is reserved for the JSON-RPC protocol — logs must go to
    # stderr or they corrupt the message stream.
    setup_logging(stream=sys.stderr)
    asyncio.run(_run_stdio())


if __name__ == "__main__":
    main()
