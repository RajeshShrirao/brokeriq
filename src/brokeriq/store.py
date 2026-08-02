"""Long-term memory store factory (LangGraph BaseStore).

Dev/test: in-memory store.
Prod: Postgres-backed store so memory survives restarts. Uses a sync
context manager (PostgresStore.from_conn_string is sync in LangGraph v1).
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from .config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def memory_store_scope() -> AsyncIterator:
    """Yield a BaseStore for the lifetime of a run.

    Usage mirrors the checkpointer pattern:
        async with memory_store_scope() as store:
            app = build_graph(checkpointer=checkpointer, store=store)
    """
    settings = get_settings()
    if settings.env == "prod":
        from langgraph.store.postgres import PostgresStore

        try:
            with PostgresStore.from_conn_string(settings.postgres_dsn) as store:
                yield store
                return
        except Exception as exc:
            logger.warning("postgres store unavailable (%s); falling back to in-memory", exc)

    from langgraph.store.memory import InMemoryStore

    yield InMemoryStore()
