"""Memory persistence tests: extracted MemoryOps land in the graph's store."""

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore

from brokeriq import llm as llm_module
from brokeriq.fake import FakeLLM
from brokeriq.graph import build_graph
from brokeriq.models import LeadInput


@pytest.fixture(autouse=True)
def fast_embeddings(monkeypatch):
    monkeypatch.setenv("BROKERIQ_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    from brokeriq.config import get_settings
    from brokeriq.rag import store

    get_settings.cache_clear()
    store.get_client.cache_clear()
    store._memory_client = None
    yield
    get_settings.cache_clear()
    store.get_client.cache_clear()
    store._memory_client = None


@pytest.mark.asyncio
async def test_memory_ops_persisted_to_store(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm_module, "complete", fake.complete)
    monkeypatch.setattr(llm_module, "complete_json", fake.complete_json)

    store = InMemoryStore()
    lead = LeadInput(company_name="Acme Widgets", industry="manufacturing")

    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = build_graph(checkpointer=checkpointer, store=store)
        result = await app.ainvoke(
            {"lead": lead, "run_id": "mem-1"},
            config={"configurable": {"thread_id": "t-mem"}},
        )

    assert result["memory_ops"], "expected extracted memory ops"

    item = await store.aget(("leads", "acme-widgets"), "profile")
    assert item is not None, "expected the lead profile to be persisted"
    assert item.value["naics"] == "5415"
    assert item.value["verdict"] == "qualified"


@pytest.mark.asyncio
async def test_memory_without_store_is_safe(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm_module, "complete", fake.complete)
    monkeypatch.setattr(llm_module, "complete_json", fake.complete_json)

    lead = LeadInput(company_name="Acme Widgets", industry="manufacturing")

    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = build_graph(checkpointer=checkpointer)  # no store attached
        result = await app.ainvoke(
            {"lead": lead, "run_id": "mem-2"},
            config={"configurable": {"thread_id": "t-nostore"}},
        )

    assert result["memory_ops"], "ops are still reported in state even without a store"
