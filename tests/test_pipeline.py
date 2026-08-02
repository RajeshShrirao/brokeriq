"""End-to-end pipeline tests driven by the deterministic FakeLLM (no API keys)."""

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from brokeriq import llm as llm_module
from brokeriq.fake import FakeLLM
from brokeriq.graph import build_graph
from brokeriq.models import LeadInput


@pytest.fixture(autouse=True)
def fast_embeddings(monkeypatch):
    """Use the small MiniLM model in tests so the suite needs no big downloads."""
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


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm_module, "complete", fake.complete)
    monkeypatch.setattr(llm_module, "complete_json", fake.complete_json)
    return fake


@pytest.mark.asyncio
async def test_full_pipeline_qualified(fake_llm):
    lead = LeadInput(company_name="Acme Widgets", industry="manufacturing", state="TX")

    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        result = await app.ainvoke(
            {"lead": lead, "run_id": "test-1"},
            config={"configurable": {"thread_id": "t-qualified"}},
        )

    assert result["research"].summary
    assert result["research"].naics_code == "5415"
    assert result["qualification"].verdict == "qualified"
    assert "general_liability" in result["qualification"].carrier_fit.lines
    assert result["brief"] is not None
    assert result["brief"].headline.startswith("Qualified lead")


@pytest.mark.asyncio
async def test_hitl_interrupt_and_approve(fake_llm):
    fake_llm.verdict = "needs_review"
    lead = LeadInput(company_name="Risky Startup", industry="technology")

    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "t-review"}}

        first = await app.ainvoke({"lead": lead, "run_id": "test-2"}, config=config)
        # the run pauses at the gate; brief is not yet produced
        assert first.get("brief") is None

        snapshot = await app.aget_state(config)
        assert snapshot.tasks, "expected the graph to be paused at an interrupt"
        interrupt_value = snapshot.tasks[0].interrupts[0].value
        assert interrupt_value["type"] == "lead_review"
        assert interrupt_value["company"] == "Risky Startup"

        second = await app.ainvoke(Command(resume={"action": "approve"}), config=config)
        assert second["qualification"].verdict == "qualified"
        assert second["brief"] is not None


@pytest.mark.asyncio
async def test_hitl_hard_disqualify(fake_llm):
    fake_llm.verdict = "disqualified"
    lead = LeadInput(company_name="Shady Ops LLC", industry="unknown")

    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "t-dq"}}

        await app.ainvoke({"lead": lead, "run_id": "test-3"}, config=config)
        result = await app.ainvoke(Command(resume={"action": "disqualify"}), config=config)

    assert result["qualification"].verdict == "disqualified"
    assert result.get("brief") is None
    assert result.get("skip_report") is True
    assert result.get("memory_ops")  # still extracted for the blacklist
