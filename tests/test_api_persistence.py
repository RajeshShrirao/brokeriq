"""API async-persistence lifecycle tests.

Covers the dev/test lifespan wiring (AsyncSqliteSaver + InMemoryStore, never
postgres) and proves a run paused at the compliance gate survives an app
restart: a fresh lifespan over the same BROKERIQ_API_DB_PATH file can resume
the thread from the persisted checkpoint.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from brokeriq import llm as llm_module
from brokeriq.api import app
from brokeriq.config import get_settings
from brokeriq.fake import FakeLLM
from brokeriq.graph import build_graph
from brokeriq.rag import store as rag_store


@pytest.fixture(autouse=True)
def api_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BROKERIQ_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("BROKERIQ_API_DB_PATH", str(tmp_path / "api.db"))

    get_settings.cache_clear()
    rag_store._memory_client = None
    yield
    get_settings.cache_clear()
    rag_store._memory_client = None


LEAD = {
    "company_name": "Acme Widgets",
    "industry": "manufacturing",
    "state": "TX",
}


def _patch_fake_after_lifespan(verdict: str = "qualified") -> FakeLLM:
    """Install the FakeLLM after the lifespan has run.

    The lifespan re-installs its own FakeLLM (default verdict) whenever no API
    keys are configured, so the per-test fake must be patched after entering
    the TestClient context for the verdict to take effect.
    """
    fake = FakeLLM(verdict=verdict)
    llm_module.complete = fake.complete
    llm_module.complete_json = fake.complete_json
    return fake


def _stream_events(client, run_id: str) -> list:
    events = []
    with client.stream("GET", f"/leads/{run_id}/stream") as resp:
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            if line.startswith("data:"):
                events.append(json.loads(line.split(":", 1)[1]))
    return events


def test_lifespan_builds_persistent_graph_with_async_sqlite_checkpointer():
    with TestClient(app) as client:
        _patch_fake_after_lifespan()

        graph = app.state.graph
        assert graph is not None
        for method in ("ainvoke", "astream", "aget_state"):
            assert callable(getattr(graph, method, None)), f"graph missing {method}"
        assert isinstance(graph.checkpointer, AsyncSqliteSaver), "dev/test must use AsyncSqliteSaver, not postgres"
        assert isinstance(graph.store, InMemoryStore)

        run_id = client.post("/leads", json=LEAD).json()["run_id"]
        events = _stream_events(client, run_id)
        names = [e for e in events if isinstance(e, str)]
        assert "run_started" in names
        assert "run_complete" in names
        complete = next(e for e in events if isinstance(e, dict) and "verdict" in e)
        assert complete["verdict"] == "qualified"
        assert complete["brief"] is not None


def test_paused_run_resumes_via_api_after_lifespan_restart():
    with TestClient(app) as first:
        _patch_fake_after_lifespan(verdict="needs_review")
        run_id = first.post("/leads", json=LEAD).json()["run_id"]
        events = _stream_events(first, run_id)
        assert "review_required" in [e for e in events if isinstance(e, str)]
        assert "run_complete" not in [e for e in events if isinstance(e, str)]

    db_path = get_settings().api_db_path
    assert Path(db_path).exists(), "paused run should be persisted to the sqlite file"

    with TestClient(app) as second:
        _patch_fake_after_lifespan(verdict="needs_review")
        body = second.post(f"/leads/{run_id}/resume", json={"action": "approve"}).json()
        assert body["verdict"] == "qualified"
        assert body["brief"] is not None


@pytest.mark.asyncio
async def test_fresh_checkpointer_resumes_persisted_thread_from_sqlite_file():
    with TestClient(app) as client:
        _patch_fake_after_lifespan(verdict="needs_review")
        run_id = client.post("/leads", json=LEAD).json()["run_id"]
        events = _stream_events(client, run_id)
        assert "review_required" in [e for e in events if isinstance(e, str)]

    thread_id = f"run-{run_id}"
    db_path = get_settings().api_db_path
    _patch_fake_after_lifespan()

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        snapshot = await graph.aget_state(config)
        assert snapshot.tasks, "expected the persisted thread to be paused at an interrupt"
        interrupt_value = snapshot.tasks[0].interrupts[0].value
        assert interrupt_value["type"] == "lead_review"
        assert interrupt_value["company"] == "Acme Widgets"

        result = await graph.ainvoke(Command(resume={"action": "approve"}), config=config)
        assert result["qualification"].verdict == "qualified"
        assert result["brief"] is not None
