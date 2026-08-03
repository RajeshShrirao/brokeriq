"""API tests: lead creation, SSE streaming, and HITL resume."""

import json

import pytest
from fastapi.testclient import TestClient

from brokeriq import llm as llm_module
from brokeriq.fake import FakeLLM
from brokeriq.rag import store as rag_store


@pytest.fixture(autouse=True)
def api_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BROKERIQ_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("BROKERIQ_API_DB_PATH", str(tmp_path / "api.db"))

    from brokeriq.config import get_settings

    get_settings.cache_clear()
    rag_store._memory_client = None
    yield
    get_settings.cache_clear()
    rag_store._memory_client = None


def _make_client(verdict: str = "qualified"):
    fake = FakeLLM(verdict=verdict)
    llm_module.complete = fake.complete
    llm_module.complete_json = fake.complete_json

    from brokeriq.api import app

    return TestClient(app)


LEAD = {
    "company_name": "Acme Widgets",
    "industry": "manufacturing",
    "state": "TX",
}


def test_healthz():
    with _make_client() as client:
        assert client.get("/healthz").json() == {"status": "ok"}


def test_create_lead_returns_run_id():
    with _make_client() as client:
        resp = client.post("/leads", json=LEAD)
        assert resp.status_code == 200
        body = resp.json()
        assert "run_id" in body
        assert body["stream_url"] == f"/leads/{body['run_id']}/stream"


def test_stream_emits_completion_events():
    with _make_client() as client:
        run_id = client.post("/leads", json=LEAD).json()["run_id"]

        events = []
        with client.stream("GET", f"/leads/{run_id}/stream") as resp:
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())
                if line.startswith("data:"):
                    events.append(json.loads(line.split(":", 1)[1]))

        names = [e for e in events if isinstance(e, str)]
        assert "run_started" in names
        assert "node" in names
        assert "run_complete" in names

        complete = next(e for e in events if isinstance(e, dict) and "verdict" in e)
        assert complete["verdict"] == "qualified"
        assert complete["brief"] is not None


def test_stream_pauses_for_review_then_resume():
    with _make_client(verdict="needs_review") as client:
        run_id = client.post("/leads", json=LEAD).json()["run_id"]

        events = []
        with client.stream("GET", f"/leads/{run_id}/stream") as resp:
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())
                if line.startswith("data:"):
                    events.append(json.loads(line.split(":", 1)[1]))

        assert "review_required" in [e for e in events if isinstance(e, str)]
        assert "run_complete" not in [e for e in events if isinstance(e, str)]

        resume = client.post(f"/leads/{run_id}/resume", json={"action": "approve"})
        assert resume.status_code == 200
        body = resume.json()
        assert body["verdict"] == "qualified"
        assert body["brief"] is not None


def test_resume_disqualify_skips_brief():
    with _make_client(verdict="disqualified") as client:
        run_id = client.post("/leads", json=LEAD).json()["run_id"]

        with client.stream("GET", f"/leads/{run_id}/stream") as resp:
            list(resp.iter_lines())

        body = client.post(f"/leads/{run_id}/resume", json={"action": "disqualify"}).json()
        assert body["verdict"] == "disqualified"
        assert body["brief"] is None


def test_unknown_run_returns_404():
    with _make_client() as client:
        assert client.get("/leads/nope").status_code == 404
        assert client.post("/leads/nope/resume", json={"action": "approve"}).status_code == 404
