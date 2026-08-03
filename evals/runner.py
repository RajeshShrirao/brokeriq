"""Pipeline runner for evals.

Two modes:
- offline: the deterministic FakeLLM replays each lead with its gold verdict,
  so the harness itself is exercised end-to-end with zero API keys.
- live: the real LLM runs (needs BROKERIQ_OPENROUTER_API_KEY or a configured
  provider key); used for genuine model evals.

Both modes write one JSON object per lead to stdout / a JSONL file.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from brokeriq import llm as llm_module
from brokeriq.fake import FakeLLM
from brokeriq.graph import build_graph
from brokeriq.models import LeadInput
from brokeriq.store import memory_store_scope

from . import dataset


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": f"eval-{run_id}"}}


async def _run_one(app, lead: LeadInput, run_id: str) -> dict:
    return await app.ainvoke({"lead": lead, "run_id": run_id}, config=_config(run_id))


def _result_dict(lead: LeadInput, state: dict) -> dict:
    """Flatten the graph state into an eval-friendly record."""
    research = state.get("research")
    qualification = state.get("qualification")
    brief = state.get("brief")
    return {
        "company": lead.company_name,
        "verdict": qualification.verdict if qualification else None,
        "icp_score": qualification.icp_score if qualification else None,
        "carrier_lines": qualification.carrier_fit.lines if qualification else [],
        "research_summary": research.summary if research else None,
        "sources": research.sources if research else [],
        "brief_headline": brief.headline if brief else None,
        "brief_summary": brief.summary if brief else None,
        "outreach_angle": brief.outreach_angle if brief else None,
        "recommended_action": brief.recommended_action if brief else None,
    }


async def run_leads(mode: str = "offline") -> list[dict]:
    """Run every lead in the dataset; returns flattened result records.

    Offline mode replays each lead with a FakeLLM configured to the gold
    verdict, so the harness itself (graph wiring, state flow, judge checks)
    is validated end-to-end with zero API keys. Live mode runs the real
    model and measures actual quality.
    """
    if mode == "live":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer, memory_store_scope() as store:
            app = build_graph(checkpointer=checkpointer, store=store)
            records = []
            for lead, _, _ in dataset.iter_leads():
                run_id = uuid.uuid4().hex[:10]
                state = await _run_one(app, lead, run_id)
                records.append(_result_dict(lead, state))
            return records
    elif mode != "offline":
        raise ValueError(f"unknown eval mode: {mode!r} (use 'offline' or 'live')")

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer, memory_store_scope() as store:
        app = build_graph(checkpointer=checkpointer, store=store)
        records = []
        for lead, gold, _ in dataset.iter_leads():
            fake = FakeLLM(verdict=gold, icp_score=82.0 if gold != "disqualified" else 30.0)
            llm_module.complete = fake.complete
            llm_module.complete_json = fake.complete_json
            run_id = uuid.uuid4().hex[:10]
            state = await _run_one(app, lead, run_id)
            records.append(_result_dict(lead, state))
        return records


def write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
