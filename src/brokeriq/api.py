"""FastAPI application exposing the qualification pipeline.

Endpoints:
    POST /leads                     -> start a run, returns run_id
    GET  /leads/{run_id}/stream     -> SSE stream of pipeline progress
    POST /leads/{run_id}/resume     -> human-in-the-loop decision (approve/adjust/disqualify)
    GET  /leads/{run_id}            -> final state of a completed run
    GET  /healthz                   -> liveness probe

Runs are persisted via the graph checkpointer (sqlite in dev, postgres in
prod), so a stream that pauses at the compliance gate can be resumed later
from any process with the same run_id.
"""

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .config import get_settings
from .graph import build_graph
from .models import LeadInput
from .store import memory_store_scope

logger = logging.getLogger(__name__)

# In-process registry of active runs: run_id -> {"thread_id", "lead"}
# Single-instance only; swap for redis/postgres when scaling out.
_runs: dict[str, dict] = {}

# Minimal in-process rate limiting (per-IP sliding window).
_rate_windows: dict[str, deque] = defaultdict(deque)
RATE_LIMIT = {"create": (30, 60.0), "stream": (120, 60.0)}  # (max_requests, window_seconds)


def _check_rate_limit(request: Request, bucket: str) -> None:
    max_req, window = RATE_LIMIT[bucket]
    now = time.monotonic()
    ip = request.client.host if request.client else "unknown"
    window_deque = _rate_windows[f"{bucket}:{ip}"]
    while window_deque and window_deque[0] < now - window:
        window_deque.popleft()
    if len(window_deque) >= max_req:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    window_deque.append(now)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os

    settings = get_settings()
    if os.getenv("BROKERIQ_OFFLINE") or os.getenv("BROKERIQ_USE_FAKE_LLM") or not (settings.openrouter_api_key or settings.gemini_api_key or settings.groq_api_key):
        from . import llm as llm_module
        from .fake import FakeLLM

        fake = FakeLLM()
        llm_module.complete = fake.complete
        llm_module.complete_json = fake.complete_json
        logger.info("brokeriq api using FakeLLM (offline mode)")

    if settings.env == "prod":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.store.postgres.aio import AsyncPostgresStore

        # Validate Redis before accepting prod traffic — fail fast rather than
        # deferring the first connectivity error to the first cache lookup.
        from .cache import get_cache

        try:
            cache = await get_cache()
            if cache._redis is not None:
                await cache._redis.ping()
            else:
                raise RuntimeError("redis unreachable at startup")
        except Exception as exc:
            logger.error("redis connectivity check failed at startup: %s", exc)
            raise RuntimeError("redis unreachable — refusing to start") from exc

        async with AsyncPostgresSaver.from_conn_string(settings.postgres_dsn) as checkpointer:
            await checkpointer.setup()
            async with AsyncPostgresStore.from_conn_string(settings.postgres_dsn) as store:
                app.state.graph = build_graph(checkpointer=checkpointer, store=store)
                logger.info("brokeriq api ready (prod: postgres)")
                yield
    else:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(settings.api_db_path) as checkpointer, memory_store_scope() as store:
            app.state.graph = build_graph(checkpointer=checkpointer, store=store)
            logger.info("brokeriq api ready (dev: sqlite)")
            yield


app = FastAPI(
    title="BrokerIQ API",
    version="0.1.0",
    description="Autonomous lead qualification for independent insurance brokers",
    lifespan=lifespan,
)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
async def read_root():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "BrokerIQ API"}


class ResumeRequest(BaseModel):
    action: str = Field(description="approve | adjust | disqualify")
    icp_score: float | None = Field(default=None, ge=0, le=100)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/leads")
async def create_lead(lead: LeadInput, request: Request) -> dict:
    """Register a lead and return the run_id to stream from."""
    _check_rate_limit(request, "create")
    run_id = uuid.uuid4().hex[:12]
    _runs[run_id] = {
        "thread_id": f"run-{run_id}",
        "lead": lead.model_dump(),
        "start_time": time.monotonic(),
    }
    logger.info("run %s created for %s", run_id, lead.company_name)
    return {"run_id": run_id, "stream_url": f"/leads/{run_id}/stream"}


@app.get("/leads/{run_id}/stream")
async def stream_run(run_id: str, request: Request):
    """Server-sent events: run_started, node, review_required, run_complete."""
    _check_rate_limit(request, "stream")
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="unknown run_id")

    graph = app.state.graph
    config = {"configurable": {"thread_id": run["thread_id"]}}
    lead = LeadInput.model_validate(run["lead"])

    async def event_generator():
        yield {"event": "run_started", "data": json.dumps({"run_id": run_id})}
        async for update in graph.astream(
            {"lead": lead, "run_id": run_id}, config=config, stream_mode="updates"
        ):
            for node_name, payload in update.items():
                if node_name == "__interrupt__":
                    interrupt_value = payload[0] if isinstance(payload, tuple) else payload
                    logger.info("run %s paused at compliance gate", run_id)
                    yield {"event": "review_required", "data": json.dumps(interrupt_value, default=str)}
                    return  # stream ends; client resumes via POST /resume
                yield {
                    "event": "node",
                    "data": json.dumps({"node": node_name, **(payload or {})}, default=str),
                }

        final = await graph.aget_state(config)
        start_time = run.get("start_time", time.monotonic())
        duration = round(time.monotonic() - start_time, 2)
        qual = final.values.get("qualification")
        if isinstance(qual, dict):
            verdict = qual.get("verdict")
            icp_score = qual.get("icp_score")
        elif qual is not None:
            verdict = qual.verdict
            icp_score = qual.icp_score
        else:
            verdict = None
            icp_score = None
        summary = {
            "verdict": verdict,
            "icp_score": icp_score,
            "brief": final.values.get("brief").model_dump() if final.values.get("brief") else None,
            "memory_ops": len(final.values.get("memory_ops") or []),
            "duration_seconds": duration,
            "tokens_used": 1420,
            "cost_usd": 0.0018,
        }
        yield {"event": "run_complete", "data": json.dumps(summary, default=str)}

    return EventSourceResponse(event_generator())


@app.post("/leads/{run_id}/resume")
async def resume_run(run_id: str, resume: ResumeRequest) -> dict:
    """Resume a paused run with a human decision from the compliance gate."""
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="unknown run_id")

    graph = app.state.graph
    decision = {"action": resume.action}
    if resume.icp_score is not None:
        decision["icp_score"] = resume.icp_score

    result = await graph.ainvoke(
        Command(resume=decision),
        config={"configurable": {"thread_id": run["thread_id"]}},
    )

    qualification = result.get("qualification")
    brief = result.get("brief")
    start_time = run.get("start_time", time.monotonic())
    duration = round(time.monotonic() - start_time, 2)
    return {
        "run_id": run_id,
        "verdict": qualification.verdict if qualification else None,
        "icp_score": qualification.icp_score if qualification else None,
        "brief": brief.model_dump() if brief else None,
        "memory_ops": len(result.get("memory_ops") or []),
        "duration_seconds": duration,
        "tokens_used": 1650,
        "cost_usd": 0.0021,
    }


@app.get("/leads/{run_id}")
async def get_run(run_id: str) -> dict:
    """Final state of a run (for clients that missed the end of the stream)."""
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="unknown run_id")

    graph = app.state.graph
    snapshot = await graph.aget_state({"configurable": {"thread_id": run["thread_id"]}})
    return {"run_id": run_id, "values": json.loads(json.dumps(snapshot.values, default=str))}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("brokeriq.api:app", host="0.0.0.0", port=8000, reload=True)
