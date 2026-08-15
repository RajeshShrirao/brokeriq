"""FastAPI application exposing the qualification pipeline."""

import json
import logging
import re
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

_runs: dict[str, dict] = {}
_runs_created: dict[str, float] = {}
_runs_ttl: float = 300.0  # 5 minutes — completed runs evicted after TTL

_rate_windows: dict[str, deque] = defaultdict(deque)
_rate_windows_last_seen: dict[str, float] = {}
_rate_windows_ttl: float = 300.0  # inactive IP keys evicted after TTL
RATE_LIMIT = {"create": (30, 60.0), "stream": (120, 60.0)}


def _evict_stale() -> None:
    now = time.monotonic()
    # Evict stale _runs entries
    stale_runs = [rid for rid, ts in _runs_created.items() if now - ts > _runs_ttl]
    for rid in stale_runs:
        _runs.pop(rid, None)
        _runs_created.pop(rid, None)
    # Evict stale _rate_windows keys
    stale_keys = [k for k, ts in _rate_windows_last_seen.items() if now - ts > _rate_windows_ttl]
    for k in stale_keys:
        _rate_windows.pop(k, None)
        _rate_windows_last_seen.pop(k, None)


def _ensure_bounded() -> None:
    """Evict stale entries from all in-memory state so the process never grows unbounded."""
    _evict_stale()


def _client_ip(request: Request) -> str:
    proxy = request.headers.get("X-Forwarded-For") or request.headers.get("CF-Connecting-IP")
    if proxy:
        candidate = proxy.split(",")[0].strip()
        if re.fullmatch(r"[0-9a-fA-F:]+", candidate):
            return candidate
    if request.client and re.fullmatch(r"[0-9a-fA-F:]+", request.client.host):
        return request.client.host
    return "unknown"


def _check_rate_limit(request: Request, bucket: str) -> None:
    _evict_stale()
    max_req, window = RATE_LIMIT[bucket]
    now = time.monotonic()
    ip = _client_ip(request)
    window_deque = _rate_windows[f"{bucket}:{ip}"]
    _rate_windows_last_seen[f"{bucket}:{ip}"] = now
    while window_deque and window_deque[0] < now - window:
        window_deque.popleft()
    if len(window_deque) >= max_req:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    window_deque.append(now)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os

    settings = get_settings()
    if os.getenv("BROKERIQ_OFFLINE") or os.getenv("BROKERIQ_USE_FAKE_LLM") or not (
        settings.openrouter_api_key or settings.gemini_api_key or settings.groq_api_key
    ):
        from .fake import FakeLLM
        from .llm import LLM

        _llm = LLM(strategy=FakeLLM())
        logger.info("brokeriq api using FakeLLM (offline mode)")
    from . import llm as llm_module

    llm_module.llm = _llm

    if settings.env == "prod":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.store.postgres.aio import AsyncPostgresStore

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
    _ensure_bounded()
    _check_rate_limit(request, "create")
    run_id = uuid.uuid4().hex[:12]
    _runs[run_id] = {
        "thread_id": f"run-{run_id}",
        "lead": lead.model_dump(),
        "start_time": time.monotonic(),
    }
    _runs_created[run_id] = time.monotonic()
    logger.info("run %s created for %s", run_id, lead.company_name)
    return {"run_id": run_id, "stream_url": f"/leads/{run_id}/stream"}


@app.get("/leads/{run_id}/stream")
async def stream_run(run_id: str, request: Request):
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
                    return
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
        }
        usage = final.values.get("usage_metadata")
        if usage:
            summary["tokens_used"] = usage.get("total_tokens")
            summary["cost_usd"] = usage.get("cost")
        yield {"event": "run_complete", "data": json.dumps(summary, default=str)}

    return EventSourceResponse(event_generator())


@app.post("/leads/{run_id}/resume")
async def resume_run(run_id: str, resume: ResumeRequest) -> dict:
    _ensure_bounded()
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
    response = {
        "run_id": run_id,
        "verdict": qualification.verdict if qualification else None,
        "icp_score": qualification.icp_score if qualification else None,
        "brief": brief.model_dump() if brief else None,
        "memory_ops": len(result.get("memory_ops") or []),
        "duration_seconds": duration,
    }
    usage = result.get("usage_metadata")
    if usage:
        response["tokens_used"] = usage.get("total_tokens")
        response["cost_usd"] = usage.get("cost")
    return response


@app.get("/leads/{run_id}")
async def get_run(run_id: str) -> dict:
    _ensure_bounded()
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="unknown run_id")

    graph = app.state.graph
    snapshot = await graph.aget_state({"configurable": {"thread_id": run["thread_id"]}})
    return {"run_id": run_id, "values": json.loads(json.dumps(snapshot.values, default=str))}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("brokeriq.api:app", host="0.0.0.0", port=8000, reload=True)
