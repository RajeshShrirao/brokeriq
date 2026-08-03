# BrokerIQ Architecture

BrokerIQ is a multi-agent lead-qualification platform for independent insurance brokers.
This document describes the system as built: agent graph, retrieval stack, persistence,
API, MCP surface, and the offline-first testing strategy.

## System overview

```
                          ┌────────────────────────────────────────────┐
   Lead input ──────────► │              FastAPI (SSE + HITL)           │
   (HTTP / CLI / MCP)     │  POST /leads → stream → resume → state     │
                          └──────────────┬─────────────────────────────┘
                                         │ checkpointer (sqlite / postgres)
                                         ▼
                          ┌────────────────────────────────────────────┐
                          │          LangGraph StateGraph              │
                          │                                            │
                          │  START ─► supervisor ─► research            │
                          │             │  ▲        │                   │
                          │             ▼  │        ▼                   │
                          │         qualification ─► gate (HITL)        │
                          │             │           │                   │
                          │             ▼           ▼                   │
                          │           report    memory ─► END           │
                          └──────────────┬─────────────────────────────┘
                                         │ tools
                                         ▼
              ┌─────────────────┬─────────────────┬───────────────────┐
              ▼                 ▼                 ▼                   ▼
        compliance RAG     web search       NAICS lookup          long-term store
        (Qdrant hybrid)    (DuckDuckGo)     (built-in map)        (PostgresStore)
```

## Agent graph

Assembled in `src/brokeriq/graph.py`. Nodes:

| Node           | Responsibility                                                        |
| -------------- | --------------------------------------------------------------------- |
| `supervisor`   | LLM-routed with deterministic rule fallback; decides next stage       |
| `research`     | Gathers company signals: web search, NAICS classification, sources    |
| `qualification`| Scores ICP fit (0–100), carrier fit, risk flags, verdict              |
| `gate`         | Compliance / HITL gate — interrupts on risky or borderline verdicts   |
| `report`       | Writes the lead brief (headline, summary, outreach angle, action)     |
| `memory`       | Extracts durable lead learnings into long-term memory                 |

Routing:

- `supervisor` → `research` / `qualification` / `report` / `memory` / `END`
- `research` → `supervisor` (loop until all stages complete)
- `qualification` → `gate`
- `gate` → `supervisor` (qualified) or `memory` (hard-disqualified; skip report)
- `report` → `supervisor`
- `memory` → `END`

### Human-in-the-loop gate

On `needs_review` verdicts the gate raises a LangGraph interrupt. The run is persisted
(checkpointer) and the API emits `review_required` on the SSE stream. A broker resumes
with one of:

- `approve` — force-qualify and continue to report
- `adjust` — override ICP score, then qualify
- `disqualify` — skip report, go straight to memory extraction

Because the graph is checkpointed per `thread_id`, resume continues from the exact
interrupted node — no recomputation of prior stages.

## Compliance RAG

`src/brokeriq/rag/` + `src/brokeriq/tools/compliance_rag.py`.

1. **Ingest** — markdown corpus (`data/corpus/`) is chunked and embedded
   (`BAAI/bge-base-en-v1.5` default; `all-MiniLM-L6-v2` for CI/laptops).
2. **Hybrid retrieval** — dense vector search + BM42 sparse search fused with
   reciprocal-rank fusion (RRF) in Qdrant.
3. **Rerank** — cross-encoder (`BAAI/bge-reranker-v2-m3`) re-scores top candidates;
   falls back gracefully when the reranker model is unavailable.
4. **Cache** — two-tier Redis semantic cache (embedding-near-duplicate keys) serves
   repeated questions without re-querying Qdrant; degrades to a no-op when Redis is down.
5. **Fallbacks** — if Qdrant is unreachable, an in-memory store serves the corpus, so
   tests and offline evals run with no infrastructure at all.

Results carry `doc_id`, `section`, and `citation` so downstream agents (and MCP clients)
can quote sources.

## Persistence

| Concern            | Dev                    | Prod                    |
| ------------------ | ---------------------- | ----------------------- |
| Graph checkpoints  | SQLite (`brokeriq.db`) | Postgres (`AsyncPostgresSaver`) |
| Long-term memory   | SQLite / in-memory     | Postgres (`PostgresStore`) |
| Semantic cache     | degraded (no Redis)    | Redis                   |
| Vector index       | in-memory fallback     | Qdrant                  |

`docker-compose.yml` runs qdrant + redis + postgres for a full prod-like stack.

## API surface

FastAPI in `src/brokeriq/api.py`. Full contract in [docs/api.md](docs/api.md).

| Method | Path                     | Purpose                                  |
| ------ | ------------------------ | ---------------------------------------- |
| POST   | `/leads`                 | Register a lead → `run_id`               |
| GET    | `/leads/{run_id}/stream` | SSE progress events                      |
| POST   | `/leads/{run_id}/resume` | Human decision at the gate               |
| GET    | `/leads/{run_id}`        | Final run state                          |
| GET    | `/healthz`               | Liveness                                 |

SSE events: `run_started`, `node`, `review_required`, `run_complete`.

## MCP server

`src/brokeriq/mcp_server.py` implements the MCP 2.0 protocol over stdio
(`mcp.server.mcpserver.MCPServer`). It exposes one tool:

- `compliance_search(query, limit)` — hybrid retrieval over the compliance corpus with
  citations. Logs go to stderr so stdout stays clean for JSON-RPC.

## Offline-first design

BrokerIQ runs and tests with **no API keys and no infrastructure**:

- `FakeLLM` (`src/brokeriq/fake.py`) replaces `complete`/`complete_json` with
  deterministic responses (supervisor routing, research, verdicts, briefs).
- MiniLM embedder avoids large model downloads.
- Unreachable Qdrant → in-memory store; unreachable Redis → degraded cache.

This is what makes CI (`.github/workflows/ci.yml`) green without secrets: ruff, pytest,
deterministic evals, and promptfoo all run on every push.

## Evals

`evals/` contains a dataset of 6 leads with gold verdicts and 5 compliance queries with
expected docs, plus:

- `runner.py` — offline mode (FakeLLM fed each lead's gold verdict) and live mode
  (real LLM with `:memory:` checkpointer).
- `judge.py` — deterministic judges (verdict accuracy, brief correctness, ICP range,
  source presence, retrieval hit-rate) plus a gated LLM-judge tier.
- `cli.py` — `python -m evals.cli --mode offline|live`.
- `promptfoo_provider.py` + `promptfoo.yaml` — compliance retrieval checks as promptfoo
  tests, runnable in CI via `npx promptfoo eval`.
