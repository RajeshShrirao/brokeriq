# BrokerIQ — Production Deployment

## Overview

BrokerIQ is a FastAPI service (port 8000) that runs an autonomous lead qualification pipeline built on LangGraph. It streams qualification progress over SSE and pauses at a compliance gate for human review before resuming.

Two runtime modes:

- **dev** (default): SQLite checkpoint + in-memory store. No external dependencies beyond an LLM provider key.
- **prod**: Postgres checkpoint + Postgres-backed LangGraph store + Redis-backed semantic cache. Redis is validated at startup and the service refuses to start if Redis is unreachable.

## Architecture

```
                        ┌──────────────┐
   Client ────────────►│  BrokerIQ    │
   (SSE / REST)        │  (uvicorn)   │
                        └──────┬───────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                 │
        ┌─────▼─────┐  ┌──────▼──────┐  ┌──────▼──────┐
        │  Postgres  │  │    Redis    │  │   Qdrant    │
        │ (checkpoint│  │ (semantic   │  │ (vector     │
        │  + store)  │  │  cache)     │  │  store)     │
        └───────────┘  └─────────────┘  └─────────────┘
```

## Prerequisites

### Required infrastructure

| Service | Purpose | Version tested |
|---|---|---|
| Postgres | LangGraph checkpoint + store (prod) | 16 |
| Redis | Semantic cache tier 1 + connectivity gate | 7 |
| Qdrant | Vector store for compliance search | 1.15.4 |
| LLM provider | OpenRouter / Gemini / Groq (at least one key) | — |

### Required env vars (prod)

| Variable | Purpose | Example |
|---|---|---|
| `POSTGRES_DSN` | Postgres connection string | `postgresql://brokeriq:pass@host:5432/brokeriq` |
| `QDRANT_URL` | Qdrant HTTP endpoint | `http://qdrant:6333` |
| `REDIS_URL` | Redis URL | `redis://redis:6379/0` |
| `OPENROUTER_API_KEY` | LLM provider key (or `GEMINI_API_KEY` / `GROQ_API_KEY`) | — |
| `BROKERIQ_ENV` | Set to `prod` | `prod` |

Optional:

| Variable | Default | Purpose |
|---|---|---|
| `BROKERIQ_LOG_LEVEL` | `INFO` | Logging level |
| `BROKERIQ_MODEL` | `openrouter/google/gemini-2.5-flash` | LLM model |
| `BROKERIQ_EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model |
| `BROKERIQ_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model |
| `BROKERIQ_RERANKER_ENABLED` | `false` | Toggle reranker |
| `BROKERIQ_LANGSMITH_TRACING` | `false` | LangSmith tracing |
| `BROKERIQ_LANGSMITH_API_KEY` | — | LangSmith key |
| `BROKERIQ_LANGSMITH_PROJECT` | `brokeriq` | LangSmith project name |
| `BROKERIQ_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `BROKERIQ_CACHE_SIMILARITY_THRESHOLD` | `0.92` | Semantic cache similarity threshold |
| `BROKERIQ_WEB_SEARCH_PROVIDER` | `ddgs` | `ddgs` or `tavily` |
| `BROKERIQ_TAVILY_API_KEY` | — | Tavily key if using tavily |
| `BROKERIQ_OFFLINE` / `BROKERIQ_USE_FAKE_LLM` | — | Force FakeLLM (no real LLM key needed) |

## Docker / Packaging

### Image

The `Dockerfile` uses `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` as base. It layers dependencies first (pyproject + lock), then source + corpus data, then syncs. The default command runs uvicorn on `0.0.0.0:8000`.

Two entrypoints are supported:

- **API**: `uv run uvicorn brokeriq.api:app --host 0.0.0.0 --port 8000`
- **MCP**: `brokeriq-mcp` (stdio server for MCP clients)

### Local stack (docker-compose.yml)

```yaml
services:
  qdrant:  # vector store
  redis:   # semantic cache
  postgres: # checkpoint + store
  api:     # BrokerIQ API, depends_on all three
```

```bash
docker compose up --build
```

The API service mounts `.env` via `env_file` and overrides `BROKERIQ_ENV=dev` plus the three infra URLs to point at the compose services. LLM keys are read from the host `.env`.

For production, run the `api` service standalone (or in a separate compose file) with `BROKERIQ_ENV=prod` and real infra URLs.

### Building the wheel

The project uses hatchling. Build with:

```bash
uv build
```

The wheel includes `src/brokeriq` and the NAICS corpus (`brokeriq/data/naics.csv`). Verify wheel contents after packaging changes.

## Redis

Redis is **required in prod** — the service validates connectivity at startup and refuses to start if Redis is unreachable. This is a deliberate fail-fast gate, not a graceful degradation.

The semantic cache has two tiers:

1. **Exact**: SHA-256 hash of normalized query → Redis GET.
2. **Semantic**: cosine similarity against a bounded in-memory fingerprint ring (max 1024 entries). Near-duplicates return cached payload without touching Qdrant.

If Redis is unreachable at runtime (after successful startup), the cache degrades to a no-op — the pipeline keeps working, cache writes/reads are silently skipped.

## Postgres (prod)

In prod mode, the lifespan:

1. Creates an `AsyncPostgresSaver` checkpointer from `POSTGRES_DSN` and calls `setup()`.
2. Creates an `AsyncPostgresStore` from the same DSN.
3. Builds the LangGraph app with both.

Both checkpoint and store share the same Postgres instance. The DSN is the same value used for both — no separate read/write split currently.

In dev/test, the lifespan uses `AsyncSqliteSaver` + in-memory `InMemoryStore` instead.

## Reverse Proxy Headers

The API reads the client IP from these headers (in order):

1. `X-Forwarded-For`
2. `CF-Connecting-IP` (Cloudflare)

Rate limiting keys on the first IP in the chain. If neither header is present, it falls back to `request.client.host`.

**When running behind a reverse proxy** (nginx, Cloudflare, load balancer), ensure:

- The proxy sets `X-Forwarded-For` (standard) or `CF-Connecting-IP` (Cloudflare).
- The proxy is trusted — BrokerIQ trusts the header unconditionally. Do not expose the service directly to the internet without a proxy that strips incoming `X-Forwarded-For` from clients.

Rate limits (per IP, sliding window):

| Endpoint | Limit | Window |
|---|---|---|
| `POST /leads` | 30 | 60s |
| `GET /leads/{run_id}/stream` | 120 | 60s |

## Production Config Guide

### Minimal prod .env

```env
BROKERIQ_ENV=prod
BROKERIQ_LOG_LEVEL=INFO
POSTGRES_DSN=postgresql://brokeriq:password@postgres-host:5432/brokeriq
QDRANT_URL=http://qdrant-host:6333
REDIS_URL=redis://redis-host:6379/0
OPENROUTER_API_KEY=sk-or-...
```

At least one LLM provider key must be present (`OPENROUTER_API_KEY`, `GEMINI_API_KEY`, or `GROQ_API_KEY`). If none is set and neither `BROKERIQ_OFFLINE` nor `BROKERIQ_USE_FAKE_LLM` is set, the service starts with FakeLLM (offline mode) — this is intentional for local dev but will produce non-functional qualifications in prod.

### Startup checklist

1. Postgres is reachable and the `brokeriq` database exists.
2. Redis is reachable (service fails fast if not).
3. Qdrant is reachable (used by compliance search nodes).
4. At least one LLM provider key is set.
5. `BROKERIQ_ENV=prod` is set (otherwise SQLite + in-memory store are used).

### Observability

- Logging: configured via `BROKERIQ_LOG_LEVEL`. The lifespan logs "brokeriq api ready (prod: postgres)" on success.
- Health: `GET /healthz` returns `{"status": "ok"}`.
- LangSmith tracing: enable with `BROKERIQ_LANGSMITH_TRACING=true` + `BROKERIQ_LANGSMITH_API_KEY`.

### Runbook — first deploy

1. Provision Postgres, Redis, Qdrant.
2. Create the `brokeriq` Postgres database.
3. Write the `.env` with prod values.
4. Build the image: `docker build -t brokeriq .`
5. Run: `docker run --env-file .env -p 8000:8000 brokeriq`
6. Verify: `curl http://localhost:8000/healthz` → `{"status":"ok"}`
7. Check logs for "brokeriq api ready (prod: postgres)".

### Running without Docker

```bash
uv sync --frozen          # install deps
export BROKERIQ_ENV=prod
export POSTGRES_DSN=...
export QDRANT_URL=...
export REDIS_URL=...
export OPENROUTER_API_KEY=...
uv run uvicorn brokeriq.api:app --host 0.0.0.0 --port 8000
```

## Notes

- The API keeps in-memory state for active runs (`_runs`, `_rate_windows`) with a 5-minute TTL on completed entries. This bounds memory growth but means run data is lost on restart — the LangGraph checkpoint (Postgres in prod) is the durable record.
- The MCP entrypoint (`brokeriq-mcp`) is a stdio server for MCP clients and does not need the web stack.
