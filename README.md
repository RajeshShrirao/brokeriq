# BrokerIQ

Autonomous lead qualification and market intelligence for independent insurance brokers.

Given a raw lead (company name + a few signals), BrokerIQ researches the company, scores it
against an ideal customer profile, checks carrier and state-regulatory fit, and produces a
ready-to-use lead brief with a recommended outreach angle — in minutes instead of hours.

Built on a LangGraph agent pipeline with hybrid RAG, human-in-the-loop review, streaming
SSE, an interactive Web Dashboard UI, and a Model Context Protocol server for compliance search.

## Features

- **S-Tier Web Dashboard UI** — single-page browser interface at `GET /` featuring:
  - **Live Mermaid.js LangGraph DAG Visualizer**: Interactive execution path rendering.
  - **Agent Thought Log & Reasoner Scratchpad**: Real-time drawer showing agent citations, NAICS lookup, and compliance facts.
  - **Execution Telemetry Badges**: Live tracking of duration (`s`), token estimates, and cost (`USD`).
  - **Model Switcher**: Dynamic provider selection (`FakeLLM Offline`, `Gemini 2.5 Flash`, `Groq Llama 3.3`, `OpenRouter`).
  - **1-Click Executive PDF Export**: Professional print stylesheet for generating broker briefs.
- **Multi-agent pipeline** — supervisor routes research → qualification → compliance gate →
  report → memory extraction agents; deterministic rule fallback keeps the graph runnable
  with no LLM configured.
- **Human-in-the-loop gate** — risky verdicts pause the run for a broker decision
  (approve / adjust score / disqualify) and resume exactly where they left off.
- **Hybrid compliance RAG** — dense + BM42 sparse retrieval with reciprocal-rank fusion,
  cross-encoder reranking, and a two-tier Redis semantic cache. Returns citation-ready
  facts (doc + section references).
- **Long-term memory** — extracted lead learnings persist across runs (Postgres store in
  prod, SQLite in dev).
- **Streaming API** — SSE progress events + resume endpoint for HITL decisions.
- **MCP 2.0 server** — `compliance_search` exposed over stdio for any MCP client.
- **Offline-first** — zero API keys required to develop and test: FakeLLM, MiniLM
  embedder, in-memory Qdrant fallback, degraded cache.
- **CI + evals** — GitHub Actions runs ruff, pytest, deterministic offline evals, and
  promptfoo compliance checks on every push.

## Quickstart

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> brokeriq && cd brokeriq
uv sync
cp .env.example .env        # add at least one LLM key for live runs (optional)

# Offline demo — FakeLLM, no keys needed
uv run brokeriq "Acme Widgets" --industry manufacturing --state TX --offline

# Run the test suite + lints
uv run ruff check src tests evals
uv run pytest -q
```

### Web Dashboard & HTTP API

Start the local API server (works offline with zero keys):

```bash
BROKERIQ_OFFLINE=1 uv run uvicorn brokeriq.api:app --reload --port 8000
```

- Open **`http://localhost:8000`** in your browser for the interactive Web Dashboard UI (live agent graph visualizer, SSE log stream, and 1-click HITL decision modal).
- REST endpoint for programmatic access:
  ```bash
  curl -X POST localhost:8000/leads \
    -H 'content-type: application/json' \
    -d '{"company_name":"Acme Widgets","state":"TX","revenue_band":"5-20M"}'
  ```

See [docs/api.md](docs/api.md) for the full contract (SSE events, resume actions, limits).

### Live run (needs an LLM key)

Set one of `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, or `GROQ_API_KEY` in `.env`, then:

```bash
uv run brokeriq "Nimbus Cyber Solutions" --domain nimbuscyber.io --industry cybersecurity --state CA
```

### MCP server

```bash
uv run brokeriq-mcp          # stdio transport
```

Register in any MCP client (e.g. Hermes `~/.hermes/config.yaml`):

```yaml
mcp_servers:
  brokeriq:
    command: "uv"
    args: ["run", "--project", "/abs/path/brokeriq", "brokeriq-mcp"]
```

### Docker

```bash
docker compose up --build    # qdrant + redis + postgres + api on :8000
```

## Evals

```bash
# Deterministic offline evals — FakeLLM fed each lead's gold verdict; no keys needed
BROKERIQ_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
BROKERIQ_QDRANT_URL=http://127.0.0.1:1 \
BROKERIQ_REDIS_URL=redis://127.0.0.1:1 \
uv run python -m evals.cli --mode offline

# promptfoo compliance checks (uses the project venv python)
PROMPTFOO_PYTHON=$PWD/.venv/bin/python npx --yes promptfoo eval -c promptfoo.yaml
```

With an LLM key set, `--mode live` runs the real graph and an optional LLM-judge tier.

## Repository layout

```
src/brokeriq/
  agents/          supervisor, research, qualification, gate, report, memory
  rag/             embeddings, hybrid store, rerank, cross-encoder, ingest
  tools/           web_search, naics_lookup, compliance_rag
  static/          Web Dashboard UI (index.html)
  graph.py         state graph assembly + routing
  api.py           FastAPI SSE + Web UI mounting + HITL resume
  mcp_server.py    MCP 2.0 stdio server (compliance_search)
  fake.py          FakeLLM for offline runs/tests
evals/             dataset, runner, judge, CLI, promptfoo provider
data/corpus/       carrier compliance markdown corpus
docs/              architecture, API contract, roadmap
```

## Docs

- [Architecture](docs/architecture.md)
- [API contract](docs/api.md)
- [Roadmap](docs/roadmap.md)

## License

Proprietary / internal — see repository owner.
