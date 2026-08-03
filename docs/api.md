# BrokerIQ HTTP API

Base URL: `http://localhost:8000` (dev). All responses are JSON unless the
endpoint streams SSE. There is no auth yet — the API is intended to sit
behind an API gateway in production.

## Endpoints

| Method | Path                        | Description                                     |
| ------ | --------------------------- | ----------------------------------------------- |
| POST   | `/leads`                    | Register a lead, get a `run_id`                 |
| GET    | `/leads/{run_id}/stream`    | Server-sent events for a run's progress         |
| POST   | `/leads/{run_id}/resume`    | Human decision for a run paused at the gate     |
| GET    | `/leads/{run_id}`           | Final state of a completed run                  |
| GET    | `/healthz`                  | Liveness probe                                  |

## POST /leads

Request body (`application/json`):

```json
{
  "company_name": "Acme Widgets",
  "domain": "acmewidgets.com",
  "industry": "manufacturing",
  "state": "TX",
  "revenue_band": "5-20M",
  "notes": ""
}
```

`revenue_band` is one of `"<1M" | "1-5M" | "5-20M" | "20M+" | "unknown"`.

Response `201`:

```json
{
  "run_id": "a1b2c3d4e5f6",
  "stream_url": "/leads/a1b2c3d4e5f6/stream"
}
```

## GET /leads/{run_id}/stream

Server-sent events (`text/event-stream`). Each event has `event:` and `data:`
fields; `data` is always JSON.

| Event              | Payload                                                                 |
| ------------------ | ----------------------------------------------------------------------- |
| `run_started`      | `{"run_id": "..."}`                                                     |
| `node`             | `{"node": "research", "research": {...}}` — one per completed graph node |
| `review_required`  | `{"type": "lead_review", "company", "icp_score", "verdict", ...}`       |
| `run_complete`     | `{"verdict", "icp_score", "brief", "memory_ops"}`                       |

If the lead needs human review, the stream **ends after `review_required`**
and the run is paused. Resume it with `POST /leads/{run_id}/resume`.

## POST /leads/{run_id}/resume

Body:

```json
{ "action": "approve" }
```

`action` is one of:

| Action        | Effect                                          |
| ------------- | ----------------------------------------------- |
| `approve`     | Treat the lead as qualified and write the brief |
| `adjust`      | Override `icp_score` (0–100), then qualify      |
| `disqualify`  | Skip the brief; record the lead for memory      |

For `adjust`, include the new score:

```json
{ "action": "adjust", "icp_score": 68 }
```

Response `200`:

```json
{
  "run_id": "a1b2c3d4e5f6",
  "verdict": "qualified",
  "icp_score": 82,
  "brief": {
    "headline": "Qualified lead: Acme Widgets",
    "summary": "...",
    "outreach_angle": "...",
    "recommended_action": "..."
  },
  "memory_ops": 1
}
```

## GET /leads/{run_id}

Full run state, useful for clients that reconnect after a stream dropped:

```json
{
  "run_id": "a1b2c3d4e5f6",
  "values": { "lead": {...}, "research": {...}, "qualification": {...}, ... }
}
```

## Limits

Per-IP sliding window: 30 requests/min on `/leads`, 120/min on `/stream`.
Excess requests get `429`.

## Running

```bash
# dev (sqlite checkpointer + in-memory store)
uvicorn brokeriq.api:app --reload --port 8000

# prod (postgres for checkpoints + long-term memory)
BROKERIQ_ENV=prod POSTGRES_DSN=postgresql://... uvicorn brokeriq.api:app --port 8000
```
