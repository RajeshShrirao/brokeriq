#!/usr/bin/env python3
"""promptfoo provider: run the BrokerIQ compliance_search tool.

promptfoo's python provider imports this module and calls
`call_api(prompt, options, context)`, which must return a dict. We run the
real hybrid RAG stack and hand back the search results as a JSON string so
the config's contains-json / javascript assertions can inspect them.

Run with the project venv python:
    PROMPTFOO_PYTHON=$PWD/.venv/bin/python npx promptfoo eval -c promptfoo.yaml
That env var is what promptfoo's python provider uses to find the
interpreter (see pythonUtils: configPath > PROMPTFOO_PYTHON > fallbacks).

Uses the MiniLM embedder so CI runs stay fast, and Qdrant falls back to the
in-memory store when no server is up.
"""

import asyncio
import json
import os

# Keep the embedder small for CI speed; safe to override via env.
os.environ.setdefault("BROKERIQ_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

from brokeriq.tools.compliance_rag import compliance_search


def call_api(prompt: str, options=None, context=None) -> dict:
    """promptfoo provider entrypoint."""
    query = prompt.strip()
    payload = asyncio.run(compliance_search(query, limit=5))
    return {"output": json.dumps(payload)}


if __name__ == "__main__":
    # Direct invocation for debugging: `echo '{"prompt": "..."}' | python3 ...`
    import sys

    raw = sys.stdin.read()
    request = json.loads(raw) if raw.strip() else {}
    print(json.dumps(call_api(request.get("prompt", ""))))
