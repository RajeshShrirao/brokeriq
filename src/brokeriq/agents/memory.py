"""Memory extractor + writer: turns run outputs into durable long-term facts.

The node extracts typed MemoryOps from the run (LLM), then applies them to
the graph's BaseStore (Postgres in prod, in-memory in dev) via get_store().
Applied ops are reflected back into state so callers can see what was written.
"""

import logging

from langgraph.config import get_store

from .. import llm
from ..models import MemoryOp
from . import prompts

logger = logging.getLogger(__name__)


async def memory_node(state: dict) -> dict:
    lead = state["lead"]
    research = state.get("research")
    qualification = state.get("qualification")
    brief = state.get("brief")
    logger.info("memory extraction: %s", lead.company_name)

    raw, usage = await llm.complete_json(
        [
            {"role": "system", "content": prompts.MEMORY},
            {
                "role": "user",
                "content": (
                    f"Company: {lead.company_name} (run {state.get('run_id', 'unknown')})\n"
                    f"Research: {research.summary if research else 'n/a'}\n"
                    f"NAICS: {research.naics_code if research else None}\n"
                    f"Verdict: {qualification.verdict if qualification else 'n/a'}, "
                    f"score {qualification.icp_score if qualification else 'n/a'}\n"
                    f"Brief headline: {brief.headline if brief else 'n/a'}"
                ),
            },
        ]
    )

    ops = [MemoryOp.model_validate(op) for op in raw.get("ops", [])]
    applied = await _apply_ops(ops)
    logger.info("memory ops: %d extracted, %d applied", len(ops), len(applied))
    existing = state.get("usage_metadata") or {"total_tokens": 0, "cost": 0.0}
    updated = {
        "total_tokens": existing.get("total_tokens", 0) + (usage or {}).get("total_tokens", 0),
        "cost": existing.get("cost", 0.0) + (usage or {}).get("cost", 0.0),
    }
    return {"memory_ops": ops, "completed_stages": {"memory"}, "usage_metadata": updated}

async def _apply_ops(ops: list[MemoryOp]) -> list[MemoryOp]:
    """Persist non-NOOP ops to the graph's store; skip cleanly if no store."""
    applied: list[MemoryOp] = []
    store = get_store()
    if store is None:
        logger.debug("no store attached; memory ops recorded in state only")
        return applied

    for op in ops:
        if op.op == "NOOP":
            continue
        try:
            if op.op == "DELETE":
                await store.adelete(op.namespace, op.key)
            else:  # ADD / UPDATE
                await store.aput(op.namespace, op.key, op.value)
            applied.append(op)
        except Exception as exc:
            logger.warning("failed to apply memory op %s/%s: %s", op.namespace, op.key, exc)
    return applied
