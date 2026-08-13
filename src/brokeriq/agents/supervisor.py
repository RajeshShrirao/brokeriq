"""Supervisor: routes the lead through the pipeline, LLM-first with a
deterministic fallback so the graph still works offline / in tests."""

import logging

from .. import llm
from . import prompts

logger = logging.getLogger(__name__)

STAGES = ("research", "qualification", "report", "memory", "done")


def _fallback_next(state: dict) -> str:
    """Rule-based routing used when no LLM is configured."""
    completed = state.get("completed_stages") or set()
    if "research" not in completed:
        return "research"
    if "qualification" not in completed:
        return "qualification"
    if "report" not in completed:
        return "report"
    if "memory" not in completed:
        return "memory"
    return "done"


async def supervisor_node(state: dict) -> dict:
    lead = state["lead"]
    research = state.get("research")
    qualification = state.get("qualification")
    logger.info("supervisor: %s", lead.company_name)

    state_summary = (
        f"Company: {lead.company_name}\n"
        f"research: {'done' if research else 'missing'}\n"
        f"qualification: {qualification.verdict if qualification else 'missing'}\n"
        f"brief: {'done' if state.get('brief') else 'missing'}\n"
        f"memory: {'done' if state.get('memory_ops') is not None else 'missing'}\n"
        f"completed_stages: {state.get('completed_stages') or set()}"
    )

    usage = None
    try:
        raw, usage = await llm.complete_json(
            [
                {"role": "system", "content": prompts.SUPERVISOR},
                {"role": "user", "content": f"State summary:\n{state_summary}"},
            ]
        )
        next_stage = str(raw.get("next", "")).strip().lower()
        if next_stage not in STAGES:
            logger.warning("supervisor returned unknown stage %r; falling back", next_stage)
            next_stage = _fallback_next(state)

        # Hard-bind routes to already-completed stages so the supervisor never
        # re-routes into a finished stage. Once the guardian fires, fall back
        # to the rule-based path (which is also bounded below).
        completed = state.get("completed_stages") or set()
        if next_stage in completed and next_stage != "done":
            logger.info(
                "supervisor tried to re-route to completed stage %r; guard firing", next_stage
            )
            next_stage = _fallback_next(state)
    except Exception as exc:
        logger.warning("supervisor LLM call failed (%s); using rule-based routing", exc)
        next_stage = _fallback_next(state)

    logger.info("supervisor routing -> %s", next_stage)
    existing = state.get("usage_metadata") or {"total_tokens": 0, "cost": 0.0}
    updated = {
        "total_tokens": existing.get("total_tokens", 0) + (usage or {}).get("total_tokens", 0),
        "cost": existing.get("cost", 0.0) + (usage or {}).get("cost", 0.0),
    }
    return {"next_stage": next_stage, "usage_metadata": updated}
