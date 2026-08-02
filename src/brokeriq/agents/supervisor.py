"""Supervisor: routes the lead through the pipeline, LLM-first with a
deterministic fallback so the graph still works offline / in tests."""

import logging

from ..llm import complete_json
from . import prompts

logger = logging.getLogger(__name__)

STAGES = ("research", "qualification", "report", "memory", "done")


def _fallback_next(state: dict) -> str:
    """Rule-based routing used when no LLM is configured."""
    if not state.get("research"):
        return "research"
    if not state.get("qualification"):
        return "qualification"
    if not state.get("brief"):
        return "report"
    if not state.get("memory_ops"):
        return "memory"
    return "done"


async def supervisor_node(state: dict) -> dict:
    lead = state["lead"]
    research = state.get("research")
    qualification = state.get("qualification")

    state_summary = (
        f"Company: {lead.company_name}\n"
        f"research: {'done' if research else 'missing'}\n"
        f"qualification: {qualification.verdict if qualification else 'missing'}\n"
        f"brief: {'done' if state.get('brief') else 'missing'}\n"
        f"memory: {'done' if state.get('memory_ops') is not None else 'missing'}"
    )

    try:
        raw = await complete_json(
            [
                {"role": "system", "content": prompts.SUPERVISOR},
                {"role": "user", "content": f"State summary:\n{state_summary}"},
            ]
        )
        next_stage = str(raw.get("next", "")).strip().lower()
        if next_stage not in STAGES:
            logger.warning("supervisor returned unknown stage %r; falling back", next_stage)
            next_stage = _fallback_next(state)
    except Exception as exc:
        logger.warning("supervisor LLM call failed (%s); using rule-based routing", exc)
        next_stage = _fallback_next(state)

    logger.info("supervisor routing -> %s", next_stage)
    return {"next_stage": next_stage}
