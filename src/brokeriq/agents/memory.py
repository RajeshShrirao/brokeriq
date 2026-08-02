"""Memory extractor: turns run outputs into durable facts for long-term storage."""

import logging

from ..llm import complete_json
from ..models import MemoryOp
from . import prompts

logger = logging.getLogger(__name__)


async def memory_node(state: dict) -> dict:
    lead = state["lead"]
    research = state.get("research")
    qualification = state.get("qualification")
    brief = state.get("brief")
    logger.info("memory extraction: %s", lead.company_name)

    raw = await complete_json(
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
    stored = [op for op in ops if op.op != "NOOP"]
    logger.info("memory ops: %d stored, %d noop", len(stored), len(ops) - len(stored))
    return {"memory_ops": ops}
