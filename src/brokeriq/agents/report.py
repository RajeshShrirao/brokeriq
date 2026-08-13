"""Report writer: produces the final actionable lead brief."""

import logging

from .. import llm
from ..models import LeadBrief
from . import prompts

logger = logging.getLogger(__name__)


async def report_node(state: dict) -> dict:
    lead = state["lead"]
    research = state.get("research")
    qualification = state.get("qualification")
    logger.info("report: %s", lead.company_name)

    raw, usage = await llm.complete_json(
        [
            {"role": "system", "content": prompts.REPORT},
            {
                "role": "user",
                "content": (
                    f"Company: {lead.company_name}\n"
                    f"Research: {research.summary if research else 'n/a'}\n"
                    f"Scores: ICP {qualification.icp_score:.0f}/100, "
                    f"carrier-fit confidence {qualification.carrier_fit.confidence:.2f}\n"
                    f"Verdict: {qualification.verdict}\n"
                    f"Carrier lines: {', '.join(qualification.carrier_fit.lines) or 'none'}\n"
                    f"Risk flags: {', '.join(qualification.risk_flags) or 'none'}"
                ),
            },
        ]
    )

    brief = LeadBrief.model_validate(raw)
    existing = state.get("usage_metadata") or {"total_tokens": 0, "cost": 0.0}
    updated = {
        "total_tokens": existing.get("total_tokens", 0) + (usage or {}).get("total_tokens", 0),
        "cost": existing.get("cost", 0.0) + (usage or {}).get("cost", 0.0),
    }
    return {
        "brief": brief,
        "completed_stages": {"report"},
        "messages": [{"role": "assistant", "content": f"Brief ready: {brief.headline}"}],
        "usage_metadata": updated,
    }
