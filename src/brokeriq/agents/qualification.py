"""Qualification agent: ICP scoring + carrier-fit grounded in the compliance corpus."""

import logging

from ..llm import complete_json
from ..models import QualificationResult
from ..tools import compliance_search
from . import prompts

logger = logging.getLogger(__name__)


async def qualification_node(state: dict) -> dict:
    lead = state["lead"]
    research = state.get("research")
    logger.info("qualification: %s", lead.company_name)

    context_lines = []
    if research and research.summary:
        context_lines.append(f"Research summary: {research.summary}")
    if research and research.funding_signals:
        context_lines.append(f"Funding signals: {', '.join(research.funding_signals)}")

    compliance_facts = await compliance_search(
        f"{lead.industry or 'business'} insurance coverage requirements "
        f"{lead.state or ''} workers compensation general liability cyber"
    )
    fact_text = "\n".join(
        f"- {f['citation']}: {f['text'][:400]}" for f in compliance_facts[:5]
    )
    if fact_text:
        context_lines.append(f"Compliance corpus facts:\n{fact_text}")

    raw = await complete_json(
        [
            {"role": "system", "content": prompts.QUALIFICATION},
            {
                "role": "user",
                "content": (
                    f"Company: {lead.company_name}\n"
                    f"Revenue band: {lead.revenue_band}\n"
                    f"State: {lead.state or 'unknown'}\n"
                    f"NAICS: {research.naics_code + ' ' + (research.naics_label or '') if research and research.naics_code else 'unknown'}\n"
                    + "\n".join(context_lines)
                ),
            },
        ]
    )

    result = QualificationResult.model_validate(raw)
    logger.info("qualification verdict=%s score=%s", result.verdict, result.icp_score)
    return {
        "qualification": result,
        "messages": [
            {
                "role": "assistant",
                "content": f"Qualification for {lead.company_name}: {result.verdict} (score {result.icp_score:.0f}).",
            }
        ],
    }
