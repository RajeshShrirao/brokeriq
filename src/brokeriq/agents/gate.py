"""Compliance gate: human-in-the-loop checkpoint for risky verdicts.

A lead flagged as disqualified or needing review pauses here. A human
approves, adjusts the score, or hard-disqualifies. Approvals flow straight
to the report writer; hard disqualifications short-circuit the pipeline.
"""

import logging

from langgraph.types import interrupt

logger = logging.getLogger(__name__)


def _interrupt_payload(state: dict) -> dict:
    lead = state["lead"]
    qual = state["qualification"]
    return {
        "type": "lead_review",
        "company": lead.company_name,
        "icp_score": qual.icp_score,
        "verdict": qual.verdict,
        "risk_flags": qual.risk_flags,
        "carrier_lines": qual.carrier_fit.lines,
        "question": (
            "Approve this lead for a broker brief, adjust the score, or "
            "disqualify it outright?"
        ),
    }


def compliance_gate_node(state: dict) -> dict:
    qual = state["qualification"]

    if qual.verdict == "qualified":
        return {}  # no human needed

    payload = _interrupt_payload(state)
    logger.warning("human review required for %s (%s)", state["lead"].company_name, qual.verdict)

    # Pauses execution; a human resumes via Command(resume=...)
    decision = interrupt(payload)

    action = (decision or {}).get("action", "approve")
    if action == "disqualify":
        adjusted = qual.model_copy(update={"verdict": "disqualified"})
        logger.info("human disqualified %s", state["lead"].company_name)
        return {"qualification": adjusted, "skip_report": True}
    if action == "adjust":
        score = float((decision or {}).get("icp_score", qual.icp_score))
        adjusted = qual.model_copy(
            update={"icp_score": max(0.0, min(100.0, score)), "verdict": "qualified"}
        )
        logger.info("human adjusted %s score to %.0f", state["lead"].company_name, score)
        return {"qualification": adjusted}

    logger.info("human approved %s", state["lead"].company_name)
    return {"qualification": qual.model_copy(update={"verdict": "qualified"})}
