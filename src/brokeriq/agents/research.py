"""Research agent: gathers company facts via web search + NAICS classification."""

import logging

from ..llm import complete_json
from ..models import ResearchReport
from ..tools import lookup_naics, web_search
from . import prompts

logger = logging.getLogger(__name__)


async def research_node(state: dict) -> dict:
    lead = state["lead"]
    logger.info("research: %s", lead.company_name)

    search_results = await web_search(f"{lead.company_name} company overview revenue employees")
    naics = lookup_naics(lead.company_name, lead.industry)

    sources = [r["url"] for r in search_results if r.get("url")]
    snippet_text = "\n".join(
        f"- {r['title']} ({r['url']}): {r['snippet'][:300]}" for r in search_results[:5]
    )
    naics_text = f"{naics['code']} {naics['label']}" if naics else "none found"

    raw = await complete_json(
        [
            {"role": "system", "content": prompts.RESEARCH},
            {
                "role": "user",
                "content": (
                    f"Company: {lead.company_name}\n"
                    f"Domain: {lead.domain or 'unknown'}\n"
                    f"Reported industry: {lead.industry or 'unknown'}\n"
                    f"NAICS guess: {naics_text}\n"
                    f"Web search results:\n{snippet_text or 'no results'}"
                ),
            },
        ]
    )

    report = ResearchReport.model_validate(raw)
    if naics and not report.naics_code:
        report.naics_code = naics["code"]
        report.naics_label = naics["label"]
    if sources:
        report.sources.extend(sources)

    return {
        "research": report,
        "messages": [
            {
                "role": "assistant",
                "content": f"Research complete for {lead.company_name}: {report.summary}",
            }
        ],
    }
