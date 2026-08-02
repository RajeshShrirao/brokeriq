"""CLI entrypoint: run the qualification pipeline on a single lead."""

import argparse
import asyncio
import json
import uuid

from .logging import setup_logging
from .models import LeadInput


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="brokeriq",
        description="Autonomous lead qualification for independent insurance brokers.",
    )
    parser.add_argument("company", help="company name of the lead")
    parser.add_argument("--domain", default=None, help="company website domain")
    parser.add_argument("--industry", default=None, help="reported industry")
    parser.add_argument("--state", default=None, help="US state abbreviation")
    parser.add_argument(
        "--revenue-band",
        choices=["<1M", "1-5M", "5-20M", "20M+", "unknown"],
        default="unknown",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="run with the deterministic fake LLM (no API key needed)",
    )
    parser.add_argument("--db", default="brokeriq.db", help="sqlite checkpoint path")
    parser.add_argument("--json", action="store_true", help="print machine-readable result")
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> dict:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from . import llm as llm_module
    from .graph import build_graph

    if args.offline:
        from .fake import FakeLLM

        fake = FakeLLM()
        llm_module.complete = fake.complete
        llm_module.complete_json = fake.complete_json

    lead = LeadInput(
        company_name=args.company,
        domain=args.domain,
        industry=args.industry,
        state=args.state,
        revenue_band=args.revenue_band,
    )
    run_id = uuid.uuid4().hex[:12]
    config = {"configurable": {"thread_id": f"cli-{run_id}"}}

    async with AsyncSqliteSaver.from_conn_string(args.db) as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        return await app.ainvoke({"lead": lead, "run_id": run_id}, config=config)

def main() -> None:
    args = _parse_args()
    setup_logging()
    result = asyncio.run(_run(args))

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    lead = result["lead"]
    research = result.get("research")
    qualification = result.get("qualification")
    brief = result.get("brief")

    print(f"\n=== BrokerIQ run: {lead.company_name} ===")
    if research:
        print(f"\nResearch: {research.summary}")
        if research.naics_code:
            print(f"NAICS: {research.naics_code} {research.naics_label or ''}")
        if research.sources:
            print("Sources:")
            for src in research.sources[:3]:
                print(f"  - {src}")
    if qualification:
        print(f"\nVerdict: {qualification.verdict} (ICP score {qualification.icp_score:.0f}/100)")
        print(f"Carrier fit: {', '.join(qualification.carrier_fit.lines) or 'none'}")
        if qualification.risk_flags:
            print(f"Risk flags: {', '.join(qualification.risk_flags)}")
    if brief:
        print(f"\nBrief: {brief.headline}")
        print(f"  {brief.summary}")
        print(f"  Angle: {brief.outreach_angle}")
        print(f"  Action: {brief.recommended_action}")
    if not brief and qualification and qualification.verdict == "disqualified":
        print("\nLead disqualified — no brief generated.")


if __name__ == "__main__":
    main()
