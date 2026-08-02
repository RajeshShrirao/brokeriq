"""CLI entrypoint: run the qualification pipeline on a single lead."""

import argparse
import asyncio
import json

from .logging import setup_logging
from .models import LeadInput


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="brokeriq", description="Lead qualification agent")
    parser.add_argument("company", help="company name to qualify")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--revenue-band", default="unknown", choices=["<1M", "1-5M", "5-20M", "20M+", "unknown"])
    parser.add_argument("--industry", default=None)
    parser.add_argument("--state", default=None)
    parser.add_argument("--offline", action="store_true", help="run without any LLM API key")
    return parser.parse_args()


async def _main() -> None:
    setup_logging()
    args = _parse_args()
    lead = LeadInput(
        company_name=args.company,
        domain=args.domain,
        revenue_band=args.revenue_band,
        industry=args.industry,
        state=args.state,
    )

    if args.offline:
        from .graph import run_offline

        steps = await run_offline(lead)
        print(json.dumps({"lead": lead.company_name, "steps": steps}, indent=2))
        return

    from langgraph.checkpoint.sqlite import SqliteSaver

    from .graph import build_graph

    async with SqliteSaver.from_conn_string("brokeriq.db") as checkpointer:
        app = await build_graph(checkpointer=checkpointer)
        result = await app.ainvoke({"lead": lead, "run_id": "cli"})
        for message in result.get("messages", []):
            print(f"[{message['role']}] {message['content']}")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
