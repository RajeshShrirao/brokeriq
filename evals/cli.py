"""Evals entrypoint.

Usage:
    python -m evals.cli --mode offline            # FakeLLM, no keys (default)
    python -m evals.cli --mode live               # real LLM (needs a provider key)
    python -m evals.cli --mode offline --json out.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json

from . import judge, runner


def _pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(prog="brokeriq-evals", description="Run BrokerIQ evals.")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--json", default=None, help="write raw run records to this JSONL path")
    parser.add_argument("--skip-retrieval", action="store_true", help="skip compliance retrieval eval")
    args = parser.parse_args()

    print(f"=== BrokerIQ evals (mode={args.mode}) ===")
    records = asyncio.run(runner.run_leads(args.mode))
    if args.json:
        runner.write_jsonl(records, args.json)
        print(f"wrote {len(records)} run records -> {args.json}")

    det = judge.deterministic_score(records)
    print("\n-- deterministic checks (no LLM) --")
    for name, stats in det.items():
        if name == "overall":
            continue
        print(f"  {name:<22} {stats['pass']}/{stats['total']}  {_pct(stats['rate'])}")
    print(f"  {'overall':<22} {det['overall']['pass']}/{det['overall']['total']}  "
          f"{_pct(det['overall']['rate'])}")

    if not args.skip_retrieval:
        rag = judge.compliance_retrieval_score()
        print("\n-- compliance retrieval (precision@5) --")
        for q in rag["queries"]:
            mark = "HIT " if q["hit"] else "MISS"
            print(f"  [{mark}] {q['query']}")
            print(f"         expected={q['expected']} found={q['found'][:3]}")
        print(f"  hit_rate {_pct(rag['hit_rate'])}")

    if args.mode == "live" and judge.has_llm():
        llm_scores = asyncio.run(judge.llm_judge_score(records))
        print("\n-- LLM judge (faithfulness / relevance) --")
        for row in llm_scores["per_lead"]:
            print(f"  {row['company']:<28} faithfulness={row['faithfulness']} relevance={row['relevance']}")
    elif args.mode == "live":
        print("\n-- LLM judge: skipped (no provider key in env) --")

    print("\nDone.")


if __name__ == "__main__":
    main()
