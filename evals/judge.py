"""Judges for the BrokerIQ evals.

Two tiers:
1. Deterministic checks — always run, no LLM needed. They catch plumbing
   failures: verdict reachable, brief produced for qualified leads, ICP
   score in range, sources present.
2. LLM judge (Ragas-style) — gated on a configured provider key. Rates
   faithfulness of the brief against the research summary and answer
   relevance to the lead. Skipped cleanly when no key exists so the
   harness never hard-fails offline.
"""

from __future__ import annotations

import os
from typing import Any

from . import dataset

# Provider keys recognised by brokeriq.llm
_PROVIDER_KEYS = ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY")


def has_llm() -> bool:
    return any(os.environ.get(k) for k in _PROVIDER_KEYS)


def _verdict_ok(rec: dict, gold: str) -> bool:
    return rec.get("verdict") == gold


def _brief_ok(rec: dict, gold: str) -> bool:
    """Qualified leads must produce a complete brief; disqualified must not.

    needs_review leads are interrupted at the HITL gate waiting for a human,
    so they legitimately have no brief yet — accept either state.
    """
    if gold == "disqualified":
        return rec.get("brief_headline") is None
    if gold == "needs_review":
        return True
    return all(
        rec.get(field)
        for field in ("brief_headline", "brief_summary", "outreach_angle", "recommended_action")
    )


def _score_in_range(rec: dict) -> bool:
    score = rec.get("icp_score")
    return score is not None and 0 <= score <= 100


def _has_sources(rec: dict) -> bool:
    return bool(rec.get("sources"))


def deterministic_score(records: list[dict]) -> dict[str, Any]:
    """Run the rule-based checks against the gold labels."""
    checks = {
        "verdict_accuracy": (_verdict_ok, []),
        "brief_correctness": (_brief_ok, []),
        "icp_score_in_range": (_score_in_range, []),
        "has_sources": (_has_sources, []),
    }
    results: dict[str, list[bool]] = {name: [] for name in checks}
    for (_, gold, _), rec in zip(dataset.iter_leads(), records):
        results["verdict_accuracy"].append(_verdict_ok(rec, gold))
        results["brief_correctness"].append(_brief_ok(rec, gold))
        results["icp_score_in_range"].append(_score_in_range(rec))
        results["has_sources"].append(_has_sources(rec))

    summary = {
        name: {"pass": sum(vals), "total": len(vals), "rate": sum(vals) / len(vals) if vals else 0.0}
        for name, vals in results.items()
    }
    summary["overall"] = {
        "pass": sum(sum(vals) for vals in results.values()),
        "total": sum(len(vals) for vals in results.values()),
        "rate": sum(sum(vals) for vals in results.values())
        / sum(len(vals) for vals in results.values()),
    }
    return summary


def compliance_retrieval_score() -> dict[str, Any]:
    """Precision@k against the sample corpus (no LLM).

    Runs the real hybrid RAG stack: in-memory Qdrant + MiniLM embedder +
    reranker, exactly like production. Verifies each gold query surfaces its
    expected doc in the top 5.
    """
    import asyncio

    from brokeriq.tools.compliance_rag import compliance_search

    async def _run() -> dict[str, Any]:
        hits_per_query: list[list[str]] = []
        for query, expected in dataset.iter_compliance():
            hits = await compliance_search(query, limit=5)
            docs = [h["doc_id"] for h in hits]
            hits_per_query.append(docs)

        per_query = []
        for (query, expected), docs in zip(dataset.iter_compliance(), hits_per_query):
            hit = any(doc in docs for doc in expected)
            per_query.append({"query": query, "expected": expected, "found": docs, "hit": hit})
        return {"queries": per_query, "hit_rate": sum(q["hit"] for q in per_query) / len(per_query)}

    return asyncio.run(_run())


async def _llm_judge_prompt(kind: str, **kw: str) -> str:
    """Compose a judge prompt for one metric."""
    if kind == "faithfulness":
        return (
            "Rate on a 0-1 scale how faithfully the summary stays grounded in the research, "
            "with no invented facts. Respond ONLY with a JSON object like "
            '{"score": 0.9, "reason": "..."}.\n\n'
            f"RESEARCH: {kw['research']}\nSUMMARY: {kw['summary']}"
        )
    if kind == "relevance":
        return (
            "Rate on a 0-1 scale how relevant this outreach angle is for the company profile. "
            "Respond ONLY with a JSON object like "
            '{"score": 0.9, "reason": "..."}.\n\n'
            f"COMPANY: {kw['company']}\nANGLE: {kw['angle']}"
        )
    raise ValueError(kind)


async def llm_judge_score(records: list[dict]) -> dict[str, Any]:
    """Ragas-style LLM judgements; must be called only when has_llm()."""
    from brokeriq import llm as llm_module

    scored = []
    for rec in records:
        faithfulness = None
        relevance = None
        if rec.get("brief_summary") and rec.get("research_summary"):
            try:
                resp = await llm_module.complete_json(
                    [
                        {"role": "system", "content": "You are a strict evaluation judge."},
                        {
                            "role": "user",
                            "content": await _llm_judge_prompt(
                                "faithfulness",
                                research=rec["research_summary"],
                                summary=rec["brief_summary"],
                            ),
                        },
                    ]
                )
                faithfulness = float(resp.get("score", 0.0))
            except Exception:
                faithfulness = None
        if rec.get("outreach_angle"):
            try:
                resp = await llm_module.complete_json(
                    [
                        {"role": "system", "content": "You are a strict evaluation judge."},
                        {
                            "role": "user",
                            "content": await _llm_judge_prompt(
                                "relevance",
                                company=rec["company"],
                                angle=rec["outreach_angle"],
                            ),
                        },
                    ]
                )
                relevance = float(resp.get("score", 0.0))
            except Exception:
                relevance = None

        scored.append(
            {
                "company": rec["company"],
                "faithfulness": faithfulness,
                "relevance": relevance,
            }
        )
    return {"mode": "llm", "per_lead": scored}
