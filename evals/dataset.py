"""Labeled evaluation dataset for the BrokerIQ pipeline.

Each lead carries a human gold label: the verdict a competent analyst would
give. The compliance queries carry the expected doc(s) that should surface.

The offline harness can replay these with the deterministic FakeLLM (which is
told the gold verdict) to validate the harness itself, or run the real model
when BROKERIQ_OPENROUTER_API_KEY / GEMINI_API_KEY / GROQ_API_KEY is set.
"""

from brokeriq.models import LeadInput

# (LeadInput, gold verdict, gold carrier-fit lines)
LEADS: list[tuple[LeadInput, str, list[str]]] = [
    (
        LeadInput(
            company_name="Nimbus Cyber Solutions",
            domain="nimbuscyber.example",
            industry="cybersecurity",
            state="TX",
            revenue_band="1-5M",
        ),
        "qualified",
        ["cyber_liability", "general_liability"],
    ),
    (
        LeadInput(
            company_name="Lone Star Logistics",
            domain="lonestarlogistics.example",
            industry="trucking / freight",
            state="TX",
            revenue_band="5-20M",
        ),
        "qualified",
        ["commercial_auto", "workers_comp", "general_liability"],
    ),
    (
        LeadInput(
            company_name="BrightPath Dental Group",
            domain="brightpathdental.example",
            industry="healthcare",
            state="CA",
            revenue_band="1-5M",
        ),
        "qualified",
        ["professional_liability", "general_liability"],
    ),
    (
        LeadInput(
            company_name="Pinnacle Roofing Co",
            domain="pinnacleroof.example",
            industry="construction",
            state="FL",
            revenue_band="<1M",
        ),
        "qualified",
        ["workers_comp", "general_liability"],
    ),
    (
        LeadInput(
            company_name="QuickCash Pawn & Jewelry",
            domain="quickcashpawn.example",
            industry="pawn shop",
            state="NV",
            revenue_band="<1M",
        ),
        "disqualified",
        [],
    ),
    (
        LeadInput(
            company_name="Downtown Bar & Grill",
            domain="downtownbar.example",
            industry="hospitality",
            state="NY",
            revenue_band="<1M",
        ),
        "needs_review",
        [],
    ),
]

# (compliance query, doc_id(s) that must appear in the top results)
COMPLIANCE_QUERIES: list[tuple[str, list[str]]] = [
    ("do carriers require MFA on email and remote access?", ["cyber-liability-basics"]),
    ("can employers in texas opt out of workers compensation?", ["workers-comp-by-state"]),
    ("is pollution coverage excluded under general liability?", ["commercial-general-liability"]),
    ("what cyber coverage is available for small businesses?", ["cyber-liability-basics"]),
]


def iter_leads():
    """Yield (lead, gold_verdict, gold_lines) tuples for the harness."""
    return list(LEADS)


def iter_compliance():
    """Yield (query, expected_docs) tuples for retrieval evals."""
    return list(COMPLIANCE_QUERIES)
