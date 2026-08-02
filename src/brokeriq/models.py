"""Pydantic models and LangGraph state for the qualification pipeline."""

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class LeadInput(BaseModel):
    """A raw lead as it arrives from any source (form, email, spreadsheet row)."""

    company_name: str = Field(min_length=1, max_length=200)
    domain: str | None = None
    revenue_band: Literal["<1M", "1-5M", "5-20M", "20M+", "unknown"] = "unknown"
    industry: str | None = None
    state: str | None = None
    notes: str = ""


class ResearchReport(BaseModel):
    """Output of the research agent: everything we learned about the company."""

    summary: str = ""
    sources: list[str] = Field(default_factory=list)
    naics_code: str | None = None
    naics_label: str | None = None
    headcount_estimate: str | None = None
    funding_signals: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class CarrierFit(BaseModel):
    """How well the lead fits the lines of business we can place."""

    lines: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class QualificationResult(BaseModel):
    """Scoring output of the qualification agent."""

    icp_score: float = Field(ge=0, le=100)
    carrier_fit: CarrierFit = Field(default_factory=CarrierFit)
    risk_flags: list[str] = Field(default_factory=list)
    verdict: Literal["qualified", "needs_review", "disqualified"] = "needs_review"


class LeadBrief(BaseModel):
    """Final deliverable: a ready-to-use lead brief."""

    headline: str = ""
    summary: str = ""
    outreach_angle: str = ""
    recommended_action: str = ""


class MemoryOp(BaseModel):
    """A typed memory operation the extractor wants to apply to long-term storage."""

    op: Literal["ADD", "UPDATE", "DELETE", "NOOP"]
    namespace: tuple[str, str]
    key: str
    value: dict = Field(default_factory=dict)


class AgentState(TypedDict, total=False):
    """State flowing through the graph. Total=False so nodes only touch what they need."""

    lead: LeadInput
    research: ResearchReport | None
    qualification: QualificationResult | None
    brief: LeadBrief | None
    memory_ops: list[MemoryOp] | None
    next_stage: str
    skip_report: bool
    messages: Annotated[list, add_messages]
    run_id: str
