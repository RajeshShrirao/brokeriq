"""Graph assembly.

For now this is a deliberately minimal two-node graph used to validate the
checkpointer wiring and the CLI. The real supervisor/specialist layout lands in
`agents/` as the pipeline grows.
"""

import logging

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .llm import complete
from .models import AgentState, LeadInput

logger = logging.getLogger(__name__)


async def _greet(state: AgentState) -> dict:
    """Node 1: acknowledge the lead and confirm we parsed it."""
    lead: LeadInput = state["lead"]
    logger.info("received lead: %s (%s)", lead.company_name, lead.industry or "unknown industry")
    return {"messages": [{"role": "assistant", "content": f"Analyzing {lead.company_name}..."}]}


async def _summarize(state: AgentState) -> dict:
    """Node 2: ask the LLM to summarize what a qualification run will do."""
    lead = state["lead"]
    text = await complete(
        messages=[
            {
                "role": "system",
                "content": "You are the planner for a lead-qualification pipeline.",
            },
            {
                "role": "user",
                "content": (
                    f"Company: {lead.company_name}\n"
                    f"Domain: {lead.domain or 'unknown'}\n"
                    f"Revenue band: {lead.revenue_band}\n"
                    "In one sentence, state what facts would determine whether this "
                    "company is worth pursuing as a commercial-insurance prospect."
                ),
            },
        ]
    )
    return {"messages": [{"role": "assistant", "content": text}]}


async def build_graph(checkpointer: SqliteSaver | None = None):
    graph = StateGraph(AgentState)
    graph.add_node("greet", _greet)
    graph.add_node("summarize", _summarize)
    graph.add_edge(START, "greet")
    graph.add_edge("greet", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile(checkpointer=checkpointer)


async def run_offline(lead: LeadInput) -> list[str]:
    """Deterministic path for local/dev runs without any API key."""
    return [
        f"received lead: {lead.company_name}",
        f"planned qualification of {lead.company_name} (offline mode)",
    ]
