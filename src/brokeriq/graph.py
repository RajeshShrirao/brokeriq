"""Graph assembly: supervisor routing over specialist agents with a HITL gate.

Flow:
    START -> supervisor -> research/qualification/report/memory -> ... -> END
    qualification -> gate (human-in-the-loop interrupt on risky verdicts)

The supervisor is LLM-routed with a deterministic rule fallback, so the graph
runs end-to-end even with no model configured (offline/demo mode).
"""

import logging

from langgraph.graph import END, START, StateGraph

from .agents.gate import compliance_gate_node
from .agents.memory import memory_node
from .agents.qualification import qualification_node
from .agents.report import report_node
from .agents.research import research_node
from .agents.supervisor import supervisor_node
from .models import AgentState

logger = logging.getLogger(__name__)

SUPERVISOR_ROUTES = {
    "research": "research",
    "qualification": "qualification",
    "report": "report",
    "memory": "memory",
    "done": END,
}


def _route_supervisor(state: AgentState) -> str:
    return state.get("next_stage", "done")


def _route_gate(state: AgentState) -> str:
    # hard-disqualified leads skip the report writer but still get memory extraction
    return "memory" if state.get("skip_report") else "supervisor"


def build_graph(checkpointer=None, store=None):
    """Assemble and compile the qualification pipeline.

    checkpointer: LangGraph BaseCheckpointSaver (SqliteSaver for dev,
        AsyncPostgresSaver for prod). A missing checkpointer disables
        persistence and human-in-the-loop resume.
    store: optional BaseStore for long-term memory.
    """
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_node)
    graph.add_node("qualification", qualification_node)
    graph.add_node("gate", compliance_gate_node)
    graph.add_node("report", report_node)
    graph.add_node("memory", memory_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", _route_supervisor, SUPERVISOR_ROUTES)
    graph.add_edge("research", "supervisor")
    graph.add_edge("qualification", "gate")
    graph.add_conditional_edges("gate", _route_gate, {"supervisor": "supervisor", "memory": "memory"})
    graph.add_edge("report", "supervisor")
    graph.add_edge("memory", END)

    return graph.compile(checkpointer=checkpointer, store=store)
