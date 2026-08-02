import pytest

from brokeriq.graph import build_graph, run_offline
from brokeriq.models import LeadInput


@pytest.mark.asyncio
async def test_offline_run_is_deterministic():
    lead = LeadInput(company_name="Acme Widgets", industry="manufacturing")
    steps = await run_offline(lead)
    assert steps[0] == "received lead: Acme Widgets"
    assert len(steps) == 2


@pytest.mark.asyncio
async def test_graph_compiles_with_inmemory_checkpointer():
    from langgraph.checkpoint.memory import InMemorySaver

    app = await build_graph(checkpointer=InMemorySaver())
    assert app is not None
