"""Integration tests for the LangGraph adapter. Skipped if langgraph absent."""
import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage  # noqa: E402

from atl import (  # noqa: E402
    CertifiedGate,
    Decision,
    HITLQueue,
    ProvenanceLog,
    RuleEngine,
    TrustLayer,
    constrain_arg,
)
from atl.integrations.langgraph import guard_tool, guarded_tool_node  # noqa: E402


def _layer():
    return TrustLayer(
        policy=RuleEngine([
            constrain_arg("transfer_funds", "amount", lambda a: a <= 10_000),
        ]),
        gate=CertifiedGate(v_floor=0.3),
        hitl=HITLQueue(resolver=lambda v: Decision.BLOCK),
        provenance=ProvenanceLog(),
    )


def test_guarded_tool_node_blocks_and_returns_toolmessage():
    layer = _layer()

    def transfer_funds(amount, to):
        return f"sent {amount}"

    transfer_funds.name = "transfer_funds"
    node = guarded_tool_node(layer, [transfer_funds], actor="executor")

    last = AIMessage(content="", tool_calls=[{
        "name": "transfer_funds",
        "args": {"amount": 250_000, "to": "x", "_risk": 0.3},
        "id": "c0", "type": "tool_call",
    }])
    out = node({"messages": [last]})
    msg = out["messages"][0]
    assert "TRUST LAYER BLOCK" in msg.content
    assert len(layer.provenance) == 1
    assert layer.provenance.verify()


def test_guarded_tool_node_allows_safe_call():
    layer = _layer()

    def transfer_funds(amount, to):
        return f"sent {amount} to {to}"

    transfer_funds.name = "transfer_funds"
    node = guarded_tool_node(layer, [transfer_funds], actor="executor")

    last = AIMessage(content="", tool_calls=[{
        "name": "transfer_funds",
        "args": {"amount": 50, "to": "y", "_risk": 0.05},
        "id": "c0", "type": "tool_call",
    }])
    out = node({"messages": [last]})
    assert out["messages"][0].content == "sent 50 to y"


def test_guard_tool_callable_returns_refusal_dict():
    layer = _layer()

    def transfer_funds(amount, to):
        return amount

    guarded = guard_tool(layer, transfer_funds, actor="executor")
    res = guarded({"amount": 999_999, "to": "x"})
    assert isinstance(res, dict) and res["trust_layer"] == "refused"
