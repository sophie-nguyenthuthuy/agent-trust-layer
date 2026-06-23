"""A *real* LangGraph multi-step agent with the trust layer in the tool loop.

This is a genuine ``StateGraph`` (agent -> tools -> agent -> ... -> END) where
the tools node is ``atl.integrations.langgraph.guarded_tool_node`` — every tool
call the agent emits is routed through policy + the certified gate + provenance
before it can execute. Blocked calls come back to the agent as ToolMessages so
the loop can recover instead of crashing.

The "LLM" here is a deterministic scripted model so the demo runs in CI with no
API keys (matching the FakeLLM-in-CI pattern). To drive it with a real OSS
model, replace ``scripted_model()`` with, e.g.::

    from langchain_ollama import ChatOllama
    model = ChatOllama(model="qwen2.5", base_url="http://localhost:11435").bind_tools(TOOLS)

Run:  pip install -e ".[langgraph]" && python examples/langgraph_agent.py
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from atl import (
    CertifiedGate,
    Decision,
    HITLQueue,
    ProvenanceLog,
    RuleEngine,
    TrustLayer,
    constrain_arg,
    deny_tools,
    require_confidence,
)
from atl.integrations.langgraph import guarded_tool_node


# ---- the agent's real tools ----------------------------------------------
@tool
def web_search(q: str) -> str:
    """Search the web for `q`."""
    return f"top results for {q!r}: [invoice#A12 €4,800; invoice#A13 €5,000]"


@tool
def transfer_funds(amount: int, to: str) -> str:
    """Transfer `amount` to account `to`."""
    return f"transferred {amount} to {to}"


@tool
def db_write(op: str, table: str) -> str:
    """Run write `op` on `table`."""
    return f"db {op} on {table} done"


TOOLS = [web_search, transfer_funds, db_write]


# ---- the trust layer ------------------------------------------------------
def build_layer() -> TrustLayer:
    policy = RuleEngine([
        deny_tools("shell_exec"),
        constrain_arg("transfer_funds", "amount", lambda a: a <= 10_000,
                      "amount exceeds 10k review cap"),
        constrain_arg("db_write", "op", lambda op: op not in {"drop", "delete_all"},
                      "destructive db op"),
        require_confidence("transfer_funds", max_risk=0.4),
    ])
    return TrustLayer(
        policy=policy,
        gate=CertifiedGate(baseline=0.15, delta=0.05, window=20, v_floor=0.25),
        hitl=HITLQueue(resolver=lambda v: Decision.BLOCK),
        provenance=ProvenanceLog(key=b"demo-key"),
    )


# ---- a deterministic, scripted "LLM" -------------------------------------
def _ai(text: str, *calls) -> AIMessage:
    tool_calls = [
        {"name": n, "args": a, "id": f"c{i}", "type": "tool_call"}
        for i, (n, a) in enumerate(calls)
    ]
    return AIMessage(content=text, tool_calls=tool_calls)


def scripted_model() -> FakeMessagesListChatModel:
    # _risk is the agent's per-step risk signal (1 - confidence); the guarded
    # node pops it before the real tool runs.
    return FakeMessagesListChatModel(responses=[
        _ai("Let me find the invoices.",
            ("web_search", {"q": "vendor invoices Q2", "_risk": 0.05})),
        _ai("I'll pay the vendor now.",
            ("transfer_funds", {"amount": 250_000, "to": "acct-x", "_risk": 0.30})),
        _ai("The transfer was refused by governance, so I stopped and flagged it "
            "for review. No funds moved."),
    ])


# ---- the graph ------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(layer: TrustLayer):
    model = scripted_model()

    def agent(state: State) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    def route(state: State) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", guarded_tool_node(layer, TOOLS, actor="executor"))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def main() -> None:
    from langchain_core.messages import HumanMessage

    layer = build_layer()
    app = build_graph(layer)
    final = app.invoke({"messages": [HumanMessage(content="Pay the Q2 vendor invoices.")]})

    print("\nLangGraph agent trace (trust layer in the tool loop)\n" + "-" * 64)
    for m in final["messages"]:
        kind = m.__class__.__name__.replace("Message", "")
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                args = {k: v for k, v in tc["args"].items() if k != "_risk"}
                print(f"  {kind:<9} → call {tc['name']}({args})")
        else:
            print(f"  {kind:<9} {m.content}")

    print("\nProvenance (every gated call, signed + chained)\n" + "-" * 64)
    for e in layer.provenance.to_list():
        print(f"  #{e['seq']} {e['actor']:>8} · {e['tool']:<15} -> "
              f"{e['decision'].upper():<8} {e['reason']}")
    print(f"\n  chain intact = {layer.provenance.verify()}\n")


if __name__ == "__main__":
    main()
