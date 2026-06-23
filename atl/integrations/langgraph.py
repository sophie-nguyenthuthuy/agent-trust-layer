"""LangGraph / LangChain integration.

Wraps the tools a LangGraph agent can call so every invocation passes through
the trust layer before it executes. No hard dependency on langgraph — the
wrapper works with any callable or LangChain ``BaseTool``; import langgraph in
your own graph and pass the guarded tools to your ``ToolNode``.

Example
-------
    from langgraph.prebuilt import ToolNode
    from atl import TrustLayer
    from atl.integrations.langgraph import guard_tools

    layer = TrustLayer(policy=..., gate=..., hitl=...)
    guarded = guard_tools(layer, [search, write_db, transfer], actor="executor")
    tool_node = ToolNode(guarded)        # drop into your StateGraph

The agent attaches its per-step confidence as a ``_risk`` value (1 - conf).
Provide ``risk_fn`` to pull it out of the tool args/state; default is 0.0.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional

from ..middleware import GateBlocked, TrustLayer
from ..types import ToolCall

RiskFn = Callable[[str, dict], float]


def _tool_name(tool: Any) -> str:
    return getattr(tool, "name", None) or getattr(tool, "__name__", None) or repr(tool)


def _invoke(tool: Any, args: dict) -> Any:
    # LangChain BaseTool exposes .invoke; plain callables are just called.
    if hasattr(tool, "invoke"):
        return tool.invoke(args)
    return tool(**args)


def guard_tool(layer: TrustLayer, tool: Any, *, actor: str = "agent",
               risk_fn: Optional[RiskFn] = None,
               on_block: Optional[Callable[[GateBlocked], Any]] = None) -> Callable:
    """Return a guarded callable wrapping a single tool.

    Args:
        risk_fn: maps (tool_name, args) -> risk in [0, 1]. Default 0.0.
        on_block: handler invoked when the layer refuses the call. Default
            returns a structured refusal dict (so the graph can keep running
            and the model can react) instead of raising.
    """
    name = _tool_name(tool)

    def runner(args: dict) -> Any:
        risk = risk_fn(name, args) if risk_fn else float(args.pop("_risk", 0.0))
        call = ToolCall(tool=name, args=args, actor=actor, risk=risk)
        try:
            return layer.execute(call, lambda **kw: _invoke(tool, kw))
        except GateBlocked as blocked:
            if on_block:
                return on_block(blocked)
            v = blocked.verdict
            return {
                "trust_layer": "refused",
                "decision": v.decision.value,
                "reason": v.reason,
                "tool": name,
            }

    runner.__name__ = name
    return runner


def guard_tools(layer: TrustLayer, tools: Iterable[Any], *, actor: str = "agent",
                risk_fn: Optional[RiskFn] = None) -> List[Callable]:
    """Guard a list of tools for use in a LangGraph ``ToolNode``."""
    return [guard_tool(layer, t, actor=actor, risk_fn=risk_fn) for t in tools]


def guarded_tool_node(layer: TrustLayer, tools: Iterable[Any], *,
                      actor: str = "agent", risk_fn: Optional[RiskFn] = None):
    """Return a LangGraph node that executes tool calls through the trust layer.

    This is the robust integration path (version-stable across langgraph 1.x):
    use it in place of the prebuilt ``ToolNode``. It reads the tool calls off
    the last ``AIMessage`` in ``state["messages"]``, runs each through
    ``layer.guard``, executes only the allowed ones, and returns one
    ``ToolMessage`` per call — refusals included, so the model can react.

        from langgraph.graph import StateGraph, END
        graph.add_node("tools", guarded_tool_node(layer, tools, actor="executor",
                                                  risk_fn=my_risk_fn))

    ``risk_fn(name, args) -> float`` supplies the per-step risk signal; if
    omitted, a ``_risk`` key is popped from the tool-call args (default 0.0).
    """
    from langchain_core.messages import ToolMessage  # local import: optional dep

    by_name = {getattr(t, "name", _tool_name(t)): t for t in tools}

    def node(state: dict) -> dict:
        messages = state["messages"]
        last = messages[-1]
        out = []
        for tc in getattr(last, "tool_calls", []) or []:
            name = tc["name"]
            args = dict(tc.get("args") or {})
            risk = risk_fn(name, args) if risk_fn else float(args.pop("_risk", 0.0))
            call = ToolCall(tool=name, args=args, actor=actor, risk=risk)
            verdict = layer.guard(call)
            if verdict.allowed:
                tool = by_name.get(name)
                result = _invoke(tool, args) if tool is not None else \
                    f"unknown tool {name}"
                content = str(result)
            else:
                content = (f"[TRUST LAYER {verdict.decision.value.upper()}] "
                           f"{verdict.reason}")
            out.append(ToolMessage(content=content, name=name,
                                   tool_call_id=tc.get("id", name)))
        return {"messages": out}

    return node
