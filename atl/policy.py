"""Policy-as-code port + a zero-dep built-in engine.

The trust layer does NOT try to out-build OPA/Cedar. ``PolicyEngine`` is a
protocol; the built-in ``RuleEngine`` covers the common cases (deny-lists,
arg constraints, per-tool confidence floors) and an OPA/Cedar adapter can be
dropped in by implementing the same ``evaluate`` method.
"""
from __future__ import annotations

from typing import Callable, List, Protocol, runtime_checkable

from .types import Decision, PolicyResult, ToolCall

# A rule: returns a PolicyResult if it fires, else None to fall through.
Rule = Callable[[ToolCall], "PolicyResult | None"]


@runtime_checkable
class PolicyEngine(Protocol):
    def evaluate(self, call: ToolCall) -> PolicyResult: ...


class RuleEngine:
    """First-match-wins rule evaluation. Default verdict is ALLOW."""

    def __init__(self, rules: "List[Rule] | None" = None) -> None:
        self.rules: List[Rule] = list(rules or [])

    def add(self, rule: Rule) -> "RuleEngine":
        self.rules.append(rule)
        return self

    def evaluate(self, call: ToolCall) -> PolicyResult:
        for rule in self.rules:
            result = rule(call)
            if result is not None:
                return result
        return PolicyResult.allow()


# ---- Rule factories -------------------------------------------------------

def deny_tools(*tools: str, reason: str = "tool on deny-list") -> Rule:
    blocked = set(tools)

    def rule(call: ToolCall) -> "PolicyResult | None":
        if call.tool in blocked:
            return PolicyResult(Decision.BLOCK, rule="deny_tools", reason=reason)
        return None

    return rule


def require_confidence(tool: str, max_risk: float) -> Rule:
    """High-stakes tools may only run when the per-step risk is low enough."""

    def rule(call: ToolCall) -> "PolicyResult | None":
        if call.tool == tool and call.risk > max_risk:
            return PolicyResult(
                Decision.ESCALATE,
                rule="require_confidence",
                reason=f"risk {call.risk:.2f} > {max_risk:.2f} for {tool}",
            )
        return None

    return rule


def constrain_arg(tool: str, arg: str, predicate: Callable[[object], bool],
                  reason: str = "argument constraint violated") -> Rule:
    """Block ``tool`` when ``arg`` fails ``predicate`` (e.g. amount caps)."""

    def rule(call: ToolCall) -> "PolicyResult | None":
        if call.tool == tool and arg in call.args:
            if not predicate(call.args[arg]):
                return PolicyResult(Decision.BLOCK, rule="constrain_arg", reason=reason)
        return None

    return rule


def allow_tools(*tools: str) -> Rule:
    """Allow-list: anything not listed is blocked. Place this LAST."""
    allowed = set(tools)

    def rule(call: ToolCall) -> "PolicyResult | None":
        if call.tool not in allowed:
            return PolicyResult(
                Decision.BLOCK, rule="allow_tools",
                reason=f"{call.tool} not in allow-list",
            )
        return None

    return rule
