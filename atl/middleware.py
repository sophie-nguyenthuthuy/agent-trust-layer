"""TrustLayer — the interceptor that sits in front of agent tool-use.

Pipeline per tool call:

    policy.evaluate ──► provenance.record ──► gate.observe ──► {ALLOW | ESCALATE→HITL | BLOCK}

Policy is a hard gate (deterministic rules). The certified gate adds a
statistical gate on accumulated risk. Whichever is stricter wins. Every
decision is logged to the tamper-evident ledger before the tool can run.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .gate import CertifiedGate
from .hitl import HITLQueue
from .policy import PolicyEngine, RuleEngine
from .provenance import ProvenanceLog
from .types import Decision, PolicyResult, ToolCall, Verdict


class GateBlocked(Exception):
    """Raised by ``execute`` when a call is not permitted."""

    def __init__(self, verdict: Verdict) -> None:
        self.verdict = verdict
        super().__init__(f"{verdict.decision.value}: {verdict.reason}")


class TrustLayer:
    def __init__(
        self,
        policy: Optional[PolicyEngine] = None,
        gate: Optional[CertifiedGate] = None,
        provenance: Optional[ProvenanceLog] = None,
        hitl: Optional[HITLQueue] = None,
        enabled: bool = True,
    ) -> None:
        # Explicit None checks, not `or`: ProvenanceLog.__len__ is 0 when empty,
        # which makes a freshly-passed log falsy and would silently drop it.
        self.policy = policy if policy is not None else RuleEngine()
        self.gate = gate if gate is not None else CertifiedGate()
        self.provenance = provenance if provenance is not None else ProvenanceLog()
        self.hitl = hitl if hitl is not None else HITLQueue()
        # When disabled the layer is a pass-through — used to measure the
        # before/after delta in the eval harness without changing call sites.
        self.enabled = enabled

    # -- core decision ------------------------------------------------------
    def guard(self, call: ToolCall) -> Verdict:
        if not self.enabled:
            # Pass-through: nothing is inspected, logged, or gated. This is the
            # "layer off" baseline the eval harness measures against.
            return Verdict(Decision.ALLOW, call, PolicyResult.allow(),
                           reason="layer-disabled")

        pol = self.policy.evaluate(call)
        cert = self.gate.observe(call.risk)
        gate_sev = self.gate.severity(cert)

        # Combine: take the strictest of policy and gate.
        decision = pol.decision
        reason = f"policy:{pol.rule}"
        order = {Decision.ALLOW: 0, Decision.ESCALATE: 1, Decision.BLOCK: 2}
        gate_decision = {
            "allow": Decision.ALLOW,
            "escalate": Decision.ESCALATE,
            "block": Decision.BLOCK,
        }[gate_sev]
        if order[gate_decision] > order[decision]:
            decision = gate_decision
            reason = (f"gate:drift mean={cert.mean_risk:.2f}"
                      f">thr={cert.threshold:.2f}@{1 - cert.delta:.0%}")
        elif pol.decision is not Decision.ALLOW:
            reason = f"policy:{pol.rule} ({pol.reason})"

        verdict = Verdict(decision, call, pol, certificate=cert, reason=reason)

        # Escalations route to a human; the resolver returns a terminal verdict.
        if verdict.decision is Decision.ESCALATE:
            final = self.hitl.escalate(verdict)
            verdict = Verdict(final, call, pol, certificate=cert,
                              reason=f"hitl:{final.value} <- {reason}")

        self.provenance.record(verdict)
        return verdict

    # -- convenience executor ----------------------------------------------
    def execute(self, call: ToolCall, fn: Callable[..., Any]) -> Any:
        """Guard then run ``fn(**call.args)`` iff allowed; else raise."""
        verdict = self.guard(call)
        if not verdict.allowed:
            raise GateBlocked(verdict)
        return fn(**call.args)

    def wrap_tool(self, name: str, fn: Callable[..., Any],
                  actor: str = "agent") -> Callable[..., Any]:
        """Return a drop-in replacement for ``fn`` that is gated.

        The wrapped tool accepts an optional ``_risk`` kwarg carrying the
        agent's per-step risk signal; it is stripped before calling ``fn``.
        """
        def wrapped(*, _risk: float = 0.0, **kwargs: Any) -> Any:
            call = ToolCall(tool=name, args=kwargs, actor=actor, risk=_risk)
            return self.execute(call, fn)

        wrapped.__name__ = name
        return wrapped
