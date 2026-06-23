"""Core types shared across the trust layer.

Zero third-party dependencies on purpose: the trust layer must be embeddable
inside any agent stack without dragging a dependency tree along.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional


class Decision(str, Enum):
    """Terminal verdict for a single tool call."""

    ALLOW = "allow"        # execute as requested
    ESCALATE = "escalate"  # route to a human before executing
    BLOCK = "block"        # refuse outright

    @property
    def is_allowed(self) -> bool:
        return self is Decision.ALLOW


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation an agent wants to make."""

    tool: str
    args: Mapping[str, Any] = field(default_factory=dict)
    actor: str = "agent"               # which agent/node in the graph
    # Optional risk signal in [0, 1] the agent attaches to *this* step.
    # 0 = fully confident, 1 = no confidence. Drives the certified gate.
    risk: float = 0.0
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PolicyResult:
    decision: Decision
    rule: str = ""          # which rule fired
    reason: str = ""

    @classmethod
    def allow(cls) -> "PolicyResult":
        return cls(Decision.ALLOW, rule="default-allow")


@dataclass
class Certificate:
    """The evidence behind a gate decision — this is the differentiator.

    The gate does not trip on a hand-tuned threshold; it trips when an
    Azuma-Hoeffding concentration bound on the accumulated risk martingale is
    violated at confidence ``1 - delta``. Every decision carries the numbers
    that justify it, so it can be audited and reproduced.
    """

    tripped: bool
    n: int                  # observations seen
    mean_risk: float        # running mean of the risk signal
    baseline: float         # tolerated baseline risk
    bound: float            # certified Azuma margin added to baseline
    threshold: float        # baseline + bound
    delta: float            # mis-trip probability ceiling
    v_floor: float          # below this we never trip (cold-start guard)

    def as_dict(self) -> dict:
        return {
            "tripped": self.tripped,
            "n": self.n,
            "mean_risk": round(self.mean_risk, 4),
            "baseline": self.baseline,
            "bound": round(self.bound, 4),
            "threshold": round(self.threshold, 4),
            "delta": self.delta,
            "v_floor": self.v_floor,
        }


@dataclass
class Verdict:
    """The full trust-layer decision for one tool call."""

    decision: Decision
    call: ToolCall
    policy: PolicyResult
    certificate: Optional[Certificate] = None
    reason: str = ""
    ts: float = field(default_factory=time.time)

    @property
    def allowed(self) -> bool:
        return self.decision.is_allowed


# A human-in-the-loop resolver: given a Verdict that escalated, return the
# final Decision (ALLOW or BLOCK). In production this is a queue + UI; in
# tests/eval it is a callable so everything stays deterministic.
HumanResolver = Callable[[Verdict], Decision]
