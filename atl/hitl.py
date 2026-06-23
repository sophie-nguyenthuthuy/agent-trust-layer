"""Human-in-the-loop escalation.

When a verdict escalates, it lands here. In production this is a durable queue
fronted by a review UI; the core only needs a resolver callable so the layer
stays synchronous and testable. A pending queue is kept for async review.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

from .types import Decision, HumanResolver, Verdict


class HITLQueue:
    def __init__(self, resolver: Optional[HumanResolver] = None,
                 default: Decision = Decision.BLOCK) -> None:
        """
        Args:
            resolver: synchronous decision function (a human, or a stand-in for
                eval/tests). If None, escalations are parked and ``default`` is
                returned (fail-closed by default).
            default: verdict used when no resolver is wired.
        """
        self.resolver = resolver
        self.default = default
        self.pending: Deque[Verdict] = deque()
        self.resolved: List[tuple[Verdict, Decision]] = []

    def escalate(self, verdict: Verdict) -> Decision:
        if self.resolver is None:
            self.pending.append(verdict)
            return self.default
        decision = self.resolver(verdict)
        if decision is Decision.ESCALATE:   # resolver must terminate
            decision = self.default
        self.resolved.append((verdict, decision))
        return decision
