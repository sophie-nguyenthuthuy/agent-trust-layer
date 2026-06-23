"""Certified confidence/drift gate — the part nobody else ships.

Most "guardrail" products gate on a threshold someone eyeballed. This gate
trips only when an Azuma-Hoeffding concentration bound on the running risk
signal is violated, so every BLOCK/ESCALATE carries a certificate stating the
confidence with which the drift is real rather than noise.

Math (Hoeffding for bounded increments r_t in [0, 1] over a window of size n):

    P( mean_emp - mean_true >= eps ) <= exp(-2 n eps^2)

Set the right-hand side to ``delta`` and solve for the margin:

    eps(n) = sqrt( ln(1/delta) / (2 n) )

We declare drift (trip the gate) when

    mean_emp >= baseline + eps(n)      (and mean_emp >= v_floor)

i.e. the observed mean exceeds the tolerated baseline by more than statistical
noise could explain at confidence ``1 - delta``. ``v_floor`` is a cold-start
guard: while risk is genuinely low we never trip, regardless of n.

This mirrors the lyapguard / lyapmon design (Azuma drift test + v_floor),
distilled to a single embeddable class with no dependencies.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque

from .types import Certificate


class CertifiedGate:
    def __init__(
        self,
        baseline: float = 0.15,
        delta: float = 0.05,
        window: int = 50,
        v_floor: float = 0.20,
        block_factor: float = 2.0,
    ) -> None:
        """
        Args:
            baseline: tolerated long-run mean risk in [0, 1].
            delta: ceiling on the probability of a false trip (lower = stricter
                evidence required). The certificate is valid at ``1 - delta``.
            window: number of recent observations the bound is computed over.
                Bounds responsiveness vs. statistical power.
            v_floor: while the running mean is below this, never trip.
            block_factor: if the mean exceeds ``threshold * block_factor`` the
                gate escalates to BLOCK instead of ESCALATE.
        """
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if not 0 <= baseline <= 1:
            raise ValueError("baseline must be in [0, 1]")
        self.baseline = baseline
        self.delta = delta
        self.window = window
        self.v_floor = v_floor
        self.block_factor = block_factor
        self._buf: Deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._buf.clear()

    @property
    def n(self) -> int:
        return len(self._buf)

    def _margin(self, n: int) -> float:
        if n <= 0:
            return float("inf")
        return math.sqrt(math.log(1.0 / self.delta) / (2.0 * n))

    def observe(self, risk: float) -> Certificate:
        """Record one risk observation and return the current certificate."""
        risk = min(1.0, max(0.0, float(risk)))
        self._buf.append(risk)
        return self.certify()

    def certify(self) -> Certificate:
        n = self.n
        mean = sum(self._buf) / n if n else 0.0
        bound = self._margin(n)
        threshold = self.baseline + bound
        tripped = (mean >= threshold) and (mean >= self.v_floor)
        return Certificate(
            tripped=tripped,
            n=n,
            mean_risk=mean,
            baseline=self.baseline,
            bound=bound,
            threshold=threshold,
            delta=self.delta,
            v_floor=self.v_floor,
        )

    def severity(self, cert: Certificate) -> str:
        """Map a tripped certificate to 'block' or 'escalate'."""
        if not cert.tripped:
            return "allow"
        if cert.mean_risk >= cert.threshold * self.block_factor:
            return "block"
        return "escalate"
