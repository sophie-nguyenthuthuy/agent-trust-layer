"""Metric helpers for the before/after harness."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


def percentile(values: List[float], p: float) -> float:
    """Nearest-rank percentile. ``p`` in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


@dataclass
class Aggregate:
    label: str
    ttft_p50_ms: float
    ttft_p95_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    hallucination_rate: float       # delivered hallucinations / delivered answers
    tool_misuse_rate: float         # executed misuse calls / executed tool calls
    over_block_rate: float          # correct answers wrongly blocked / correct total
    overhead_ms_mean: float         # mean trust-layer time per request
    n_requests: int = 0
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {
            "label": self.label,
            "ttft_p50_ms": round(self.ttft_p50_ms, 2),
            "ttft_p95_ms": round(self.ttft_p95_ms, 2),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "tool_misuse_rate": round(self.tool_misuse_rate, 4),
            "over_block_rate": round(self.over_block_rate, 4),
            "overhead_ms_mean": round(self.overhead_ms_mean, 4),
            "n_requests": self.n_requests,
        }
