"""Agent Trust & Governance Layer (atl).

A dependency-free middleware that sits in front of an agent's tool-use:
policy-as-code + tamper-evident provenance + a *certified* drift/confidence
gate + human-in-the-loop escalation.

The differentiator is the gate: it trips on an Azuma-Hoeffding bound, not a
hand-tuned threshold, so every decision carries an auditable certificate.
"""
from .gate import CertifiedGate
from .hitl import HITLQueue
from .middleware import GateBlocked, TrustLayer
from .policy import (
    RuleEngine,
    allow_tools,
    constrain_arg,
    deny_tools,
    require_confidence,
)
from .provenance import ProvenanceLog
from .types import Certificate, Decision, PolicyResult, ToolCall, Verdict

__version__ = "0.1.0"

__all__ = [
    "TrustLayer",
    "GateBlocked",
    "CertifiedGate",
    "RuleEngine",
    "deny_tools",
    "allow_tools",
    "require_confidence",
    "constrain_arg",
    "ProvenanceLog",
    "HITLQueue",
    "ToolCall",
    "Verdict",
    "Decision",
    "Certificate",
    "PolicyResult",
]
