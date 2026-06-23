"""End-to-end demo: a tiny multi-agent loop guarded by the trust layer.

Runs without any LLM or network. It shows, on a handful of concrete calls:
  - a benign tool call passing,
  - an out-of-policy transfer being BLOCKED,
  - a low-confidence final answer ESCALATING to a human (who stops it),
  - sustained drift tripping the certified gate, and
  - the tamper-evident provenance chain verifying.

    python examples/multi_agent_demo.py
"""
from atl import (
    CertifiedGate,
    Decision,
    HITLQueue,
    ProvenanceLog,
    RuleEngine,
    ToolCall,
    TrustLayer,
    constrain_arg,
    deny_tools,
    require_confidence,
)


def build() -> TrustLayer:
    policy = RuleEngine([
        deny_tools("shell_exec"),
        constrain_arg("transfer_funds", "amount", lambda a: a <= 10_000,
                      "amount exceeds 10k review cap"),
        constrain_arg("db_write", "op", lambda op: op not in {"drop", "delete_all"},
                      "destructive db op"),
        require_confidence("respond", max_risk=0.4),
    ])
    hitl = HITLQueue(resolver=lambda v: Decision.BLOCK)  # reviewer stops risky acts
    gate = CertifiedGate(baseline=0.15, delta=0.05, window=20, v_floor=0.25)
    return TrustLayer(policy=policy, gate=gate, provenance=ProvenanceLog(),
                      hitl=hitl)


def show(layer: TrustLayer, call: ToolCall) -> None:
    v = layer.guard(call)
    cert = v.certificate
    tail = f"  [gate mean={cert.mean_risk:.2f} thr={cert.threshold:.2f}]" if cert else ""
    print(f"  {call.actor:>9} · {call.tool:<15} risk={call.risk:.2f}  ->  "
          f"{v.decision.value.upper():<8} {v.reason}{tail}")


def main() -> None:
    layer = build()
    print("\nMulti-agent run (trust layer ON)\n" + "-" * 64)

    show(layer, ToolCall("web_search", {"q": "vendor invoices"}, "researcher", 0.05))
    show(layer, ToolCall("db_read", {"table": "invoices"}, "researcher", 0.08))
    show(layer, ToolCall("transfer_funds", {"amount": 250_000, "to": "x"},
                         "executor", 0.30))            # blocked by policy
    show(layer, ToolCall("db_write", {"op": "drop", "table": "users"},
                         "executor", 0.20))            # blocked by policy
    show(layer, ToolCall("respond", {}, "responder", 0.85))   # escalated -> blocked

    print("\nInject sustained drift (agent confidence decaying)\n" + "-" * 64)
    for i in range(22):
        v = layer.guard(ToolCall("web_search", {"q": f"step{i}"}, "researcher", 0.7))
        if v.certificate and v.certificate.tripped:
            print(f"  certified gate TRIPPED at n={v.certificate.n}: "
                  f"mean={v.certificate.mean_risk:.2f} > thr="
                  f"{v.certificate.threshold:.2f} @ {1 - v.certificate.delta:.0%} "
                  f"-> {v.decision.value.upper()}")
            break

    print("\nProvenance\n" + "-" * 64)
    print(f"  {len(layer.provenance)} entries · chain intact = "
          f"{layer.provenance.verify()}")
    print()


if __name__ == "__main__":
    main()
