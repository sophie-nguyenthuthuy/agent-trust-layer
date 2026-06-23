"""VN-grounded governance + live Prometheus metrics.

Shows the Vietnam policy pack (AML large-value transfers, foreign-transfer
review, NĐ 13/2023 PII egress) gating an agent, with every decision streamed to
a Prometheus ``/metrics`` endpoint for the dashboard.

    python examples/vn_governance_demo.py
"""
from atl import (
    CertifiedGate,
    Decision,
    HITLQueue,
    MetricsServer,
    PrometheusSink,
    ProvenanceLog,
    RuleEngine,
    ToolCall,
    TrustLayer,
)
from atl.packs.vn import vn_pack


def build():
    sink = PrometheusSink()
    layer = TrustLayer(
        policy=RuleEngine(vn_pack()),               # AML + FX + PII, VN-grounded
        gate=CertifiedGate(baseline=0.15, delta=0.05, v_floor=0.25),
        hitl=HITLQueue(resolver=lambda v: Decision.BLOCK),
        provenance=ProvenanceLog(key=b"vn-demo", sink=sink),
    )
    return layer, sink


def show(layer, call):
    v = layer.guard(call)
    print(f"  {call.tool:<15} -> {v.decision.value.upper():<9} {v.reason}")


def main():
    layer, sink = build()
    print("\nVN governance pack (Luật PCRT 2022 · QĐ 11/2023 · NĐ 13/2023)\n" + "-" * 70)
    show(layer, ToolCall("transfer_funds", {"amount_vnd": 250_000, "country": "VN"}))
    show(layer, ToolCall("transfer_funds", {"amount_vnd": 500_000_000, "country": "VN"}))
    show(layer, ToolCall("transfer_funds", {"amount_vnd": 1_000_000, "country": "US"}))
    show(layer, ToolCall("send_email", {"body": "Khách CCCD 001090123456, sđt 0987654321"}))
    show(layer, ToolCall("send_email", {"body": "CCCD 001090123456", "consent": True}))

    server = MetricsServer(sink, port=0).start()
    print(f"\n/metrics (served at http://127.0.0.1:{server.port}/metrics)\n" + "-" * 70)
    print("\n".join("  " + ln for ln in sink.render().splitlines()
                     if ln and not ln.startswith("#")))
    server.stop()
    print(f"\n  provenance chain intact = {layer.provenance.verify()}\n")


if __name__ == "__main__":
    main()
