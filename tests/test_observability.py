import urllib.request

from atl import (
    Decision,
    MetricsServer,
    PrometheusSink,
    ProvenanceLog,
    RuleEngine,
    ToolCall,
    TrustLayer,
    deny_tools,
    multi_sink,
)


def _layer_with_sink(sink):
    return TrustLayer(
        policy=RuleEngine([deny_tools("rm_rf")]),
        provenance=ProvenanceLog(sink=sink),
        hitl=None,
    )


def test_prometheus_sink_counts_decisions():
    sink = PrometheusSink()
    layer = _layer_with_sink(sink)
    layer.guard(ToolCall("ls", actor="a"))
    layer.guard(ToolCall("rm_rf", actor="a"))
    text = sink.render()
    assert 'atl_decisions_total{tool="ls",actor="a",decision="allow"} 1' in text
    assert 'atl_blocked_total{tool="rm_rf"} 1' in text
    assert "atl_provenance_entries_total 2" in text


def test_prometheus_sink_tracks_gate():
    sink = PrometheusSink()
    layer = _layer_with_sink(sink)
    for _ in range(5):
        layer.guard(ToolCall("ls", risk=0.05))
    text = sink.render()
    assert "atl_gate_trips_total 0" in text
    assert "atl_gate_mean_risk" in text


def test_multi_sink_fans_out():
    seen_a, seen_b = [], []
    sink = multi_sink(seen_a.append, seen_b.append)
    layer = _layer_with_sink(sink)
    layer.guard(ToolCall("ls"))
    assert len(seen_a) == 1 and len(seen_b) == 1


def test_metrics_server_serves_exposition():
    sink = PrometheusSink()
    layer = _layer_with_sink(sink)
    layer.guard(ToolCall("ls"))
    server = MetricsServer(sink, port=0).start()
    try:
        url = f"http://127.0.0.1:{server.port}/metrics"
        body = urllib.request.urlopen(url, timeout=5).read().decode()
    finally:
        server.stop()
    assert "atl_decisions_total" in body
    assert "atl_provenance_entries_total 1" in body
