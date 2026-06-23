"""Observability sinks for the provenance ledger.

``ProvenanceLog`` accepts a ``sink`` callable invoked with each entry dict.
This module provides:

  - ``PrometheusSink`` — zero-dependency, renders the Prometheus text exposition
    format directly (no prometheus_client needed). The dashboard story.
  - ``MetricsServer`` — a stdlib HTTP server exposing ``/metrics``.
  - ``otel_sink`` — optional OpenTelemetry span emitter (only if otel installed).
  - ``multi_sink`` — fan a single entry out to several sinks.

Everything here is stdlib-only; OTel is imported lazily and only if you use it.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Tuple

Sink = Callable[[dict], None]


def multi_sink(*sinks: Sink) -> Sink:
    """Combine several sinks into one (e.g. Prometheus + OTel + a log file)."""
    def fan(entry: dict) -> None:
        for s in sinks:
            s(entry)
    return fan


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


class PrometheusSink:
    """Accumulates trust-layer metrics and renders Prometheus exposition text.

    Exposed series:
      - atl_decisions_total{tool,actor,decision}
      - atl_blocked_total{tool}
      - atl_gate_trips_total
      - atl_provenance_entries_total
      - atl_gate_mean_risk            (last observed running mean risk)
    """

    def __init__(self, namespace: str = "atl") -> None:
        self.ns = namespace
        self._decisions: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._blocked: Dict[str, int] = defaultdict(int)
        self._gate_trips = 0
        self._entries = 0
        self._last_mean_risk = 0.0
        self._lock = threading.Lock()

    def __call__(self, entry: dict) -> None:
        with self._lock:
            self._entries += 1
            tool = entry.get("tool", "")
            actor = entry.get("actor", "")
            decision = entry.get("decision", "")
            self._decisions[(tool, actor, decision)] += 1
            if decision != "allow":
                self._blocked[tool] += 1
            cert = entry.get("certificate")
            if cert:
                self._last_mean_risk = cert.get("mean_risk", self._last_mean_risk)
                if cert.get("tripped"):
                    self._gate_trips += 1

    def render(self) -> str:
        ns = self.ns
        out = []
        with self._lock:
            out.append(f"# HELP {ns}_decisions_total Trust-layer decisions.")
            out.append(f"# TYPE {ns}_decisions_total counter")
            for (tool, actor, decision), n in sorted(self._decisions.items()):
                out.append(
                    f'{ns}_decisions_total{{tool="{_escape(tool)}",'
                    f'actor="{_escape(actor)}",decision="{_escape(decision)}"}} {n}'
                )
            out.append(f"# HELP {ns}_blocked_total Non-allowed decisions by tool.")
            out.append(f"# TYPE {ns}_blocked_total counter")
            for tool, n in sorted(self._blocked.items()):
                out.append(f'{ns}_blocked_total{{tool="{_escape(tool)}"}} {n}')
            out.append(f"# HELP {ns}_gate_trips_total Certified-gate trips.")
            out.append(f"# TYPE {ns}_gate_trips_total counter")
            out.append(f"{ns}_gate_trips_total {self._gate_trips}")
            out.append(f"# HELP {ns}_provenance_entries_total Ledger entries.")
            out.append(f"# TYPE {ns}_provenance_entries_total counter")
            out.append(f"{ns}_provenance_entries_total {self._entries}")
            out.append(f"# HELP {ns}_gate_mean_risk Last running mean risk.")
            out.append(f"# TYPE {ns}_gate_mean_risk gauge")
            out.append(f"{ns}_gate_mean_risk {self._last_mean_risk}")
        return "\n".join(out) + "\n"


class MetricsServer:
    """Serve a ``PrometheusSink`` on ``http://host:port/metrics`` (stdlib)."""

    def __init__(self, sink: PrometheusSink, host: str = "127.0.0.1",
                 port: int = 9464) -> None:
        self.sink = sink
        self.host = host
        self.port = port
        self._httpd: "ThreadingHTTPServer | None" = None
        self._thread: "threading.Thread | None" = None

    def _handler(self):
        sink = self.sink

        class H(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path.rstrip("/") in ("/metrics", ""):
                    body = sink.render().encode()
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "text/plain; version=0.0.4")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *a):  # silence
                pass

        return H

    def start(self) -> "MetricsServer":
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._handler())
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()


def otel_sink(tracer) -> Sink:
    """Return a sink that emits one OpenTelemetry span per decision.

    ``tracer`` is an ``opentelemetry.trace.Tracer``. Imported lazily so OTel
    stays an optional dependency.
    """
    def sink(entry: dict) -> None:
        with tracer.start_as_current_span("atl.decision") as span:
            span.set_attribute("atl.tool", entry.get("tool", ""))
            span.set_attribute("atl.actor", entry.get("actor", ""))
            span.set_attribute("atl.decision", entry.get("decision", ""))
            span.set_attribute("atl.reason", entry.get("reason", ""))
            cert = entry.get("certificate") or {}
            if "mean_risk" in cert:
                span.set_attribute("atl.gate.mean_risk", cert["mean_risk"])
                span.set_attribute("atl.gate.tripped", bool(cert.get("tripped")))
    return sink
