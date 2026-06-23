# agent-trust-layer

**Certified trust & governance middleware that sits in front of an agent's tool-use.**

Policy-as-code guardrails · tamper-evident provenance · a **certified** drift/confidence gate · human-in-the-loop escalation — in one dependency-free package you can drop into any agent stack.

> Most "agent guardrail" products gate on a threshold someone eyeballed. This one gates on an **Azuma–Hoeffding concentration bound**, so every BLOCK/ESCALATE carries an auditable *certificate* stating the confidence with which the drift is real rather than noise. That gate is the moat; policy, provenance, and HITL are deliberately thin and pluggable.

```
tool_call ─► policy.evaluate ─► gate.observe ─► provenance.record ─► { ALLOW | ESCALATE→human | BLOCK }
```

---

## Why it exists

The four pillars of "agent governance" are not equally defensible:

| Pillar | Market | This repo |
|---|---|---|
| Policy-as-code | Commoditized (OPA/Cedar/NeMo) | thin built-in engine + pluggable port |
| Provenance | Commoditizing (OTel/Langfuse) | hash-chained, HMAC-signed, offline-verifiable |
| **Confidence/drift gating** | **Wide open** | **certified gate — the differentiator** |
| HITL escalation | Easy glue | sync resolver + async queue |

You bring your own policy engine and tracer; the trust layer owns the **trust decision**.

## Install

```bash
pip install -e .            # core is stdlib-only
pip install -e ".[langgraph]"   # optional LangGraph adapter
pip install -e ".[dev]"         # tests
```

## 60-second demo

```bash
python examples/multi_agent_demo.py
```

```
   executor · transfer_funds  risk=0.30 -> BLOCK   policy:constrain_arg (amount exceeds 10k review cap)
  responder · respond         risk=0.85 -> BLOCK   hitl:block <- policy:require_confidence (risk 0.85 > 0.40)
  certified gate TRIPPED at n=12: mean=0.53 > thr=0.50 @ 95% -> BLOCK
  12 entries · chain intact = True
```

## Quickstart

```python
from atl import (TrustLayer, RuleEngine, CertifiedGate, HITLQueue, ProvenanceLog,
                 ToolCall, Decision, constrain_arg, deny_tools, require_confidence)

layer = TrustLayer(
    policy=RuleEngine([
        deny_tools("shell_exec"),
        constrain_arg("transfer_funds", "amount", lambda a: a <= 10_000),
        require_confidence("respond", max_risk=0.4),   # low-confidence answers escalate
    ]),
    gate=CertifiedGate(baseline=0.15, delta=0.05, window=50, v_floor=0.25),
    hitl=HITLQueue(resolver=my_review_fn),             # sync human-in-the-loop
    provenance=ProvenanceLog(key=b"my-signing-key"),
)

# Guard a call, or wrap a tool so it's gated automatically.
verdict = layer.guard(ToolCall("transfer_funds", {"amount": 250_000}, risk=0.3))
assert verdict.decision is Decision.BLOCK
print(verdict.certificate.as_dict())   # the evidence behind the decision
```

### LangGraph

Drop `guarded_tool_node` into a real `StateGraph` in place of the prebuilt
`ToolNode` — every tool call the agent emits is gated before it runs, and
refusals return as `ToolMessage`s so the loop can recover:

```python
from langgraph.graph import StateGraph, START, END
from atl.integrations.langgraph import guarded_tool_node

g = StateGraph(State)
g.add_node("agent", agent)
g.add_node("tools", guarded_tool_node(layer, TOOLS, actor="executor"))
g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
g.add_edge("tools", "agent")
```

Runnable end-to-end (deterministic scripted model — no API key; swap in
`ChatOllama` for a live OSS model):

```bash
pip install -e ".[langgraph]"
python examples/langgraph_agent.py
```

```
  AI    → call web_search({'q': 'vendor invoices Q2'})
  Tool  top results for 'vendor invoices Q2': [...]
  AI    → call transfer_funds({'amount': 250000, 'to': 'acct-x'})
  Tool  [TRUST LAYER BLOCK] policy:constrain_arg (amount exceeds 10k review cap)
  AI    The transfer was refused by governance, so I stopped ... No funds moved.
  #0 executor · web_search     -> ALLOW   ·  #1 executor · transfer_funds -> BLOCK   (chain intact)
```

`guard_tools(layer, tools)` is also available to wrap plain callables.

## The certified gate

A risk signal `r_t ∈ [0,1]` (e.g. `1 − confidence`) is observed each step. For bounded increments, Hoeffding gives `P(mean_emp − mean_true ≥ ε) ≤ exp(−2nε²)`. Setting the RHS to `delta` yields the margin

```
ε(n) = sqrt( ln(1/delta) / (2n) )
```

The gate trips when `mean_emp ≥ baseline + ε(n)` **and** `mean_emp ≥ v_floor`. So:

- **Cold-start safe** — at small `n` the margin is huge; one bad reading can't trip it.
- **Auditable** — each decision returns a `Certificate{n, mean_risk, threshold, bound, delta, v_floor}`.
- **Tunable on evidence, not vibes** — `delta` is a literal false-trip probability ceiling.

This is the [lyapguard](https://github.com/sophie-nguyenthuthuy/lyapguard) / lyapmon Azuma-drift design distilled to one embeddable class.

## VN-grounded policy pack

Off-the-shelf guardrails know nothing about Vietnamese money-movement or data
law. `atl.packs.vn` ships rules anchored to real instruments — the differentiator
for a VSF / Vietnam-market pitch:

```python
from atl import RuleEngine, TrustLayer
from atl.packs.vn import vn_pack

layer = TrustLayer(policy=RuleEngine(vn_pack()))   # AML + FX + PII, cited
```

| Rule | Trigger | Grounding |
|---|---|---|
| `aml_large_transfer` | transfer ≥ **400.000.000 VND** → ESCALATE | Luật PCRT 2022, **QĐ 11/2023/QĐ-TTg** |
| `foreign_transfer_review` | outward FX transfer without license → ESCALATE | Pháp lệnh Ngoại hối, NĐ 70/2014/NĐ-CP |
| `vn_pii_guard` | egress tool carrying CCCD/phone/email without consent → ESCALATE/BLOCK | **NĐ 13/2023/NĐ-CP** |
| `vn_pii_guard(include_mst=True)` | egress carrying a **checksum-valid MST** (tax code) → ESCALATE/BLOCK | TT 105/2020/TT-BTC |

Each verdict's `reason` carries the citation, so the provenance ledger is
audit-ready. `python examples/vn_governance_demo.py` runs it end-to-end.

The PII detectors are **format-validated, not bare regex** — a CCCD candidate
must carry a real province code (Thông tư 07/2016) and a plausible birth year, a
phone must use a real post-2018 mobile prefix, an MST must pass its mod-11
checksum — so order ids and random digit runs don't false-positive. Matches
carry a confidence (`high`/`low`); the unreliable 9-digit CMND and the business
MST are opt-in (`include_cmnd=True` / `include_mst=True`); and
`vn_pii_guard(allowlist=[...])` skips known-safe values (hotlines, fixtures).

> Engineering controls, not legal advice — tune thresholds/citations with your compliance team.

## Observability — the dashboard story

Every decision can fan out to metrics/tracing via the provenance `sink`:

```python
from atl import ProvenanceLog, PrometheusSink, MetricsServer, multi_sink, otel_sink

prom = PrometheusSink()
log = ProvenanceLog(sink=multi_sink(prom, otel_sink(tracer)))   # OTel optional
MetricsServer(prom, port=9464).start()        # GET /metrics  (stdlib, zero-dep)
```

Exposes `atl_decisions_total{tool,actor,decision}`, `atl_blocked_total{tool}`,
`atl_gate_trips_total`, `atl_provenance_entries_total`, `atl_gate_mean_risk` —
scrape into Prometheus/Grafana. OTel spans (`atl.decision`) are emitted only if
`opentelemetry` is installed (`pip install -e ".[otel]"`).

## Eval harness — does it actually help?

A reproducible synthetic multi-agent workload (no LLM, no network — deterministic fault injection so the layer OFF and ON see *identical* tasks) measures TTFT, P95 latency, hallucination rate, and tool-misuse rate before/after.

```bash
python -m eval.harness --n 400 --json results.json
```

| metric | OFF | ON | delta |
|---|--:|--:|--:|
| TTFT p50 (ms) | 80.6 | 80.6 | +0.0% |
| TTFT p95 (ms) | 115.8 | 115.8 | +0.0% |
| Latency p95 (ms) | 410.5 | 410.6 | +0.0% |
| **Hallucination rate** | 0.185 | **0.084** | **−54%** |
| **Tool-misuse rate** | 0.107 | **0.000** | **−100%** |
| Over-block rate | 0.000 | 0.000 | — |
| Trust overhead/req | 0.001 ms | **0.043 ms** | +43 µs |

Reading it honestly:

- **Latency is free.** The layer adds ~43 microseconds per request — invisible against ~80–400 ms of agent work, so TTFT and P95 don't move.
- **Tool-misuse → 0.** Every out-of-policy destructive call is blocked deterministically.
- **Hallucination halves, doesn't vanish.** The residual 8.4% are *confident* hallucinations — wrong answers the model reports with low risk. A confidence gate cannot see those by construction; closing that gap needs a better risk signal (self-consistency, retrieval grounding), not a better gate. The harness reports it rather than hiding it.
- **Zero over-blocking** in this workload (correct answers carry low risk, so they never escalate).

> The numbers above come from a **synthetic fault model**, not a live LLM — its purpose is a deterministic, CI-runnable measurement of the *layer's* behavior. Swap in a real agent via the LangGraph adapter to benchmark your own stack; `eval/workload.py` is where the fault model lives.

## Layout

```
atl/
  types.py         ToolCall, Verdict, Certificate, Decision
  gate.py          CertifiedGate — Azuma-bounded drift gate  ★ the moat
  policy.py        RuleEngine + rule factories + PolicyEngine port (OPA/Cedar)
  provenance.py    hash-chained, HMAC-signed, offline-verifiable ledger
  observability.py PrometheusSink + /metrics server + OTel sink (zero-dep core)
  hitl.py          HITLQueue — sync resolver + async pending queue
  middleware.py    TrustLayer — the interceptor wiring it all together
  packs/vn.py      Vietnam-grounded policy pack (AML / FX / NĐ 13/2023 PII)
  integrations/langgraph.py   guarded_tool_node() + guard_tools()
eval/              before/after harness (metrics, workload, runner)
examples/          multi_agent_demo · langgraph_agent (real StateGraph) · vn_governance_demo
tests/             44 tests, stdlib + pytest (langgraph tests skip if not installed)
```

## License

Apache-2.0.
