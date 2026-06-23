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
  hitl.py          HITLQueue — sync resolver + async pending queue
  middleware.py    TrustLayer — the interceptor wiring it all together
  integrations/langgraph.py   guard_tools() for LangGraph ToolNode
eval/              before/after harness (metrics, workload, runner)
examples/          multi_agent_demo.py (no deps) · langgraph_agent.py (real StateGraph)
tests/             24 tests, stdlib + pytest (langgraph tests skip if not installed)
```

## License

Apache-2.0.
