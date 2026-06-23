import pytest

from atl import (
    CertifiedGate,
    Decision,
    GateBlocked,
    HITLQueue,
    RuleEngine,
    ToolCall,
    TrustLayer,
    constrain_arg,
    deny_tools,
    require_confidence,
)


def make_layer(**kw):
    policy = RuleEngine([
        deny_tools("rm_rf"),
        constrain_arg("transfer", "amount", lambda a: a <= 100),
        require_confidence("respond", 0.4),
    ])
    return TrustLayer(policy=policy,
                      gate=CertifiedGate(v_floor=0.3),
                      hitl=HITLQueue(resolver=lambda v: Decision.BLOCK),
                      **kw)


def test_policy_block_propagates():
    layer = make_layer()
    v = layer.guard(ToolCall("rm_rf"))
    assert v.decision is Decision.BLOCK
    assert not v.allowed


def test_escalation_routes_to_hitl():
    layer = make_layer()
    v = layer.guard(ToolCall("respond", risk=0.9))
    # require_confidence escalates; HITL resolver blocks.
    assert v.decision is Decision.BLOCK
    assert v.reason.startswith("hitl:")


def test_allow_passes_and_executes():
    layer = make_layer()
    out = layer.execute(ToolCall("transfer", {"amount": 50}),
                        lambda amount: amount * 2)
    assert out == 100


def test_blocked_execute_raises():
    layer = make_layer()
    with pytest.raises(GateBlocked):
        layer.execute(ToolCall("transfer", {"amount": 5000}), lambda amount: amount)


def test_disabled_layer_is_passthrough():
    layer = make_layer(enabled=False)
    v = layer.guard(ToolCall("rm_rf"))
    assert v.allowed
    assert len(layer.provenance) == 0   # nothing logged when off


def test_every_decision_is_logged():
    layer = make_layer()
    layer.guard(ToolCall("transfer", {"amount": 50}))
    layer.guard(ToolCall("rm_rf"))
    assert len(layer.provenance) == 2
    assert layer.provenance.verify()


def test_wrap_tool_strips_risk():
    layer = make_layer()
    tool = layer.wrap_tool("transfer", lambda amount: amount + 1)
    assert tool(_risk=0.1, amount=10) == 11
