from atl import (
    Decision,
    RuleEngine,
    ToolCall,
    allow_tools,
    constrain_arg,
    deny_tools,
    require_confidence,
)


def test_deny_tools_blocks():
    eng = RuleEngine([deny_tools("rm_rf")])
    assert eng.evaluate(ToolCall("rm_rf")).decision is Decision.BLOCK
    assert eng.evaluate(ToolCall("ls")).decision is Decision.ALLOW


def test_require_confidence_escalates():
    eng = RuleEngine([require_confidence("respond", max_risk=0.4)])
    assert eng.evaluate(ToolCall("respond", risk=0.6)).decision is Decision.ESCALATE
    assert eng.evaluate(ToolCall("respond", risk=0.1)).decision is Decision.ALLOW


def test_constrain_arg_caps_amount():
    eng = RuleEngine([constrain_arg("transfer", "amount", lambda a: a <= 100)])
    assert eng.evaluate(ToolCall("transfer", {"amount": 500})).decision is Decision.BLOCK
    assert eng.evaluate(ToolCall("transfer", {"amount": 50})).decision is Decision.ALLOW


def test_allow_list_blocks_unknown():
    eng = RuleEngine([allow_tools("search", "read")])
    assert eng.evaluate(ToolCall("search")).decision is Decision.ALLOW
    assert eng.evaluate(ToolCall("delete")).decision is Decision.BLOCK


def test_first_match_wins():
    eng = RuleEngine([deny_tools("x"), allow_tools("y")])
    # 'x' is denied by the first rule even though allow_tools would also block.
    assert eng.evaluate(ToolCall("x")).rule == "deny_tools"
