from atl import Decision, RuleEngine, ToolCall
from atl.packs.vn import (
    AML_LARGE_TXN_VND,
    detect_vn_pii,
    vn_money_movement_pack,
    vn_pack,
    vn_pii_guard,
)


def test_aml_threshold_escalates_at_400m():
    eng = RuleEngine(vn_money_movement_pack())
    big = eng.evaluate(ToolCall("transfer_funds", {"amount_vnd": AML_LARGE_TXN_VND}))
    assert big.decision is Decision.ESCALATE
    assert "QĐ 11/2023" in big.reason
    small = eng.evaluate(ToolCall("transfer_funds", {"amount_vnd": 399_000_000}))
    assert small.decision is Decision.ALLOW


def test_foreign_transfer_needs_license():
    eng = RuleEngine(vn_money_movement_pack())
    v = eng.evaluate(ToolCall("transfer_funds",
                              {"amount_vnd": 1000, "country": "US"}))
    assert v.decision is Decision.ESCALATE
    ok = eng.evaluate(ToolCall("transfer_funds",
                               {"amount_vnd": 1000, "country": "US",
                                "fx_license": "GP-123"}))
    assert ok.decision is Decision.ALLOW


def test_detect_vn_pii_kinds():
    assert "cccd" in detect_vn_pii("CCCD 012345678901 của khách")
    assert "phone" in detect_vn_pii("liên hệ 0987654321")
    assert "email" in detect_vn_pii("gửi tới a.b@example.com")
    assert detect_vn_pii("không có gì nhạy cảm ở đây") == []


def test_pii_guard_escalates_without_consent():
    eng = RuleEngine([vn_pii_guard()])
    leak = eng.evaluate(ToolCall("send_email",
                                 {"body": "CCCD 012345678901, sđt 0987654321"}))
    assert leak.decision is Decision.ESCALATE
    assert "NĐ 13/2023" in leak.reason
    with_consent = eng.evaluate(ToolCall("send_email",
                                         {"body": "CCCD 012345678901",
                                          "consent": True}))
    assert with_consent.decision is Decision.ALLOW


def test_pii_guard_blocks_when_configured():
    eng = RuleEngine([vn_pii_guard(action=Decision.BLOCK)])
    v = eng.evaluate(ToolCall("http_post", {"data": "email x@y.com"}))
    assert v.decision is Decision.BLOCK


def test_vn_pack_combines_both():
    rules = vn_pack()
    eng = RuleEngine(rules)
    assert eng.evaluate(
        ToolCall("transfer_funds", {"amount_vnd": 500_000_000})
    ).decision is Decision.ESCALATE
    assert eng.evaluate(
        ToolCall("send_email", {"body": "0987654321"})
    ).decision is Decision.ESCALATE
