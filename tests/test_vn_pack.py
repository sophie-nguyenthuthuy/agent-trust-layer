from atl import Decision, RuleEngine, ToolCall
from atl.packs.vn import (
    AML_LARGE_TXN_VND,
    detect_vn_pii,
    find_vn_pii,
    vn_money_movement_pack,
    vn_pack,
    vn_pii_guard,
)

# Valid fixtures: province 001 (Hà Nội), 19xx male, born 1990; Viettel 098 mobile.
VALID_CCCD = "001090123456"
VALID_PHONE = "0987654321"


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
    assert "cccd" in detect_vn_pii(f"CCCD {VALID_CCCD} của khách")
    assert "phone" in detect_vn_pii(f"liên hệ {VALID_PHONE}")
    assert "email" in detect_vn_pii("gửi tới a.b@example.com")
    assert detect_vn_pii("không có gì nhạy cảm ở đây") == []


def test_invalid_cccd_rejected():
    # Bad province code (999) and an implausible (future) birth year are dropped.
    assert detect_vn_pii("mã đơn 999090123456") == []        # province invalid
    assert detect_vn_pii("order 001990123456") == []          # year 2099 > today


def test_random_12_digit_not_flagged():
    # A plain 12-digit order id should not masquerade as a CCCD.
    assert "cccd" not in detect_vn_pii("invoice 100000000000")


def test_invalid_phone_prefix_rejected():
    assert "phone" not in detect_vn_pii("ref 0123456789")      # 012 not a prefix
    assert "phone" in detect_vn_pii("+84987654321")            # +84 form valid


def test_cmnd_off_by_default_and_low_confidence():
    assert "cmnd" not in detect_vn_pii("so 012345678")
    hits = find_vn_pii("so 012345678", include_cmnd=True)
    assert any(m.kind == "cmnd" and m.confidence == "low" for m in hits)


def test_allowlist_skips_known_values():
    assert detect_vn_pii(f"hotline {VALID_PHONE}", allowlist=[VALID_PHONE]) == []
    # +84 / 0 forms are matched canonically.
    assert detect_vn_pii("call +84987654321", allowlist=["0987654321"]) == []


def test_pii_guard_escalates_without_consent():
    eng = RuleEngine([vn_pii_guard()])
    leak = eng.evaluate(ToolCall("send_email",
                                 {"body": f"CCCD {VALID_CCCD}, sđt {VALID_PHONE}"}))
    assert leak.decision is Decision.ESCALATE
    assert "NĐ 13/2023" in leak.reason
    with_consent = eng.evaluate(ToolCall("send_email",
                                         {"body": f"CCCD {VALID_CCCD}",
                                          "consent": True}))
    assert with_consent.decision is Decision.ALLOW


def test_pii_guard_blocks_when_configured():
    eng = RuleEngine([vn_pii_guard(action=Decision.BLOCK)])
    v = eng.evaluate(ToolCall("http_post", {"data": "email x@y.com"}))
    assert v.decision is Decision.BLOCK


def test_pii_guard_respects_allowlist():
    eng = RuleEngine([vn_pii_guard(allowlist=[VALID_PHONE])])
    v = eng.evaluate(ToolCall("send_email", {"body": f"hotline {VALID_PHONE}"}))
    assert v.decision is Decision.ALLOW


def test_vn_pack_combines_both():
    rules = vn_pack()
    eng = RuleEngine(rules)
    assert eng.evaluate(
        ToolCall("transfer_funds", {"amount_vnd": 500_000_000})
    ).decision is Decision.ESCALATE
    assert eng.evaluate(
        ToolCall("send_email", {"body": "0987654321"})
    ).decision is Decision.ESCALATE
