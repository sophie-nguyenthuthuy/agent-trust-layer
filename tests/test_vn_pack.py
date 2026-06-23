from atl import Decision, RuleEngine, ToolCall
from atl.packs.vn import (
    AML_LARGE_TXN_VND,
    crc16_ccitt,
    detect_vn_pii,
    find_vn_pii,
    parse_vietqr,
    validate_mst,
    validate_napas_bin,
    validate_vietqr,
    vn_money_movement_pack,
    vn_pack,
    vn_pii_guard,
)


def _tlv(tag, value):
    return f"{tag}{len(value):02d}{value}"


def _make_vietqr(bin_code="970415", account="0123456789", amount=None):
    """Build a well-formed VietQR payload with a correct EMVCo CRC."""
    merchant = _tlv("00", "A000000727") + _tlv(
        "01", _tlv("00", bin_code) + _tlv("01", account)) + _tlv("02", "QRIBFTTA")
    body = (_tlv("00", "01") + _tlv("01", "11") + _tlv("38", merchant)
            + _tlv("53", "704"))
    if amount is not None:
        body += _tlv("54", str(amount))
    body += _tlv("58", "VN") + "6304"
    return body + f"{crc16_ccitt(body):04X}"

# Valid fixtures: province 001 (Hà Nội), 19xx male, born 1990; Viettel 098 mobile.
VALID_CCCD = "001090123456"
VALID_PHONE = "0987654321"
VALID_MST = "0101245486"          # real 10-digit MST (check digit 6)


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


def test_validate_mst_checksum():
    assert validate_mst(VALID_MST)              # Vingroup
    assert validate_mst("0101248141")           # FPT
    assert not validate_mst("0101245487")       # wrong check digit
    assert not validate_mst("12345")            # wrong length


def test_validate_mst_13_digit_branch():
    assert validate_mst("0101245486-001")       # hyphenated branch form
    assert validate_mst("0101245486001")        # 13 contiguous digits
    assert not validate_mst("0101245487-001")   # bad core check digit


def test_mst_off_by_default_then_opt_in():
    assert "mst" not in detect_vn_pii(f"đối tác {VALID_MST}")
    hits = find_vn_pii(f"đối tác {VALID_MST}", include_mst=True)
    assert any(m.kind == "mst" and m.confidence == "high" for m in hits)


def test_mst_guard_via_pack():
    eng = RuleEngine(vn_pack(include_mst=True))
    v = eng.evaluate(ToolCall("send_email", {"body": f"xuất hóa đơn MST {VALID_MST}"}))
    assert v.decision is Decision.ESCALATE
    # A non-checksum 10-digit number is not mistaken for an MST.
    ok = eng.evaluate(ToolCall("send_email", {"body": "mã 1234567890"}))
    assert ok.decision is Decision.ALLOW


def test_crc16_ccitt_standard_check_vector():
    # The canonical CRC-16/CCITT-FALSE check value proves the implementation.
    assert crc16_ccitt("123456789") == 0x29B1


def test_napas_bin_validation():
    assert validate_napas_bin("970436")        # Vietcombank
    assert not validate_napas_bin("123456")
    assert not validate_napas_bin("97043")      # wrong length


def test_vietqr_roundtrip_and_parse():
    qr = _make_vietqr(bin_code="970415", account="0123456789", amount=50000)
    assert validate_vietqr(qr)
    p = parse_vietqr(qr)
    assert p["valid_crc"] and p["bin"] == "970415"
    assert p["account"] == "0123456789" and p["bank"] == "VietinBank"
    assert p["amount"] == "50000"


def test_vietqr_tampered_crc_rejected():
    qr = _make_vietqr()
    tampered = qr[:20] + ("9" if qr[20] != "9" else "8") + qr[21:]
    assert not validate_vietqr(tampered)


def test_napas_transfer_guard_in_pack():
    eng = RuleEngine(vn_money_movement_pack())
    good_qr = _make_vietqr()
    assert eng.evaluate(
        ToolCall("transfer_funds", {"vietqr": good_qr})).decision is Decision.ALLOW
    assert eng.evaluate(
        ToolCall("transfer_funds", {"vietqr": good_qr[:-1] + "0"})
    ).decision is Decision.BLOCK
    assert eng.evaluate(
        ToolCall("transfer_funds", {"bank_bin": "999999"})).decision is Decision.BLOCK
    assert eng.evaluate(
        ToolCall("transfer_funds", {"bank_bin": "970436"})).decision is Decision.ALLOW


def test_vn_pack_combines_both():
    rules = vn_pack()
    eng = RuleEngine(rules)
    assert eng.evaluate(
        ToolCall("transfer_funds", {"amount_vnd": 500_000_000})
    ).decision is Decision.ESCALATE
    assert eng.evaluate(
        ToolCall("send_email", {"body": "0987654321"})
    ).decision is Decision.ESCALATE
