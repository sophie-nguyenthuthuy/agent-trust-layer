"""Vietnam-grounded policy pack — the differentiator vs. NeMo / Lakera.

Off-the-shelf guardrails know nothing about Vietnamese money-movement or data
rules. This pack ships rules anchored to real instruments so an agent operating
on VND and Vietnamese PII trips the right control:

  - **AML / large-value transactions** — Luật Phòng, chống rửa tiền 2022
    (Law on AML, No. 14/2022/QH15) and Quyết định 11/2023/QĐ-TTg, which sets the
    large-value transaction reporting threshold at **400.000.000 VND**. Transfers
    at/above this escalate for a "báo cáo giao dịch giá trị lớn" + review.
  - **Foreign transfers** — Pháp lệnh Ngoại hối / Nghị định 70/2014/NĐ-CP:
    outward foreign-currency transfers route to review unless licensed.
  - **Personal data** — Nghị định 13/2023/NĐ-CP (Bảo vệ dữ liệu cá nhân): a
    tool that would egress Vietnamese PII (CCCD/CMND, số điện thoại, email)
    without a consent flag is escalated.

These are engineering controls, not legal advice — tune thresholds and citations
with your compliance team. Citations are attached to each verdict's reason so
the provenance ledger is audit-ready.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable, List

from ..types import Decision, PolicyResult, ToolCall

Rule = Callable[[ToolCall], "PolicyResult | None"]

# Quyết định 11/2023/QĐ-TTg — ngưỡng giao dịch giá trị lớn phải báo cáo.
AML_LARGE_TXN_VND = 400_000_000


# ---- money movement -------------------------------------------------------
def aml_large_transfer(tool: str = "transfer_funds", arg: str = "amount_vnd",
                       threshold: int = AML_LARGE_TXN_VND) -> Rule:
    """Escalate transfers at/above the AML large-value reporting threshold."""
    def rule(call: ToolCall) -> "PolicyResult | None":
        if call.tool == tool and arg in call.args:
            try:
                amount = float(call.args[arg])
            except (TypeError, ValueError):
                return None
            if amount >= threshold:
                return PolicyResult(
                    Decision.ESCALATE, rule="aml_large_transfer",
                    reason=(f"{int(amount):,} VND ≥ {threshold:,} ngưỡng giá trị lớn "
                            f"(QĐ 11/2023/QĐ-TTg, Luật PCRT 2022) — báo cáo + review"),
                )
        return None
    return rule


def foreign_transfer_review(tool: str = "transfer_funds",
                            country_arg: str = "country",
                            license_arg: str = "fx_license") -> Rule:
    """Escalate outward foreign transfers lacking an FX license flag."""
    def rule(call: ToolCall) -> "PolicyResult | None":
        if call.tool != tool:
            return None
        country = str(call.args.get(country_arg, "VN")).upper()
        if country not in ("VN", "VIETNAM", "VIỆT NAM") and not call.args.get(license_arg):
            return PolicyResult(
                Decision.ESCALATE, rule="foreign_transfer_review",
                reason=(f"chuyển ngoại tệ ra {country} thiếu giấy phép — "
                        f"Pháp lệnh Ngoại hối / NĐ 70/2014/NĐ-CP"),
            )
        return None
    return rule


def vn_money_movement_pack(threshold: int = AML_LARGE_TXN_VND) -> List[Rule]:
    return [aml_large_transfer(threshold=threshold), foreign_transfer_review()]


# ---- personal data (NĐ 13/2023/NĐ-CP) ------------------------------------
# Hardened detectors: bare regex over-matches (any 12-digit order id looks like
# a CCCD). Each candidate is format-validated to cut false positives, and every
# match carries a confidence so callers can choose how aggressively to act.

# Mã tỉnh/thành 3 chữ số dùng trong CCCD (Thông tư 07/2016/TT-BCA), 63 tỉnh.
CCCD_PROVINCE_CODES = frozenset({
    "001", "002", "004", "006", "008", "010", "011", "012", "014", "015",
    "017", "019", "020", "022", "024", "025", "026", "027", "030", "031",
    "033", "034", "035", "036", "037", "038", "040", "042", "044", "045",
    "046", "048", "049", "051", "052", "054", "056", "058", "060", "062",
    "064", "066", "067", "068", "070", "072", "074", "075", "077", "079",
    "080", "082", "083", "084", "086", "087", "089", "091", "092", "093",
    "094", "095", "096",
})

# Đầu số di động VN hợp lệ (sau quy hoạch 2018), gồm cả số 0 đứng đầu.
VALID_PHONE_PREFIXES = frozenset({
    "032", "033", "034", "035", "036", "037", "038", "039",          # Viettel
    "086", "096", "097", "098",                                       # Viettel
    "070", "076", "077", "078", "079", "089", "090", "093",          # MobiFone
    "081", "082", "083", "084", "085", "088", "091", "094",          # VinaPhone
    "052", "056", "058", "092",                                       # Vietnamobile
    "059", "099",                                                     # Gmobile
    "087",                                                            # iTel
})

CONFIDENCE_RANK = {"low": 1, "medium": 2, "high": 3}

_CCCD_RE = re.compile(r"(?<!\d)\d{12}(?!\d)")
_CMND_RE = re.compile(r"(?<!\d)\d{9}(?!\d)")
_PHONE_RE = re.compile(r"(?<![\d+])(?:\+84|0)\d{9}(?!\d)")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}")


@dataclass(frozen=True)
class PIIMatch:
    kind: str          # cccd | cmnd | phone | email
    value: str
    confidence: str    # high | medium | low


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _canon_phone(s: str) -> str:
    d = _digits(s)
    if d.startswith("84"):
        d = "0" + d[2:]
    return d


def _valid_cccd(s: str) -> bool:
    """Validate CCCD structure: province code + century/gender + plausible year."""
    if len(s) != 12 or s[:3] not in CCCD_PROVINCE_CODES:
        return False
    century_digit = int(s[3])              # 0/1=19xx, 2/3=20xx, 4/5=21xx, ...
    base_year = 1900 + (century_digit // 2) * 100
    full_year = base_year + int(s[4:6])
    return 1900 <= full_year <= date.today().year


def _valid_phone(s: str) -> bool:
    canon = _canon_phone(s)
    return len(canon) == 10 and canon[:3] in VALID_PHONE_PREFIXES


def find_vn_pii(text: str, *, include_cmnd: bool = False,
                allowlist: Iterable[str] = ()) -> List[PIIMatch]:
    """Return validated Vietnamese-PII matches in ``text``.

    Args:
        include_cmnd: also scan for 9-digit CMND. Off by default — 9-digit
            validation is unreliable (collides with order ids etc.), so it is
            reported only at ``low`` confidence when enabled.
        allowlist: literal values to ignore (e.g. a company's published hotline
            or test fixtures); phones/ids are compared in canonical digit form.
    """
    allow_raw = set(allowlist)
    allow_phone = {_canon_phone(a) for a in allowlist}
    allow_digits = {_digits(a) for a in allowlist}
    out: List[PIIMatch] = []
    seen = set()

    def add(kind: str, value: str, conf: str) -> None:
        if (kind, value) not in seen:
            seen.add((kind, value))
            out.append(PIIMatch(kind, value, conf))

    for m in _CCCD_RE.finditer(text):
        v = m.group()
        if v in allow_raw or v in allow_digits or not _valid_cccd(v):
            continue
        add("cccd", v, "high")
    for m in _PHONE_RE.finditer(text):
        v = m.group()
        if _canon_phone(v) in allow_phone or not _valid_phone(v):
            continue
        add("phone", v, "high")
    for m in _EMAIL_RE.finditer(text):
        v = m.group()
        if v in allow_raw:
            continue
        add("email", v, "high")
    if include_cmnd:
        for m in _CMND_RE.finditer(text):
            v = m.group()
            if v in allow_raw or v in allow_digits:
                continue
            add("cmnd", v, "low")
    return out


def detect_vn_pii(text: str, *, include_cmnd: bool = False,
                  allowlist: Iterable[str] = ()) -> List[str]:
    """Return the (deduped, ordered) PII kinds found — convenience over find_vn_pii."""
    kinds: List[str] = []
    for m in find_vn_pii(text, include_cmnd=include_cmnd, allowlist=allowlist):
        if m.kind not in kinds:
            kinds.append(m.kind)
    return kinds


def vn_pii_guard(egress_tools: Iterable[str] = ("send_email", "http_post",
                                                "webhook", "external_api"),
                 consent_arg: str = "consent",
                 action: Decision = Decision.ESCALATE,
                 min_confidence: str = "high",
                 include_cmnd: bool = False,
                 allowlist: Iterable[str] = ()) -> Rule:
    """Gate egress tools that carry Vietnamese PII without a consent flag.

    NĐ 13/2023/NĐ-CP requires a lawful basis (typically consent) to process /
    transfer personal data. Calls carrying ``consent=True`` pass; otherwise a
    call whose args contain PII at/above ``min_confidence`` is escalated (or
    blocked, if ``action`` is BLOCK).
    """
    tools = set(egress_tools)
    threshold = CONFIDENCE_RANK[min_confidence]
    allow = tuple(allowlist)

    def rule(call: ToolCall) -> "PolicyResult | None":
        if call.tool not in tools or call.args.get(consent_arg):
            return None
        blob = " ".join(str(v) for v in call.args.values())
        hits = [m for m in find_vn_pii(blob, include_cmnd=include_cmnd,
                                       allowlist=allow)
                if CONFIDENCE_RANK[m.confidence] >= threshold]
        if hits:
            kinds = sorted({m.kind for m in hits})
            return PolicyResult(
                action, rule="vn_pii_guard",
                reason=(f"PII ({', '.join(kinds)}) qua {call.tool} thiếu consent — "
                        f"NĐ 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân"),
            )
        return None
    return rule


def vn_pii_pack(action: Decision = Decision.ESCALATE) -> List[Rule]:
    return [vn_pii_guard(action=action)]


def vn_pack(threshold: int = AML_LARGE_TXN_VND,
            pii_action: Decision = Decision.ESCALATE) -> List[Rule]:
    """The full Vietnam pack: money-movement + personal-data rules."""
    return vn_money_movement_pack(threshold=threshold) + vn_pii_pack(action=pii_action)
