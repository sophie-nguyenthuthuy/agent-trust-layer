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
_PII_PATTERNS = {
    # CCCD (12 chữ số) — căn cước công dân. Bounded to avoid matching longer ids.
    "cccd": re.compile(r"(?<!\d)\d{12}(?!\d)"),
    # CMND (9 chữ số) — chứng minh nhân dân (legacy).
    "cmnd": re.compile(r"(?<!\d)\d{9}(?!\d)"),
    # Số điện thoại di động VN.
    "phone": re.compile(r"(?<!\d)(?:\+84|0)(?:3|5|7|8|9)\d{8}(?!\d)"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}


def detect_vn_pii(text: str) -> List[str]:
    """Return the kinds of Vietnamese PII found in ``text`` (deduped, ordered)."""
    found = [kind for kind, pat in _PII_PATTERNS.items() if pat.search(text)]
    return found


def vn_pii_guard(egress_tools: Iterable[str] = ("send_email", "http_post",
                                                "webhook", "external_api"),
                 consent_arg: str = "consent",
                 action: Decision = Decision.ESCALATE) -> Rule:
    """Gate egress tools that carry Vietnamese PII without a consent flag.

    NĐ 13/2023/NĐ-CP requires a lawful basis (typically consent) to process /
    transfer personal data. If the call already carries ``consent=True`` it
    passes; otherwise it is escalated (or blocked, if ``action`` is BLOCK).
    """
    tools = set(egress_tools)

    def rule(call: ToolCall) -> "PolicyResult | None":
        if call.tool not in tools or call.args.get(consent_arg):
            return None
        blob = " ".join(str(v) for v in call.args.values())
        kinds = detect_vn_pii(blob)
        if kinds:
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
