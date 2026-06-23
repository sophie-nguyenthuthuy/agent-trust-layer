"""Tamper-evident provenance log.

Every trust decision is appended to a hash-chained, HMAC-signed ledger. Any
edit to a past entry breaks the chain, so the audit trail is verifiable
offline with a single shared key (self-hostable — no external service).

Entries are also emitted as plain dicts so they can be forwarded to OTel /
Langfuse / a SIEM without coupling the core to any of them.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from .types import Verdict

GENESIS = "0" * 64


@dataclass
class Entry:
    seq: int
    ts: float
    actor: str
    tool: str
    args_hash: str
    decision: str
    reason: str
    certificate: Optional[dict]
    prev_hash: str
    entry_hash: str = ""

    def signing_payload(self) -> bytes:
        body = {k: v for k, v in asdict(self).items() if k != "entry_hash"}
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ProvenanceLog:
    def __init__(self, key: bytes = b"atl-dev-key",
                 sink: "Optional[Callable[[dict], None]]" = None) -> None:
        self.key = key
        self.sink = sink          # optional OTel/Langfuse forwarder
        self._entries: List[Entry] = []

    @property
    def head(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS

    def record(self, verdict: Verdict) -> Entry:
        call = verdict.call
        args_hash = _sha(json.dumps(call.args, sort_keys=True, default=str).encode())
        entry = Entry(
            seq=len(self._entries),
            ts=verdict.ts or time.time(),
            actor=call.actor,
            tool=call.tool,
            args_hash=args_hash,
            decision=verdict.decision.value,
            reason=verdict.reason,
            certificate=verdict.certificate.as_dict() if verdict.certificate else None,
            prev_hash=self.head,
        )
        entry.entry_hash = hmac.new(
            self.key, entry.signing_payload(), hashlib.sha256
        ).hexdigest()
        self._entries.append(entry)
        if self.sink:
            self.sink(asdict(entry))
        return entry

    def verify(self) -> bool:
        """Return True iff the full chain is intact and correctly signed."""
        prev = GENESIS
        for entry in self._entries:
            if entry.prev_hash != prev:
                return False
            expected = hmac.new(
                self.key, entry.signing_payload(), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(expected, entry.entry_hash):
                return False
            prev = entry.entry_hash
        return True

    def __len__(self) -> int:
        return len(self._entries)

    def to_list(self) -> List[dict]:
        return [asdict(e) for e in self._entries]
