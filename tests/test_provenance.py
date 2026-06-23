from atl import Decision, PolicyResult, ProvenanceLog, ToolCall, Verdict


def _verdict(tool="t", decision=Decision.ALLOW):
    return Verdict(decision, ToolCall(tool), PolicyResult.allow(), reason="x")


def test_chain_intact_after_appends():
    log = ProvenanceLog(key=b"k")
    for _ in range(5):
        log.record(_verdict())
    assert len(log) == 5
    assert log.verify()


def test_tampering_breaks_chain():
    log = ProvenanceLog(key=b"k")
    for _ in range(3):
        log.record(_verdict())
    # Mutate a past entry's decision in place.
    log._entries[1].decision = "block"
    assert not log.verify()


def test_reorder_breaks_chain():
    log = ProvenanceLog(key=b"k")
    log.record(_verdict("a"))
    log.record(_verdict("b"))
    log._entries[0], log._entries[1] = log._entries[1], log._entries[0]
    assert not log.verify()


def test_sink_receives_entries():
    seen = []
    log = ProvenanceLog(key=b"k", sink=seen.append)
    log.record(_verdict())
    assert seen and seen[0]["tool"] == "t"
