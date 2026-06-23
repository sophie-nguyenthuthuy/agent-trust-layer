from atl import CertifiedGate


def test_cold_start_never_trips():
    g = CertifiedGate(baseline=0.1, delta=0.05, v_floor=0.2)
    # A single high reading must not trip: the Azuma margin is huge at n=1.
    cert = g.observe(0.9)
    assert not cert.tripped
    assert cert.bound > 0.5


def test_sustained_drift_trips():
    g = CertifiedGate(baseline=0.1, delta=0.05, window=50, v_floor=0.2)
    tripped = False
    for _ in range(60):
        cert = g.observe(0.7)
        tripped = tripped or cert.tripped
    assert tripped
    assert cert.mean_risk > cert.threshold


def test_low_risk_stays_below_v_floor():
    g = CertifiedGate(baseline=0.1, delta=0.05, v_floor=0.25)
    for _ in range(200):
        cert = g.observe(0.05)
    assert not cert.tripped
    assert cert.mean_risk < g.v_floor


def test_severity_escalate_vs_block():
    g = CertifiedGate(baseline=0.1, delta=0.05, window=30, v_floor=0.2,
                      block_factor=2.0)
    for _ in range(40):
        cert = g.observe(0.95)
    assert g.severity(cert) == "block"


def test_certificate_is_serializable():
    g = CertifiedGate()
    d = g.observe(0.3).as_dict()
    assert {"tripped", "n", "mean_risk", "threshold", "delta"} <= d.keys()
