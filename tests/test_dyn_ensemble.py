"""Pins for §§77-78: coupling replication table + M0-M3 ensemble table.
Frozen data + deterministic OLS: values are regression pins. No claim that
any member 'should' win; ensemble asserts utility, never mechanism.
"""

import pytest

import apedp_dyn.epistemic.active_disambiguation.e5_market as M
from apedp_dyn.epistemic.active_disambiguation.drift_e5 import classify
from apedp_dyn.epistemic.active_disambiguation.ensemble_m import run_asset

ASSETS = ("BTC", "ETH", "SOL", "BNB", "DOGE", "AVAX")
PANELS = None


def _panels():
    global PANELS
    if PANELS is None:
        frames = M.load_frames()
        PANELS = {a: M.build_asset_panel(frames["raw"], a) for a in ASSETS}
    return PANELS


def test_coupling_replication_classes():
    expected = {"BTC": "stable-ambiguous", "ETH": "drifting", "SOL": "drifting",
                "BNB": "drifting", "DOGE": "drifting", "AVAX": "drifting"}
    for a in ASSETS:
        _, traj = M.run_coupling(_panels(), a)
        assert classify(traj) == expected[a]


def test_coupling_ha_counts():
    expected = {"BTC": 1, "ETH": 1, "SOL": 1, "BNB": 0, "DOGE": 3, "AVAX": 5}
    for a in ASSETS:
        rows, _ = M.run_coupling(_panels(), a)
        assert sum(s == "H-A" for s in rows[-1].rank_sequence) == expected[a]


def test_ensemble_table():
    expected_d = {"BTC": 0.1361, "ETH": 0.0393, "SOL": 0.0577,
                  "BNB": 0.0342, "DOGE": 0.0064, "AVAX": -0.0979}
    for a in ASSETS:
        r = run_asset({a: _panels()[a]}, a)
        assert r.n_origins == 11
        assert abs(r.r2["M_hist"]) < 1e-9
        assert abs(r.delta_m3_m0 - expected_d[a]) < 1e-3
        assert abs(r.r2["M2"] - r.r2["M3"]) < 0.03  # rankV adds ~nothing once H-A present


def test_coupling_no_promotions():
    for a in ASSETS:
        rows, _ = M.run_coupling(_panels(), a)
        assert all(r.promotions == 0 for r in rows)


def test_ensemble_determinism():
    a = run_asset({"BTC": _panels()["BTC"]}, "BTC")
    b = run_asset({"BTC": _panels()["BTC"]}, "BTC")
    assert a == b


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
