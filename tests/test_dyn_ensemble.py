"""Pins for §§77-78: coupling replication table + M0-M3 ensemble table.

Re-minted by ERR-INC-74-79: re-derived under the corrected harness — F1 d5
forward-shift fix (``68a3dadb``), forward-overlap purge (``3419425``) and A01
no-leak train blocks (``9a73170``). Frozen data + deterministic OLS: values
are regression pins. No claim that any member 'should' win; ensemble asserts
utility, never mechanism. Substantive updates carried by the re-derivation:
the coupling table now splits 3 stable / 3 drifting, H-A rank-1 counts rose
across the board, and the "M2≈M3 everywhere" claim inverts on ETH/AVAX
(rankV earns an incremental place; the spread is capped and pinned).
Forensics doc §80c carries the erratum table.
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
    expected = {"BTC": "stable-ambiguous", "ETH": "stable-ambiguous",
                "SOL": "stable-ambiguous", "BNB": "drifting",
                "DOGE": "drifting", "AVAX": "drifting"}
    for a in ASSETS:
        _, traj = M.run_coupling(_panels(), a)
        assert classify(traj) == expected[a]


def test_coupling_ha_counts():
    expected = {"BTC": 3, "ETH": 3, "SOL": 5, "BNB": 1, "DOGE": 7, "AVAX": 6}
    for a in ASSETS:
        rows, _ = M.run_coupling(_panels(), a)
        assert sum(s == "H-A" for s in rows[-1].rank_sequence) == expected[a]


def test_ensemble_table():
    expected_d = {"BTC": 0.0913, "ETH": 0.0643, "SOL": 0.0906,
                  "BNB": 0.0970, "DOGE": -0.0237, "AVAX": -0.1120}
    for a in ASSETS:
        r = run_asset({a: _panels()[a]}, a)
        assert r.n_origins == 11
        assert abs(r.r2["M_hist"]) < 1e-9
        assert abs(r.delta_m3_m0 - expected_d[a]) < 1e-3
        # ERR-INC-74-79: M2≈M3 no longer holds universally under the
        # corrected target — ETH (+.0408) and AVAX (+.0721) earn an
        # incremental rankV place. The spread is capped and pinned.
        assert abs(r.r2["M2"] - r.r2["M3"]) < 0.08


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
