"""E5-R1 cross-asset replication pins (doc S75). Frozen data/seeds; the table
below is a deterministic regression pin — any change must be intentional.
Assets chosen before results: BTC, ETH, SOL, BNB, DOGE, AVAX (no PAXG in
this file; heterogeneous majors/meme/L1 substitute set).
"""

import pytest

import apedp_dyn.epistemic.active_disambiguation.e5_market as M
from apedp_dyn.epistemic.active_disambiguation.drift_e5 import classify
from apedp_dyn.epistemic.active_disambiguation.trading_e4t1_v2 import ORIGINS

ASSETS = ("BTC", "ETH", "SOL", "BNB", "DOGE", "AVAX")
PANELS = None


def _panels():
    global PANELS
    if PANELS is None:
        frames = M.load_frames()
        PANELS = {a: M.build_asset_panel(frames["raw"], a) for a in ASSETS}
    return PANELS


def test_r1_market_classes():
    expected = dict.fromkeys(ASSETS, "drifting")
    for a in ASSETS:
        _, traj = M.run_trajectory(_panels(), a)
        assert classify(traj) == expected[a]


def test_r1_null_controls():
    # Frozen-seed (5150) nulls: 4/6 stable-ambiguous; ETH+BNB misclassify on
    # this seed only (pass on 5151-53) -> ~12% trajectory-level false-positive
    # rate, quantified not hidden. Market 6/6 vs null 3/24 recorded in doc.
    expected = {"BTC": "stable-ambiguous", "ETH": "drifting", "SOL": "stable-ambiguous",
                "BNB": "drifting", "DOGE": "stable-ambiguous", "AVAX": "stable-ambiguous"}
    for a in ASSETS:
        _, ntraj = M.run_null(_panels(), a)
        assert classify(ntraj) == expected[a]


def test_r1_no_promotions():
    for a in ASSETS:
        rows, _ = M.run_trajectory(_panels(), a)
        assert all(r.promotions == 0 for r in rows)


def test_r1_calendar_descriptive():
    seqs = {}
    for a in ASSETS:
        rows, _ = M.run_trajectory(_panels(), a)
        seqs[a] = rows[-1].rank_sequence
    cal = M.rotation_calendar(seqs, ORIGINS)
    assert len(cal) == len(ORIGINS) - 1
    assert any(len(changed) >= 3 for _, changed in cal)  # Mar-Jun clustering
    assert any(len(changed) == 0 for _, changed in cal)  # Nov-22 none


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
