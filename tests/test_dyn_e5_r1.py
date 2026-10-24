"""E5-R1 cross-asset replication pins (doc S75, re-minted by ERR-INC-74-79).

Re-derived under the corrected harness — F1 d5 forward-shift fix (``68a3dadb``)
and A01 no-leak train-block mask (``9a73170``). Assets chosen before results:
BTC, ETH, SOL, BNB, DOGE, AVAX (no PAXG in this file; heterogeneous
majors/meme/L1 substitute set). Under honest forward targets no asset clears
the kernel resolution floor through this microscope: the market table is
6/6 stable-ambiguous (the 6/6 "drifting" contrast required the contaminated
pre-fix target), and the seed-5150 null set carries exactly one trajectory
false-positive (BNB), quantified not hidden. The descriptive rotation
calendar still rotates everywhere (no empty boundary) with the 2023H1
clustering preserved. Doc erratum: Forensics §80c.
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
    # ERR-INC-74-79: under the corrected forward target the market table is
    # 6/6 stable-ambiguous (was 6/6 drifting under the F1-buggy target).
    expected = dict.fromkeys(ASSETS, "stable-ambiguous")
    for a in ASSETS:
        _, traj = M.run_trajectory(_panels(), a)
        assert classify(traj) == expected[a]


def test_r1_null_controls():
    # Frozen-seed (5150) nulls re-derived (ERR-INC-74-79): 5/6 stable-
    # ambiguous; BNB misclassifies on this seed (drifting) — a quantified
    # trajectory-level false-positive, not hidden. The seed 5151-53 pass
    # observation applies to the old target and is no longer the claim.
    expected = {"BTC": "stable-ambiguous", "ETH": "stable-ambiguous",
                "SOL": "stable-ambiguous", "BNB": "drifting",
                "DOGE": "stable-ambiguous", "AVAX": "stable-ambiguous"}
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
    # Exact deterministic boundary sizes under the corrected harness.
    assert [len(changed) for _, changed in cal] == [2, 4, 3, 4, 3, 4, 2, 4, 4, 4]
    # 2023H1 mild clustering persists: several boundaries move >= 4 assets.
    assert any(len(changed) >= 4 for _, changed in cal)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
