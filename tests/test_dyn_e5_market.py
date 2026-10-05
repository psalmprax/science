"""E5-market pins (doc S74): diagnostic-only trajectory signatures.

Frozen data + frozen seeds: market BTC trajectory classifies drifting while
the pipeline-matched stationary null classifies stable-ambiguous (control
passes: the machinery does not manufacture rotation). No status, no
strategy, no selection use — microscope only.
"""

import pytest

import apedp_dyn.epistemic.active_disambiguation.e5_market as M
from apedp_dyn.epistemic.active_disambiguation.drift_e5 import classify

PANELS = None


def _panels():
    global PANELS
    if PANELS is None:
        frames = M.load_frames()
        PANELS = {a: M.build_asset_panel(frames["raw"], a) for a in ("BTC", "ETH", "SOL")}
    return PANELS


def test_market_drifting_null_stable():
    _, traj = M.run_trajectory(_panels())
    _, ntraj = M.run_null(_panels())
    assert classify(traj) == "drifting"
    assert classify(ntraj) == "stable-ambiguous"
    assert classify(traj) != classify(ntraj)


def test_no_promotions_anywhere_oracle_multi():
    rows, _ = M.run_trajectory(_panels())
    assert all(r.promotions == 0 for r in rows)
    assert all("multi" in r.oracle_status.lower() for r in rows)


def test_record_completeness():
    rows, _ = M.run_trajectory(_panels())
    assert len(rows) == 11
    for r in rows:
        assert len(r.rank_sequence) == 11 or r.origin
        assert r.peak_margin >= r.cumul_margin - 1e-12


def test_determinism():
    a, _ = M.run_trajectory(_panels())
    b, _ = M.run_trajectory(_panels())
    assert [r.cumul_margin for r in a] == [r.cumul_margin for r in b]
    assert [r.rank_sequence for r in a] == [r.rank_sequence for r in b]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
