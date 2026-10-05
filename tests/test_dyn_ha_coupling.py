"""H-A/E5 coupling pins (doc S76). Identifiability case study, not performance.
Frozen data; rank sequence is a deterministic regression pin. No assertion
that any hypothesis 'should' win — the questions are rank persistence,
floor margins, rotation, and consolidation.
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


def test_coupling_class():
    _, traj = M.run_coupling(_panels(), "BTC")
    assert classify(traj) == "stable-ambiguous"


def test_coupling_rank_sequence():
    rows, _ = M.run_coupling(_panels(), "BTC")
    seq = rows[-1].rank_sequence
    assert seq == ("H-RV", "H-RV", "H-A", "H-P", "H-P", "H-P", "H-P", "H-P", "H-P", "H-P", "H-P")
    assert sum(s == "H-A" for s in seq) == 1


def test_coupling_no_consolidation():
    rows, _ = M.run_coupling(_panels(), "BTC")
    assert all(r.promotions == 0 for r in rows)


def test_coupling_determinism():
    a, _ = M.run_coupling(_panels(), "BTC")
    b, _ = M.run_coupling(_panels(), "BTC")
    assert [r.rank_sequence for r in a] == [r.rank_sequence for r in b]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
