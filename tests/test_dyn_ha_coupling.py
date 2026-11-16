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
    # The sequence pinned here until 2026-11-16, H-RV,H-RV,H-A,H-P x8, was
    # recorded before FLAW-A01 (2026-10-21) removed the train blocks that read
    # forward through the five-day shift in `_d5`; the leak carried the H-P
    # plurality. CHANGELOG 2026-11-02 ("a stale pin corrected against the
    # record") describes this correction, but the test file was not committed
    # with it and the old pin kept failing.
    rows, _ = M.run_coupling(_panels(), "BTC")
    seq = rows[-1].rank_sequence
    assert seq == ("H-RV", "H-RV", "H-A", "H-A", "H-A", "H-RV", "H-RV", "H-RV", "H-RV", "H-RV", "H-RV")
    assert sum(s == "H-A" for s in seq) == 3


#: H-A rank-1 counts over the 11 origins, from the re-derivation table in
#: docs/TRADING_EQUATION_FORENSICS.md (row "§77 coupling classes", corrected
#: harness column). The pre-fix column read 1/1/1 for the same three assets.
RECORDED_H_A_RANK_1_COUNTS = {"BTC": 3, "ETH": 3, "SOL": 5}


@pytest.mark.parametrize("asset", sorted(RECORDED_H_A_RANK_1_COUNTS))
def test_coupling_reproduces_the_recorded_post_fix_counts(asset):
    """The pin above is the documented value, not a snapshot of today's code.

    The forensics record was re-derived on the corrected harness before this
    test existed, so agreeing with it is independent evidence; the leaking
    train blocks returning would give 1 for every asset here.
    """
    rows, traj = M.run_coupling(_panels(), asset)
    seq = rows[-1].rank_sequence
    assert sum(s == "H-A" for s in seq) == RECORDED_H_A_RANK_1_COUNTS[asset]
    assert classify(traj) == "stable-ambiguous"


def test_coupling_no_consolidation():
    rows, _ = M.run_coupling(_panels(), "BTC")
    assert all(r.promotions == 0 for r in rows)


def test_coupling_determinism():
    a, _ = M.run_coupling(_panels(), "BTC")
    b, _ = M.run_coupling(_panels(), "BTC")
    assert [r.rank_sequence for r in a] == [r.rank_sequence for r in b]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
