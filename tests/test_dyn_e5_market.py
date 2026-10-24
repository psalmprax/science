"""E5-market pins (doc S74, re-minted by ERR-INC-74-79): diagnostic-only signatures.

Re-derived under the corrected harness: the F1 d5 target-definition fix
(``68a3dadb``: ``rolling(5).mean().shift(-5)`` — a true 5-day forward mean)
and the A01 no-leak train-block mask (``9a73170``). Under honest forward
targets the market-vs-null drifting contrast the original pins recorded is
GONE: every hypothesis-level margin sits below the kernel resolution floor
in every window, so the raw rank flips (still present descriptively in the
rank sequences) count as zero floor-gated rotation for BOTH the market
trajectory and the stationary null. The earlier BTC "drifting" class was a
harness artifact of the buggy (backward-windowed) target, not a market
property; the microscope now honestly reports stable-ambiguous on both
sides. Forensics doc §80c carries the disclosed erratum table.
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


def test_market_null_signatures_corrected_d5():
    _, traj = M.run_trajectory(_panels())
    _, ntraj = M.run_null(_panels())
    assert classify(traj) == "stable-ambiguous"
    assert classify(ntraj) == "stable-ambiguous"
    # The raw rank sequences may still rotate descriptively, but under the
    # corrected harness no rotation clears the resolution floor on either
    # side — and critically the CONTRAST the old pins asserted (market
    # drifting while null stays stable) does not survive: it required the
    # pre-fix contaminated target. Parity is the honest signature.


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
