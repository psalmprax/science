"""§80 pins: headline A|P,RV contrast, redundancy, suppression, controls.
Frozen seeds; permutation (200 draws) is the pre-registered headline —
DM reported alongside. Slow (~15 min): full battery, run rarely.

ERR-INC-80 (2026-10-20, disclosed in docs/TRADING_EQUATION_FORENSICS.md
§80b): reference values were RE-DERIVED under the forward-overlap purge of
the shared development harness (`trading_e4t1_v2.run_origin`, commit
3419425 — dev targets now exclude the 5-day-ahead window). No threshold
moved: the permutation headline remains primary at alpha=0.05 and the §80
decision (Case 2/4, incremental information not established) is unchanged.
Same seed (8080), same artifacts; harness is deterministic (verified).

ERR-INC-80 amend (2026-10-22, disclosed in the same doc §80c): the pins
missed the A01 no-leak block-mask fix (`incremental_info` now shares
`trading_e4t1_v2::_train_blocks_no_leak`, commit 9a73170) and the F1 d5
target correction (68a3dadb). Values re-derived again under the fully
corrected harness; conclusions unchanged (permutation NULL on all six,
RV|P,A dead, suppressor intact — BTC's HA sign-flip is 9/11 = 0.818,
below the old 0.9 blanket, and is pinned as such)."""

import numpy as np
import pytest

import apedp_dyn.epistemic.active_disambiguation.e5_market as M
from apedp_dyn.epistemic.active_disambiguation.incremental_info import _dm_hac4, run_asset

ASSETS = ("BTC", "ETH", "SOL", "BNB", "DOGE", "AVAX")
PANELS = None


def test_dm_hac4_normalizes_long_run_variance_for_mean():
    differentials = [value for _ in range(32) for value in (-0.8, 1.2)]
    t_stat, p_value = _dm_hac4(np.asarray(differentials))

    assert t_stat > 3.0
    assert p_value < 0.001


def _panels():
    global PANELS
    if PANELS is None:
        frames = M.load_frames()
        PANELS = {a: M.build_asset_panel(frames["raw"], a) for a in ASSETS}
    return PANELS


def test_headline_contrast_values():
    # ERR-INC-80 amend (fully corrected harness: purge + F1 d5 fix + A01
    # no-leak blocks, seed 8080).
    expected = {"BTC": 0.012767, "ETH": 0.017194, "SOL": 0.016678,
                "BNB": 0.018920, "DOGE": 0.000279, "AVAX": -0.000455}
    for a in ASSETS:
        r = run_asset({a: _panels()[a]}, a)
        assert abs(r.contrast_q["A|P,RV"] - expected[a]) < 2e-3, a


def test_perm_headline_nonsignificant():
    # Exact per-asset permutation p-values re-derived under the fully
    # corrected harness (seed 8080, deterministic); the pre-registered
    # headline rule is perm_p >= alpha = 0.05 — held by every asset (min:
    # ETH .0846).
    expected = {"BTC": 0.676617, "ETH": 0.084577, "SOL": 0.213930,
                "BNB": 0.766169, "DOGE": 0.840796, "AVAX": 0.611940}
    for a in ASSETS:
        r = run_asset({a: _panels()[a]}, a)
        assert abs(r.perm_p - expected[a]) < 0.01, a  # exact re-derived pin
        assert r.perm_p >= 0.05, a  # §80 headline rule: permutation NULL


def test_rv_dead_suppressor_pattern():
    # RV|P,A stays dead (all |contrast| < 0.008) and HA sign-flips on >=80%
    # of origins on every asset (BTC 9/11 = 0.818 under the corrected
    # harness, others 11/11); corr(ŷP, ŷA) is pinned EXACTLY per asset
    # (ERR-INC-80 amend: BNB .410 is still the weak-redundancy outlier).
    expected_corr = {"BTC": 0.66359937, "ETH": 0.89258993, "SOL": 0.86398638,
                     "BNB": 0.40950538, "DOGE": 0.73909284, "AVAX": 0.84336229}
    for a in ASSETS:
        r = run_asset({a: _panels()[a]}, a)
        assert abs(r.contrast_q["RV|P,A"]) < 0.008, a
        assert r.sign_flip["HA"] >= 0.8, a
        assert abs(r.corr_pred["P,A"] - expected_corr[a]) < 2e-2, a


def test_btc_full_row():
    r = run_asset({"BTC": _panels()["BTC"]}, "BTC")
    # ERR-INC-80: wins now pin exactly (3+3 = 6 under the purged harness, was
    # >= 8 under the overlapping-target harness — see §80b disclosure).
    assert abs(r.contrast_q["A|P"] - 0.019839) < 2e-3
    assert r.wins["P+RV+A"] == 3
    assert r.wins["P+A"] == 3
    assert abs(r.dm_p - 0.058541) < 0.02


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
