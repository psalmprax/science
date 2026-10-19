"""§80 pins: headline A|P,RV contrast, redundancy, suppression, controls.
Frozen seeds; permutation (200 draws) is the pre-registered headline —
DM reported alongside. Slow (~15 min): full battery, run rarely.
"""

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
    expected = {"BTC": 0.0287, "ETH": 0.0093, "SOL": 0.0078,
                "BNB": 0.0053, "DOGE": -0.0043, "AVAX": -0.0071}
    for a in ASSETS:
        r = run_asset({a: _panels()[a]}, a)
        assert abs(r.contrast_q["A|P,RV"] - expected[a]) < 2e-3, a


def test_perm_headline_nonsignificant():
    for a in ASSETS:
        r = run_asset({a: _panels()[a]}, a)
        assert r.perm_p >= 0.24, a


def test_rv_dead_suppressor_pattern():
    for a in ASSETS:
        r = run_asset({a: _panels()[a]}, a)
        assert abs(r.contrast_q["RV|P,A"]) < 0.008, a
        assert r.sign_flip["HA"] >= 0.9, a
        assert r.corr_pred["P,A"] > 0.7, a


def test_btc_full_row():
    r = run_asset({"BTC": _panels()["BTC"]}, "BTC")
    assert abs(r.contrast_q["A|P"] - 0.0321) < 2e-3
    assert r.wins["P+RV+A"] + r.wins["P+A"] >= 8
    assert abs(r.dm_p - 0.036) < 0.02


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
