"""§79 pins: gated ensemble, QLIKE/R2 agreement, permutation control.
Frozen data/seeds (PERM_SEED=6161): permutation p-values are regression
pins. Headline result pinned here: pooled M2-M0 gains do NOT survive the
month-block permutation control on any asset (all p >= 0.24).
"""

import pytest

import apedp_dyn.epistemic.active_disambiguation.e5_market as M
from apedp_dyn.epistemic.active_disambiguation.ensemble_gated import run_asset

ASSETS = ("BTC", "ETH", "SOL", "BNB", "DOGE", "AVAX")
PANELS = None


def _panels():
    global PANELS
    if PANELS is None:
        frames = M.load_frames()
        PANELS = {a: M.build_asset_panel(frames["raw"], a) for a in ASSETS}
    return PANELS


def test_gate_counts():
    expected = {"BTC": 11, "ETH": 9, "SOL": 11, "BNB": 9, "DOGE": 11, "AVAX": 2}
    for a in ASSETS:
        r = run_asset({a: _panels()[a]}, a)
        assert sum(o.gate_open for o in r.origins) == expected[a]
        assert len(r.origins) == 11


def test_perm_control_nonsignificant():
    for a in ASSETS:
        r = run_asset({a: _panels()[a]}, a)
        assert r.perm_p >= 0.24, a


def test_adaptive_is_per_origin_mixture():
    for a in ASSETS:
        r = run_asset({a: _panels()[a]}, a)
        for o in r.origins:
            want = o.q_m2 if o.gate_open else o.q_m0
            assert o.q_ad == want
            assert (o.r2_ad == o.r2_m2) if o.gate_open else (o.r2_ad == o.r2_m0)


def test_btc_per_origin_spot():
    r = run_asset({"BTC": _panels()["BTC"]}, "BTC")
    o = {x.origin: x for x in r.origins}
    assert abs(o["2022-10-01"].r2_m2 - 0.638) < 0.01
    assert abs(o["2023-06-01"].r2_m2 - (-0.304)) < 0.01


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
