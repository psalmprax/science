"""§79 pins: gated ensemble, QLIKE/R2 agreement, permutation control.

Re-minted by ERR-INC-74-79: re-derived under the corrected harness — F1 d5
forward-shift fix (``68a3dadb``), forward-overlap purge (``3419425``) and
A01 no-leak train blocks (``9a73170``). Frozen data/seeds (PERM_SEED=6161);
permutation p-values are regression pins. Headline under the corrected
harness: pooled M2-M0 gains stay within temporal-reshuffling noise on 5/6
assets (min other p = .1244), but ETH is a single marginal survivor at
p = .0498 (~1/201) — disclosed and pinned, not hidden. The old blanket
"all p >= 0.24" required the contaminated pre-fix target. Doc erratum:
Forensics §80c.
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
    expected = {"BTC": 10, "ETH": 6, "SOL": 8, "BNB": 1, "DOGE": 0, "AVAX": 0}
    for a in ASSETS:
        r = run_asset({a: _panels()[a]}, a)
        assert sum(o.gate_open for o in r.origins) == expected[a]
        assert len(r.origins) == 11


def test_perm_control_with_one_marginal_eth():
    for a in ASSETS:
        r = run_asset({a: _panels()[a]}, a)
        if a == "ETH":
            # Exact pin for the single marginal survivor: deterministic at
            # PERM_SEED=6161 under the corrected harness.
            assert r.perm_p is not None and abs(r.perm_p - 0.049751) < 1e-4, a
        else:
            assert r.perm_p is not None and r.perm_p >= 0.12, a


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
    assert abs(o["2022-10-01"].r2_m2 - 0.0749) < 0.01
    assert abs(o["2023-02-01"].r2_m2 - 0.6400) < 0.01
    assert abs(o["2023-06-01"].r2_m2 - (-0.5755)) < 0.01


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
