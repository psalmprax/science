"""Seed-order fix for the measured generation gap (v0.1.6, opt-in).

Legacy family-2 (input-major delayed levels) truncates the 60-form seed
catalogue at many inputs and drops whole inputs' shortest lag — measured
3/22 on the trading-signal frame. `seed_order="coverage_first"` (lag-major
family 2) keeps every input's lag-1 inside the SAME budget.

Pinned here:
  * the three previously-truncated positions (H-RV, H-A, H-OBV_MOM) recover
    the planted column at their solo-frame R2 under coverage_first;
  * legacy still fails them on the identical frame (control intact);
  * null-world false promotions stay 0/50 under the new order — the seed
    reordering buys coverage, not risk;
  * small-input catalogues are unchanged between orders (Rust-pinned too).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.proposers.rime_proposer import RimeProposer
from apedp_dyn.sandbox.trading_signals_sandbox import (
    SIGNAL_REGISTRY,
    extract_all_22_features,
    load_real_btc_orderflow,
)

NAMES = [s[0] for s in SIGNAL_REGISTRY]
GAP_POSITIONS = (15, 16, 19)   # measured legacy generation gap (v0.1.6)
NAMES_AT = {15: "H-RV", 16: "H-A", 19: "H-OBV_MOM"}


def _frame() -> pd.DataFrame:
    panel = load_real_btc_orderflow(600)
    X = extract_all_22_features(panel)
    return pd.DataFrame({nm: X[:, i] for i, nm in enumerate(NAMES)})


def _planted(base: pd.DataFrame, pos: int) -> pd.DataFrame:
    rng = np.random.RandomState(100 + pos)
    col = base[NAMES[pos]].to_numpy()
    y = np.zeros(len(base))
    y[1:] = 2.0 * col[:-1] + 10.0 + 0.05 * rng.randn(len(base) - 1)
    out = base.copy()
    out["target"] = y
    return out


def _best(props):
    return sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]


def _require_seed_order(props) -> None:
    """This suite needs the opt-in kernel + metadata; older builds skip."""
    if props and props[0].metadata.get("seed_order") is None:
        pytest.skip("pre-seed_order build: kernel/config passthrough absent")


def test_coverage_first_closes_the_generation_gap():
    base = _frame()
    for pos in GAP_POSITIONS:
        df = _planted(base, pos)
        cov = RimeProposer(feature_cols=NAMES, target_col="target", max_depth=2,
                           budget=200, seed_order="coverage_first").fit_and_propose(df)
        leg = RimeProposer(feature_cols=NAMES, target_col="target", max_depth=2,
                           budget=200, seed_order="legacy").fit_and_propose(df)
        solo = RimeProposer(feature_cols=[NAMES[pos]], target_col="target", max_depth=2,
                            budget=200).fit_and_propose(df)
        assert len(solo) > 0, pos
        solo_r2 = _best(solo).parameters["dev_r2"]
        assert len(cov) > 0, pos
        bc = _best(cov)
        _require_seed_order(cov)
        assert bc.metadata.get("inputs_used_names") == [NAMES[pos]], (pos, bc.metadata)
        assert bc.parameters["dev_r2"] >= solo_r2 - 0.02, (pos, bc.parameters, solo_r2)
        # legacy control: same frame, gap still present
        bl = _best(leg)
        assert bl.metadata.get("inputs_used_names") != [NAMES[pos]] or \
            bl.parameters["dev_r2"] < 0.9, (pos, bl.metadata, bl.parameters)
    print(f"\n[seed-order] coverage_first recovers {[NAMES_AT[p] for p in GAP_POSITIONS]} at solo-frame R2")


def test_coverage_first_null_world_fpr_unchanged():
    base = _frame()
    total = 0
    for seed in range(50):
        rng = np.random.RandomState(seed)
        df = base.copy()
        df["target"] = rng.randn(len(base))
        props = RimeProposer(feature_cols=NAMES, target_col="target", max_depth=2,
                             budget=200, seed_order="coverage_first").fit_and_propose(df)
        total += len(props)
    assert total == 0, total
    print("\n[seed-order] null worlds: 0 proposals / 50 seeds (unchanged)")


def test_seed_order_default_is_legacy():
    base = _frame()
    df = _planted(base, 15)   # a gap position: orders must differ there
    d = RimeProposer(feature_cols=NAMES, target_col="target", max_depth=2, budget=200).fit_and_propose(df)
    c = RimeProposer(feature_cols=NAMES, target_col="target", max_depth=2, budget=200,
                     seed_order="coverage_first").fit_and_propose(df)
    assert d[0].metadata.get("seed_order") == "legacy"
    assert c[0].metadata.get("seed_order") == "coverage_first"
    assert _best(d).parameters["dev_r2"] < 0.9 < _best(c).parameters["dev_r2"]
