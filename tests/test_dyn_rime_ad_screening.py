"""Screening-gated univariate fan-out in RimeAdProposer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer


def _frame():
    rng = np.random.RandomState(7)
    n = 200
    data = {f"x{k}": rng.randn(n) for k in range(10)}
    data["target"] = data["x3"] ** 2 + 0.01 * rng.randn(n)
    return pd.DataFrame(data)


def test_screening_ranks_signal_feature_first():
    df = _frame()
    prop = RimeAdProposer()
    ranked = prop._screen_features(
        df, [c for c in df.columns if c != "target"],
        df["target"].to_numpy(float))
    assert ranked[0] == "x3", ranked
    assert len(ranked) == 5


def test_screen_top_k_bounds_fan_out():
    df = _frame()
    prop = RimeAdProposer(screen_top_k=2)
    ranked = prop._screen_features(
        df, [c for c in df.columns if c != "target"],
        df["target"].to_numpy(float))
    assert ranked == ["x3", ranked[1]] and len(ranked) == 2


def test_fan_out_emits_candidates_on_signal_feature():
    df = _frame()
    cands = RimeAdProposer().fit_and_propose(df)
    assert any("x3" in c.formula for c in cands)
