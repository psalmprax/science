"""Coverage challenge: per-position planted recovery on the 22-signal frame
+ funnel invariants + null worlds + the generation-vs-qualification split.

Planted dependence at each of 22 input positions: y[t] = 2*x[t-1] + 10 +
noise (lag-1, inside the frozen LAG_SET). The search must recover the exact
column. Measured (seeds fixed, this file's own RNG):

  exact recovery           17/22
  collinear substitute      2/22  (H-OFI->H-CVD, H-STOCH->H-WILLR: in-grammar
                                 twins fitting identically; MULTI, not UNIQUE)
  truncation miss           3/22  (H-RV, H-A, H-OBV_MOM: solo-frame control
                                 recovers the SAME dependence at R2 0.96-0.9995,
                                 so the form is in the catalogue but its seed is
                                 cut by the 60-form budget at 22 inputs ->
                                 generation gap, not a scoring failure)

Funnel invariants asserted on every run: rejected_score + accepted ==
evaluated, evaluated <= budget, dropped_* >= 0.
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
SUBSTITUTE = {0: "H-CVD", 12: "H-WILLR"}       # collinear twins (in-grammar)
TRUNCATED = (15, 16, 19)                        # 22-input frame vs solo control


def _frame() -> pd.DataFrame:
    panel = load_real_btc_orderflow(600)
    X = extract_all_22_features(panel)
    return pd.DataFrame({nm: X[:, i] for i, nm in enumerate(NAMES)})


def _best(props, pos):
    return sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]


def _require_instrumentation(props) -> None:
    """Coverage claims need structural inputs_used_names; older builds skip."""
    if props and props[0].metadata.get("inputs_used_names") is None:
        pytest.skip("pre-instrumentation build: inputs_used_names absent")


def _planted(df_named: pd.DataFrame, pos: int) -> pd.DataFrame:
    rng = np.random.RandomState(100 + pos)
    col = df_named[NAMES[pos]].to_numpy()
    y = np.zeros(len(df_named))
    y[1:] = 2.0 * col[:-1] + 10.0 + 0.05 * rng.randn(len(df_named) - 1)
    out = df_named.copy()
    out["target"] = y
    return out


def _funnel_invariants(m) -> None:
    if m.get("rejected_score") is None:
        return  # pre-instrumentation kernel (frozen snapshot): skip
    assert m.get("rejected_score", 0) + m.get("accepted", 0) == m.get("evaluated", -1), m
    assert m.get("evaluated", 0) <= 200, m
    for k in ("rejected_score", "dropped_dedup", "dropped_scope"):
        assert m.get(k, 0) >= 0, (k, m)


def test_coverage_per_position_recovery():
    base = _frame()
    exact = 0
    for pos in range(22):
        df = _planted(base, pos)
        props = RimeProposer(feature_cols=NAMES, target_col="target",
                             max_depth=2, budget=200).fit_and_propose(df)
        _require_instrumentation(props)
        if pos in TRUNCATED:
            continue  # measured separately (isolation control below)
        assert len(props) > 0, pos
        b = _best(props, pos)
        _funnel_invariants(b.metadata)
        used = b.metadata.get("inputs_used_names")
        assert b.parameters["dev_r2"] > 0.45, (pos, b.parameters["dev_r2"])
        if pos in SUBSTITUTE:
            assert used == [SUBSTITUTE[pos]], (pos, used)  # twin, not planted
        else:
            assert used == [NAMES[pos]], (pos, used)        # exact
            assert b.parameters["dev_r2"] > 0.98, (pos, b.parameters["dev_r2"])
            exact += 1
    assert exact == 17, exact
    print(f"\n[coverage] exact={exact}/22; substitutes={len(SUBSTITUTE)}; truncation-miss={len(TRUNCATED)}")


def test_coverage_truncated_positions_recover_in_solo_frame():
    # Isolation control: SAME dependence, single-column frame. If it recovers
    # here but not at 22 inputs, the loss is seed-catalogue truncation
    # (generation), not scoring.
    base = _frame()
    for pos in TRUNCATED:
        df = _planted(base, pos)
        solo = RimeProposer(feature_cols=[NAMES[pos]], target_col="target",
                            max_depth=2, budget=200).fit_and_propose(df)
        assert len(solo) > 0, pos
        b = _best(solo, pos)
        assert b.metadata.get("inputs_used_names") == [NAMES[pos]], (pos, b.metadata.get("inputs_used_names"))
        assert b.parameters["dev_r2"] > 0.95, (pos, b.parameters["dev_r2"])
    print(f"\n[coverage] solo-frame control recovers {[NAMES[p] for p in TRUNCATED]} at R2>0.95")


def test_coverage_null_worlds_abstain():
    base = _frame()
    for seed in (301, 302):
        rng = np.random.RandomState(seed)
        df = base.copy()
        df["target"] = rng.randn(len(base))
        props = RimeProposer(feature_cols=NAMES, target_col="target",
                             max_depth=2, budget=200).fit_and_propose(df)
        assert len(props) == 0, seed
    print("\n[coverage] null worlds: 0 proposals at both seeds")
