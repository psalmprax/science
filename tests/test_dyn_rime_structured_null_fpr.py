"""Layered false-promotion measurement: proposer vs referee on STRUCTURED nulls.

Why this file exists: the system's null-world FPR (0/50, 0/120, 1/120) was
measured on IID Gaussian null worlds. On those, the mean predictor is optimal
and nothing can be promoted — the battery is easy by construction. It does not
describe the proposer's behaviour on STRUCTURED targets that have no dependence
on the available feature, which is the null population that matters.

Measured here (fixed seeds, 40 structured null worlds: AR(1), oscillation,
polynomial trend, heteroskedastic bursts; feature x is iid and irrelevant):

  * PROPOSER-level: 14/40 worlds promote at least one candidate, and every
    promoted equation references the irrelevant feature. Mechanism identified:
    the `State` node turns iid noise into a persistent random walk, which
    correlates with any persistent or trending target.
  * SYSTEM-level (proposer -> whiteness guard |rho1| < 0.5): 0/40. Every one
    of those promotions carries |rho1| >= 0.5 (measured up to +0.906), i.e.
    the guard refuses exactly the ones the proposer wrongly accepted.
  * Grammar expansion (square_seeds + rational_ops + stem_ops + power_ops)
    does not change either number.

Consequence for the record: "RIME has 0% null-world FPR" is a true statement
about the *referee-gated pipeline on IID nulls*. It is not a statement about
the proposer on structured nulls, where it is ~35%. The safety comes from the
referee layer, and the two FPRs are different quantities that must be
reported separately.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.epistemic.adjudicate import RESID_RHO_FLAG
from apedp_dyn.proposers.rime_proposer import RimeProposer

ALL_OPS = dict(square_seeds=True, rational_ops=True, stem_ops=True, power_ops=True)


def _structured_null(n: int, seed: int):
    """Target has structure but no dependence on x (the only feature)."""
    r = np.random.RandomState(seed)
    x = r.randn(n)
    kind = seed % 4
    if kind == 0:                      # AR(1)
        e = np.zeros(n)
        for t in range(1, n):
            e[t] = 0.9 * e[t - 1] + r.randn()
        y = e
    elif kind == 1:                    # oscillation
        t = np.arange(n) / n
        y = np.sin(2 * np.pi * 7 * t) + 0.1 * r.randn(n)
    elif kind == 2:                    # trend + curvature
        t = np.arange(n) / n
        y = 3 * t + 2 * t ** 2 - t ** 3 + 0.05 * r.randn(n)
    else:                              # heteroskedastic bursts
        s = np.where(np.arange(n) > n // 2, 0.1, 5.0)
        y = s * r.randn(n)
    return x, y


def _rho1(resid: np.ndarray) -> float:
    r = resid[np.isfinite(resid)]
    if r.size < 10 or float(np.std(r)) == 0.0:
        return float("nan")
    return float(np.corrcoef(r[:-1], r[1:])[0, 1])


def _run(config: dict) -> tuple[int, int]:
    """Return (proposer promotions, promotions surviving the whiteness guard)."""
    promoted = surviving = 0
    for seed in range(40):
        x, y = _structured_null(600, seed)
        ntr = 400
        tr = pd_like = __import__("pandas").DataFrame({"x": x[:ntr], "target": y[:ntr]})
        te = __import__("pandas").DataFrame({"x": x[ntr:], "target": y[ntr:]})
        props = RimeProposer(feature_cols=["x"], target_col="target",
                             max_depth=3, budget=200, **config).fit_and_propose(tr)
        for q in props:
            promoted += 1
            yp = np.asarray(q.compute_score(te), dtype=float)
            yt = te["target"].to_numpy()
            m = np.isfinite(yt) & np.isfinite(yp)
            r1 = _rho1((yt - yp)[m])
            if np.isfinite(r1) and abs(r1) < RESID_RHO_FLAG:
                surviving += 1
    return promoted, surviving


def test_proposer_fpr_on_structured_nulls_is_nonzero():
    promoted, surviving = _run(dict())
    assert promoted >= 5, f"expected the proposer to leak on structured nulls, got {promoted}"
    assert surviving == 0, f"referee guard must refuse all of them, got {surviving}"
    print(f"\n[layered-fpr] frozen grammar: proposer {promoted}/40, system {surviving}/40")


def test_expanded_grammar_does_not_change_system_level_fpr():
    promoted, surviving = _run(ALL_OPS)
    assert promoted >= 5, promoted
    assert surviving == 0, surviving
    print(f"\n[layered-fpr] ALL ops ON: proposer {promoted}/40, system {surviving}/40")


def test_iid_nulls_stay_silent_both_layers():
    import pandas as pd
    promoted = surviving = 0
    for seed in range(20):
        r = np.random.RandomState(9000 + seed)
        df = pd.DataFrame({"x": r.randn(600), "target": r.randn(600)})
        props = RimeProposer(feature_cols=["x"], target_col="target",
                             max_depth=3, budget=200).fit_and_propose(df)
        promoted += len(props)
        surviving += len(props)  # if they were emitted, the guard judgement is separate;
        # but on IID nulls nothing is emitted at all, so this stays zero.
    assert promoted == 0, promoted
    print("\n[layered-fpr] IID nulls: 0/20 emitted at either layer")
