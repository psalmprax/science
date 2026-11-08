"""End-to-end probes: a sum-named column must not lose its coefficient.

The coefficient-prefix audit found eight proposers that format a fitted
coefficient onto a **caller-supplied name** — a library term, a DataFrame
column, a feature label. Every name in use today is atomic, so the defect was
latent, and the fix (`apedp_dyn/grammar/term_format.scaled`) was pinned by a
formatter unit test and a registry entry. That is a proof about the formatter,
not about the proposers: nothing showed that a proposer given a name containing
an operator still emits one shared coefficient.

These tests do. Each builds the proposer's real input with a column literally
named ``"a + b"``, calls the emitting entry point, and asserts on the formula
that comes out. The hazard is exact: before the fix, ``f"{c}*{name}"`` produced
``1.2345*a + b`` — coefficient 1.2345 on ``a`` and 1 on ``b`` — which is a
different function from the one the scorer computes.

Two details keep these probes honest:

- every probe asserts it produced a proposal, so an empty list cannot pass;
- ``assert_coefficient_binds_whole_column`` is itself tested against the
  unparenthesised form, so a helper that accepted everything would fail.

Not covered here, with the reason: `CompartmentalSirProposer` needs a
compartment panel whose column names are already fixed by the epidemiology
contract, `RimeAdProposer._propose_univariate` fans out to four Rust
representations after the emitting method, and `SINDyProposer` and
`qed.library.render` already have behavioural probes with sum-named inputs in
`tests/test_term_builder_structure_audit.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sympy as sp

SUM_COLUMN = "a + b"


def assert_coefficient_binds_whole_column(formula: str, column: str = SUM_COLUMN) -> None:
    """Every appearance of ``column`` must be wrapped as a coefficient operand."""
    assert f"*({column})" in formula, (
        f"no coefficient was formatted onto the parenthesised column: {formula}")
    remainder = formula.replace(f"*({column})", "")
    assert f"*{column}" not in remainder, (
        "a coefficient was formatted onto the bare column name, so it binds to "
        f"the first addend only: {formula}")


def _panel(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A frame whose feature column is literally named ``a + b``.

    The column is a random walk plus noise, so its trailing standard deviation
    actually varies and the regime split has two populated sides; the target is
    a fixed multiple of it plus small noise, so any proposer that fits the
    column has to put a real coefficient on it.
    """
    rng = np.random.default_rng(seed)
    column = np.cumsum(rng.normal(size=n)) + rng.normal(size=n)
    y = 0.5 * column + rng.normal(scale=0.01, size=n)
    return pd.DataFrame({SUM_COLUMN: column, "y": y,
                         "vol": np.abs(rng.normal(size=n)) + 1.0})


# --------------------------------------------------------------- the helper


def test_the_helper_rejects_the_unparenthesised_form():
    """A helper that accepted the hazard would make every probe below useless."""
    with pytest.raises(AssertionError):
        assert_coefficient_binds_whole_column("+1.2345*a + b", SUM_COLUMN)
    assert_coefficient_binds_whole_column("+1.2345*(a + b)", SUM_COLUMN)


# ------------------------------------------------------ regime switching


def test_regime_switching_binds_a_sum_named_column():
    from apedp_dyn.proposers.regime_switching_proposer import RegimeSwitchingProposer

    panel = _panel()
    proposals = RegimeSwitchingProposer().fit_and_propose(
        panel, feature_cols=[SUM_COLUMN], target_col="y", vol_col="vol")

    assert proposals, "the proposer returned nothing; the probe proves nothing"
    formula = proposals[0].formula
    assert formula.count(f"*({SUM_COLUMN})") == 2, (
        f"expected one bound coefficient per regime branch: {formula}")
    assert_coefficient_binds_whole_column(formula)


# --------------------------------------------------- distributed lag terms


def test_distributed_lag_binds_a_sum_named_column():
    from apedp_dyn.proposers.distributed_lag_interaction_proposer import (
        DistributedLagInteractionProposer,
    )

    panel = _panel()
    proposals = DistributedLagInteractionProposer().fit_and_propose(
        panel, feature_cols=[SUM_COLUMN], target_col="y")

    assert proposals, "the proposer returned nothing; the probe proves nothing"
    formula = proposals[0].formula
    assert "intercept(" in formula, formula
    assert_coefficient_binds_whole_column(formula)


# ------------------------------------------------ sparse microstructure


def test_sparse_microstructure_binds_a_sum_named_column(monkeypatch):
    """The column names come from the feature extractor, so the extractor is
    swapped for one that names a sum; everything downstream is untouched."""
    from apedp_dyn.proposers import sparse_microstructure_proposer as module

    panel = _panel()
    y = panel["y"].to_numpy(dtype=float)
    feats = panel[[SUM_COLUMN]].copy()

    monkeypatch.setattr(module, "extract_microstructure_features",
                        lambda _df: (feats, y))

    proposals = module.SparseMicrostructureProposer().fit_and_propose(
        {"panel": panel}, target_col="y")

    assert proposals, "the proposer returned nothing; the probe proves nothing"
    formula = proposals[0].formula
    assert_coefficient_binds_whole_column(formula)

    # The formula must describe the arithmetic: the scorer weights the column by
    # the same coefficient, and the fitted weight is not 1 (which is what the
    # unparenthesised form would have implied for the second addend).
    parsed = sp.sympify(formula.replace(" ", ""))
    # One coefficient, applied to the whole sum: it is the coefficient of each
    # symbol inside the brackets, and there is no coefficient of its own on the
    # second addend.
    coefficient = float(parsed.coeff(sp.Symbol("a")))
    scored = proposals[0].compute_score(panel)
    expected = coefficient * panel[SUM_COLUMN].to_numpy(dtype=float)
    # The formula prints four decimals while the scorer keeps full precision, so
    # this is a structural comparison, not a bit-for-bit one: what it rules out
    # is the unparenthesised form, which would put a coefficient on `a` alone
    # and score a different function (~2x off for a target built from the sum).
    assert np.allclose(scored, expected, rtol=1e-3, atol=1e-4)


# ------------------------------------------------------- RIME-AD additive


def test_rime_ad_multivariate_binds_a_sum_named_column():
    """The emitting method itself: `_propose_multivariate` is where the
    coefficient meets the column name. The public entry point reaches it
    unchanged for a multi-feature frame; the fan-out that follows is a
    different site and a different test."""
    from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer

    panel = _panel()
    panel["c"] = np.random.default_rng(1).normal(size=len(panel))
    y = panel["y"].to_numpy(dtype=float)

    proposals = RimeAdProposer()._propose_multivariate(
        panel, [SUM_COLUMN, "c"], y)

    assert proposals, "the proposer returned nothing; the probe proves nothing"
    formula = proposals[0].formula
    assert_coefficient_binds_whole_column(formula)
    # The interaction column built here is named "a + b*a + b" (the two feature
    # labels joined by "*"); it carries a top-level operator for the same
    # reason, so any coefficient on it is also bound as a unit.
    assert "*a + b" not in formula.replace(f"*({SUM_COLUMN})", ""), formula
