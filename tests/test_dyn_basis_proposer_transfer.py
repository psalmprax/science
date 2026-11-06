"""Frozen-coefficient transfer checks for the polynomial basis proposer.

The two systems below are synthetic equation families, not empirical physics
datasets. Each proposer sees training observations only; its returned scorer is
then evaluated on distinct, out-of-range observations without refitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.proposers.basis_regression_proposer import BasisRegressionProposer


@pytest.mark.parametrize(
    ("domain", "intercept", "linear", "quadratic"),
    [
        ("mechanical_oscillator", 0.4, -2.0, -0.5),
        ("chemical_kinetics", 1.2, 0.8, -0.03),
    ],
)
def test_polynomial_proposer_transfers_frozen_fit_to_unseen_regime(
    domain: str,
    intercept: float,
    linear: float,
    quadratic: float,
):
    """A fitted degree-two structure should recover the same law out of range."""
    x_train = np.linspace(-1.0, 1.0, 81)
    x_holdout = np.linspace(1.05, 2.0, 41)

    def target(x: np.ndarray) -> np.ndarray:
        return intercept + linear * x + quadratic * x**2

    train = pd.DataFrame({"state": x_train})
    holdout = pd.DataFrame({"state": x_holdout})
    proposals = BasisRegressionProposer(basis="poly", degree=2).fit_and_propose(
        train,
        target(x_train),
        variable_names=["state"],
    )
    assert len(proposals) == 1, domain

    prediction = np.asarray(proposals[0].compute_score(holdout), dtype=float)
    expected = target(x_holdout)
    assert not np.intersect1d(x_train, x_holdout).size
    assert np.all(np.isfinite(prediction)), domain
    np.testing.assert_allclose(prediction, expected, rtol=1e-5, atol=1e-6, err_msg=domain)
