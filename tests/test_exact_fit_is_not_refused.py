"""An exact fit is accepted, not refused as unassessable (SYSTEM_FLAWS_AUDIT M-01).

Before 2026-11-19 a holdout residual of exact zeros made both whiteness checks
return "residual series is constant ... unassessable", so noise-free data from
the right law was refused whenever rounding happened to leave no error, and the
verdict depended on the platform's floating point.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine
from apedp_dyn.epistemic.adjudicate import EpistemicStatus, _sole_survivor_whiteness
from apedp_dyn.epistemic.whiteness_calibration import check_whiteness

CHECKS = [lambda r: _sole_survivor_whiteness(r, 0.5), lambda r: check_whiteness(r, 0.5)]


@pytest.mark.parametrize("check", CHECKS, ids=["adjudicate", "whiteness_calibration"])
def test_exact_zero_residuals_pass_as_an_exact_fit(check):
    ok, reason = check(np.zeros(60))
    assert ok and "exact fit" in reason


@pytest.mark.parametrize("check", CHECKS, ids=["adjudicate", "whiteness_calibration"])
def test_a_constant_offset_is_still_refused(check):
    ok, _ = check(np.full(60, 0.25))
    assert not ok


@pytest.mark.parametrize("check", CHECKS, ids=["adjudicate", "whiteness_calibration"])
def test_one_nonzero_residual_goes_through_the_ordinary_test(check):
    resid = np.zeros(60)
    resid[17] = 1e-3
    _ok, reason = check(resid)
    assert "exact fit" not in reason


def test_noise_free_data_from_the_true_law_is_promoted():
    """The case the blind benchmark lost nine textbook laws to: p = m v, no noise."""
    rng = np.random.default_rng(0)
    X = {"mass": rng.uniform(1, 10, 120), "vel": rng.uniform(1, 10, 120), "other": rng.uniform(1, 10, 120)}
    y = X["mass"] * X["vel"]
    result = L2DiscoveryEngine(operators=("+", "-", "*", "/", "pow")).discover(
        X=X, Y=y, dimensions={"mass": Dimension(m=1), "vel": Dimension(l=1, t=-1), "other": Dimension(t=1)},
        target_dimension=Dimension(m=1, l=1, t=-1), target_name="y", seed=1,
    )
    adj = result.adjudication
    assert adj.best_rmse == 0.0  # the case is the exact-zero one, not rounding noise
    assert adj.status is EpistemicStatus.UNIQUE_SUPPORT, adj.reason
    assert adj.promoted.candidate.metadata["canonical_expr"] == "mass*vel"
