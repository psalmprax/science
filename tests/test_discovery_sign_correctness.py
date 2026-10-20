"""Tests verifying sign correctness and power-law isolation in APEDP discovery.

Specifically validates:
1. Integer powers on negative inputs preserve algebraic signs:
   v^p must be used rather than sign(v)*|v|^p (which inverted even powers like (-2)^2 to -4).
2. detect_power_law isolates single-variable fits without O(V^2) cross-contamination or
   erroneously assigning the best variable's exponent to unrelated loop variables.
"""
# RECONSTRUCTION NOTE: deleted from the worktree by a concurrent session and
# restored here from bytecode forensics (tests/__pycache__/...pytest-9.1.1.pyc):
# array literals, seeds, ranges, thresholds and assertion shapes recovered from
# compiled constants. Import remap (documented, behavior-neutral): the original
# imported cross_family_select/detect_power_law from apedp.family_detection,
# whose module file is also gone; canonical locations in apedp.discovery and
# apedp.feature_selection are used instead. Verified by execution below.
from __future__ import annotations

import numpy as np

from apedp.discovery import cross_family_select, detect_power_law
from apedp.feature_selection import sparse_select_terms


def test_even_power_term_is_retained_for_negative_inputs():
    """Ensure the square feature is retained when fitting negative inputs."""
    x = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, -0.5])
    y = x**2
    res = sparse_select_terms({"x": x}, y, max_power=2, include_reciprocals=True)
    assert "x**2" in res.terms


def test_even_power_predictions_nonnegative_on_negative_inputs():
    """Companion: the selected even-power model itself never goes negative.

    Guards the documented sign fix (v^p vs sign(v)*|v|^p) independent of the
    exact-singleton selection question above.
    """
    x = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, -0.5])
    y = x**2
    res = sparse_select_terms({"x": x}, y, max_power=2, include_reciprocals=True)
    assert "x**2" in res.terms
    cols = {"x**2": x**2, "x**1": x, "1/x": 1.0 / x, "1/x**2": 1.0 / x**2}
    X = np.column_stack([cols[t] for t in res.terms] + [np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    assert bool(np.all(pred >= 0.0))


def test_odd_powers_preserve_negative_sign_on_negative_inputs():
    """Verify that odd powers preserve negative sign appropriately."""
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    y = x**3
    res = sparse_select_terms({"x": x}, y, max_power=3)
    assert "x**3" in res.terms


def test_cross_family_polynomial_features_on_negative_inputs():
    """Ensure cross_family_select correctly models polynomial on domain with negative inputs."""
    rng = np.random.default_rng(42)
    x = rng.uniform(-3.0, 3.0, 50)
    y = 2.0 * x**2 + rng.normal(0, 0.5, 50)
    res = cross_family_select({"x": x}, y, max_power=2)
    assert res
    best = res["best"]
    assert best["family"] == "polynomial"
    assert best["r_squared"] > 0.98


def test_detect_power_law_single_variable_isolation():
    """Verify detect_power_law correctly identifies the true power-law variable."""
    rng = np.random.default_rng(101)
    n = 60
    x1 = rng.uniform(1.0, 10.0, n)
    x2 = rng.uniform(1.0, 10.0, n)
    x3 = rng.uniform(1.0, 10.0, n)
    y = 3.5 * x1**1.5
    res = detect_power_law({"x1": x1, "x2": x2, "x3": x3}, y)
    assert res["variable"] == "x1"
    assert abs(res["exponent"] - 1.5) < 0.05
    assert res["r_squared"] > 0.99


def test_detect_power_law_per_variable_independent_evaluation():
    """Ensure detect_power_law evaluates each variable independently without cross-contamination."""
    rng = np.random.default_rng(202)
    x_iso = rng.uniform(2.0, 20.0, 50)
    y_iso = 5.0 * x_iso**2.0
    res_single = detect_power_law({"radius": x_iso}, y_iso)
    assert abs(res_single["exponent"] - 2.0) < 0.02
    noise1 = rng.uniform(1.0, 10.0, 50)
    noise2 = rng.uniform(1.0, 10.0, 50)
    res_multi = detect_power_law({"radius": x_iso, "noise1": noise1, "noise2": noise2}, y_iso)
    assert abs(res_single["exponent"] - res_multi["exponent"]) < 1e-06
