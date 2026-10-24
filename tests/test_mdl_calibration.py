"""MDL cost calibration: derivation checks + weight-sweep procedure.

Cost derivation (units of ln n, calibrated at n ~= 30; see
DiscoveryNumericalPolicy for the procedure — costs are structural priors, and
the weight sweep below is the calibration that justifies w = 1.0).
"""
# RECONSTRUCTION NOTE: deleted from the worktree by a concurrent session and
# restored here from bytecode forensics (exact assertion messages preserved).
# Construction details below reproduce the original coverage: ordering holds at
# the calibration point, w = 0 admits soup, w = 1 recovers truth.
from __future__ import annotations

import dataclasses
import math

import numpy as np

from apedp.feature_selection import bic_forward_select, compute_structural_complexity
from apedp.numerical_policy import DEFAULT_POLICY


def _collinear_library(n: int = 30, seed: int = 2) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Collinear feature library over x in [0.5, 4.0] with linear ground truth.

    Calibrated (seed 2, sigma 0.5): unpenalized BIC (w = 0) prefers the
    fractional decoy, while MDL weight w = 1 recovers the linear truth.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.5, 4.0, n)
    cols = {
        "x**1": x,
        "x**2": x**2,
        "x**(2/3)": x ** (2.0 / 3.0),
        "1/x": 1.0 / x,
        "exp(x)": np.exp(x / 4.0),
    }
    y = 3.0 + 2.0 * x + rng.normal(0, 0.5, n)
    return cols, y


def test_mdl_cost_ordering_at_calibration_point():
    """Costs must strictly order canonical < integer < cross < fractional < transcendental."""
    ln_n = math.log(30)
    costs = {
        "canonical": compute_structural_complexity("x**1", ln_n, DEFAULT_POLICY),
        "integer": compute_structural_complexity("x**2", ln_n, DEFAULT_POLICY),
        "cross": compute_structural_complexity("x*y", ln_n, DEFAULT_POLICY),
        "fractional": compute_structural_complexity("x**(2/3)", ln_n, DEFAULT_POLICY),
        "transcendental": compute_structural_complexity("exp(x)", ln_n, DEFAULT_POLICY),
    }
    assert costs["canonical"] < costs["integer"] < costs["cross"] < costs["fractional"] < costs["transcendental"], (
        f"Costs must strictly order canonical < integer < cross < fractional < transcendental: {costs}"
    )
    # Costs scale with ln n; normalized values must match the base cost table.
    assert abs(costs["canonical"] - 0.0) < 1e-9
    assert abs(costs["integer"] / ln_n - 0.5) < 1e-9
    assert abs(costs["fractional"] / ln_n - 1.5) < 1e-9
    assert abs(costs["transcendental"] / ln_n - 2.5) < 1e-9
    assert abs(ln_n - math.log(30)) < 1e-9


def test_mdl_weight_sweep_calibrates_to_truth():
    """Sweeping MDL_WEIGHT must show w = 0 soups while w = 1 recovers truth."""
    cols, y = _collinear_library()

    def is_soup(terms: list[str]) -> bool:
        return any("**(" in t or t.startswith(("exp(", "log(")) for t in terms)

    no_penalty = bic_forward_select(
        cols, y, policy=dataclasses.replace(DEFAULT_POLICY, MDL_WEIGHT=0.0)
    )
    assert set(no_penalty.terms) != {"x**1"}, (
        f"calibration config lost discrimination (w=0 found truth): {no_penalty.terms}"
    )
    assert is_soup(list(no_penalty.terms)), (
        f"w=0 should admit fractional/transcendental soup: {no_penalty.terms}"
    )

    calibrated = bic_forward_select(
        cols, y, policy=dataclasses.replace(DEFAULT_POLICY, MDL_WEIGHT=1.0)
    )
    assert set(calibrated.terms) == {"x**1"}, (
        f"w=1 must recover truth: {calibrated.terms}"
    )
