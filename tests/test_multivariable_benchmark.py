"""Multi-variable separable benchmark (Level 1 & Level 2).

Evaluates the integrated pipeline on a 3-variable coupled/separable target:
    y = 3.2*x1**2 + 1.7*exp(-0.8*x2) + 0.4/x3 + epsilon
Verifies:
1. SeparabilityAnalyzer discovers full additive coordinate partitioning [[x1], [x2], [x3]].
2. Independent coordinate fitting with VarPro recovers the ground-truth parameters.
3. Identifiability diagnostics confirm well-conditioned bases.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from apedp.separability import SeparabilityAnalyzer
from apedp.varpro import VarProOptimizer


@pytest.fixture
def multivar_dataset():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.uniform(0.5, 3.0, n)
    x2 = rng.uniform(0.2, 2.5, n)
    x3 = rng.uniform(0.5, 3.0, n)
    # Target: 3.2*x1^2 + 1.7*exp(-0.8*x2) + 0.4/x3
    y_clean = 3.2 * (x1**2) + 1.7 * np.exp(-0.8 * x2) + 0.4 / x3
    return {"x1": x1, "x2": x2, "x3": x3}, y_clean


def test_level1_synthetic_exact_separability(multivar_dataset):
    x_data, y_clean = multivar_dataset
    analyzer = SeparabilityAnalyzer(tolerance=0.35, ridge_alpha=1e-3)
    decomp = analyzer.decompose(x_data, y_clean)

    # 1. Verification of coordinate decomposition
    assert decomp.is_fully_additive
    assert len(decomp.additive_groups) == 3
    assert all(len(group) == 1 for group in decomp.additive_groups)


def test_level2_noisy_synthetic_separability(multivar_dataset):
    x_data, y_clean = multivar_dataset
    rng = np.random.default_rng(101)
    # Add 2% Gaussian noise
    noise_sigma = 0.02 * np.std(y_clean)
    y_noisy = y_clean + rng.normal(0, noise_sigma, size=len(y_clean))

    analyzer = SeparabilityAnalyzer(tolerance=0.40, ridge_alpha=1e-2)
    decomp = analyzer.decompose(x_data, y_noisy)

    assert decomp.is_fully_additive
    assert len(decomp.additive_groups) == 3


def test_varpro_multi_variable_recovery(multivar_dataset):
    x_data, y_clean = multivar_dataset
    x1, x2, x3 = sp.Symbol("x1"), sp.Symbol("x2"), sp.Symbol("x3")
    p0, p1, p2, a0 = sp.Symbol("p0"), sp.Symbol("p1"), sp.Symbol("p2"), sp.Symbol("a0")

    # Separated skeleton discovered from additive partitioning
    skeleton = p0 * (x1**2) + p1 * sp.exp(-a0 * x2) + p2 * (1 / x3)
    params = [p0, p1, p2, a0]
    variables = [x1, x2, x3]

    solver = VarProOptimizer()
    res = solver.fit(
        skeleton,
        params,
        variables,
        x_data,
        y_clean,
        alpha_bounds={a0: (0.1, 3.0)},
    )

    assert res.mse < 1e-6
    assert abs(res.linear_params[p0] - 3.2) < 1e-2
    assert abs(res.linear_params[p1] - 1.7) < 1e-2
    assert abs(res.linear_params[p2] - 0.4) < 1e-2
    assert abs(res.nonlinear_params[a0] - 0.8) < 1e-2
    assert res.identifiability.status == "IDENTIFIABLE"
