"""Tests for numerical and symbolic separability analysis under clean and noisy regimes."""
from __future__ import annotations

import numpy as np
import pytest

from apedp.separability import (
    MultiplicativeDomainCheck,
    SeparabilityAnalyzer,
    SeparabilityPairResult,
    SeparabilityType,
    VariableDecomposition,
)


@pytest.fixture
def grid_2d():
    x = np.linspace(1.0, 4.0, 20)
    y = np.linspace(1.0, 4.0, 20)
    X, Y = np.meshgrid(x, y)
    return X.ravel(), Y.ravel()


def test_additive_separability_exact(grid_2d):
    x, y = grid_2d
    z = x**2 + y**2
    analyzer = SeparabilityAnalyzer(tolerance=0.25)
    res = analyzer.analyze_pair(x, y, z, "x", "y")
    assert isinstance(res, SeparabilityPairResult)
    assert res.verdict == SeparabilityType.ADDITIVE
    assert res.score_additive < 0.15


def test_multiplicative_separability_exact(grid_2d):
    x, y = grid_2d
    z = (x**2) * (y**3)
    analyzer = SeparabilityAnalyzer(tolerance=0.25)
    res = analyzer.analyze_pair(x, y, z, "x", "y")
    assert isinstance(res, SeparabilityPairResult)
    assert res.verdict == SeparabilityType.MULTIPLICATIVE
    assert res.score_multiplicative < 0.25
    assert res.multiplicative_check.status == "VALID"


def test_multiplicative_domain_check_invalid_on_sign_crossings():
    # Signal with zero crossings and sign flips
    x = np.linspace(-3.0, 3.0, 50)
    y = x  # cross zero at x=0
    check = SeparabilityAnalyzer.inspect_multiplicative_domain(y)
    assert isinstance(check, MultiplicativeDomainCheck)
    assert check.status == "INVALID"
    assert "SIGN_FLIPS" in check.reason


def test_non_separable_form(grid_2d):
    x, y = grid_2d
    # Strongly coupled cross-term: sin(x * y) + 2.0
    z = np.sin(x * y) + 2.0
    analyzer = SeparabilityAnalyzer(tolerance=0.25)
    res = analyzer.analyze_pair(x, y, z, "x", "y")
    assert res.verdict == SeparabilityType.NONE


def test_variable_decomposition_3d():
    x = np.linspace(1.0, 3.0, 10)
    X1, X2, X3 = np.meshgrid(x, x, x)
    x1, x2, x3 = X1.ravel(), X2.ravel(), X3.ravel()
    y = x1**2 + x2**2 + x3**3

    analyzer = SeparabilityAnalyzer(tolerance=0.30)
    decomp = analyzer.decompose({"x1": x1, "x2": x2, "x3": x3}, y)
    assert isinstance(decomp, VariableDecomposition)
    assert decomp.is_fully_additive
    assert len(decomp.additive_groups) == 3


@pytest.mark.parametrize("noise_level", [0.0, 0.001, 0.01, 0.05])
def test_noise_stress_additive_separability(grid_2d, noise_level):
    """Stress test: verify additive separability is robust up to 5% noise."""
    x, y = grid_2d
    rng = np.random.default_rng(42)
    clean_z = x**2 + y**2
    sigma_noise = noise_level * np.std(clean_z)
    noisy_z = clean_z + rng.normal(0, sigma_noise, size=clean_z.shape)

    analyzer = SeparabilityAnalyzer(tolerance=0.40, ridge_alpha=1e-2)
    res = analyzer.analyze_pair(x, y, noisy_z, "x", "y")
    assert res.verdict == SeparabilityType.ADDITIVE
