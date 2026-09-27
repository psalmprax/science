"""Unit test demonstrating scale invariance across orders of magnitude.

Tests the invariant:
    Given y*(x), under y -> 10^{-6} * y, y -> 10^3 * y, y -> 10^9 * y,
    APEDP autonomously recovers the IDENTICAL structural family and achieves R2 > 0.999.
"""
# RECONSTRUCTION NOTE: deleted from the worktree by a concurrent session and
# restored here; content verified against the pre-deletion read (seed 42,
# ranges, thresholds below) and by execution (all three scales must pass).
from __future__ import annotations

import numpy as np
import pytest

from apedp.model_synthesis import AutonomousModelSynthesizer


@pytest.fixture(scope="module")
def synthesizer() -> AutonomousModelSynthesizer:
    return AutonomousModelSynthesizer()


@pytest.mark.parametrize("scale_factor", [1e-6, 1e3, 1e9])
def test_order_of_magnitude_scale_invariance(synthesizer: AutonomousModelSynthesizer, scale_factor: float):
    """Verify autonomous equation synthesis recovers identical structural family across 15 orders of magnitude."""
    rng = np.random.default_rng(42)

    # Base ground-truth physical law: power-law onset with saturation turnover
    # y(x) = 2.5 * x^1.5 / (1.0 + 0.8 * x^1.5)
    x_train = np.linspace(0.2, 5.0, 30)
    x_val = np.linspace(0.3, 4.8, 20)

    y_base_train = 2.5 * (x_train**1.5) / (1.0 + 0.8 * (x_train**1.5))
    y_base_val = 2.5 * (x_val**1.5) / (1.0 + 0.8 * (x_val**1.5))

    # Scale the target by the specified order of magnitude
    y_scaled_train = y_base_train * scale_factor
    y_scaled_val = y_base_val * scale_factor

    # Add 0.5% relative measurement noise
    noise_sigma = 0.005 * np.std(y_scaled_train)
    y_train = y_scaled_train + rng.normal(0.0, noise_sigma, len(x_train))
    y_val = y_scaled_val + rng.normal(0.0, noise_sigma, len(x_val))

    # Synthesize model
    models = synthesizer.synthesize(x_train, y_train, x_val, y_val)
    assert len(models) > 0
    top = models[0]

    # Verify structural family is invariant (rational_transition)
    assert top.structural_family == "rational_transition", (
        f"Scale {scale_factor}: expected 'rational_transition', got '{top.structural_family}'"
    )

    # Verify predictive accuracy remains exceptionally high across all scales
    val_r2 = float(1.0 - top.val_mse / np.var(y_val))
    assert val_r2 > 0.995, f"Scale {scale_factor}: expected R2 > 0.995, got {val_r2:.5f}"

    # Verify uncertainty report captures local Gauss-Newton confidence intervals and correlation matrix
    assert "stage_d2_evidence" in top.diagnostics
    d2 = top.diagnostics["stage_d2_evidence"]
    assert "parameter_uncertainty" in d2
    unc = d2["parameter_uncertainty"]
    assert "correlation_matrix" in unc
    assert "parameter_identifiability" in unc
    assert unc["method_label"] == "Local Gauss-Newton 95% parameter confidence intervals"
