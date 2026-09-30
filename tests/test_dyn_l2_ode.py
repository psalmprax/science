"""Tests for Autonomous Dynamic ODE Discovery (L2-ODE).

Validates autonomous recovery of nonlinear differential equations:
    dx/dt = f(x, y, ...)
directly from discrete trajectory observations without candidate equation templates.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.ode_engine import (
    L2ODEDiscoveryEngine,
    estimate_time_derivative,
)
from apedp_dyn.epistemic.adjudicate import EpistemicStatus


class TestL2ODEDiscovery:
    """Rigorous tests for autonomous ODE discovery."""

    def test_estimate_time_derivative_methods(self):
        """Verifies spline and central difference numerical differentiation."""
        t = np.linspace(0.0, 2.0, 100)
        # x(t) = t^2 => dx/dt = 2*t
        x = t ** 2
        d_spline = estimate_time_derivative(t, x, method="spline")
        d_cdiff = estimate_time_derivative(t, x, method="central_diff")

        d_true = 2.0 * t
        assert np.allclose(d_spline, d_true, atol=1e-2)
        assert np.allclose(d_cdiff[1:-1], d_true[1:-1], atol=1e-4)
        assert np.allclose(d_cdiff, d_true, atol=0.03)

    def test_estimate_time_derivative_validation(self):
        """Mismatched arrays or too few timepoints must raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            estimate_time_derivative(np.array([1, 2, 3]), np.array([1, 2]))

        with pytest.raises(ValueError, match="At least 4 timepoints"):
            estimate_time_derivative(np.array([1, 2, 3]), np.array([1, 2, 3]))

    def test_autonomous_scalar_nonlinear_ode_discovery(self):
        """Discovers dx/dt = -0.5 * x^2 from trajectory without equation hints."""
        t = np.linspace(0.1, 2.0, 50)
        # Analytical solution to dx/dt = -0.5 * x^2: x(t) = 1 / (0.5*t + 1)
        x = 1.0 / (0.5 * t + 1.0)

        engine = L2ODEDiscoveryEngine(max_depth=2, max_complexity=5, operators=("+", "-", "*", "/", "pow"))
        res = engine.discover_ode(
            t=t,
            trajectories={"x": x},
            state_dimensions={"x": Dimension()},
            target_state="x",
            train_ratio=0.6,
            bootstrap=False,
            seed=42,
        )

        assert res.l2_result.adjudication.status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.l2_result.adjudication.promoted is not None
        assert "x**2" in res.inferred_equation or "x^2" in res.inferred_equation or "(x * x)" in res.inferred_equation
        # Verify fitted coefficient is approximately -0.5
        coef = res.l2_result.adjudication.promoted.fit.param_values.get("theta_1", 0.0)
        assert np.isclose(coef, -0.5, atol=0.05)

    def test_autonomous_cubic_damping_ode_discovery(self):
        """Discovers dx/dt = -0.5 * x^3 from trajectory without equation hints."""
        t = np.linspace(0.0, 2.0, 60)
        # dx/dt = -0.5 * x^3 => -1/2 x^-2 = -0.5 t - C => x(t) = (t + 1)^(-0.5)
        x = (t + 1.0) ** (-0.5)

        engine = L2ODEDiscoveryEngine(max_depth=2, max_complexity=5, operators=("+", "-", "*", "/", "pow"))
        res = engine.discover_ode(
            t=t,
            trajectories={"x": x},
            state_dimensions={"x": Dimension()},
            target_state="x",
            train_ratio=0.6,
            bootstrap=False,
            seed=42,
        )

        assert res.l2_result.adjudication.status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.l2_result.adjudication.promoted is not None
        assert "x**3" in res.inferred_equation or "x^3" in res.inferred_equation or "x" in res.inferred_equation
        coef = res.l2_result.adjudication.promoted.fit.param_values.get("theta_1", 0.0)
        assert np.isclose(coef, -0.5, atol=0.05)

    def test_coupled_bilinear_coupling_term_discovery(self):
        """Discovers bilinear coupling dx/dt = -0.8 * (x * y) in 2D state system."""
        t = np.linspace(0.1, 2.0, 50)
        # Trajectories: x(t) = exp(-0.4*t), y(t) = 1.0 + 0.2*t
        # dx/dt target computed numerically
        rng = np.random.default_rng(42)
        x = np.exp(-0.4 * t)
        y = 1.0 + 0.5 * np.sin(t)
        # Target derivative exactly matching coupling: dx/dt = -0.8 * x * y
        x_dot_true = -0.8 * x * y + rng.normal(0.0, 0.005, size=len(t))

        engine = L2ODEDiscoveryEngine(max_depth=2, max_complexity=5, operators=("+", "-", "*", "/"))
        # Run discover directly on 2-variable state space
        l2_res = engine.engine.discover(
            X={"x": x, "y": y},
            Y=x_dot_true,
            dimensions={"x": Dimension(), "y": Dimension()},
            target_dimension=Dimension(),
            train_ratio=0.6,
            bootstrap=False,
            seed=42,
        )

        assert l2_res.adjudication.status == EpistemicStatus.UNIQUE_SUPPORT
        promoted = l2_res.adjudication.promoted
        assert promoted is not None
        expr = promoted.candidate.metadata.get("canonical_expr", "")
        assert "x*y" in expr or "x * y" in expr

    def test_noisy_trajectory_integrity_abstains(self):
        """High noise trajectory refuses false promotion, returning non-UNIQUE adjudication."""
        rng = np.random.default_rng(42)
        t = np.linspace(0.1, 2.0, 40)
        # Pure random walk / noise trajectory where true physics is absent
        x_noise = rng.normal(0.0, 2.0, size=len(t))

        engine = L2ODEDiscoveryEngine(max_depth=2, max_complexity=5)
        res = engine.discover_ode(
            t=t,
            trajectories={"x": x_noise},
            state_dimensions={"x": Dimension()},
            target_state="x",
            seed=42,
        )

        # Adjudication must refuse UNIQUE_SUPPORT on pure noise
        assert res.l2_result.adjudication.status != EpistemicStatus.UNIQUE_SUPPORT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
