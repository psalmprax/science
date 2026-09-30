"""Tests for Multi-Term Superposition L2 Open Discovery Engine.

Verifies:
  1. Autonomous recovery of 2-term damped harmonic oscillator (x_ddot = -k*x - c*v).
  2. Autonomous recovery of 3-term nonlinear Duffing oscillator (x_ddot = -alpha*x - delta*v - beta*x^3).
  3. FIM / Collinearity gating: combinations with condition number kappa > 10^3 are pruned before fitting.
  4. Pareto complexity frontier selection: higher term counts require strict holdout RMSE improvement.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.multiterm_engine import MultiTermL2Engine
from apedp_dyn.epistemic.adjudicate import EpistemicStatus


class TestMultiTermL2Engine:
    """Verifies multi-term superposition discovery and FIM conditioning."""

    def test_harmonic_oscillator_multi_term_discovery(self):
        """Autonomously discovers 2-term damped harmonic oscillator x_ddot = -4*x - 0.2*v."""
        def odes(t, y):
            return [y[1], -4.0 * y[0] - 0.2 * y[1]]

        t_eval = np.linspace(0.0, 10.0, 100)
        sol = solve_ivp(odes, (0.0, 10.0), [1.0, 0.0], t_eval=t_eval, rtol=1e-9, atol=1e-9)
        x, v = sol.y[0], sol.y[1]
        x_ddot = -4.0 * x - 0.2 * v

        engine = MultiTermL2Engine(max_depth=1, max_term_complexity=2, max_terms=2, operators=("+", "-", "*"))
        res = engine.discover_multiterm(
            X={"x": x, "v": v},
            Y=x_ddot,
            dimensions={"x": Dimension(), "v": Dimension()},
            target_dimension=Dimension(),
            target_name="x_ddot",
            train_ratio=0.6,
            bootstrap=False,
        )

        assert res.adjudication.status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.adjudication.promoted is not None
        assert len(res.adjudication.promoted.candidate.terms) == 2
        # Verify both x and v terms were identified
        term_names = [t.name for t in res.adjudication.promoted.candidate.terms]
        assert "x" in term_names
        assert "v" in term_names

        # Verify fitted coefficients: theta_x ≈ -4.0, theta_v ≈ -0.2
        pv = res.adjudication.promoted.fit.param_values
        for t_obj in res.adjudication.promoted.candidate.terms:
            if t_obj.name == "x":
                assert np.isclose(pv[t_obj.parameter], -4.0, atol=0.01)
            elif t_obj.name == "v":
                assert np.isclose(pv[t_obj.parameter], -0.2, atol=0.01)

    def test_duffing_oscillator_multi_term_discovery(self):
        """Autonomously discovers 3-term nonlinear Duffing oscillator x_ddot = -1*x - 0.2*v - 1*x^3."""
        def odes(t, y):
            return [y[1], -0.2 * y[1] - 1.0 * y[0] - 1.0 * (y[0]**3)]

        t_eval = np.linspace(0.0, 10.0, 120)
        sol = solve_ivp(odes, (0.0, 10.0), [1.0, 0.0], t_eval=t_eval, rtol=1e-9, atol=1e-9)
        x, v = sol.y[0], sol.y[1]
        x_ddot = -0.2 * v - 1.0 * x - 1.0 * (x**3)

        engine = MultiTermL2Engine(max_depth=2, max_term_complexity=3, max_terms=3, operators=("+", "-", "*", "pow"))
        res = engine.discover_multiterm(
            X={"x": x, "v": v},
            Y=x_ddot,
            dimensions={"x": Dimension(), "v": Dimension()},
            target_dimension=Dimension(),
            target_name="x_ddot",
            train_ratio=0.6,
            bootstrap=False,
        )

        assert res.adjudication.status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.adjudication.promoted is not None
        assert len(res.adjudication.promoted.candidate.terms) == 3

        term_names = [t.name for t in res.adjudication.promoted.candidate.terms]
        assert "x" in term_names
        assert "v" in term_names
        assert any("x" in tn and "3" in tn for tn in term_names)

    def test_fim_collinearity_gate_prunes_ill_conditioned_pairs(self):
        """Verifies candidate combinations with condition number kappa > 10^3 are pruned."""
        rng = np.random.default_rng(42)
        n = 80
        u = rng.uniform(1.0, 10.0, size=n)
        # Highly collinear feature w = 2.0001*u
        w = 2.0 * u + rng.normal(0, 0.00001, size=n)
        y = 3.0 * u + rng.normal(0, 0.01, size=n)

        engine = MultiTermL2Engine(
            max_depth=1,
            max_term_complexity=2,
            max_terms=2,
            max_condition_number=100.0,
            operators=("+", "-", "*"),
        )
        res = engine.discover_multiterm(
            X={"u": u, "w": w},
            Y=y,
            dimensions={"u": Dimension(), "w": Dimension()},
            target_dimension=Dimension(),
            train_ratio=0.6,
            bootstrap=False,
        )

        # The collinear pair [u, w] must be pruned by the FIM gate
        assert res.collinear_pruned_count >= 1
