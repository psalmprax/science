"""Tests for Dimensionless Transcendentals (sin, cos, sqrt) in L2 Open Discovery.

Verifies:
  1. Dimensional Type Enforcement: sin and cos strictly require dimensionless arguments
     [0, 0, ...] (via Buckingham Pi / dimensional analysis) and raise DimensionalError on dimensional inputs.
  2. Gate B Physical Pruning: transcendentals whose output dimension does not match the target LHS are pruned without fitting.
  3. CAS Algebraic Reduction: Pythagorean trigonometric identities (sin^2 + cos^2 = 1) are canonicalized to trivial constants and pruned.
  4. Autonomous Nonlinear Pendulum Recovery: ddot(theta) = -(g/L)*sin(theta) at large angles is promoted to UNIQUE_SUPPORT.
  5. Small-Angle Epistemic Abstention: In the quasi-linear regime (theta_0 small), the adjudicator detects observational equivalence with theta and returns MULTI_HYPOTHESIS.
  6. Dimensionless Ratio Discovery: Learns y = c*cos(x / L0) where x and L0 have length dimensions [L=1], autonomously forming the dimensionless ratio.
  7. Multi-Term Damped Pendulum Discovery: Autonomously discovers ddot(theta) = -gamma*omega - omega0^2*sin(theta).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from apedp_dyn.discovery_l2.ast_generator import (
    BinaryOpNode,
    PowNode,
    UnaryOpNode,
    VariableNode,
    _canonical_key_and_validity,
)
from apedp_dyn.discovery_l2.dimensional_types import (
    DIMENSIONLESS,
    Dimension,
    DimensionalError,
)
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine
from apedp_dyn.discovery_l2.multiterm_engine import MultiTermL2Engine
from apedp_dyn.discovery_l2.pruning import gate_b_physical_validity
from apedp_dyn.epistemic.adjudicate import EpistemicStatus


class TestDimensionlessTranscendentals:
    """Rigorous verification of trigonometric transcendentals under dimensional and epistemic sovereignty."""

    def test_transcendental_dimensional_type_enforcement(self):
        """sin and cos must reject dimensioned inputs and accept strictly dimensionless inputs."""
        dim_length = Dimension(l=1)
        dim_momentum = Dimension(m=1, l=1, t=-1)
        x_node = VariableNode("x", dim_length)
        p_node = VariableNode("p", dim_momentum)

        # Applying sin or cos to dimensioned nodes must raise DimensionalError.
        # Message generalized from "Transcendental operator ..." to
        # "Operator ..." when sat/inv_quad joined the dimensionless-gated set;
        # the offending dimension is now included.
        with pytest.raises(DimensionalError, match="Operator 'sin' requires strictly dimensionless"):
            UnaryOpNode("sin", x_node)

        with pytest.raises(DimensionalError, match="Operator 'cos' requires strictly dimensionless"):
            UnaryOpNode("cos", p_node)

        # Dimensionless node (e.g. angle in radians) succeeds
        theta_node = VariableNode("theta", DIMENSIONLESS)
        sin_node = UnaryOpNode("sin", theta_node)
        cos_node = UnaryOpNode("cos", theta_node)
        assert sin_node.dimension == DIMENSIONLESS
        assert cos_node.dimension == DIMENSIONLESS

    def test_gate_b_dimensional_pruning(self):
        """Transcendental expressions whose output dimension mismatches target LHS are pruned at Gate B."""
        theta_node = VariableNode("theta", DIMENSIONLESS)
        sin_node = UnaryOpNode("sin", theta_node)

        # Target requires length (e.g. position in meters)
        target_dim = Dimension(l=1)
        ok, reason = gate_b_physical_validity(sin_node, target_dim)
        assert not ok
        assert "dimension mismatch" in reason.lower()

    def test_trigonometric_pythagorean_identity_cancellation(self):
        """sin(theta)^2 + cos(theta)^2 is canonicalized to constant 1 and pruned as non-informative."""
        theta = VariableNode("theta", DIMENSIONLESS)
        s = UnaryOpNode("sin", theta)
        c = UnaryOpNode("cos", theta)
        s2 = PowNode(s, Fraction(2, 1))
        c2 = PowNode(c, Fraction(2, 1))
        pythag = BinaryOpNode("+", s2, c2)

        is_valid, key = _canonical_key_and_validity(pythag)
        assert not is_valid
        assert key == "1"

    def test_autonomous_nonlinear_pendulum_discovery(self):
        """Discovers ddot(theta) = -9.81 * sin(theta) at large angles with UNIQUE_SUPPORT."""
        rng = np.random.default_rng(42)

        def odes(t, y):
            return [y[1], -9.81 * np.sin(y[0])]

        t_eval = np.linspace(0.0, 4.0, 100)
        sol = solve_ivp(odes, (0.0, 4.0), [2.0, 0.0], t_eval=t_eval, rtol=1e-9, atol=1e-9)
        theta = sol.y[0]
        noise = rng.normal(0, 0.02, size=len(theta))
        theta_ddot = -9.81 * np.sin(theta) + noise

        engine = L2DiscoveryEngine(
            operators=("+", "-", "*", "/", "pow", "sin", "cos"),
            max_depth=1,
            max_complexity=3,
        )
        res = engine.discover(
            X={"theta": theta},
            Y=theta_ddot,
            dimensions={"theta": Dimension()},
            target_dimension=Dimension(),
            train_ratio=0.6,
            bootstrap=False,
            seed=42,
        )

        assert res.adjudication.status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.adjudication.promoted is not None
        assert res.adjudication.promoted.candidate.candidate_id == "sin(theta)"
        coef = res.adjudication.promoted.fit.param_values.get("theta_1", 0.0)
        assert np.isclose(coef, -9.81, atol=0.05)

    def test_small_angle_quasi_linear_epistemic_abstention(self):
        """In the quasi-linear regime (small initial angle), system abstains with MULTI_HYPOTHESIS."""
        rng = np.random.default_rng(42)

        def odes(t, y):
            return [y[1], -9.81 * np.sin(y[0])]

        t_eval = np.linspace(0.0, 4.0, 80)
        sol = solve_ivp(odes, (0.0, 4.0), [0.5, 0.0], t_eval=t_eval, rtol=1e-9, atol=1e-9)
        theta = sol.y[0]
        noise = rng.normal(0, 0.02, size=len(theta))
        theta_ddot = -9.81 * np.sin(theta) + noise

        engine = L2DiscoveryEngine(
            operators=("+", "-", "*", "/", "pow", "sin", "cos"),
            max_depth=1,
            max_complexity=3,
        )
        res = engine.discover(
            X={"theta": theta},
            Y=theta_ddot,
            dimensions={"theta": Dimension()},
            target_dimension=Dimension(),
            train_ratio=0.6,
            bootstrap=False,
            seed=42,
        )

        # In small-angle oscillation, theta and sin(theta) are within tau_epi -> MULTI_HYPOTHESIS
        assert res.adjudication.status == EpistemicStatus.MULTI_HYPOTHESIS
        surviving_ids = [c.candidate.candidate_id for c in res.pareto_candidates]
        assert "theta" in surviving_ids
        assert "sin(theta)" in surviving_ids

    def test_dimensionless_ratio_transcendental_discovery(self):
        """Discovers y = 3.5 * cos(x / L0) where x and L0 have dimension [L=1]."""
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(0.0, 10.0, size=n)
        L0 = np.full(n, 2.0)
        y_true = 3.5 * np.cos(x / L0)
        noise = rng.normal(0, 0.01, size=n)
        y = y_true + noise

        engine = L2DiscoveryEngine(
            operators=("+", "-", "*", "/", "pow", "sin", "cos"),
            max_depth=2,
            max_complexity=5,
        )
        res = engine.discover(
            X={"x": x, "L0": L0},
            Y=y,
            dimensions={"x": Dimension(l=1), "L0": Dimension(l=1)},
            target_dimension=Dimension(),
            train_ratio=0.6,
            bootstrap=False,
            seed=42,
        )

        assert res.adjudication.status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.adjudication.promoted is not None
        canon = res.adjudication.promoted.candidate.metadata.get("canonical_expr", "")
        assert "cos(x/L0)" in canon or "cos((x / L0))" in canon or "cos(0.5*x)" in canon
        coef = res.adjudication.promoted.fit.param_values.get("theta_1", 0.0)
        assert np.isclose(coef, 3.5, atol=0.05)

    def test_multiterm_damped_nonlinear_pendulum_discovery(self):
        """Discovers ddot(theta) = -0.4 * omega - 5.0 * sin(theta) via MultiTermL2Engine."""
        rng = np.random.default_rng(42)

        def odes(t, y):
            return [y[1], -5.0 * np.sin(y[0]) - 0.4 * y[1]]

        t_eval = np.linspace(0.0, 6.0, 100)
        sol = solve_ivp(odes, (0.0, 6.0), [1.8, 0.0], t_eval=t_eval, rtol=1e-9, atol=1e-9)
        theta = sol.y[0]
        omega = sol.y[1]
        noise = rng.normal(0, 0.01, size=len(theta))
        theta_ddot = -5.0 * np.sin(theta) - 0.4 * omega + noise

        engine = MultiTermL2Engine(
            max_depth=1,
            max_term_complexity=2,
            max_terms=2,
            operators=("+", "-", "*", "sin", "cos"),
        )
        res = engine.discover_multiterm(
            X={"theta": theta, "omega": omega},
            Y=theta_ddot,
            dimensions={"theta": Dimension(), "omega": Dimension()},
            target_dimension=Dimension(),
            train_ratio=0.6,
            bootstrap=False,
        )

        assert res.adjudication.status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.adjudication.promoted is not None
        assert len(res.adjudication.promoted.candidate.terms) == 2
        term_names = [t.name for t in res.adjudication.promoted.candidate.terms]
        assert "omega" in term_names
        assert "sin(theta)" in term_names

        pv = res.adjudication.promoted.fit.param_values
        for t_obj in res.adjudication.promoted.candidate.terms:
            if t_obj.name == "omega":
                assert np.isclose(pv[t_obj.parameter], -0.4, atol=0.02)
            elif t_obj.name == "sin(theta)":
                assert np.isclose(pv[t_obj.parameter], -5.0, atol=0.02)
