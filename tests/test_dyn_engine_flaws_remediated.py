"""Comprehensive verification for the four core engine upgrades:

1. Noise-robust velocity derivative estimation with regularized polynomial filtering (Savitzky-Golay).
2. Order flow, velocity relaxation, and rational saturation terms in mechanical grammar.
3. Standardized innovation residual whitening under heteroskedasticity (volatility clustering).
4. Rational saturation basis operators ('sat', 'inv_quad') in L2 open discovery.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.discovery_l2.ast_generator import (
    UnaryOpNode,
    VariableNode,
)
from apedp_dyn.discovery_l2.dimensional_types import Dimension, DimensionalError
from apedp_dyn.discovery_l2.multiterm_engine import MultiTermL2Engine
from apedp_dyn.discovery_l2.pruning import check_residual_whiteness
from apedp_dyn.epistemic.adjudicate import _sole_survivor_whiteness
from apedp_dyn.gates import dimensional_validity
from apedp_dyn.grammar.candidate_ir import TERM_FAMILIES, CandidateDynamics, Term
from apedp_dyn.grammar.mechanical_grammar import (
    _TERM_BUILDERS,
    materialize,
)
from apedp_dyn.problems.finance import _estimate_velocity, spec_from_series


class TestNoiseRobustDerivativeEstimation:
    """Fix 1: Mitigation of finite-difference O(1/dt) noise amplification."""

    def test_smooth_derivative_suppresses_high_frequency_noise(self):
        t = np.linspace(0, 10, 100)
        dt = float(t[1] - t[0])
        # Clean signal x(t) = sin(t), true v(t) = cos(t)
        clean_x = np.sin(t)
        true_v = np.cos(t)
        
        # Add high-frequency noise
        np.random.seed(42)
        noise = 0.05 * np.random.randn(len(t))
        noisy_x = clean_x + noise

        raw_v = np.gradient(noisy_x, dt)
        smooth_v = _estimate_velocity(noisy_x, dt, smooth=True)

        raw_rmse = float(np.sqrt(np.mean((raw_v - true_v) ** 2)))
        smooth_rmse = float(np.sqrt(np.mean((smooth_v - true_v) ** 2)))

        # Smoothed derivative must significantly outperform raw finite differences on noisy signal
        assert smooth_rmse < raw_rmse
        assert smooth_rmse < 0.6 * raw_rmse

    def test_spec_from_series_smooth_parameter(self):
        prices = 100.0 * np.exp(0.001 * np.arange(60))
        prob_default = spec_from_series(prices, (0, 40), (40, 60), smooth_derivatives=False)
        prob_smooth = spec_from_series(prices, (0, 40), (40, 60), smooth_derivatives=True)

        assert prob_default.train["truth_y"].shape == (40, 2)
        assert prob_smooth.train["truth_y"].shape == (40, 2)
        assert np.all(np.isfinite(prob_smooth.train["truth_y"]))


class TestMechanicalGrammarSaturationAndRelaxation:
    """Fix 2: Order flow rate laws and saturation operators in grammar."""

    def test_saturation_and_relaxation_in_term_builders(self):
        assert "sat_x" in _TERM_BUILDERS
        assert "sat_v" in _TERM_BUILDERS
        assert "sat_u" in _TERM_BUILDERS
        assert "inv_x2" in _TERM_BUILDERS
        assert "u_relax" in _TERM_BUILDERS

    def test_saturation_family_registered(self):
        assert "saturation" in TERM_FAMILIES
        assert "sat_x" in TERM_FAMILIES["saturation"]
        assert "sat_u" in TERM_FAMILIES["saturation"]

    def test_materialize_and_dimensional_validity_with_saturation(self):
        prices = 100.0 * np.exp(0.001 * np.arange(50))
        of = np.random.randn(50)
        prob = spec_from_series(prices, (0, 30), (30, 50), orderflow=of)

        c = CandidateDynamics("xddot", (Term("v", "a0"), Term("sat_u", "a1"), Term("sat_x", "a2")))
        model = materialize(prob, c)
        assert len(model.equations) == 2
        assert model.equations[0].state == "x"
        assert model.equations[1].state == "v"

        # Check Gate 1 dimensional validity
        g1 = dimensional_validity(model)
        assert g1.passed, f"Gate 1 failed: {g1.detail}"


class TestStandardizedInnovationWhitening:
    """Fix 3: Standardized innovation testing under volatility clustering (heteroskedasticity)."""

    def test_heteroskedastic_whitening_in_pruning(self):
        np.random.seed(42)
        n = 120
        # GARCH-like clustered volatility innovations
        sigmas = np.ones(n)
        for t in range(1, n):
            sigmas[t] = 0.7 * sigmas[t - 1] + 0.3 * (3.0 if (20 <= t <= 50 or 80 <= t <= 100) else 0.5)
        raw_residuals = sigmas * np.random.randn(n)

        # Raw autocorrelation may be high or fail due to variance shifts
        _ok_raw, _rho1_raw = check_residual_whiteness(raw_residuals, threshold=0.3, allow_standardized=False)
        ok_std, rho1_std = check_residual_whiteness(raw_residuals, threshold=0.3, allow_standardized=True)

        # Standardized innovations correctly recognize underlying uncorrelated innovations
        assert ok_std is True
        assert abs(rho1_std) < 0.3

    def test_adjudicator_sole_survivor_whiteness_accepts_standardized_innovations(self):
        np.random.seed(123)
        n = 100
        # Clustered error bursts
        errors = np.random.randn(n)
        errors[30:60] *= 5.0  # volatility burst

        ok, note = _sole_survivor_whiteness(errors, threshold=0.35)
        assert ok is True
        assert "vs flag" in note


class TestL2DiscoveryRationalAndSaturationBasis:
    """Fix 4: Rational and saturation operators in L2 discovery."""

    def test_ast_unary_saturation_evaluation(self):
        x_node = VariableNode("x", Dimension())
        sat_n = UnaryOpNode("sat", x_node)
        inv_n = UnaryOpNode("inv_quad", x_node)

        env = {"x": np.array([-10.0, -1.0, 0.0, 1.0, 10.0])}
        sat_out = sat_n.evaluate(env)
        inv_out = inv_n.evaluate(env)

        # Bounded between -1 and 1
        assert np.all(np.abs(sat_out) < 1.0)
        assert sat_out[2] == 0.0
        assert np.isclose(sat_out[3], 0.5)

        # Bounded between 0 and 1
        assert np.all(inv_out > 0.0)
        assert np.all(inv_out <= 1.0)
        assert inv_out[2] == 1.0
        assert np.isclose(inv_out[3], 0.5)

    def test_ast_rejects_dimensional_input_to_saturation(self):
        # Saturation requires dimensionless argument
        length_node = VariableNode("x", Dimension(l=1))
        with pytest.raises(DimensionalError):
            UnaryOpNode("sat", length_node)
        with pytest.raises(DimensionalError):
            UnaryOpNode("inv_quad", length_node)

    def test_multiterm_l2_engine_with_saturation_operators(self):
        np.random.seed(99)
        x = np.linspace(-3, 3, 50)
        # Synthetic process governed by saturation: y = 2.0 * (x / (1 + |x|))
        y = 2.0 * (x / (1.0 + np.abs(x)))

        eng = MultiTermL2Engine(
            operators=("sat", "+", "*"),
            max_depth=1,
            max_terms=1,
        )
        res = eng.discover_multiterm(
            X={"x": x},
            Y=y,
            dimensions={"x": Dimension()},
            target_dimension=Dimension(),
            target_name="y",
            seed=42,
        )

        assert len(res.pareto_candidates) > 0
        best_cand = res.pareto_candidates[0]
        assert best_cand.holdout_rmse < 1e-3
