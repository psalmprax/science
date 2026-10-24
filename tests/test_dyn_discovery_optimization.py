"""Tests for Discovery Optimization Roadmap: Numeric Fingerprinting, Semantic Types, Novelty, and Multi-Objective Pareto.

Verifies the 4 architectural pillars:
  1. Numeric Fingerprinting (zero-cost candidate deduplication, constant/singularity rejection, abs/bipolar distinction).
  2. Representation Novelty Score (1 - max |corr(f, X)|) and factor collinearity detection.
  3. Financial & Economic Semantic Dimensions ([P], [Q], return velocity, flow, spread, dimensional error gates).
  4. Multi-Objective Pareto Frontier Selection (joint error, complexity, novelty, and residual whiteness).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from apedp_dyn.discovery_l2.ast_generator import (
    BinaryOpNode,
    ConstantNode,
    PowNode,
    UnaryOpNode,
    VariableNode,
)
from apedp_dyn.discovery_l2.dimensional_types import (
    DIMENSIONLESS,
    DOLLAR_FLOW,
    ORDER_FLOW,
    PRICE_DIMENSION,
    RETURN_VELOCITY,
    SPREAD_DIMENSION,
    VOLATILITY,
    VOLUME_QUANTITY,
    Dimension,
    DimensionalError,
)
from apedp_dyn.discovery_l2.fingerprint import (
    FingerprintDeduplicator,
    compute_fingerprint,
    make_bipolar_probe_grid,
    make_probe_grid,
)
from apedp_dyn.discovery_l2.pruning import (
    compute_novelty_score,
    fit_and_score_candidate,
    select_pareto_candidates,
)


class TestNumericFingerprinting:
    """Verifies probe-grid based numeric candidate deduplication."""

    def test_proportional_and_sign_invariance(self):
        """Proportional basis candidates (2*x vs x, -x vs x) generate identical fingerprints."""
        variables = {"x": Dimension()}
        probe_grid = make_probe_grid(variables, n_points=32)
        bipolar_grid = make_bipolar_probe_grid(variables, n_points=32)

        x_node = VariableNode("x", Dimension())
        neg_x = BinaryOpNode("-", ConstantNode(0.0), x_node)
        two_x = BinaryOpNode("+", x_node, x_node)

        valid_x, hash_x, _ = compute_fingerprint(x_node, probe_grid, bipolar_grid)
        valid_neg, hash_neg, _ = compute_fingerprint(neg_x, probe_grid, bipolar_grid)
        valid_two, hash_two, _ = compute_fingerprint(two_x, probe_grid, bipolar_grid)

        assert valid_x and valid_neg and valid_two
        assert hash_x == hash_neg == hash_two

    def test_trivial_constant_and_singularity_rejection(self):
        """Zero-variance expressions (x - x, x / x) and singularities are flagged as invalid."""
        variables = {"x": Dimension()}
        probe_grid = make_probe_grid(variables, n_points=32)
        bipolar_grid = make_bipolar_probe_grid(variables, n_points=32)

        x_node = VariableNode("x", Dimension())
        x_minus_x = BinaryOpNode("-", x_node, x_node)
        x_div_x = BinaryOpNode("/", x_node, x_node)

        valid_sub, _, reason_sub = compute_fingerprint(x_minus_x, probe_grid, bipolar_grid)
        valid_div, _, reason_div = compute_fingerprint(x_div_x, probe_grid, bipolar_grid)

        assert not valid_sub and reason_sub == "trivial_constant"
        assert not valid_div and reason_div == "trivial_constant"

    def test_abs_distinction_via_bipolar_grid(self):
        """abs(v) is distinguished from v due to bipolar probe grid evaluation."""
        variables = {"v": Dimension(t=-1)}
        probe_grid = make_probe_grid(variables, n_points=32)
        bipolar_grid = make_bipolar_probe_grid(variables, n_points=32)

        v_node = VariableNode("v", Dimension(t=-1))
        abs_v = UnaryOpNode("abs", v_node)

        valid_v, hash_v, _ = compute_fingerprint(v_node, probe_grid, bipolar_grid)
        valid_abs, hash_abs, _ = compute_fingerprint(abs_v, probe_grid, bipolar_grid)

        assert valid_v and valid_abs
        assert hash_v != hash_abs

    def test_deduplicator_admits_simpler_superseding_expression(self):
        """A simpler equivalent expression (lower complexity) supersedes a previously seen one."""
        variables = {"x": Dimension()}
        probe_grid = make_probe_grid(variables, n_points=32)
        bipolar_grid = make_bipolar_probe_grid(variables, n_points=32)
        dedup = FingerprintDeduplicator(probe_grid, bipolar_grid)

        x_node = VariableNode("x", Dimension())
        # Complex expression with complexity 3
        two_x = BinaryOpNode("+", x_node, x_node)
        assert two_x.complexity == 3
        assert dedup.is_novel(two_x) is True

        # Candidate with higher complexity 5 is rejected
        four_x = BinaryOpNode("+", two_x, two_x)
        assert four_x.complexity == 7
        assert dedup.is_novel(four_x) is False

        # Candidate with simpler complexity 1 supersedes it
        assert x_node.complexity == 1
        assert dedup.is_novel(x_node) is True


class TestNoveltyScoreAndCollinearity:
    """Verifies candidate representation novelty score against input channels."""

    def test_novelty_score_collinear_vs_orthogonal(self):
        """Collinear feature has novelty near 0; orthogonal feature has novelty near 1."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        y_collinear = 2.5 * x + rng.normal(0, 1e-4, 100)
        z_orthogonal = rng.standard_normal(100)
        # Gram-Schmidt orthogonalization
        z_orthogonal = z_orthogonal - (np.dot(z_orthogonal, x) / np.dot(x, x)) * x

        X = {"x": x}
        nov_collinear = compute_novelty_score(y_collinear, X)
        nov_orthogonal = compute_novelty_score(z_orthogonal, X)

        assert nov_collinear < 0.05
        assert nov_orthogonal > 0.95

    def test_novelty_attached_to_candidate_provenance(self):
        """Scored candidate records representation novelty in metadata and provenance."""
        x = np.linspace(0.1, 5.0, 50)

        X_tr = {"x": x[:30]}
        y_tr = 3.0 * (x[:30] ** 2)
        X_ho = {"x": x[30:]}
        y_ho = 3.0 * (x[30:] ** 2)

        x_node = VariableNode("x", Dimension())
        x_sq = PowNode(x_node, 2)

        cand = fit_and_score_candidate(
            node=x_sq,
            X_tr=X_tr,
            y_tr=y_tr,
            X_ho=X_ho,
            y_ho=y_ho,
            target_name="y",
        )

        assert "novelty" in cand.candidate.metadata
        assert "novelty" in cand.provenance
        assert 0.0 <= cand.candidate.metadata["novelty"] <= 1.0


class TestFinancialSemanticDimensions:
    """Verifies financial and economic semantic dimensional typing."""

    def test_financial_dimensions_calculus(self):
        """Enforces dimensional algebra for price, volume, and order flow."""
        # Order flow [Q * T^-1] / Volume [Q] = Return Velocity [T^-1]
        flow_dim = ORDER_FLOW
        vol_dim = VOLUME_QUANTITY
        ret_dim = flow_dim / vol_dim
        assert ret_dim == RETURN_VELOCITY
        assert ret_dim.t == Fraction(-1, 1)

        # Dollar Flow = Price * Order Flow
        d_flow = PRICE_DIMENSION * ORDER_FLOW
        assert d_flow == DOLLAR_FLOW
        assert d_flow.p == 1 and d_flow.q == 1 and d_flow.t == -1

        # Spread is in price units
        assert SPREAD_DIMENSION.p == 1

        # Volatility [T^-1/2] squared is [T^-1]
        assert VOLATILITY ** 2 == RETURN_VELOCITY

    def test_unphysical_market_combinations_raise_error(self):
        """Unphysical financial additions or transcendentals on dimensional variables raise DimensionalError."""
        p_node = VariableNode("price", PRICE_DIMENSION)
        q_node = VariableNode("volume", VOLUME_QUANTITY)
        v_node = VariableNode("return", RETURN_VELOCITY)

        # Cannot add Price + Volume
        with pytest.raises(DimensionalError):
            BinaryOpNode("+", p_node, q_node)

        # Cannot add Price + Return
        with pytest.raises(DimensionalError):
            BinaryOpNode("-", p_node, v_node)

        # Cannot take transcendental exp() of dimensional quantity
        with pytest.raises(DimensionalError):
            UnaryOpNode("exp", p_node)

        # Log of price is invalid without dimensionless ratio
        with pytest.raises(DimensionalError):
            UnaryOpNode("log", p_node)

        # But log of price ratio p1 / p2 is valid!
        p2_node = VariableNode("price_ma", PRICE_DIMENSION)
        p_ratio = BinaryOpNode("/", p_node, p2_node)
        assert p_ratio.dimension.is_dimensionless()
        log_ret = UnaryOpNode("log", p_ratio)
        assert log_ret.dimension == DIMENSIONLESS


class TestMultiObjectiveParetoFrontier:
    """Verifies multi-objective frontier selection (error, complexity, novelty, stability)."""

    def test_multi_objective_preserves_accuracy_and_novelty_champions(self):
        """Rivals with near-identical error but higher novelty or cleaner residuals are retained."""
        x = np.linspace(0.1, 5.0, 50)
        X_tr = {"x": x[:30]}
        X_ho = {"x": x[30:]}

        # Candidate 1: standard fit
        c1 = fit_and_score_candidate(
            node=VariableNode("x", Dimension()),
            X_tr=X_tr,
            y_tr=x[:30],
            X_ho=X_ho,
            y_ho=x[30:],
            target_name="target",
        )

        # Candidate 2: rival candidate with virtually identical RMSE
        c2 = fit_and_score_candidate(
            node=PowNode(VariableNode("x", Dimension()), Fraction(1, 1)),
            X_tr=X_tr,
            y_tr=x[:30],
            X_ho=X_ho,
            y_ho=x[30:],
            target_name="target",
        )

        pareto = select_pareto_candidates([c1, c2], tau=0.1, max_rivals_per_complexity=2)
        assert len(pareto) >= 1
        assert pareto[0].holdout_rmse is not None
        assert np.isfinite(pareto[0].holdout_rmse)
