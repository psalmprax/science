"""Tests for APEDP Level 2 Open Discovery Milestone.

Verifies:
  1. Core L2 Contract:
     - The discovery engine receives ONLY raw data D = {X, t, Y}, variable names,
       SI dimensions, and primitive operators (+, -, *, /, pow, exp, log).
     - No expected equation families, expected exponents, or hand-written templates.
  2. Physically Typed AST Construction:
     - Enforces SI dimensional type calculus during construction:
       * [u] * [v] = [u + v]
       * [u] / [v] = [u - v]
       * [u] + [v] and [u] - [v] require [u] == [v]
       * exp(u) and log(u) require strictly dimensionless [0, 0, 0, ...]
       * pow(u, p) requires constant rational exponent p
  3. Three-Stage Pruning Pipeline:
     - Gate A: Algebraic validity (finite, non-singular, non-constant).
     - Gate B: Physical validity against target LHS dimension.
     - Gate C: Real least-squares parameter fitting, holdout RMSE, residual whiteness,
       and Pareto complexity selection.
  4. Adjudication & 4-State Epistemic Suite:
     - Dataset A (Recoverable): UNIQUE_SUPPORT
     - Dataset B (Observationally Collinear / Equivalent): MULTI_HYPOTHESIS
     - Dataset C (Too noisy / sparse): INSUFFICIENT_EVIDENCE
     - Dataset D (Out of model space): NO_VALID_CANDIDATE
  5. Provenance & Honesty:
     - Clean BenchmarkLevel.L2_OPEN_DISCOVERY threading.
     - Zero mock losses, real parameter fitting, real holdout residuals.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from apedp_dyn.discovery_l2.ast_generator import (
    ROLL_WINDOW,
    BinaryOpNode,
    PowNode,
    RollingOpNode,
    UnaryOpNode,
    VariableNode,
    collect_variable_names,
    generate_dimensionally_typed_ast,
)
from apedp_dyn.discovery_l2.dimensional_types import (
    DIMENSIONLESS,
    Dimension,
    DimensionalError,
)
from apedp_dyn.discovery_l2.l2_engine import (
    L2DiscoveryEngine,
)
from apedp_dyn.discovery_l2.pruning import (
    check_residual_whiteness,
    gate_a_algebraic_validity,
    gate_b_physical_validity,
)
from apedp_dyn.epistemic.adjudicate import (
    BenchmarkLevel,
    DataOrigin,
    EpistemicStatus,
)

# ── Data Generators for the 4-State Suite ───────────────────────────────────


def generate_dataset_a_recoverable(
    n_points: int = 60,
    noise_sigma: float = 0.02,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Dimension], Dimension]:
    """Dataset A: Recoverable physical system (Newton's second law: F = m * a).

    Independent mass m and acceleration a with clean physical ground truth.
    """
    rng = np.random.RandomState(seed)
    m = rng.uniform(1.0, 5.0, n_points)
    a = rng.uniform(2.0, 8.0, n_points)
    v = rng.uniform(5.0, 20.0, n_points)
    t = rng.uniform(1.0, 10.0, n_points)

    F_clean = 1.0 * (m * a)
    noise = rng.normal(0, noise_sigma * np.mean(F_clean), n_points)
    F_obs = F_clean + noise

    dimensions = {
        "m": Dimension(m=1),
        "a": Dimension(l=1, t=-2),
        "v": Dimension(l=1, t=-1),
        "t": Dimension(t=1),
    }
    target_dimension = Dimension(m=1, l=1, t=-2)
    features = {"m": m, "a": a, "v": v, "t": t}
    return features, F_obs, dimensions, target_dimension


def generate_dataset_b_collinear(
    n_points: int = 60,
    noise_sigma: float = 0.02,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Dimension], Dimension]:
    """Dataset B: Observationally collinear rivals (two coupled accelerations a1 ~ a2).

    Structural unidentifiability: both (m * a1) and (m * a2) have identical physical
    dimension and near-identical empirical predictive power on the available intervention.
    """
    rng = np.random.RandomState(seed)
    m = rng.uniform(1.0, 5.0, n_points)
    a1 = rng.uniform(2.0, 8.0, n_points)
    # a2 is strongly collinear with a1 in the observed domain
    a2 = a1 + rng.normal(0, 0.001, n_points)

    F_clean = 1.0 * (m * a1)
    noise = rng.normal(0, noise_sigma * np.mean(F_clean), n_points)
    F_obs = F_clean + noise

    dimensions = {
        "m": Dimension(m=1),
        "a1": Dimension(l=1, t=-2),
        "a2": Dimension(l=1, t=-2),
    }
    target_dimension = Dimension(m=1, l=1, t=-2)
    features = {"m": m, "a1": a1, "a2": a2}
    return features, F_obs, dimensions, target_dimension


def generate_dataset_c_noisy(
    n_points: int = 60,
    noise_sigma: float = 8.0,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Dimension], Dimension]:
    """Dataset C: Too noisy / low SNR regime.

    Noise is multiple times the underlying signal, rendering empirical evidence
    insufficient for scientific promotion.
    """
    rng = np.random.RandomState(seed)
    m = rng.uniform(1.0, 5.0, n_points)
    a = rng.uniform(2.0, 8.0, n_points)

    F_clean = 1.0 * (m * a)
    # Massive noise dominating signal
    noise = rng.normal(0, noise_sigma * np.std(F_clean), n_points)
    F_obs = F_clean + noise

    dimensions = {
        "m": Dimension(m=1),
        "a": Dimension(l=1, t=-2),
    }
    target_dimension = Dimension(m=1, l=1, t=-2)
    features = {"m": m, "a": a}
    return features, F_obs, dimensions, target_dimension


def generate_dataset_d_out_of_space(
    n_points: int = 60,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Dimension], Dimension]:
    """Dataset D: Out of model space / unexplainable by input variables.

    Target is Force (M L T^-2), but available inputs lack mass M entirely (only L, T, Theta).
    Physical dimensional gating must reject all proposals without unscientific hallucination.
    """
    rng = np.random.RandomState(seed)
    length = rng.uniform(1.0, 10.0, n_points)
    temp = rng.uniform(250.0, 350.0, n_points)
    time = rng.uniform(1.0, 5.0, n_points)

    # Some arbitrary process that cannot be expressed with length, temp, time
    F_obs = 100.0 + rng.normal(0, 1.0, n_points)

    dimensions = {
        "length": Dimension(l=1),
        "temp": Dimension(th=1),
        "time": Dimension(t=1),
    }
    target_dimension = Dimension(m=1, l=1, t=-2)
    features = {"length": length, "temp": temp, "time": time}
    return features, F_obs, dimensions, target_dimension


# ── Test Suite ───────────────────────────────────────────────────────────────


class TestPhysicallyTypedAST:
    """Verifies dimension-aware expression construction and type calculus."""

    def test_multiplication_adds_exponents(self):
        d_mass = Dimension(m=1)
        d_accel = Dimension(l=1, t=-2)
        d_force = d_mass * d_accel
        assert d_force == Dimension(m=1, l=1, t=-2)

    def test_division_subtracts_exponents(self):
        d_force = Dimension(m=1, l=1, t=-2)
        d_mass = Dimension(m=1)
        d_accel = d_force / d_mass
        assert d_accel == Dimension(l=1, t=-2)

    def test_addition_requires_matching_dimensions(self):
        d_len1 = Dimension(l=1)
        d_len2 = Dimension(l=1)
        d_time = Dimension(t=1)

        # Matching dimensions succeed
        d_sum = d_len1 + d_len2
        assert d_sum == Dimension(l=1)

        # Incompatible dimensions raise DimensionalError
        with pytest.raises(DimensionalError, match="incompatible dimensions"):
            _ = d_len1 + d_time

    def test_transcendental_requires_dimensionless(self):
        v_dim = VariableNode("x", Dimension(l=1))
        v_dimless = VariableNode("theta", DIMENSIONLESS)

        # Dimensional input raises DimensionalError at construction
        with pytest.raises(DimensionalError, match="strictly dimensionless"):
            _ = UnaryOpNode("exp", v_dim)

        with pytest.raises(DimensionalError, match="strictly dimensionless"):
            _ = UnaryOpNode("log", v_dim)

        # Dimensionless input constructs successfully
        exp_node = UnaryOpNode("exp", v_dimless)
        assert exp_node.dimension.is_dimensionless()

        log_node = UnaryOpNode("log", v_dimless)
        assert log_node.dimension.is_dimensionless()

    def test_pow_requires_rational_exponent(self):
        v_len = VariableNode("L", Dimension(l=1))

        # Rational powers scale exponents
        p_half = PowNode(v_len, Fraction(1, 2))
        assert p_half.dimension == Dimension(l=Fraction(1, 2))

        p_inv = PowNode(v_len, Fraction(-1, 1))
        assert p_inv.dimension == Dimension(l=-1)

        # Non-rational exponent raises TypeError
        with pytest.raises(TypeError, match="constant rational exponent"):
            _ = PowNode(v_len, "non_rational")  # type: ignore


class TestRollingOperators:
    """Causal rolling mean/std: dimension-preserving, strictly causal,
    non-nesting, and reachable from generation."""

    def test_preserves_dimension(self):
        v_dim = VariableNode("x", Dimension(l=1))
        assert RollingOpNode("rollmean", v_dim).dimension == Dimension(l=1)
        assert RollingOpNode("rollstd", v_dim).dimension == Dimension(l=1)

    def test_causal_values_match_numpy(self):
        v = VariableNode("x", DIMENSIONLESS)
        x = np.arange(1.0, 101.0)
        m = RollingOpNode("rollmean", v).evaluate({"x": x})
        s = RollingOpNode("rollstd", v).evaluate({"x": x})
        assert bool(np.isnan(m[: ROLL_WINDOW - 1]).all())
        assert abs(float(m[ROLL_WINDOW - 1]) - float(x[:ROLL_WINDOW].mean())) < 1e-9
        assert abs(float(s[ROLL_WINDOW - 1]) - float(x[:ROLL_WINDOW].std())) < 1e-9

    def test_rejects_nesting_and_unknown_ops(self):
        v = VariableNode("x", DIMENSIONLESS)
        with pytest.raises(ValueError, match="do not nest"):
            _ = RollingOpNode("rollmean", RollingOpNode("rollmean", v))
        with pytest.raises(ValueError, match="Unsupported rolling"):
            _ = RollingOpNode("median", v)

    def test_generation_reaches_rolling_forms(self):
        nodes = generate_dimensionally_typed_ast(
            {"volume": DIMENSIONLESS},
            operators=("+", "-", "/", "log", "rollmean", "rollstd"),
            max_depth=4, max_complexity=10)
        strs = [n.to_str() for n in nodes]
        assert any("rollmean40(log(volume))" in s and "rollstd40(log(volume))" in s
                   for s in strs)


class TestSymmetricPairGeneration:
    """Binary-operator pair enumeration must be symmetric.

    The frontier x pool enumeration used to build only (n_new, n_old) pairs, so
    n_old/n_new was unreachable and the commutative "the reverse pair will
    create it" guard dropped BOTH orderings; self-products (x*x) were excluded
    outright, making squares impossible whenever `pow` is absent from the
    grammar.
    """

    def test_division_orderings_both_reachable(self):
        nodes = generate_dimensionally_typed_ast(
            {"v": DIMENSIONLESS, "w": DIMENSIONLESS},
            operators=("/",), max_depth=2, max_complexity=6,
        )
        strs = {n.to_str() for n in nodes}
        assert "(v / w)" in strs, sorted(strs)
        assert "(w / v)" in strs, sorted(strs)

    def test_self_multiplication_reachable_without_pow(self):
        nodes = generate_dimensionally_typed_ast(
            {"v": DIMENSIONLESS, "w": DIMENSIONLESS},
            operators=("*",), max_depth=2, max_complexity=6,
        )
        strs = {n.to_str() for n in nodes}
        assert "(v * v)" in strs, sorted(strs)
        assert "(w * w)" in strs, sorted(strs)
        # ...and the squares compose further (v*v*w is reachable at depth 3).
        deep = generate_dimensionally_typed_ast(
            {"v": DIMENSIONLESS, "w": DIMENSIONLESS},
            operators=("*",), max_depth=3, max_complexity=8,
        )
        assert any(n.to_str() == "((v * v) * w)" for n in deep)

    def test_subtraction_orderings_are_sign_equivalent_by_design(self):
        """w - v is deduped against v - w because the fitted coefficient spans the sign.

        The basis fingerprint orients the first significant sample positive
        (fingerprint.py), so theta*f and -theta*(-f) are one basis function.
        That is a deliberate completeness statement, not an oversight: the
        subtraction arm of the audit finding does not apply, while the division
        and self-product arms do (both covered above).
        """
        nodes = generate_dimensionally_typed_ast(
            {"v": DIMENSIONLESS, "w": DIMENSIONLESS},
            operators=("-",), max_depth=2, max_complexity=6,
        )
        strs = {n.to_str() for n in nodes}
        assert "(v - w)" in strs
        import numpy as np
        env = {"v": np.array([1.0, 2.0, 3.0, 4.0]), "w": np.array([4.0, 1.0, 3.0, 2.0])}
        fwd = BinaryOpNode("-", VariableNode("v", DIMENSIONLESS), VariableNode("w", DIMENSIONLESS)).evaluate(env)
        rev = BinaryOpNode("-", VariableNode("w", DIMENSIONLESS), VariableNode("v", DIMENSIONLESS)).evaluate(env)
        assert np.allclose(fwd, -rev)

    def test_rolling_variables_are_collected(self):
        """A rolling term depends on its child's variables (was frozenset())."""
        v = VariableNode("v", DIMENSIONLESS)
        roll = RollingOpNode("rollmean", v, 5)
        assert collect_variable_names(roll) == frozenset(("v",))
        composed = BinaryOpNode("*", roll, VariableNode("w", DIMENSIONLESS))
        assert collect_variable_names(composed) == frozenset(("v", "w"))


class TestPruningGates:
    """Verifies the three-stage pruning pipeline (Gate A, B, C)."""

    def test_gate_a_rejects_singular_expressions(self):
        v1 = VariableNode("x", DIMENSIONLESS)
        v2 = VariableNode("zero", DIMENSIONLESS)
        div_node = BinaryOpNode("/", v1, v2)

        env = {"x": np.array([1.0, 2.0, 3.0]), "zero": np.array([0.0, 0.0, 0.0])}
        ok, reason, _ = gate_a_algebraic_validity(div_node, env)
        assert not ok
        assert "Singular" in reason

    def test_gate_a_rejects_constant_when_variance_expected(self):
        v = VariableNode("x", DIMENSIONLESS)
        sub_node = BinaryOpNode("-", v, v)

        env = {"x": np.array([1.0, 2.0, 3.0, 4.0])}
        ok, reason, _ = gate_a_algebraic_validity(sub_node, env)
        assert not ok
        assert "zero variance" in reason

    def test_gate_b_matches_target_lhs_dimension(self):
        target = Dimension(m=1, l=1, t=-2)
        valid_node = VariableNode("F", target)
        invalid_node = VariableNode("v", Dimension(l=1, t=-1))

        ok1, _ = gate_b_physical_validity(valid_node, target)
        assert ok1

        ok2, reason2 = gate_b_physical_validity(invalid_node, target)
        assert not ok2
        assert "mismatch" in reason2

    def test_residual_whiteness_metric(self):
        rng = np.random.RandomState(42)
        # White i.i.d. noise
        white_resid = rng.normal(0, 1.0, 100)
        is_white, rho1 = check_residual_whiteness(white_resid, threshold=0.5)
        assert is_white
        assert abs(rho1) < 0.3

        # Structured / autoregressive series
        structured = np.zeros(100)
        for i in range(1, 100):
            structured[i] = 0.9 * structured[i - 1] + rng.normal(0, 0.1)
        is_white_struct, rho1_struct = check_residual_whiteness(structured, threshold=0.5)
        assert not is_white_struct
        assert abs(rho1_struct) > 0.5


class TestEpistemicAdjudicationSuite:
    """Verifies all four epistemic outcomes autonomously without hand-crafted vocabularies."""

    def test_dataset_a_recovers_unique_support(self):
        """Dataset A (Recoverable): Discovers F = m * a and earns UNIQUE_SUPPORT."""
        X, Y, dims, target_dim = generate_dataset_a_recoverable(
            n_points=60, noise_sigma=0.02, seed=42
        )

        engine = L2DiscoveryEngine(operators=("+", "-", "*", "/", "pow"))
        res = engine.discover(
            X=X,
            Y=Y,
            dimensions=dims,
            target_dimension=target_dim,
            target_name="Force",
            noise_sigma=0.02 * float(np.mean(Y)),
            data_origin=DataOrigin.SIMULATION,
        )

        assert res.adjudication.status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.adjudication.promoted is not None
        cand_id = res.adjudication.promoted.candidate.candidate_id
        assert "a" in cand_id and "m" in cand_id and "*" in cand_id
        assert res.adjudication.benchmark_level == BenchmarkLevel.L2_OPEN_DISCOVERY
        assert res.adjudication.data_origin == DataOrigin.SIMULATION
        assert res.adjudication.best_rmse is not None and res.adjudication.best_rmse < 1.0

    def test_dataset_b_abstains_with_multi_hypothesis(self):
        """Dataset B (Collinear / Equivalent): Identifies collinear rivals and abstains with MULTI_HYPOTHESIS."""
        X, Y, dims, target_dim = generate_dataset_b_collinear(
            n_points=60, noise_sigma=0.02, seed=42
        )

        engine = L2DiscoveryEngine(operators=("+", "-", "*", "/", "pow"))
        res = engine.discover(
            X=X,
            Y=Y,
            dimensions=dims,
            target_dimension=target_dim,
            target_name="Force",
            noise_sigma=0.02 * float(np.mean(Y)),
            data_origin=DataOrigin.SIMULATION,
        )

        assert res.adjudication.status == EpistemicStatus.MULTI_HYPOTHESIS
        assert res.adjudication.promoted is None
        assert res.adjudication.delta_rmse is not None
        assert res.adjudication.tau_epi is not None
        assert res.adjudication.delta_rmse <= res.adjudication.tau_epi
        # Reason reworded with the prespecified-resolution form; assert the
        # claim itself rather than the retired "structures ... under available
        # interventions" phrasing that no longer exists in adjudicate().
        assert "observationally indistinguishable" in res.adjudication.reason

    def test_dataset_c_abstains_with_insufficient_evidence(self):
        """Dataset C (Too noisy): Signal overwhelmed by noise; epistemic layer returns INSUFFICIENT_EVIDENCE."""
        X, Y, dims, target_dim = generate_dataset_c_noisy(n_points=60, noise_sigma=8.0, seed=42)

        engine = L2DiscoveryEngine(operators=("+", "-", "*", "/", "pow"))
        res = engine.discover(
            X=X,
            Y=Y,
            dimensions=dims,
            target_dimension=target_dim,
            target_name="Force",
            noise_sigma=8.0 * float(np.std(Y)),
            data_origin=DataOrigin.SIMULATION,
        )

        assert res.adjudication.status == EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert res.adjudication.promoted is None

    def test_dataset_d_detects_out_of_model_space(self):
        """Dataset D (Out of model space): Variables lack mass; Gate B yields 0 candidates -> NO_VALID_CANDIDATE."""
        X, Y, dims, target_dim = generate_dataset_d_out_of_space(n_points=60, seed=42)

        engine = L2DiscoveryEngine(operators=("+", "-", "*", "/", "pow"))
        res = engine.discover(
            X=X,
            Y=Y,
            dimensions=dims,
            target_dimension=target_dim,
            target_name="Force",
            data_origin=DataOrigin.SIMULATION,
        )

        assert res.adjudication.status == EpistemicStatus.NO_VALID_CANDIDATE
        assert res.adjudication.promoted is None
        assert len(res.gate_b_passed) == 0
        assert res.adjudication.benchmark_level == BenchmarkLevel.L2_OPEN_DISCOVERY


class TestL2EngineHonestyAndProvenance:
    """Verifies no mock losses, real parameter fitting, and correct metadata threading."""

    def test_candidates_have_real_least_squares_fits(self):
        X, Y, dims, target_dim = generate_dataset_a_recoverable(n_points=50, seed=123)
        engine = L2DiscoveryEngine()
        res = engine.discover(
            X=X,
            Y=Y,
            dimensions=dims,
            target_dimension=target_dim,
            target_name="Force",
            data_origin=DataOrigin.EXPERIMENTAL,
        )

        assert len(res.candidates) > 0
        for c in res.candidates:
            assert c.train_loss is not None and np.isfinite(c.train_loss) and c.train_loss > 0.0
            assert c.holdout_rmse is not None and np.isfinite(c.holdout_rmse) and c.holdout_rmse > 0.0
            g4 = c.provenance.get("gate4_status", {})
            assert "holdout_residuals" in g4
            assert len(g4["holdout_residuals"]) == 25  # 50% holdout split
            # Real least-squares holdout: the recorded RMSE must equal the RMSE
            # recomputed from the recorded signed residuals (no mock values).
            recomputed_rmse = float(np.sqrt(np.mean(np.asarray(g4["holdout_residuals"]) ** 2)))
            assert abs(recomputed_rmse - c.holdout_rmse) < 1e-9
            assert c.provenance["benchmark_level"] == BenchmarkLevel.L2_OPEN_DISCOVERY
            assert c.provenance["data_origin"] == DataOrigin.EXPERIMENTAL

    def test_unconstrained_ast_construction_without_templates(self):
        """AST generator produces candidates purely from raw variables, SI dimensions, and primitives."""
        variables = {
            "m": Dimension(m=1),
            "v": Dimension(l=1, t=-1),
            "t": Dimension(t=1),
        }
        nodes = generate_dimensionally_typed_ast(
            variables, operators=("*", "/", "pow"), max_depth=2
        )
        assert len(nodes) > 10

        # Discovered forms like m * v (momentum), v / t (accel), m * (v / t) (force), (v^2)
        expr_strs = [n.to_str() for n in nodes]
        assert any("m" in s and "v" in s for s in expr_strs)
        assert any("t" in s for s in expr_strs)

    def test_t_parameter_integrated_in_discovery(self):
        """Pass t as an explicit temporal array in D = {X, t, Y}."""
        rng = np.random.RandomState(42)
        n_points = 60
        m = rng.uniform(1.0, 5.0, n_points)
        v = rng.uniform(5.0, 20.0, n_points)
        t = rng.uniform(1.0, 10.0, n_points)
        F_clean = m * (v / t)
        noise = rng.normal(0, 0.01, n_points)
        F_obs = F_clean + noise

        engine = L2DiscoveryEngine(operators=("*", "/", "pow"))
        # t is passed as explicit argument, NOT in X
        res = engine.discover(
            X={"m": m, "v": v},
            t=t,
            Y=F_obs,
            dimensions={"m": Dimension(m=1), "v": Dimension(l=1, t=-1)},
            target_dimension=Dimension(m=1, l=1, t=-2),
            target_name="Force",
            noise_sigma=0.01,
        )
        assert res.adjudication.status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.adjudication.promoted is not None
        assert "t" in res.adjudication.promoted.candidate.candidate_id

    def test_algebraic_canonicalization_prevents_spurious_multi_hypothesis(self):
        """Verify that 3-variable laws (p = m * x / t) achieve UNIQUE_SUPPORT.

        Syntactic variants such as ((m / t) * x) and ((m * x) / t) are recognized
        as identical algebraic structures and never compete as epistemic rivals.
        """
        rng = np.random.RandomState(42)
        n_points = 60
        m = rng.uniform(1.0, 5.0, n_points)
        x = rng.uniform(1.0, 10.0, n_points)
        t = rng.uniform(1.0, 10.0, n_points)
        p_clean = (m * x) / t
        noise = rng.normal(0, 0.01, n_points)
        p_obs = p_clean + noise

        engine = L2DiscoveryEngine(operators=("*", "/", "pow"))
        res = engine.discover(
            X={"m": m, "x": x, "t": t},
            Y=p_obs,
            dimensions={"m": Dimension(m=1), "x": Dimension(l=1), "t": Dimension(t=1)},
            target_dimension=Dimension(m=1, l=1, t=-1),
            target_name="p",
            noise_sigma=0.01,
        )
        assert res.adjudication.status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.adjudication.promoted is not None

    def test_structural_constant_elimination(self):
        """Expressions like x - x and x / x are pruned at the structural level."""
        dims = {"x": Dimension(l=1)}
        nodes = generate_dimensionally_typed_ast(dims, operators=("+", "-", "*", "/"), max_depth=1)
        expr_strs = [n.to_str() for n in nodes]
        assert "(x - x)" not in expr_strs
        assert "(x / x)" not in expr_strs

    def test_mismatched_sample_lengths_raise_error(self):
        engine = L2DiscoveryEngine()
        with pytest.raises(ValueError, match="mismatch"):
            engine.discover(
                X={"x": np.array([1.0, 2.0, 3.0])},
                Y=np.array([1.0, 2.0]),
                dimensions={"x": Dimension(l=1)},
                target_dimension=Dimension(l=1),
            )

    def test_insufficient_sample_size_returns_graceful_adjudication(self):
        engine = L2DiscoveryEngine()
        res = engine.discover(
            X={"x": np.array([1.0, 2.0])},
            Y=np.array([1.0, 2.0]),
            dimensions={"x": Dimension(l=1)},
            target_dimension=Dimension(l=1),
        )
        assert res.adjudication.status == EpistemicStatus.NO_VALID_CANDIDATE
        assert res.adjudication.promoted is None
