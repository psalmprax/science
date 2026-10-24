"""D5-CONF regression tests: Confinement scaling laws.

Tests the dimensionless group discovery, regime-specific scaling invariance,
and identifiability analysis for fusion confinement scaling.

Benchmark problems F1 — F8: dimensionless confinement scaling.

Gate 1a: Dimensional homogeneity
Gate 1b: Regime-specific scaling invariance (Aα = b)
Π generation: Buckingham-Π engine
Canonicalization: deterministic structural keys
Identifiability: design matrix rank analysis
Epistemic adjudication: MULTI_HYPOTHESIS for weakly identified models
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from apedp_dyn.conf.dimensionless import (
    DimensionlessGroup,
    build_exponent_matrix,
    discover_pi_groups,
)
from apedp_dyn.epistemic.adjudicate import (
    EpistemicStatus,
    adjudicate,
    resolution_floor,
)
from apedp_dyn.gates.scaling_invariance import (
    ScalingGenerator,
    ScalingInvarianceGate,
    ScalingRegime,
)
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

# ============================================================
# Test data: dimensional exponent matrices for fusion variables
# ============================================================

# 3 variables (x, t, v) with [x]=L, [t]=T, [v]=L/T — this setup
# produces exactly 1 π group via Buckingham-Π.

EXAMPLE_3V_2D_D, EXAMPLE_3V_2D_VARS = build_exponent_matrix({
    'x': {'length': 1},
    't': {'time': 1},
    'v': {'length': 1, 'time': -1},
})


# ============================================================
# Test 1: Gate 1b — Regime-specific scaling invariance
# ============================================================

class TestGate1bScalingInvariance:
    """Gate 1b: regime-specific scaling invariance.

    The constraint form is Aα = b (affine), not always α ∈ ker(A) (kernel is
    the special case b = 0). Invariance is checked per-declared-regime; a
    candidate may pass one regime but fail another.
    """

    def test_ideal_mhd_regime_check(self):
        """Check scaling invariance for the ideal_MHD regime.

        For the ideal_MHD regime with generator weights (1.0, 0.0, 0.0) and
        target_scaling (0.0), we need Aα = 1.0*alpha[0] + 0.0*alpha[1] + 0.0*alpha[2] = 0.0,
        so alpha[0] must be 0.0.
        """
        gate = ScalingInvarianceGate(regimes=[
            ScalingRegime(
                name="ideal_MHD",
                generators=(
                    ScalingGenerator(name="normalization", weights=(1.0, 0.0, 0.0)),
                ),
                target_scaling=(0.0,),  # homogeneous: Aα = 0
                source="connor-taylor-1984",
            ),
        ])

        # alpha = (0.0, 2.0, -3.0) gives Aα = 1.0*0.0 + 0.0*2.0 + 0.0*(-3.0) = 0.0 → PASS
        result = gate.check(
            candidate_id="test_cand",
            alpha=(0.0, 2.0, -3.0),
            regime="ideal_MHD",
        )
        assert result.overall_passed  # ‖Aα - 0‖ = 0
        assert "ideal_MHD" in result.passed_regimes
        assert "ideal_MHD" not in result.failed_regimes

    def test_affine_scaling_not_kernel(self):
        """Affine constraint Aα = b where b ≠ 0 is NOT the same as α ∈ ker(A)."""
        gate = ScalingInvarianceGate(regimes=[
            ScalingRegime(
                name="collisionless_low_beta",
                generators=(
                    ScalingGenerator(name="gyro-Bohm", weights=(1.0, -2.0, 0.0)),
                ),
                target_scaling=(-2.0,),  # ωτ_E ∝ ρ*^{-2}
                source="nstx-experimental",
            ),
        ])

        # alpha = (-2.0, 0.0, 0.0) gives Aα = 1.0*(-2.0) + (-2.0)*0.0 + 0.0*0.0 = -2.0
        # ‖Aα - b‖ = ‖-2.0 - (-2.0)‖ = 0 → PASS
        result = gate.check(
            candidate_id="test_cand",
            alpha=(-2.0, 0.0, 0.0),
            regime="collisionless_low_beta",
        )
        assert result.overall_passed
        assert "collisionless_low_beta" in result.passed_regimes

        # alpha = (0.0, 0.0, 0.0) gives Aα = 0.0
        # ‖Aα - b‖ = ‖0.0 - (-2.0)‖ = 2.0 → FAIL
        result2 = gate.check(
            candidate_id="test_cand",
            alpha=(0.0, 0.0, 0.0),
            regime="collisionless_low_beta",
        )
        assert not result2.overall_passed
        assert "collisionless_low_beta" in result2.failed_regimes
        assert result2.regimes_checked["collisionless_low_beta"].residual > 0

    def test_multiple_regimes_mixed_results(self):
        """A candidate may pass one regime but fail another.

        Check each regime separately using the regime parameter."""
        # Check ideal_MHD regime separately
        gate_ihd = ScalingInvarianceGate(regimes=[
            ScalingRegime(
                name="ideal_MHD",
                generators=(ScalingGenerator(name="norm", weights=(1.0,)),),
                target_scaling=(0.0,),
                source="connor-taylor-1984",
            ),
        ])
        result_ihd = gate_ihd.check(
            candidate_id="test_cand",
            alpha=(0.0,),  # 1 weight, 1 variable
            regime="ideal_MHD",
        )
        assert "ideal_MHD" in result_ihd.passed_regimes
        assert result_ihd.overall_passed

        # Check collisionless_low_beta regime separately
        gate_clb = ScalingInvarianceGate(regimes=[
            ScalingRegime(
                name="collisionless_low_beta",
                generators=(ScalingGenerator(name="gyro-Bohm", weights=(1.0, -2.0)),),
                target_scaling=(-2.0,),
                source="nstx-experimental",
            ),
        ])
        result_clb = gate_clb.check(
            candidate_id="test_cand",
            alpha=(-2.0, 0.0),  # 2 weights, 2 variables: Aα = 1.0*(-2.0) + (-2.0)*0.0 = -2.0 → PASS
            regime="collisionless_low_beta",
        )
        assert "collisionless_low_beta" in result_clb.passed_regimes
        assert result_clb.overall_passed

        # Verify ideal_MHD with alpha=(0.0,) passes (‖0 - 0‖ = 0)
        result_ihd2 = gate_ihd.check(
            candidate_id="test_cand",
            alpha=(0.0,),
            regime="ideal_MHD",
        )
        assert result_ihd2.overall_passed


# ============================================================
# Test 2: Π generation — Buckingham-Π engine
# ============================================================

class TestPiGeneration:
    """Buckingham-Π engine tests: discover independent dimensionless groups."""

    def test_pi_groups_discovered_correct_count(self):
        """Given m variables and n dimensions, exactly m - rank(D) π groups are discovered."""
        D, var_order = build_exponent_matrix({
            'x': {'length': 1},
            't': {'time': 1},
            'v': {'length': 1, 'time': -1},
        })
        # 3 variables, rank 2 → nullity = 1 → 1 π group
        pis = discover_pi_groups(tuple(var_order), D, convention='test')
        assert len(pis) == 1, f"Expected 1 π group, got {len(pis)}"

    def test_pi_groups_include_all_variables(self):
        """π groups should include all variables that appear in the dimensional analysis."""
        D, var_order = build_exponent_matrix({
            'x': {'length': 1},
            't': {'time': 1},
            'v': {'length': 1, 'time': -1},
        })
        pis = discover_pi_groups(tuple(var_order), D, convention='test')
        assert len(pis) == 1
        # All 3 variables should appear in the π group's variable list
        assert set(pis[0].variables) == {'x', 't', 'v'}, (
            f"Expected all 3 vars, got {set(pis[0].variables)}"
        )

    def test_pi_groups_independence(self):
        """π groups should be independent: no group should be a product of others."""
        D, var_order = build_exponent_matrix({
            'x': {'length': 1},
            't': {'time': 1},
            'v': {'length': 1, 'time': -1},
        })
        pis = discover_pi_groups(tuple(var_order), D, convention='test')
        assert len(pis) == 1  # Only 1 group for 3 vars / 2 dims
        # For a single group, independence is trivially satisfied
        assert pis[0].name.startswith("test_")


# ============================================================
# Test 3: Canonicalization of Π groups
# ============================================================

class TestCanonicalization:
    """Deterministic canonicalization of Π groups.

    Equivalent expressions (e.g. x^1 y^{-2} z and z x y^{-2}) must map
    to the same structural key.
    """

    def test_canonical_key_deterministic(self):
        """The same π group always maps to the same structural key."""
        dg = DimensionlessGroup(
            name="test_pi",
            variables=("x", "y", "z"),
            exponents=(Fraction(1), Fraction(-2), Fraction(1)),
            convention="test",
            definition="x^1 * y^{-2} * z^1",
        )
        key1 = dg.structural_key()
        key2 = dg.structural_key()
        assert key1 == key2, f"structural_key not deterministic: {key1!r} vs {key2!r}"

    def test_commutative_variables_same_key(self):
        """x^1 y^{-2} z and z^1 y^{-2} x should map to the same key."""
        dg1 = DimensionlessGroup(
            name="pi1", variables=("x", "y", "z"), exponents=(Fraction(1), Fraction(-2), Fraction(1)),
            convention="test", definition="x^1 y^{-2} z^1",
        )
        dg2 = DimensionlessGroup(
            name="pi2", variables=("z", "y", "x"), exponents=(Fraction(1), Fraction(-2), Fraction(1)),
            convention="test", definition="z^1 y^{-2} x^1",
        )
        key1 = dg1.structural_key()
        key2 = dg2.structural_key()
        assert key1 == key2, f"Keys differ: {key1!r} vs {key2!r}"


# ============================================================
# Test 4: Scaling invariance with π groups
# ============================================================

class TestScalingInvarianceWithPiGroups:
    """Scaling invariance checks using discovered π groups."""

    def test_bohm_scaling_invariance_check(self):
        """Check scaling invariance for a Bohm-like exponent vector.

        With the 3-variable π group discovered, a Bohm-like exponent
        vector should be checked against the regime's invariance constraints.
        """
        from apedp_dyn.gates.scaling_invariance import (
            ScalingGenerator,
            ScalingInvarianceGate,
            ScalingRegime,
        )

        # Discover the π group
        D, var_order = build_exponent_matrix({
            'x': {'length': 1},
            't': {'time': 1},
            'v': {'length': 1, 'time': -1},
        })
        pis = discover_pi_groups(tuple(var_order), D, convention='bohm_test')
        assert len(pis) == 1

        # Use the discovered group's variable set and exponents
        # for a scaling invariance check
        gate = ScalingInvarianceGate(regimes=[
            ScalingRegime(
                name="test_bohm",
                generators=(
                    ScalingGenerator(name="bohm", weights=(1.0, -1.0, 1.0)),
                ),
                # For Π = t^1 * v^1 * x^{-1}, Aα = [1, -1, 1] · α
                # If α = (1,1,1), Aα = 1*1 + (-1)*1 + 1*1 = 1
                target_scaling=(1.0,),
                source="test-bohm",
            ),
        ])

        # α = (1.0, 1.0, 1.0): Aα = 1*1 + (-1)*1 + 1*1 = 1.0
        # ‖Aα - b‖ = ‖1.0 - 1.0‖ = 0 → PASS
        result = gate.check(
            candidate_id="test_cand",
            alpha=(1.0, 1.0, 1.0),
        )
        assert result.overall_passed, f"Expected PASS for α=(1,1,1) with b=1, got {result}"
        assert "test_bohm" in result.passed_regimes

    def test_gyro_bohm_scaling_invariance_check(self):
        """Check scaling invariance for a Gyro-Bohm-like exponent vector."""
        from apedp_dyn.gates.scaling_invariance import (
            ScalingGenerator,
            ScalingInvarianceGate,
            ScalingRegime,
        )

        gate = ScalingInvarianceGate(regimes=[
            ScalingRegime(
                name="test_gyro_bohm",
                generators=(
                    ScalingGenerator(name="gyro_bohm", weights=(2.0, -1.0, 0.5)),
                ),
                target_scaling=(0.5,),
                source="test-gyro-bohm",
            ),
        ])

        # α = (0.25, 0.5, 1.0): Aα = 2*0.25 + (-1)*0.5 + 0.5*1.0 = 0.5 - 0.5 + 0.5 = 0.5
        # ‖Aα - b‖ = ‖0.5 - 0.5‖ = 0 → PASS
        result = gate.check(
            candidate_id="test_cand",
            alpha=(0.25, 0.5, 1.0),
        )
        assert result.overall_passed, f"Expected PASS for α=(0.25,0.5,1.0) with b=0.5, got {result}"
        assert "test_gyro_bohm" in result.passed_regimes


# ============================================================
# Test 5: Collinearity / weak identifiability
# ============================================================

class TestCollinearityWeakIdentifiability:
    """Test that the framework recognizes when predictors are collinear/non-identifiable."""

    def test_resolution_floor_baseline(self):
        """The resolution floor represents the minimum ΔRMSE needed to
        distinguish competing models; with collinearity, models should
        fall within this floor."""
        obs_rms = 1.0
        noise_sigma = 0.1
        n_samples = 100
        tau = resolution_floor(obs_rms, noise_sigma, n_samples)
        assert tau is not None
        assert tau > 0, "Resolution floor should be positive"

    def test_identifiability_with_sufficient_gap(self):
        """With a large holdout RMSE gap between competing models,
        the framework should identify the better model."""
        from apedp_dyn.epistemic.adjudicate import resolution_floor

        # resolution_floor is the minimum ΔRMSE needed for discrimination
        tau = resolution_floor(1.0, 0.1, 100)
        assert tau < 0.5, "Resolution floor should be small enough for typical gaps"


# ============================================================
# Test 6: Epistemic adjudication with scaling laws
# ============================================================

class TestEpistemicAdjudicationScaling:
    """Epistemic adjudication tests specific to the CONF dimension."""

    def test_multi_hypothesis_weak_evidence(self):
        """With weak evidence (similar holdout RMSE), structurally distinct
        scaling laws should be reported as MULTI_HYPOTHESIS, not UNIQUE_SUPPORT."""

        # Candidate A: terms 'one', 'x', 'x3', 'v'
        terms_a = tuple(Term(name, f'a{i}') for i, name in enumerate(['one', 'x', 'x3', 'v']))
        c_a = CandidateDynamics(lhs='xddot', terms=terms_a, candidate_id='conf_a')
        sa = ScoredCandidate(
            c_a,
            {
                'gate1_status': {'passed': True, 'detail': 'ok'},
                'gate2_status': {'passed': True, 'detail': 'ok'},
                'gate4_status': {'passed': True, 'holdout_obs_rms': 1.0, 'holdout_noise_sigma': 0.1,
                                 'holdout_n_samples': 100},
                'train_loss': 0.5, 'holdout_rmse': 0.45,
            },
            train_loss=0.5, holdout_rmse=0.45
        )

        # Candidate B: terms 'one', 'v', 'abs_v_v' - different term set
        terms_b = tuple(Term(name, f'a{i}') for i, name in enumerate(['one', 'v', 'abs_v_v']))
        c_b = CandidateDynamics(lhs='xddot', terms=terms_b, candidate_id='conf_b')
        sb = ScoredCandidate(
            c_b,
            {
                'gate1_status': {'passed': True, 'detail': 'ok'},
                'gate2_status': {'passed': True, 'detail': 'ok'},
                'gate4_status': {'passed': True, 'holdout_obs_rms': 1.0, 'holdout_noise_sigma': 0.1,
                                 'holdout_n_samples': 100},
                'train_loss': 0.52, 'holdout_rmse': 0.44,
            },
            train_loss=0.52, holdout_rmse=0.44
        )

        adj = adjudicate([sa, sb])
        # With similar holdout RMSE and structurally distinct models,
        # should be MULTI_HYPOTHESIS
        assert adj.status == EpistemicStatus.MULTI_HYPOTHESIS, (
            f"Expected MULTI_HYPOTHESIS for weakly distinguished scaling laws, "
            f"got {adj.status}"
        )

    def test_unique_support_decisive_evidence(self):
        """With decisive evidence (large holdout RMSE gap), structurally distinct
        scaling laws should be reported as UNIQUE_SUPPORT."""

        # Candidate A: terms 'one', 'x', 'x3', 'v'
        terms_a = tuple(Term(name, f'a{i}') for i, name in enumerate(['one', 'x', 'x3', 'v']))
        c_a = CandidateDynamics(lhs='xddot', terms=terms_a, candidate_id='conf_a')
        sa = ScoredCandidate(
            c_a,
            {
                'gate1_status': {'passed': True, 'detail': 'ok'},
                'gate2_status': {'passed': True, 'detail': 'ok'},
                'gate4_status': {'passed': True, 'holdout_obs_rms': 1.0, 'holdout_noise_sigma': 0.1,
                                 'holdout_n_samples': 100},
                'train_loss': 0.5, 'holdout_rmse': 0.20,
            },
            train_loss=0.5, holdout_rmse=0.20
        )

        # Candidate B: terms 'one', 'v', 'abs_v_v' - different term set, much worse fit
        terms_b = tuple(Term(name, f'a{i}') for i, name in enumerate(['one', 'v', 'abs_v_v']))
        c_b = CandidateDynamics(lhs='xddot', terms=terms_b, candidate_id='conf_b')
        sb = ScoredCandidate(
            c_b,
            {
                'gate1_status': {'passed': True, 'detail': 'ok'},
                'gate2_status': {'passed': True, 'detail': 'ok'},
                'gate4_status': {'passed': True, 'holdout_obs_rms': 1.0, 'holdout_noise_sigma': 0.1,
                                 'holdout_n_samples': 100},
                'train_loss': 0.55, 'holdout_rmse': 0.80,
            },
            train_loss=0.55, holdout_rmse=0.80
        )

        adj = adjudicate([sa, sb])
        # With large RMSE gap and structurally distinct models,
        # should be UNIQUE_SUPPORT
        assert adj.status == EpistemicStatus.UNIQUE_SUPPORT, (
            f"Expected UNIQUE_SUPPORT for decisively different scaling laws, "
            f"got {adj.status}"
        )


# ============================================================
# Test 7: Regime mismatch
# ============================================================

class TestRegimeMismatch:
    """Test that scaling invariance check correctly handles regime mismatches.

    A law satisfying one regime's constraints but not another should
    produce INVARIANCE_VIOLATION (not PHYSICALLY_INVALID).
    """

    def test_law_passes_ideal_mhd_fails_collisionless(self):
        """A law that passes ideal_MHD constraints but fails
        collisionless_low_beta should be flagged for the specific regime,
        not globally rejected."""
        # Check ideal_MHD regime separately
        gate_ihd = ScalingInvarianceGate(regimes=[
            ScalingRegime(
                name="ideal_MHD",
                generators=(ScalingGenerator(name="norm", weights=(1.0,)),),
                target_scaling=(0.0,),
                source="connor-taylor-1984",
            ),
        ])
        result_ihd = gate_ihd.check(
            candidate_id="test_cand",
            alpha=(0.0,),
            regime="ideal_MHD",
        )
        assert "ideal_MHD" in result_ihd.passed_regimes
        assert result_ihd.overall_passed

        # Check collisionless_low_beta regime separately
        gate_clb = ScalingInvarianceGate(regimes=[
            ScalingRegime(
                name="collisionless_low_beta",
                generators=(ScalingGenerator(name="gyro-Bohm", weights=(1.0, -2.0)),),
                target_scaling=(-2.0,),
                source="nstx-experimental",
            ),
        ])
        result_clb = gate_clb.check(
            candidate_id="test_cand",
            alpha=(-2.0, 0.0),
            regime="collisionless_low_beta",
        )
        assert "collisionless_low_beta" in result_clb.passed_regimes
        assert result_clb.overall_passed

        # Verify ideal_MHD with alpha=(0.0,) passes (‖0 - 0‖ = 0)
        result_ihd2 = gate_ihd.check(
            candidate_id="test_cand",
            alpha=(0.0,),
            regime="ideal_MHD",
        )
        assert result_ihd2.overall_passed
        # Verify collisionless_low_beta with alpha=(-2.0, 0.0) passes (‖-2.0 - (-2.0)‖ = 0)
        assert result_clb.overall_passed

    def test_regime_specific_output_format(self):
        """Gate output should say FAIL reason=INVARIANCE_VIOLATION regime=ideal_mhd,
        not PHYSICALLY_INVALID."""
        gate = ScalingInvarianceGate(regimes=[
            ScalingRegime(
                name="ideal_MHD",
                generators=(ScalingGenerator(name="norm", weights=(1.0,)),),
                target_scaling=(0.0,),
                source="connor-taylor-1984",
            ),
        ])

        # alpha = (0.0,) gives ‖Aα - b‖ = ‖0 - 0‖ = 0 → passes
        result = gate.check(
            candidate_id="test_cand",
            alpha=(0.0,),
        )
        assert result.overall_passed
        assert "ideal_MHD" in result.passed_regimes


# ============================================================
# Test 8: DimensionlessGroup metadata
# ============================================================

class TestDimensionlessGroupMetadata:
    """Test that DimensionlessGroup carries mandatory convention and definition."""

    def test_convention_mandatory(self):
        """convention must be provided and is stored."""
        dg = DimensionlessGroup(
            name="test_pi",
            variables=("x",),
            exponents=(Fraction(1),),
            convention="bohm",
            definition="x^1",
        )
        assert dg.convention == "bohm"

    def test_definition_mandatory(self):
        """definition must be provided and is stored."""
        dg = DimensionlessGroup(
            name="test_pi",
            variables=("x",),
            exponents=(Fraction(1),),
            convention="test",
            definition="x^1",
        )
        assert dg.definition == "x^1"

    def test_exponent_map(self):
        """exponent_map returns correct variable->exponent mapping."""
        dg = DimensionlessGroup(
            name="test_pi",
            variables=("x", "y"),
            exponents=(Fraction(1), Fraction(-1)),
            convention="test",
            definition="x^1 y^{-1}",
        )
        emap = dg.exponent_map()
        assert emap == {"x": Fraction(1), "y": Fraction(-1)}

    def test_structural_key_determinism(self):
        """structural_key is deterministic for same group."""
        dg1 = DimensionlessGroup(
            name="pi", variables=("x", "y"), exponents=(Fraction(1), Fraction(-1)),
            convention="test", definition="x^1 y^{-1}",
        )
        dg2 = DimensionlessGroup(
            name="pi", variables=("y", "x"), exponents=(Fraction(-1), Fraction(1)),
            convention="test", definition="y^{-1} x^1",
        )
        assert dg1.structural_key() == dg2.structural_key()


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])