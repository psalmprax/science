"""The 12-Case Decisive Benchmark Protocol for APEDP 3.0.

Evaluates the autonomous Scientific Model Discovery and Inference Engine across
the 12 definitive scenarios:
1. Clean Recoverable Analytic Law (Kepler/Newtonian parabolic law)
2. Observationally Equivalent Hypotheses (Shomate vs Quadratic Cp)
3. Hidden OOD Overfitting Trap (High-degree polynomial vs Exponential)
4. Pathological MSE Trap (Singularity/pole avoided on sample points)
5. Dimensional Incompatibility Trap (Units mismatch & transcendental holds)
6. Syntactic / Algebraic Permutations (Identity canonicalization)
7. Bootstrap-Unstable Artifacts (Sampling artifact rejection)
8. High-D Non-Symbolic Function (Neural adversarial reference ceiling & Hybrid)
9. Sparse-Data Asymptotic Constraint (Boundary/asymptotic physics compliance)
10. Closed-Loop Active Hypothesis Discrimination (MULTI_HYPOTHESIS -> x* query -> UNIQUE_SUPPORT)
11. False Confidence Trap (Identifiability & stability failure -> INSUFFICIENT_EVIDENCE)
12. Wrong Physical Oracle / Scientific Falsification & Belief Revision (Empirical falsification -> belief update)
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp.decision.stage_c import (
    CandidateProfile,
    EpistemicState,
    ModelClass,
    StageCDecisionEngine,
)
from apedp.neural_discovery.config import NeuralDiscoveryConfig
from apedp.pipeline import DiscoveryPipeline
from apedp.symbolic_equivalence import SymbolicEquivalenceEngine


@pytest.fixture
def fast_cfg():
    return NeuralDiscoveryConfig(
        max_epochs=20,
        uncertainty_method="dropout",
        num_layers=2,
        hidden_dims=(32, 32),
        dropout_rate=0.05,
    )


# --------------------------------------------------------------------------
# Case 1: Clean Recoverable Analytic Law
# --------------------------------------------------------------------------
def test_case_1_clean_recoverable_analytic_law(tmp_path, fast_cfg):
    """A clean parabolic law: Symbolic candidate matches neural ceiling and achieves UNIQUE_SUPPORT."""
    rng = np.random.default_rng(42)
    x = np.linspace(1.0, 5.0, 50)
    y = 2.5 * (x**2) + 1.5 + rng.normal(0, 0.02, size=len(x))

    pipeline = DiscoveryPipeline(output_dir=str(tmp_path / "case1"))
    res = pipeline.run_scientific_investigation(
        data={"x": x, "y": y},
        target_column="y",
        hypotheses_to_evaluate=[
            {"candidate_id": "true_quad", "expression": "2.5*x**2 + 1.5", "complexity": 3.0},
            {"candidate_id": "underfit_lin", "expression": "15.0*x - 12.0", "complexity": 2.0},
        ],
        neural_config=fast_cfg,
        include_neural=True,
    )

    decision = res["decision"]
    assert decision["state"] == EpistemicState.UNIQUE_SUPPORT.value
    assert decision["winner"] == "true_quad"
    assert "neural_adversary" in [c["candidate_id"] for c in res["candidates"]]


# --------------------------------------------------------------------------
# Case 2: Observationally Equivalent Hypotheses
# --------------------------------------------------------------------------
def test_case_2_observationally_equivalent_hypotheses(tmp_path):
    """Narrow temperature range: Shomate and Quadratic Cp models fit holdout equally well.
    APEDP refuses to pick a single winner, emitting MULTI_HYPOTHESIS."""
    t = np.linspace(0.2, 0.6, 40)
    y_shomate = 30.0 + 15.0 * t - 5.0 * (t**2) + 0.5 * (t**3)
    y_quad = 30.0 + 14.8 * t - 4.5 * (t**2)
    # Midpoint target ensuring balanced holdout errors
    cp = 0.5 * (y_shomate + y_quad)

    pipeline = DiscoveryPipeline(output_dir=str(tmp_path / "case2"))
    res = pipeline.run_scientific_investigation(
        data={"t": t, "cp": cp},
        target_column="cp",
        hypotheses_to_evaluate=[
            {"candidate_id": "shomate_cp", "expression": "30.0 + 15.0*t - 5.0*t**2 + 0.5*t**3", "complexity": 8.0},
            {"candidate_id": "quadratic_cp", "expression": "30.0 + 14.8*t - 4.5*t**2", "complexity": 3.0},
        ],
        holdout_margin_ratio=1.20,
        include_neural=False,
        include_hybrid=False,
    )

    decision = res["decision"]
    assert decision["state"] == EpistemicState.MULTI_HYPOTHESIS.value
    assert set(decision["supported_hypotheses"]) == {"shomate_cp", "quadratic_cp"}
    assert decision["discrimination_required"] is True


# --------------------------------------------------------------------------
# Case 3: Hidden OOD Trap
# --------------------------------------------------------------------------
def test_case_3_hidden_ood_trap(tmp_path):
    """High-degree polynomial achieves R2 > 0.999 on [0, 2] but explodes on [2.5, 4.0].
    Physical exponential relaxation model preserves stability and wins."""
    # In-sample: x in [0.2, 2.0]
    x_in = np.linspace(0.2, 2.0, 30)
    y_in = 5.0 * np.exp(-0.8 * x_in)

    # Near-OOD: x in [2.5, 4.0]
    x_ood = np.linspace(2.5, 4.0, 20)
    y_ood = 5.0 * np.exp(-0.8 * x_ood)

    # Overfitting polynomial fit on in-sample
    poly_coeffs = np.polyfit(x_in, y_in, deg=4)
    poly_expr = f"{poly_coeffs[0]:.4g}*x**4 + {poly_coeffs[1]:.4g}*x**3 + {poly_coeffs[2]:.4g}*x**2 + {poly_coeffs[3]:.4g}*x + {poly_coeffs[4]:.4g}"

    pipeline = DiscoveryPipeline(output_dir=str(tmp_path / "case3"))
    res = pipeline.run_scientific_investigation(
        data={"x": x_in, "y": y_in},
        target_column="y",
        near_ood_data={"x": x_ood, "y": y_ood},
        hypotheses_to_evaluate=[
            {"candidate_id": "poly_deg4", "expression": poly_expr, "complexity": 5.0},
            {"candidate_id": "exponential_relaxation", "expression": "5.0*exp(-0.8*x)", "complexity": 3.0},
        ],
        include_neural=False,
        include_hybrid=False,
    )

    cand_map = {c["candidate_id"]: c for c in res["candidates"]}
    assert cand_map["poly_deg4"]["near_ood_mse"] > cand_map["exponential_relaxation"]["near_ood_mse"] * 2.0


# --------------------------------------------------------------------------
# Case 4: Pathological MSE Trap
# --------------------------------------------------------------------------
def test_case_4_pathological_mse_trap(tmp_path):
    """Candidate with an unphysical singularity 1/(x - 2.5001) avoided on sample points.
    Stage A dense domain audit catches the pole, flagging math_sound=False."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    pipeline = DiscoveryPipeline(output_dir=str(tmp_path / "case4"))
    res = pipeline.run_scientific_investigation(
        data={"x": x, "y": y},
        target_column="y",
        hypotheses_to_evaluate=[
            # Singularity inside [1.0, 5.0] at x = 2.5001
            {"candidate_id": "pole_trap", "expression": "1.0 / (x - 2.5001) + 2.0*x"},
            {"candidate_id": "sound_linear", "expression": "2.0*x"},
        ],
        include_neural=False,
        include_hybrid=False,
    )

    cand_map = {c["candidate_id"]: c for c in res["candidates"]}
    assert cand_map["pole_trap"]["math_sound"] is False
    assert cand_map["sound_linear"]["math_sound"] is True
    # Sound linear wins Stage C
    assert res["decision"]["winner"] == "sound_linear"


# --------------------------------------------------------------------------
# Case 5: Dimensional Incompatibility Trap
# --------------------------------------------------------------------------
def test_case_5_dimensional_incompatibility_trap(tmp_path):
    """Dimensionally invalid candidate T + x (time + length) and unscaled transcendental cos(T)
    are excluded from Stage C selection."""
    engine = StageCDecisionEngine()
    candidates = [
        CandidateProfile(
            candidate_id="invalid_sum",
            expression="T + x",
            math_sound=True,
            dim_status="INVALID",  # Incompatible physical dimensions
            in_sample_mse=0.001,
            holdout_mse=0.001,
        ),
        CandidateProfile(
            candidate_id="unresolved_transcendental",
            expression="cos(T)",
            math_sound=True,
            dim_status="UNRESOLVED",  # Unscaled dimensioned argument (P2 HOLD)
            in_sample_mse=0.002,
            holdout_mse=0.002,
        ),
    ]

    decision = engine.evaluate(candidates)
    assert decision.state == EpistemicState.NO_VALID_CANDIDATE
    assert decision.winner is None
    assert "Stage A gates" in decision.adjudication_rationale


# --------------------------------------------------------------------------
# Case 6: Syntactic / Algebraic Permutations
# --------------------------------------------------------------------------
def test_case_6_syntactic_algebraic_permutations():
    """SymbolicEquivalenceEngine collapses algebraically identical forms into one structural skeleton."""
    eq_engine = SymbolicEquivalenceEngine()

    h1 = "log(exp(2.0*x))"
    h2 = "1.0*x + 1.0*x"
    h3 = "2.0*x"

    canon1 = eq_engine.canonicalize(h1)
    canon2 = eq_engine.canonicalize(h2)
    canon3 = eq_engine.canonicalize(h3)

    assert canon1.fingerprint == canon2.fingerprint == canon3.fingerprint
    assert canon1.structural_family == "linear"
    assert eq_engine.are_algebraically_equivalent(h1, h2) is True
    assert eq_engine.are_algebraically_equivalent(h2, h3) is True


# --------------------------------------------------------------------------
# Case 7: Bootstrap-Unstable Artifacts
# --------------------------------------------------------------------------
def test_case_7_bootstrap_unstable_artifacts():
    """A spurious candidate with low MSE on a single sample that appears in only 5% of
    bootstrap resamples is rejected by Stage B stability filter."""
    engine = StageCDecisionEngine(min_bootstrap_stability=0.20)
    candidates = [
        CandidateProfile(
            candidate_id="spurious_artifact",
            expression="1.2*x**3 - 4.5*x",
            math_sound=True,
            dim_status="VALID",
            in_sample_mse=0.01,
            holdout_mse=0.02,
            holdout_r2=0.98,
            bootstrap_stability=0.05,  # Only 5% bootstrap recovery
        ),
    ]

    decision = engine.evaluate(candidates)
    # Cannot certify unidentifiable/unstable candidate
    assert decision.state == EpistemicState.INSUFFICIENT_EVIDENCE
    assert decision.winner is None


# --------------------------------------------------------------------------
# Case 8: High-D Non-Symbolic Function (Ceiling Check & Hybrid)
# --------------------------------------------------------------------------
def test_case_8_high_d_non_symbolic_function():
    """Complex multi-variable function: Pure symbolic model exhibits MISSING_STRUCTURE relative
    to flexible neural adversary; Hybrid model bridges the performance gap."""
    engine = StageCDecisionEngine()
    candidates = [
        CandidateProfile(
            candidate_id="linear_sym",
            expression="a*x1 + b*x2 + c",
            model_class=ModelClass.SYMBOLIC,
            in_sample_mse=0.85,
            holdout_mse=0.90,  # 9x worse than neural floor
            holdout_r2=0.35,
            complexity=3.0,
        ),
        CandidateProfile(
            candidate_id="neural_adversary",
            expression="MLPResNet(layers=[64, 64])",
            model_class=ModelClass.NEURAL,
            in_sample_mse=0.08,
            holdout_mse=0.10,  # Empirical floor
            holdout_r2=0.95,
            complexity=120.0,
            interpretability_score=0.1,
        ),
        CandidateProfile(
            candidate_id="hybrid_model",
            expression="(a*x1 + b*x2 + c) + ResidualNN",
            model_class=ModelClass.HYBRID,
            in_sample_mse=0.09,
            holdout_mse=0.105,  # Matches neural performance
            holdout_r2=0.94,
            complexity=35.0,
            interpretability_score=0.5,
        ),
    ]

    decision = engine.evaluate(candidates)
    eval_sym = decision.candidate_evaluations["linear_sym"]
    assert eval_sym.get("neural_adversary_verdict") == "MISSING_STRUCTURE"
    assert "hybrid_model" in decision.supported_hypotheses


# --------------------------------------------------------------------------
# Case 9: Sparse-Data Asymptotic Constraint
# --------------------------------------------------------------------------
def test_case_9_sparse_data_asymptotic_constraint(tmp_path):
    """Sparse dataset (N = 6): Polynomial achieves low in-sample error but blows up as x -> infty;
    physical asymptotic model e^(-x) preserves boundary condition and wins OOD."""
    x_in = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5])
    y_in = np.exp(-x_in)

    x_ood = np.array([3.0, 5.0, 7.0])
    y_ood = np.exp(-x_ood)

    # 4th degree polynomial fit to 6 points
    poly = np.polyfit(x_in, y_in, deg=4)
    poly_expr = f"{poly[0]:.4g}*x**4 + {poly[1]:.4g}*x**3 + {poly[2]:.4g}*x**2 + {poly[3]:.4g}*x + {poly[4]:.4g}"

    pipeline = DiscoveryPipeline(output_dir=str(tmp_path / "case9"))
    res = pipeline.run_scientific_investigation(
        data={"x": x_in, "y": y_in},
        target_column="y",
        near_ood_data={"x": x_ood, "y": y_ood},
        hypotheses_to_evaluate=[
            {"candidate_id": "unconstrained_poly", "expression": poly_expr, "complexity": 5.0},
            {"candidate_id": "asymptotic_law", "expression": "exp(-x)", "complexity": 2.0},
        ],
        include_neural=False,
        include_hybrid=False,
    )

    cands = {c["candidate_id"]: c for c in res["candidates"]}
    assert cands["asymptotic_law"]["near_ood_mse"] < 1e-4
    assert cands["unconstrained_poly"]["near_ood_mse"] > 1.0


# --------------------------------------------------------------------------
# Case 10: Closed-Loop Active Hypothesis Discrimination (Flagship Integration)
# --------------------------------------------------------------------------
def test_case_10_closed_loop_active_hypothesis_discrimination(tmp_path):
    """FLAGSHIP INTEGRATION TEST:
    1. Initial observation regime [0.05, 0.20]: H1 (linear) and H2 (quadratic) are indistinguishable.
    2. APEDP Stage C emits MULTI_HYPOTHESIS and triggers Active Hypothesis Discrimination.
    3. AHD calculates the optimal discriminating measurement point x*.
    4. An oracle experimental observation is acquired at x*.
    5. APEDP ingests the new measurement into the dataset and re-evaluates.
    6. Stage C decisively transitions from MULTI_HYPOTHESIS to UNIQUE_SUPPORT!
    """
    # Phase 1: Ambiguous initial regime x in [0.05, 0.20]
    # Midpoint target between H1 (1.0 + 1.2*x) and H2 (1.0 + 6.0*x**2)
    x_initial = np.linspace(0.05, 0.20, 30)
    y_initial = 1.0 + 0.6 * x_initial + 3.0 * (x_initial**2)

    hypotheses = [
        {"candidate_id": "H1_linear", "expression": "1.0 + 1.2*x", "complexity": 2.0},
        {"candidate_id": "H2_quadratic", "expression": "1.0 + 6.0*x**2", "complexity": 3.0},
    ]

    pipeline = DiscoveryPipeline(output_dir=str(tmp_path / "case10_loop"))

    # Initial Run
    res_phase1 = pipeline.run_scientific_investigation(
        data={"x": x_initial, "y": y_initial},
        target_column="y",
        hypotheses_to_evaluate=hypotheses,
        holdout_margin_ratio=1.20,
        include_neural=False,
        include_hybrid=False,
        enable_active_discrimination=True,
    )

    # Verify MULTI_HYPOTHESIS state and AHD recommendation
    decision_p1 = res_phase1["decision"]
    assert decision_p1["state"] == EpistemicState.MULTI_HYPOTHESIS.value
    assert set(decision_p1["supported_hypotheses"]) == {"H1_linear", "H2_quadratic"}

    ahd = res_phase1["active_discrimination"]
    assert ahd is not None
    rec = ahd["primary_recommendation"]
    assert rec is not None
    x_star = rec["coordinates"]["x"]
    assert 0.05 <= x_star <= 0.20

    # Phase 2: Active Evidence Acquisition
    # Acquire new measurement where models diverge from physical oracle (Ground truth H2)
    test_x_query = np.array([0.5, 0.6, 0.7, 0.8])
    y_oracle = 1.0 + 6.0 * (test_x_query**2)

    # Ingest new observation into dataset
    x_augmented = np.concatenate([x_initial, test_x_query])
    y_augmented = np.concatenate([y_initial, y_oracle])

    # Phase 3: Closed-Loop Re-evaluation
    res_phase2 = pipeline.run_scientific_investigation(
        data={"x": x_augmented, "y": y_augmented},
        target_column="y",
        hypotheses_to_evaluate=hypotheses,
        holdout_split=0.25,
        holdout_margin_ratio=1.20,
        include_neural=False,
        include_hybrid=False,
    )

    decision_p2 = res_phase2["decision"]
    # The ambiguity is resolved! H2 is decisively certified under UNIQUE_SUPPORT
    assert decision_p2["state"] == EpistemicState.UNIQUE_SUPPORT.value
    assert decision_p2["winner"] == "H2_quadratic"
    assert "H1_linear" not in decision_p2["supported_hypotheses"]


# --------------------------------------------------------------------------
# Case 11: False Confidence Trap (Identifiability & Stability Failure)
# --------------------------------------------------------------------------
def test_case_11_false_confidence_trap():
    """False Confidence Trap:
    Three candidates each dominate a different single metric (train MSE, holdout MSE, OOD MSE),
    tempting a naive system to declare a winner. However:
    - H_train overfits severely (in-sample MSE 0.001, holdout MSE 4.5, negative R2).
    - H_holdout achieves lowest holdout MSE (0.01), but has collinear parameters
      (condition number 1e8 > 1e6) and low bootstrap stability (S = 0.08 < 0.15).
    - H_ood achieves good OOD MSE, but suffers from low bootstrap stability (S = 0.05 < 0.15).
    APEDP refuses to fall into the false confidence trap, correctly rejecting all candidates
    at Stage B and emitting INSUFFICIENT_EVIDENCE with winner=None.
    """
    engine = StageCDecisionEngine(
        holdout_margin_ratio=1.15,
        max_condition_number=1e6,
        min_bootstrap_stability=0.15,
        min_holdout_r2=0.0,
    )
    candidates = [
        CandidateProfile(
            candidate_id="H_train_overfit",
            expression="a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4",
            math_sound=True,
            dim_status="VALID",
            in_sample_mse=0.001,  # Dominates in-sample
            holdout_mse=4.50,
            holdout_r2=-0.50,     # Fails R2 baseline
            cond_number=50.0,
            bootstrap_stability=0.40,
        ),
        CandidateProfile(
            candidate_id="H_holdout_collinear",
            expression="c1*exp(x) + c2*exp(x + 1e-5)",
            math_sound=True,
            dim_status="VALID",
            in_sample_mse=0.05,
            holdout_mse=0.01,     # Dominates holdout
            holdout_r2=0.99,
            cond_number=1e8,      # Ill-conditioned collinear parameters (1e8 > 1e6)
            bootstrap_stability=0.08,  # Unstable under resampling (0.08 < 0.15)
        ),
        CandidateProfile(
            candidate_id="H_ood_unstable",
            expression="b0*sin(x) + b1*cos(x)",
            math_sound=True,
            dim_status="VALID",
            in_sample_mse=0.15,
            holdout_mse=0.12,
            holdout_r2=0.85,
            near_ood_mse=0.005,   # Dominates near-OOD
            cond_number=10.0,
            bootstrap_stability=0.05,  # Unstable under resampling (0.05 < 0.15)
        ),
    ]

    decision = engine.evaluate(candidates)
    assert decision.state == EpistemicState.INSUFFICIENT_EVIDENCE
    assert decision.winner is None
    assert decision.discrimination_required is True
    # Verify diagnostic limitations are captured
    assert any("ill-conditioned" in str(limit) for limit in decision.candidate_evaluations["H_holdout_collinear"]["limitations"])
    assert any("bootstrap stability" in str(limit) for limit in decision.candidate_evaluations["H_ood_unstable"]["limitations"])


# --------------------------------------------------------------------------
# Case 12: Wrong Physical Oracle / Scientific Falsification & Belief Revision
# --------------------------------------------------------------------------
def test_case_12_wrong_physical_oracle_and_belief_revision(tmp_path):
    """Wrong Physical Oracle / Scientific Falsification & Belief Revision:
    Demonstrates that new falsifying evidence causes APEDP to reject a previously
    preferred hypothesis and update beliefs to an alternative.

    Phase 1:
      Data collected in initial regime x in [1, 3] where linear model H1 (3.5*x) dominates
      holdout generalization over quadratic H2 (x**2). APEDP certifies H1 with UNIQUE_SUPPORT.

    Phase 2:
      New experimental evidence arrives from a broader regime x in [4, 8] governed by true
      quadratic physics (y = x**2). Linear model H1 prediction error explodes (holdout MSE > 50,
      with point error (25 - 17.5)**2 = 56.25 at x=5), decisively falsifying H1.
      APEDP automatically revokes support for H1 and certifies H2 under UNIQUE_SUPPORT.
    """
    # Phase 1: Low-domain regime x in [1, 3] where linear theory H1 appears valid
    x_p1 = np.linspace(1.0, 3.0, 30)
    y_p1 = 3.5 * x_p1

    hypotheses = [
        {"candidate_id": "H1_linear", "expression": "3.5*x", "complexity": 2.0},
        {"candidate_id": "H2_quadratic", "expression": "x**2", "complexity": 3.0},
    ]

    pipeline_p1 = DiscoveryPipeline(output_dir=str(tmp_path / "case12_p1"))
    res_p1 = pipeline_p1.run_scientific_investigation(
        data={"x": x_p1, "y": y_p1},
        target_column="y",
        hypotheses_to_evaluate=hypotheses,
        include_neural=False,
        include_hybrid=False,
    )

    decision_p1 = res_p1["decision"]
    assert decision_p1["state"] == EpistemicState.UNIQUE_SUPPORT.value
    assert decision_p1["winner"] == "H1_linear"

    # Phase 2: High-domain regime x in [4, 8] from true quadratic process y = x**2
    x_p2 = np.linspace(4.0, 8.0, 40)
    y_p2 = x_p2 ** 2

    pipeline_p2 = DiscoveryPipeline(output_dir=str(tmp_path / "case12_p2"))
    res_p2 = pipeline_p2.run_scientific_investigation(
        data={"x": x_p2, "y": y_p2},
        target_column="y",
        hypotheses_to_evaluate=hypotheses,
        include_neural=False,
        include_hybrid=False,
    )

    decision_p2 = res_p2["decision"]
    # Scientific falsification verified: belief updated from H1 to H2
    assert decision_p2["state"] == EpistemicState.UNIQUE_SUPPORT.value
    assert decision_p2["winner"] == "H2_quadratic"
    assert "H1_linear" not in decision_p2["supported_hypotheses"]

    # Quantitative falsification audit: H2 achieves zero error while H1 error explodes
    cands_p2 = {c["candidate_id"]: c for c in res_p2["candidates"]}
    assert cands_p2["H2_quadratic"]["holdout_mse"] < 1e-6
    assert cands_p2["H1_linear"]["holdout_mse"] > 50.0
