"""Tests for Stage C Epistemic Decision Engine.

Verifies the 4-state epistemic machine:
1. NO_VALID_CANDIDATE: All candidates rejected or held at Stage A.
2. INSUFFICIENT_EVIDENCE: Candidates admissible but empirical support inadequate.
3. UNIQUE_SUPPORT: Single decisive candidate without competitive alternatives.
4. MULTI_HYPOTHESIS: Multiple competitive candidates with trade-offs. The engine
   refuses to pick a single winner.
"""
from __future__ import annotations

from apedp.decision import (
    CandidateProfile,
    EpistemicState,
    StageCDecisionEngine,
)


def test_stage_c_no_valid_candidate_when_all_rejected_or_unresolved():
    engine = StageCDecisionEngine()
    candidates = [
        CandidateProfile(
            candidate_id="cos_nest",
            expression="cos(a*T + b)",
            math_sound=True,
            dim_status="UNRESOLVED",  # P2 HOLD
        ),
        CandidateProfile(
            candidate_id="incompatible_add",
            expression="T + dx",
            math_sound=True,
            dim_status="INVALID",  # P2 REJECT
        ),
        CandidateProfile(
            candidate_id="singular_pole",
            expression="1 / (T - 300)",
            math_sound=False,  # Unsound math
            dim_status="VALID",
        ),
    ]

    decision = engine.evaluate(candidates)
    assert decision.state == EpistemicState.NO_VALID_CANDIDATE
    assert decision.winner is None
    assert len(decision.supported_hypotheses) == 0
    assert "Stage A gates" in decision.adjudication_rationale
    assert decision.discrimination_required is False


def test_stage_c_insufficient_evidence_when_holdout_missing_or_unidentifiable():
    engine = StageCDecisionEngine(max_condition_number=1e5, min_bootstrap_stability=0.2)
    candidates = [
        CandidateProfile(
            candidate_id="h1_unidentified",
            expression="a*T**3 + b*T**2 + c/T**2",
            math_sound=True,
            dim_status="VALID",
            in_sample_mse=0.01,
            holdout_mse=float("inf"),  # Missing holdout
            holdout_r2=-1.0,
            cond_number=1e9,  # Severe collinearity
        ),
        CandidateProfile(
            candidate_id="h2_unstable",
            expression="a*T + b",
            math_sound=True,
            dim_status="VALID",
            in_sample_mse=0.5,
            holdout_mse=0.6,
            holdout_r2=0.5,
            bootstrap_stability=0.05,  # Unstable under resamples
        ),
    ]

    decision = engine.evaluate(candidates)
    assert decision.state == EpistemicState.INSUFFICIENT_EVIDENCE
    assert decision.winner is None
    assert decision.discrimination_required is True
    assert len(decision.missing_evidence) > 0


def test_stage_c_unique_support_when_one_candidate_decisively_dominates():
    engine = StageCDecisionEngine(holdout_margin_ratio=1.15)
    candidates = [
        CandidateProfile(
            candidate_id="clear_winner",
            expression="a*T + b",
            math_sound=True,
            dim_status="VALID",
            in_sample_mse=0.10,
            holdout_mse=0.11,
            holdout_r2=0.98,
            complexity=2.0,
            cond_number=10.0,
            bootstrap_stability=0.95,
        ),
        CandidateProfile(
            candidate_id="inferior_model",
            expression="a*T**2 + b*T + c",
            math_sound=True,
            dim_status="VALID",
            in_sample_mse=0.25,
            holdout_mse=0.30,  # > 15% worse
            holdout_r2=0.90,
            complexity=3.0,
            cond_number=50.0,
            bootstrap_stability=0.80,
        ),
    ]

    decision = engine.evaluate(candidates)
    assert decision.state == EpistemicState.UNIQUE_SUPPORT
    assert decision.winner == "clear_winner"
    assert decision.supported_hypotheses == ["clear_winner"]
    assert decision.discrimination_required is False
    assert "uniquely supported" in decision.adjudication_rationale


def test_stage_c_multi_hypothesis_preserves_ambiguity_on_tradeoffs():
    """Validates the core methane / ethanol review case:
    Shomate has best training fit, but Quadratic has competitive holdout and lower complexity.
    Stage C must NOT pick a winner; it must emit MULTI_HYPOTHESIS and request discrimination."""
    engine = StageCDecisionEngine(holdout_margin_ratio=1.15)
    candidates = [
        CandidateProfile(
            candidate_id="quadratic",
            expression="a + b*T + c*T**2",
            math_sound=True,
            dim_status="VALID",
            in_sample_mse=0.81,
            holdout_mse=0.82,  # Best holdout
            holdout_r2=0.95,
            near_ood_mse=1.10,
            complexity=3.0,
            cond_number=250.0,
            bootstrap_stability=0.88,
            aic=-8.3,
            bic=-4.3,  # BIC prefers quadratic
        ),
        CandidateProfile(
            candidate_id="shomate",
            expression="A + B*T + C*T**2 + D*T**3 + E/T**2",
            math_sound=True,
            dim_status="VALID",
            in_sample_mse=0.62,  # Best training fit
            holdout_mse=0.89,  # Within 8.5% of quadratic (< 15% margin)
            holdout_r2=0.94,
            near_ood_mse=1.45,
            complexity=7.0,
            cond_number=4500.0,
            bootstrap_stability=0.75,
            aic=-8.6,  # AIC marginally prefers shomate
            bic=-1.2,
        ),
        CandidateProfile(
            candidate_id="cos_nest",
            expression="cos(a*T + b)",
            math_sound=True,
            dim_status="UNRESOLVED",  # Correctly held out
        ),
    ]

    decision = engine.evaluate(candidates)

    # Invariant: Must emit MULTI_HYPOTHESIS, not pick a winner!
    assert decision.state == EpistemicState.MULTI_HYPOTHESIS
    assert decision.winner is None
    assert set(decision.supported_hypotheses) == {"quadratic", "shomate"}
    assert "cos_nest" not in decision.supported_hypotheses
    assert decision.discrimination_required is True
    assert "Observed trade-offs" in decision.adjudication_rationale
    assert "Forcing a winner would encode unevidenced bias" in decision.adjudication_rationale


def test_stage_c_tri_hypothesis_neural_adversary_flags_missing_structure():
    """Validates Neural Network as an adversarial ceiling challenger:
    If a flexible neural network achieves much lower holdout error than the symbolic model,
    Stage C must flag MISSING_STRUCTURE."""
    from apedp.decision import ModelClass

    engine = StageCDecisionEngine()
    candidates = [
        CandidateProfile(
            candidate_id="linear_sym",
            expression="a*x + b",
            model_class=ModelClass.SYMBOLIC,
            in_sample_mse=0.55,
            holdout_mse=0.58,  # > 5x worse than NN
            holdout_r2=0.45,
            complexity=2.0,
        ),
        CandidateProfile(
            candidate_id="mlp_resnet_nn",
            expression="MLPResNet(layers=[64, 64], act=relu)",
            model_class=ModelClass.NEURAL,
            in_sample_mse=0.08,
            holdout_mse=0.10,  # Empirical floor
            holdout_r2=0.95,
            complexity=150.0,
            interpretability_score=0.1,
        ),
    ]

    decision = engine.evaluate(candidates)
    eval_sym = decision.candidate_evaluations["linear_sym"]
    assert eval_sym.get("neural_adversary_verdict") == "MISSING_STRUCTURE"
    assert eval_sym.get("gap_to_neural_ceiling") >= 5.0
    assert "missing structure" in decision.adjudication_rationale


def test_stage_c_tri_hypothesis_symbolic_matches_neural_ceiling():
    """When a simple symbolic formula matches the predictive performance of a flexible neural net,
    Stage C notes MATCHES_NEURAL_CEILING."""
    from apedp.decision import ModelClass

    engine = StageCDecisionEngine(holdout_margin_ratio=1.15)
    candidates = [
        CandidateProfile(
            candidate_id="true_quadratic_sym",
            expression="2.0*x**2 + 1.0",
            model_class=ModelClass.SYMBOLIC,
            in_sample_mse=0.102,
            holdout_mse=0.104,
            holdout_r2=0.97,
            complexity=3.0,
            interpretability_score=1.0,
        ),
        CandidateProfile(
            candidate_id="mlp_nn",
            expression="MLP(hidden=[64, 32])",
            model_class=ModelClass.NEURAL,
            in_sample_mse=0.098,
            holdout_mse=0.100,  # Only 4% lower
            holdout_r2=0.97,
            complexity=100.0,
            interpretability_score=0.1,
        ),
    ]

    decision = engine.evaluate(candidates)
    eval_sym = decision.candidate_evaluations["true_quadratic_sym"]
    assert eval_sym.get("neural_adversary_verdict") == "MATCHES_NEURAL_CEILING"
    # Both are within 15% margin
    assert decision.state == EpistemicState.MULTI_HYPOTHESIS
    assert "Cross-class competition" in decision.adjudication_rationale
    assert "Occam's razor strongly favors symbolic parsimony" in decision.adjudication_rationale


def test_stage_c_paired_se_marks_otherwise_outside_margin_competitive():
    """Paired-SE indistinguishability promotes a candidate outside the 15% margin.

    Exercises the holdout_residuals metadata channel wired through synthesis.
    """
    engine = StageCDecisionEngine()
    resid = [0.010, -0.012, 0.008, -0.009, 0.011, -0.010, 0.009, -0.011, 0.010, -0.009]
    best = CandidateProfile(
        candidate_id="best",
        expression="a*x**2",
        math_sound=True,
        dim_status="VALID",
        in_sample_mse=0.10,
        holdout_mse=0.10,
        holdout_r2=0.99,
        complexity=3.0,
        cond_number=10.0,
        bootstrap_stability=0.9,
        metadata={"holdout_residuals": list(resid)},
    )
    rival = CandidateProfile(
        candidate_id="rival",
        expression="a*x**2 + b",
        math_sound=True,
        dim_status="VALID",
        in_sample_mse=0.10,
        holdout_mse=0.13,  # 30% worse: outside the 15% margin
        holdout_r2=0.985,
        complexity=4.0,
        cond_number=12.0,
        bootstrap_stability=0.9,
        metadata={"holdout_residuals": list(resid)},  # identical residuals -> SE-equivalent
    )
    decision = engine.evaluate([best, rival])
    assert decision.state == EpistemicState.MULTI_HYPOTHESIS
    assert set(decision.supported_hypotheses) == {"best", "rival"}


def test_stage_c_pole_inside_observed_data_refused():
    """A pole strictly inside observed data (outside gaps) is refused at Stage A."""
    engine = StageCDecisionEngine()
    pole = CandidateProfile(
        candidate_id="pole_model",
        expression="1.65*x + 0.12/Abs(x - 0.6)**0.25",
        math_sound=True,
        dim_status="VALID",
        in_sample_mse=0.003,
        holdout_mse=0.003,
        holdout_r2=0.99,
        complexity=5.0,
        cond_number=10.0,
        bootstrap_stability=0.9,
        metadata={"x_obs_min": 0.0, "x_obs_max": 1.0, "x_gaps": []},
    )
    linear = CandidateProfile(
        candidate_id="linear_model",
        expression="1.65*x + 0.12",
        math_sound=True,
        dim_status="VALID",
        in_sample_mse=0.004,
        holdout_mse=0.004,
        holdout_r2=0.985,
        complexity=2.0,
        cond_number=10.0,
        bootstrap_stability=0.9,
        metadata={"x_obs_min": 0.0, "x_obs_max": 1.0, "x_gaps": []},
    )
    decision = engine.evaluate([pole, linear])
    assert "pole_model" not in decision.supported_hypotheses
    assert "linear_model" in decision.supported_hypotheses


def test_stage_c_gap_pole_exempt_from_refusal():
    """A pole inside a known sampling gap (physical critical point) is exempt."""
    engine = StageCDecisionEngine()
    crit = CandidateProfile(
        candidate_id="crit_model",
        expression="2.0*Abs(x - 5.0)**(-0.5)",
        math_sound=True,
        dim_status="VALID",
        in_sample_mse=0.001,
        holdout_mse=0.001,
        holdout_r2=0.999,
        complexity=3.0,
        cond_number=10.0,
        bootstrap_stability=0.9,
        metadata={"x_obs_min": 0.0, "x_obs_max": 10.0, "x_gaps": [[4.0, 6.0]]},
    )
    other = CandidateProfile(
        candidate_id="other_model",
        expression="0.5*x + 1.0",
        math_sound=True,
        dim_status="VALID",
        in_sample_mse=0.05,
        holdout_mse=0.05,
        holdout_r2=0.90,
        complexity=2.0,
        cond_number=10.0,
        bootstrap_stability=0.9,
        metadata={"x_obs_min": 0.0, "x_obs_max": 10.0, "x_gaps": [[4.0, 6.0]]},
    )
    decision = engine.evaluate([crit, other])
    assert decision.state == EpistemicState.UNIQUE_SUPPORT
    assert decision.winner == "crit_model"
