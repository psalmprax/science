"""Tests for Adversarial Science Benchmark Suite & Novelty Prior-Art Audit."""
from __future__ import annotations

from apedp.adversarial_benchmarks import (
    AdversarialScienceBenchmarkSuite,
    AdversarialTrapResult,
    AdversarialTrapType,
)
from apedp.epistemic import (
    NoveltyAuditResult,
    NoveltyStatus,
    PriorArtAuditor,
    ScientificMaturityLevel,
    TestableModelPackage,
    VerificationGate,
)
from apedp.gated_assurance import EpistemicActionStance


def test_novelty_prior_art_auditor_langmuir_rediscovery():
    """Validates that a synthesized Langmuir/Michaelis-Menten law is recognized as re-discovered prior art."""
    # Synthesized from observations without formula name: v = 15*S / (2.5 + S)
    result = PriorArtAuditor.audit_novelty("15.0 * S / (2.5 + S)", variable_name="S")
    assert isinstance(result, NoveltyAuditResult)
    assert result.status == NoveltyStatus.KNOWN_MODEL
    assert result.closest_reference_id == "LANGMUIR_MICHAELIS_MENTEN"
    assert "novelty claim is refused" in result.epistemic_disclaimer.lower()

    # Serialization
    d = result.to_dict()
    assert d["status"] == "KNOWN_MODEL"
    assert d["searched_corpus_size"] >= 10


def test_novelty_prior_art_auditor_related_structural_families():
    """Validates that offset rationals and Hill equations are categorized as related structures."""
    # Offset rational: y = (2 + 5*x) / (1 + 3*x) -> non-zero intercept
    offset_result = PriorArtAuditor.audit_novelty("(2 + 5*x) / (1 + 3*x)", variable_name="x")
    assert offset_result.status == NoveltyStatus.RELATED_STRUCTURE
    assert "OFFSET_RATIONAL" in offset_result.isomorphism_type

    # Hill cooperative binding: y = 10*x^2 / (1 + 4*x^2)
    hill_result = PriorArtAuditor.audit_novelty("10 * x**2 / (1 + 4 * x**2)", variable_name="x")
    assert hill_result.status == NoveltyStatus.RELATED_STRUCTURE
    assert hill_result.closest_reference_id == "HILL_EQUATION"


def test_novelty_prior_art_auditor_apparently_novel_within_corpus():
    """Validates that genuinely unseen functional forms receive explicit corpus disclaimer."""
    # Candidate: y = x * sin(log(1 + x^2)) -> not in standard corpus
    novel_result = PriorArtAuditor.audit_novelty("x * sin(log(1 + x**2))", variable_name="x")
    assert novel_result.status == NoveltyStatus.APPARENTLY_NOVEL_WITHIN_CORPUS
    assert novel_result.closest_reference_id is None
    # Crucial epistemic check: NEVER claims "new scientific law"
    assert "new physical law" not in novel_result.status.value.lower()
    assert "apparently novel within the searched reference corpus" in novel_result.epistemic_disclaimer.lower()


def test_testable_model_package_canonical_tuple_with_novelty_and_maturity():
    """Validates TestableModelPackage canonical tuple T = (M, P, D, C, L, H, F, E, G) with maturity & novelty."""
    novelty_audit = PriorArtAuditor.audit_novelty("15.0 * x / (2.5 + x)", variable_name="x")

    package = TestableModelPackage(
        model_id="APEDP-BIO-001",
        equation_str="v(S) = Vmax * S / (Km + S)",
        parameters={"Vmax": 15.0, "Km": 2.5},
        domain={"substrate_mM": (0.0, 100.0)},
        constraints=("Strict positivity v >= 0", "Monotonically increasing dv/dS >= 0", "Asymptotic saturation v <= Vmax"),
        limiting_behavior={"S_to_zero": "Vmax * S / Km", "S_to_inf": "Vmax"},
        recovered_limits={"low_substrate": "First-order linear kinetics"},
        predictions=("Initial reaction velocity", "Half-saturation concentration"),
        required_data={"observable": "velocity_uM_s", "min_points": 20},
        holdout_test={"partition": "high_concentration_holdout", "hash": "sha256:abc1234"},
        falsification_criteria=("dv/dS < 0", "v > 1.2 * Vmax"),
        competing_models=("v(S) = Vmax * S^2 / (Km^2 + S^2)",),
        maximal_discrimination_point={"substrate_mM": 1.25},
        novelty_audit=novelty_audit,
        maturity_level=ScientificMaturityLevel.LEVEL_4_TESTABLE_MODEL_SYNTHESIS,
    )

    data = package.to_dict()
    assert data["maturity_level"] == "LEVEL_4_TESTABLE_MODEL_SYNTHESIS"
    assert data["novelty_audit"]["status"] == "KNOWN_MODEL"
    assert VerificationGate.NOVELTY_AUDIT.value == "NOVELTY_AUDIT"


def test_adversarial_suite_trap_a_dimensional_decoy():
    """Validates Trap A: dimensional decoy E_k = 0.5*m*v^3/c is caught and refused.

    The defence must come from the real positivity invariant, and the decoy must
    genuinely pass the dimensional gate (that is what makes it a decoy).
    """
    suite = AdversarialScienceBenchmarkSuite()
    result = suite.audit_trap_a_dimensional_decoy()
    assert isinstance(result, AdversarialTrapResult)
    assert result.trap_type == AdversarialTrapType.DIMENSIONAL_DECOY
    assert result.passed_audit is True
    assert result.epistemic_stance == EpistemicActionStance.REFUSE
    assert "PositivityViolationProbe" in result.defense_mechanism

    # Evidence produced by the real components, not by the benchmark fixture.
    details = result.details
    assert details["dimensionally_homogeneous"] is True
    assert details["dimensional_counterexamples"] == []
    assert details["positivity_violation_detected"] is True
    assert details["negative_energy_value"] < 0.0
    assert details["system_verdict"] == "REFUSED_CANNOT_REPAIR"
    # The falsification report is embedded verbatim for audit
    assert details["falsification_report"]["survived"] is False


def test_adversarial_suite_trap_b_overfitting_trap():
    """Validates Trap B: degree-9 polynomial with R^2 > 0.999 is rejected on holdout."""
    suite = AdversarialScienceBenchmarkSuite()
    result = suite.audit_trap_b_overfitting_trap()
    assert result.trap_type == AdversarialTrapType.OVERFITTING_TRAP
    assert result.passed_audit is True
    assert result.epistemic_stance == EpistemicActionStance.REFUSE
    assert result.details["r2_train"] > 0.99
    assert result.details["mse_holdout"] > 10.0

    # The rejection must be driven by the real empirical axis on the holdout,
    # not by an in-sample score.
    details = result.details
    assert details["empirical_counterexamples"] == ["EmpiricalResidualProbe"]
    assert details["holdout_r_squared"] < 0.0
    # Fits the interpolation points but fails the extrapolation points, so the
    # locked holdout is not fully covered.
    assert details["holdout_coverage_within_5pct"] < 0.9
    assert details["n_free_parameters"] >= result.details["n_train_points"]
    assert details["improvement_metric"]["is_net_positive"] is False
    assert details["system_verdict"] == "FALSIFIED"


def test_adversarial_suite_trap_c_extrapolation_trap():
    """Validates Trap C: pole singularity in operational range leads to ABSTAIN."""
    suite = AdversarialScienceBenchmarkSuite()
    result = suite.audit_trap_c_extrapolation_trap()
    assert result.trap_type == AdversarialTrapType.EXTRAPOLATION_TRAP
    assert result.passed_audit is True
    assert result.epistemic_stance == EpistemicActionStance.ABSTAIN
    assert 10.0 in result.details["poles_detected"]
    # The pole must be located by the real denominator-root probe inside the
    # claimed envelope.
    assert result.details["pole_in_claimed_envelope"] is True
    assert result.details["system_verdict"] == "FALSIFIED"


def test_adversarial_suite_trap_d_confounded_mechanisms():
    """Validates Trap D: indistinguishable mechanisms return AGNOSTIC + x* discrimination coordinate."""
    suite = AdversarialScienceBenchmarkSuite()
    result = suite.audit_trap_d_confounded_mechanisms()
    assert result.trap_type == AdversarialTrapType.CONFOUNDED_MECHANISMS
    assert result.passed_audit is True
    assert result.epistemic_stance == EpistemicActionStance.AGNOSTIC
    assert result.details["optimal_discrimination_t_star"] > 5.0
    assert result.details["max_snr_at_t_star"] > 2.0


def test_adversarial_suite_trap_e_synthetic_novelty_trap():
    """Validates Trap E: re-discovered known law is audited and false novelty is refused."""
    suite = AdversarialScienceBenchmarkSuite()
    result = suite.audit_trap_e_synthetic_novelty_trap()
    assert result.trap_type == AdversarialTrapType.SYNTHETIC_NOVELTY_TRAP
    assert result.passed_audit is True
    assert result.details["status"] == "KNOWN_MODEL"
    assert "refused" in result.details["epistemic_disclaimer"].lower()


def test_adversarial_suite_trap_f_hidden_contradiction():
    """Validates Trap F: unsatisfiable constraint pairs are refused with a certificate."""
    suite = AdversarialScienceBenchmarkSuite()
    result = suite.audit_trap_f_hidden_contradiction()
    assert result.trap_type == AdversarialTrapType.HIDDEN_CONTRADICTION
    assert result.passed_audit is True
    assert result.epistemic_stance == EpistemicActionStance.REFUSE
    assert result.details["is_mathematically_impossible"] is True
    # The refusal must come from the real conflict detector, with a certificate.
    assert result.details["system_refused"] is True
    assert result.details["system_verdict"] == "REFUSED_INCOMPATIBLE_CONSTRAINTS"
    certificate = result.details["infeasibility_certificate"]
    assert certificate is not None and certificate["is_infeasible"] is True


def test_adversarial_suite_end_to_end_refusal_to_overclaim_score():
    """Validates that the entire 6-trap adversarial benchmark achieves Refusal-to-Overclaim Score of 100%."""
    suite = AdversarialScienceBenchmarkSuite()
    summary = suite.run_all_traps()

    assert summary["total_traps"] == 6
    assert summary["passed_traps"] == 6
    assert summary["failed_traps"] == 0
    assert summary["failed_trap_ids"] == []
    assert summary["refusal_to_overclaim_score"] == 1.0
    assert summary["all_traps_defended"] is True

    # Every trap must name the component that produced its verdict, so the score
    # is auditable rather than self-declared.
    for trap in summary["results"]:
        assert trap["passed_audit"] is True
        assert trap["defense_mechanism"].strip()

    # The traps routed through the IndependentAdversarialFalsifier must carry its
    # report, and it must have rejected the candidate.
    falsifier_traps = [
        t for t in summary["results"]
        if "IndependentAdversarialFalsifier" in t["defense_mechanism"]
        or "ContradictoryConstraintEngine" in t["defense_mechanism"]
    ]
    assert len(falsifier_traps) == 4
    for trap in falsifier_traps:
        assert trap["details"]["system_verdict"] in (
            "FALSIFIED", "REFUSED_CANNOT_REPAIR", "REFUSED_INCOMPATIBLE_CONSTRAINTS"
        )
        assert trap["details"]["falsification_report"]["survived"] is False
