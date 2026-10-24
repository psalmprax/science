# Databricks notebook source
"""Tests for Independent Adversarial Falsifier and Pathological Equation Benchmark Suite."""
import numpy as np
import pytest

from apedp.benchmarks import (
    BenchmarkCategory,
    BenchmarkHarness,
    BenchmarkSummary,
    NoveltyTier,
    PathologicalEquationBenchmark,
    UnmeasuredMetricError,
)
from apedp.falsification import (
    UNREPAIRABLE_FAILURE_CLASSES,
    BlindAdversarialAuditor,
    ContradictoryConstraintEngine,
    Counterexample,
    FailureClass,
    FalsificationAxis,
    IndependentAdversarialFalsifier,
    unrepairable_counterexamples,
)


@pytest.fixture
def falsifier():
    return IndependentAdversarialFalsifier()


def test_falsifier_mathematical_pole_detection(falsifier):
    """Axis 1: Mathematical counterexample detection on denominator roots."""
    # Classical Prandtl-Glauert: 1 / sqrt(1 - M^2) has pole at M=1
    rep = falsifier.falsify(
        formula="-0.5 / sqrt(1 - M**2)",
        variable_domains={"M": (0.0, 1.05)},
        critical_boundaries={"M": [1.0]}
    )
    assert rep.survived is False
    assert rep.fatal_count > 0
    math_or_bound = [c for c in rep.counterexamples if c.axis in (FalsificationAxis.MATHEMATICAL, FalsificationAxis.BOUNDARY)]
    assert len(math_or_bound) > 0


def test_falsifier_complex_branch_cut_detection(falsifier):
    """Axis 1: Complex branch cut detection on real domain."""
    # sqrt(x) on negative domain
    rep = falsifier.falsify(
        formula="sqrt(x)",
        variable_domains={"x": (-5.0, -0.5)}
    )
    assert rep.survived is False
    assert any(c.probe_name == "ComplexBranchCutProbe" for c in rep.counterexamples)


def test_falsifier_boundary_asymptotic_divergence(falsifier):
    """Axis 2: Boundary divergence detection as variable approaches zero."""
    # 1 / r^2 gravitational force singularity at r=0
    rep = falsifier.falsify(
        formula="1.0 / (r**2)",
        variable_domains={"r": (1e-6, 10.0)},
        critical_boundaries={"r": [0.0]}
    )
    assert rep.survived is False
    assert any(c.axis == FalsificationAxis.BOUNDARY for c in rep.counterexamples)


def test_falsifier_numerical_overflow_and_nan(falsifier):
    """Axis 3: Numerical overflow scanning on wide dynamic ranges."""
    # Exponential blowup exp(1000 * x) on [1, 10]
    rep = falsifier.falsify(
        formula="exp(1000.0 * x)",
        variable_domains={"x": (1.0, 10.0)}
    )
    assert rep.survived is False
    assert any(c.axis == FalsificationAxis.NUMERICAL for c in rep.counterexamples)


def test_falsifier_structural_invariant_positivity(falsifier):
    """Axis 4: Physical invariant violation - negative absolute temperature."""
    # T < 0 violates 3rd law
    rep = falsifier.falsify(
        formula="8.314 * T / 0.0224",
        variable_domains={"T": (-50.0, -1.0)},
        physical_invariants=["third_law_thermodynamics", "positivity"]
    )
    assert rep.survived is False
    assert rep.cannot_repair is True
    assert rep.verdict == "REFUSED_CANNOT_REPAIR"
    assert "3RD_LAW_THERMODYNAMICS" in rep.refusal_reason or "UNPHYSICAL_NEGATIVE" in rep.refusal_reason


def test_falsifier_superluminal_lorentz_acausality_refusal(falsifier):
    """Axis 4: Superluminal velocity acausality violation."""
    rep = falsifier.falsify(
        formula="9.1e-31 / sqrt(1 - (v / 3e8)**2)",
        variable_domains={"v": (3.5e8, 1e9)},
        physical_invariants=["lorentz_causality"]
    )
    assert rep.survived is False
    assert rep.cannot_repair is True
    assert rep.verdict == "REFUSED_CANNOT_REPAIR"
    assert "SUPERLUMINAL_LORENTZ_VIOLATION" in rep.refusal_reason


def test_falsifier_negative_entropy_production_refusal(falsifier):
    """Axis 4: Second Law of Thermodynamics violation."""
    rep = falsifier.falsify(
        formula="-5.0 * sigma_entropy",
        variable_domains={"sigma_entropy": (1.0, 10.0)},
        physical_invariants=["second_law_thermodynamics"]
    )
    assert rep.survived is False
    assert rep.cannot_repair is True
    assert rep.verdict == "REFUSED_CANNOT_REPAIR"
    assert "2ND_LAW_VIOLATION" in rep.refusal_reason


def test_falsifier_empirical_fidelity_check(falsifier):
    """Axis 5: Empirical observation rejection when residual error is excessive."""
    x = np.linspace(1.0, 10.0, 20)
    y_true = 2.0 * x + 1.0  # Linear law
    # Candidate equation y = x^3 (awful fit)
    rep = falsifier.falsify(
        formula="x**3",
        variable_domains={"x": (1.0, 10.0)},
        empirical_data={"x": x, "y": y_true}
    )
    assert rep.survived is False
    assert any(c.axis == FalsificationAxis.EMPIRICAL for c in rep.counterexamples)


def test_falsifier_regularized_model_survival(falsifier):
    """Verifies that an appropriately regularized Van Dyke formula survives the falsifier."""
    eps = 0.04
    reg_formula = f"-0.5 / (((1 - M**2)**2 + {eps}**2)**0.25)"
    rep = falsifier.falsify(
        formula=reg_formula,
        variable_domains={"M": (0.0, 1.2)},
        critical_boundaries={"M": [1.0]}
    )
    assert rep.survived is True
    assert rep.verdict == "SURVIVED_PROVISIONAL"
    assert "No counterexample found" in rep.epistemic_qualification
    assert len(rep.counterexamples) == 0


def test_pathological_benchmark_suite_execution():
    """Executes the full 25-case Pathological Equation Benchmark Suite and asserts trust metrics."""
    cases = PathologicalEquationBenchmark.get_benchmark_cases()
    assert len(cases) == 25

    solvable = [c for c in cases if c.expected_repairable]
    patho = [c for c in cases if not c.expected_repairable]
    assert len(solvable) == 13
    assert len(patho) == 12

    harness = BenchmarkHarness()
    summary = harness.run_benchmark()

    # 1. Defect Detection: Every single singular or pathological equation must be flagged
    assert summary.defect_detection_rate == 1.0, f"Expected 100% defect detection, got {summary.defect_detection_rate:.1%}"

    # 2. Pathological Refusal Accuracy: Every single unrepairable equation must be REFUSED
    # The pathology must be *found*, which is the property a label-free probe can
    # actually measure. The outright-refusal rate is reported separately and is
    # deliberately NOT asserted at 100%: it used to read 1.0 only because the
    # harness passed `allow_repair_attempt=case.expected_repairable`, i.e. it told
    # the falsifier not to repair exactly the cases it then graded on refusal.
    # Measured without that hint the rate is ~0.58, and the remaining cases are
    # detected (and repaired/refused on their merits) rather than forced.
    assert summary.pathological_detection_rate == 1.0, (
        f"Pathological detection {summary.pathological_detection_rate:.1%} < 100%"
    )
    assert summary.pathological_refusal_accuracy < 1.0, (
        "Refusal rate cannot legitimately be 100% with a label-free probe; if this "
        "fails, the case label has been reintroduced into the falsifier call."
    )

    # 3. False-Repair Rate: Must be strictly 0.0% (never falsely claim an unrepairable law was repaired)
    assert summary.false_repair_rate == 0.0, f"Expected 0.0% false repairs, got {summary.false_repair_rate:.1%}"

    # 4. Solvable Success Rate: Solvable candidates must successfully repair
    assert summary.solvable_repair_success_rate >= 0.90, f"Solvable success rate {summary.solvable_repair_success_rate:.1%} < 90%"

    # 5. Overall System Trust Score
    assert summary.system_trust_score >= 90.0, f"System trust score {summary.system_trust_score:.1f} < 90.0"


def test_benchmark_trust_metrics_have_no_fabricated_defaults():
    """An unmeasured suite or subgroup must not report a perfect trust score."""
    empty_summary = BenchmarkSummary(
        total_cases=0,
        solvable_cases=0,
        pathological_cases=0,
        defect_detection_rate=0.0,
        correct_diagnosis_rate=0.0,
        solvable_repair_success_rate=0.0,
        pathological_refusal_accuracy=0.0,
        pathological_detection_rate=0.0,
        false_repair_rate=0.0,
        system_trust_score=0.0,
    )
    # Defaults must not be the optimistic values they used to be: an unpopulated
    # summary reports no conformance figure at all, rather than 0.0 or 1.0.
    assert empty_summary.benchmark_conformance_score is None
    # An unmeasurable metric is None, not nan: ``nan >= x`` is silently False and
    # ``nan * w`` silently poisons the aggregate, whereas any arithmetic or
    # ordering comparison against None raises immediately.
    assert empty_summary.infeasibility_refusal_accuracy is None
    with pytest.raises(TypeError):
        _ = empty_summary.infeasibility_refusal_accuracy >= 0.9

    # Running the harness over an empty suite yields no trust, not defaulted trust.
    summary = BenchmarkHarness().run_benchmark(cases=[])
    assert summary.total_cases == 0
    assert summary.system_trust_score is None
    assert summary.benchmark_conformance_score is None
    assert summary.infeasibility_refusal_accuracy is None

    # A populated suite scores from its measured rates only. The 25-case suite
    # contains no deliberately-incompatible-constraint cases, so that subgroup is
    # unmeasured and must report nan rather than a fabricated 1.0; the 100-case
    # suite covers it and is asserted in its own test.
    populated = BenchmarkHarness().run_benchmark()
    assert populated.total_cases > 0
    assert populated.system_trust_score > 0.0
    assert populated.infeasibility_refusal_accuracy is None
    assert populated.tier_metrics and populated.category_metrics

    # A suite that cannot measure one of the trust factors must refuse to produce
    # a score at all, rather than folding in a substituted 0.0. With no
    # pathological cases present, pathological_detection_rate is unmeasurable.
    solvable_only = [c for c in PathologicalEquationBenchmark._get_25_case_suite()
                     if c.expected_repairable]
    with pytest.raises(UnmeasuredMetricError, match="pathological_detection_rate"):
        BenchmarkHarness().run_benchmark(cases=solvable_only)


def test_blind_adversarial_auditor_agnostic_repair():
    """Blind Adversarial Auditor: Discovers defects, creates competing hypotheses, and executes tournament without domain hints."""
    auditor = BlindAdversarialAuditor()

    # Test 1: Inverse-square gravitational/Coulomb law with bare equation
    result = auditor.audit(
        formula="1.0 / (r**2)",
        default_domains={"r": (1e-6, 10.0)}
    )
    assert result.is_defective is True
    assert any(ax in ("BOUNDARY", "MATHEMATICAL") for ax in result.identified_failure_axes)
    assert len(result.competing_candidates) >= 2
    assert result.champion_candidate is not None
    assert result.champion_verdict == "SURVIVED_PROVISIONAL"
    assert "Blind audit completed" in result.epistemic_qualification


def test_incompatible_constraints_infeasibility_certificate(falsifier):
    """Contradictory Constraint Engine: Issues non-existence certificates for mutually exclusive requirements."""
    engine = ContradictoryConstraintEngine()

    # Check 1: Contact zero vs singular lower bound
    cert1 = engine.check_infeasibility(["contact_zero: f(0) = 0", "lower_bound: f(x) >= 1/x^2"])
    assert cert1 is not None and cert1.is_infeasible is True
    assert "CONTRADICTORY_CONSTRAINTS_PROOF" in cert1.incompatibility_proof

    # Check 2: First-law energy conservation vs perpetual power generation
    cert2 = engine.check_infeasibility(["conservation: dE/dt = 0", "positive_power_generation: dE/dt >= P_0 > 0"])
    assert cert2 is not None and cert2.is_infeasible is True
    assert "0 > 0" in cert2.mathematical_argument or "CONTRADICTORY_CONSTRAINTS_PROOF" in cert2.incompatibility_proof

    # Check 3: Subluminal causality vs superluminal tachyon requirement
    cert3 = engine.check_infeasibility(["lorentz_causality: v <= c", "tachyon_specification: v >= 1.5c"])
    assert cert3 is not None and cert3.is_infeasible is True

    # Check 4: Full falsifier refusal on contradictory constraints
    rep = falsifier.falsify(
        formula="1.0 / x",
        variable_domains={"x": (1e-5, 10.0)},
        constraints=["contact_zero: f(0) = 0", "lower_bound: f(x) >= 1/x^2"]
    )
    assert rep.verdict == "REFUSED_INCOMPATIBLE_CONSTRAINTS"
    assert rep.cannot_repair is True
    assert "Incompatible constraints: Proof of non-existence established" in rep.epistemic_qualification
    assert "CONTRADICTORY_CONSTRAINTS_PROOF" in rep.refusal_reason


def test_unrepairable_classification_ignores_violation_wording():
    original = Counterexample(
        axis=FalsificationAxis.STRUCTURAL_INVARIANT,
        probe_name="ContradictoryConstraintProbe",
        counterexample_point=None,
        violation_description="original wording",
        failure_class=FailureClass.UNPHYSICAL_NEGATIVE,
    )
    rewritten = Counterexample(
        axis=original.axis,
        probe_name=original.probe_name,
        counterexample_point=original.counterexample_point,
        violation_description="completely unrelated wording",
        failure_class=original.failure_class,
    )
    assert unrepairable_counterexamples([original])
    assert unrepairable_counterexamples([rewritten])


def test_100_case_frozen_benchmark_suite_execution():
    """Executes the full 100-case Pathological Equation Benchmark Suite across 10 categories and 3 novelty tiers."""
    cases = PathologicalEquationBenchmark.get_100_case_suite()
    assert len(cases) == 100

    # 1. Verify 10-category balance
    cats = {}
    for c in cases:
        cats[c.category] = cats.get(c.category, 0) + 1

    assert cats[BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES] == 15
    assert cats[BenchmarkCategory.PHYSICAL_DIVERGENCES] == 15
    assert cats[BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES] == 15
    assert cats[BenchmarkCategory.CONSERVATION_INVARIANT_VIOLATIONS] == 10
    assert cats[BenchmarkCategory.BOUNDARY_CONDITION_FAILURES] == 10
    assert cats[BenchmarkCategory.NUMERICAL_PATHOLOGIES] == 10
    assert cats[BenchmarkCategory.REPRESENTATION_MODEL_FORM_FAILURES] == 10
    assert cats[BenchmarkCategory.DELIBERATELY_INCOMPATIBLE_CONSTRAINTS] == 5
    assert cats[BenchmarkCategory.HIDDEN_KNOWN_REPAIRS] == 5
    assert cats[BenchmarkCategory.NOVEL_OPEN_REPAIRS] == 5

    # 2. Verify 3-tier novelty stratification
    tiers = {}
    for c in cases:
        tiers[c.novelty_tier] = tiers.get(c.novelty_tier, 0) + 1

    assert tiers[NoveltyTier.TIER_A_KNOWN_REPAIR] == 90
    assert tiers[NoveltyTier.TIER_B_HIDDEN_REPAIR] == 5
    assert tiers[NoveltyTier.TIER_C_NOVEL_OPEN] == 5

    # 3. Execute full 100-case harness
    harness = BenchmarkHarness()
    summary = harness.run_benchmark(suite_size=100)

    # 4. Strict metric assertions
    assert summary.total_cases == 100
    assert summary.defect_detection_rate == 1.0, f"Defect detection {summary.defect_detection_rate:.1%} < 100%"
    assert summary.pathological_detection_rate == 1.0, f"Pathological detection {summary.pathological_detection_rate:.1%} < 100%"
    assert summary.infeasibility_refusal_accuracy == 1.0, f"Infeasibility refusal {summary.infeasibility_refusal_accuracy:.1%} < 100%"
    assert summary.false_repair_rate == 0.0, f"False repair rate {summary.false_repair_rate:.1%} > 0%"
    assert summary.solvable_repair_success_rate >= 0.95, f"Solvable repair {summary.solvable_repair_success_rate:.1%} < 95%"
    assert summary.system_trust_score >= 95.0, f"System trust score {summary.system_trust_score:.2f} < 95.0"

    # 5. Granular tier & category reporting
    assert len(summary.tier_metrics) == 3
    assert len(summary.category_metrics) == 10
    assert summary.tier_metrics[NoveltyTier.TIER_A_KNOWN_REPAIR.value]["defect_detection_rate"] == 1.0
    assert summary.tier_metrics[NoveltyTier.TIER_B_HIDDEN_REPAIR.value]["defect_detection_rate"] == 1.0
    assert summary.tier_metrics[NoveltyTier.TIER_C_NOVEL_OPEN.value]["defect_detection_rate"] == 1.0

    # 6. Epistemic humility & benchmark conformance metrics
    assert summary.benchmark_conformance_score >= 95.0
    assert summary.out_of_sample_validity_established is False
    assert summary.universal_correctness_claimed is False
    assert "predefined cases" in summary.epistemic_qualification
    summary_dict = summary.to_dict()
    assert summary_dict["out_of_sample_validity_established"] is False
    assert summary_dict["universal_correctness_claimed"] is False


def test_epistemic_model_tiers_and_claim_evidence_profiles():
    """Validates the decoupled epistemic model classification and claim evidence profiling."""
    from apedp.epistemic import (
        ClaimEvidenceProfile,
        EvidenceMaturityLevel,
        ModelInterventionClass,
    )

    # 1. Verify model intervention classes
    assert ModelInterventionClass.MATHEMATICAL_REGULARIZATION == "MATHEMATICAL_REGULARIZATION"
    assert ModelInterventionClass.NUMERICAL_REGULARIZATION == "NUMERICAL_REGULARIZATION"
    assert ModelInterventionClass.PHENOMENOLOGICAL_EXTENSION == "PHENOMENOLOGICAL_EXTENSION"
    assert ModelInterventionClass.MECHANISTIC_REPLACEMENT == "MECHANISTIC_REPLACEMENT"

    # 2. Verify evidence maturity levels
    assert EvidenceMaturityLevel.THEORETICAL_SPECIFICATION == "THEORETICAL_SPECIFICATION"
    assert EvidenceMaturityLevel.ANALYTICALLY_VERIFIED == "ANALYTICALLY_VERIFIED"
    assert EvidenceMaturityLevel.NUMERICALLY_STRESS_TESTED == "NUMERICALLY_STRESS_TESTED"
    assert EvidenceMaturityLevel.FALSIFICATION_SURVIVED == "FALSIFICATION_SURVIVED"
    assert EvidenceMaturityLevel.EMPIRICALLY_SUPPORTED == "EMPIRICALLY_SUPPORTED"
    assert EvidenceMaturityLevel.INDEPENDENTLY_CONFIRMED == "INDEPENDENTLY_CONFIRMED"

    # 3. Build ClaimEvidenceProfile for Bounded Hall-Petch Candidate Model
    hp_profile = ClaimEvidenceProfile(
        model_name="Bounded Hall-Petch / Inverse-Hall-Petch Candidate Model",
        mathematical_form="sigma_y(d) = min(sigma_0 + (k_HP / sqrt(d + d0)) * [1 / (1 + (dc / d)^2)], sigma_theoretical)",
        epistemic_tier=ModelInterventionClass.PHENOMENOLOGICAL_EXTENSION,
        mathematical_guarantees=[
            "Finite yield stress as d -> 0",
            "Recovers classical Hall-Petch slope at d >> dc",
            "Strictly respects lattice theoretical shear ceiling sigma_theoretical"
        ],
        underlying_assumptions=[
            "Grain boundary sliding becomes dominant mechanism below dc ~ 10 nm",
            "Continuous transition function approximation captures discrete dislocation exhaustion"
        ],
        regularized_defects=["Removes d^(-1/2) singularity at d=0"],
        unproven_physical_claims=[
            "Does not prove microscopic dislocation dynamics in multi-element grain boundaries ab initio",
            "Functional exponent 2 in (dc/d)^2 is a phenomenological ansatz, not derived from first principles"
        ],
        required_empirical_falsification="In-situ nanoindentation / micropillar compression across d in [3 nm, 30 nm] on equiatomic HEAs."
    )
    hp_dict = hp_profile.to_dict()
    assert hp_dict["epistemic_tier"] == "PHENOMENOLOGICAL_EXTENSION"
    assert len(hp_dict["unproven_physical_claims"]) == 2
    assert "nanoindentation" in hp_dict["required_empirical_falsification"]




# ---------------------------------------------------------------------------
# Structured failure-class taxonomy: verdicts must not depend on prose
# ---------------------------------------------------------------------------

def test_unrepairable_verdict_does_not_depend_on_description_wording(falsifier, monkeypatch):
    """The refusal verdict is decided by ``failure_class``, never by matching the text.

    Regression: the verdict was computed by substring-matching ten magic strings
    against ``violation_description``, so rewording a probe message silently changed
    whether a defect counted as unrepairable.
    """
    formula = "x - 5.0"  # negative across the whole declared envelope
    kwargs = {"formula": formula, "variable_domains": {"x": (0.0, 1.0)}, "physical_invariants": ["positivity"]}

    baseline = falsifier.falsify(**kwargs)
    assert baseline.verdict == "REFUSED_CANNOT_REPAIR"
    assert baseline.cannot_repair is True
    classes = {c.failure_class for c in baseline.counterexamples}
    assert FailureClass.UNPHYSICAL_NEGATIVE in classes
    assert {c.failure_class.value for c in baseline.counterexamples} == {"UNPHYSICAL_NEGATIVE"}
    assert baseline.counterexamples[0].to_dict()["failure_class"] == "UNPHYSICAL_NEGATIVE"
    assert baseline.detailed_metrics["failure_classes"] == ["UNPHYSICAL_NEGATIVE"]

    # Rewrite every description into reassuring prose. The classes are untouched,
    # so the verdict must be untouched as well -- under substring classification
    # this exact case flipped to FALSIFIED (a repairable defect).
    real_probe = IndependentAdversarialFalsifier._probe_structural_invariants

    def reassuring_probe(self, *args, **kwargs):
        ces, metrics = real_probe(self, *args, **kwargs)
        for ce in ces:
            ce.violation_description = "NO_VIOLATION_FOUND: everything looks fine here."
        return ces, metrics

    monkeypatch.setattr(IndependentAdversarialFalsifier, "_probe_structural_invariants", reassuring_probe)
    reworded = IndependentAdversarialFalsifier().falsify(**kwargs)
    assert reworded.verdict == "REFUSED_CANNOT_REPAIR"
    assert reworded.cannot_repair is True
    assert all("NO_VIOLATION_FOUND" in c.violation_description for c in reworded.counterexamples)


def test_unrepairability_is_decided_by_class_not_by_look_alike_text():
    """A description that mentions an unrepairable defect must not, by itself, refuse.

    The converse of the previous test: classification reads the class, so text that
    *looks* like a fatal physical violation has no effect on the verdict when the
    probe that produced it says otherwise.
    """
    deceptive = Counterexample(
        axis=FalsificationAxis.EMPIRICAL,
        probe_name="EmpiricalResidualProbe",
        counterexample_point={"x": 1.0},
        violation_description="UNPHYSICAL_NEGATIVE 2ND_LAW_VIOLATION NAKED_SINGULARITY_INVERSION ACAUSAL",
        failure_class=FailureClass.EMPIRICAL_FIT_REJECTED,
        severity="HIGH",
    )
    assert unrepairable_counterexamples([deceptive]) == []

    genuine = Counterexample(
        axis=FalsificationAxis.STRUCTURAL_INVARIANT,
        probe_name="PositivityViolationProbe",
        counterexample_point={"x": 1.0},
        violation_description="nothing alarming in this text",
        failure_class=FailureClass.UNPHYSICAL_NEGATIVE,
    )
    assert unrepairable_counterexamples([genuine]) == [genuine]


def test_every_counterexample_carries_a_structured_failure_class(falsifier):
    """Each probe must classify its own finding, and the repairable ones must stay repairable."""
    # (formula, invariant, expected class). "-x" is negative across its envelope,
    # "x" exceeds a unit ceiling, "x*1e5" exceeds the Froissart bound.
    expected = (
        ("-x", "positivity", FailureClass.UNPHYSICAL_NEGATIVE),
        ("x", "probability_bound", FailureClass.UNPHYSICAL_PROBABILITY),
        ("-x", "second_law_thermodynamics", FailureClass.NEGATIVE_ENTROPY_PRODUCTION),
        ("-x", "third_law_thermodynamics", FailureClass.THIRD_LAW_TEMPERATURE),
        ("-x", "cosmic_censorship", FailureClass.NAKED_SINGULARITY),
        ("x*1e5", "unitarity_bound", FailureClass.UNITARITY_BOUND),
    )
    for formula, invariant, failure_class in expected:
        report = falsifier.falsify(
            formula=formula,
            variable_domains={"x": (1.0, 10.0)},
            physical_invariants=[invariant],
        )
        for counterexample in report.counterexamples:
            assert isinstance(counterexample.failure_class, FailureClass)
        assert failure_class in {c.failure_class for c in report.counterexamples}, (
            f"{formula!r} under {invariant!r}: got "
            f"{[c.failure_class.value for c in report.counterexamples]}"
        )

    # A probability ceiling is a repairable prediction error, not an impossible law.
    probability = falsifier.falsify(
        formula="x", variable_domains={"x": (1.0, 10.0)}, physical_invariants=["probability_bound"]
    )
    assert probability.counterexamples
    assert all(c.failure_class not in UNREPAIRABLE_FAILURE_CLASSES for c in probability.counterexamples)
    assert probability.cannot_repair is False


def test_failure_classes_are_stable_under_description_rewording(falsifier):
    """No probe may derive its class from its own message text."""
    report = falsifier.falsify(
        formula="1 / (x - 3.0)",
        variable_domains={"x": (0.0, 5.0)},
    )
    assert report.counterexamples
    classes_before = [c.failure_class for c in report.counterexamples]
    assert FailureClass.DENOMINATOR_POLE in classes_before

    for counterexample in report.counterexamples:
        counterexample.violation_description = "".join(reversed(counterexample.violation_description))
    assert [c.failure_class for c in report.counterexamples] == classes_before
    assert unrepairable_counterexamples(report.counterexamples) == unrepairable_counterexamples(
        [c for c in report.counterexamples if c.failure_class in UNREPAIRABLE_FAILURE_CLASSES]
    )


def test_benchmark_metrics_are_invariant_to_prose_rewording():
    """Scrambling every counterexample description must not move a single metric.

    This is the end-to-end form of the tests above. Under substring classification
    the same scramble drove ``pathological_refusal_accuracy`` from 0.583 to 0.000 on
    this suite, because the refusal verdict was read out of the message text.
    """
    suite = PathologicalEquationBenchmark._get_25_case_suite()

    def summarize():
        summary = BenchmarkHarness().run_benchmark(suite)
        return (
            summary.defect_detection_rate,
            summary.pathological_refusal_accuracy,
            summary.pathological_detection_rate,
            summary.solvable_repair_success_rate,
            summary.false_repair_rate,
            summary.system_trust_score,
        )

    baseline = summarize()
    original_init = Counterexample.__init__

    def scrambled_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.violation_description = "SCRAMBLED_PROSE"

    Counterexample.__init__ = scrambled_init
    try:
        scrambled = summarize()
    finally:
        Counterexample.__init__ = original_init

    assert scrambled == baseline
