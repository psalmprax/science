"""Evidence tiers: what each verdict's supporting evidence actually is.

The point of the taxonomy is that a consumer can tell a proved refutation from a
measured one from a thresholded one, and that a *clearance* can never be dressed up
as a proof. These tests check the taxonomy's own consistency first (every probe
declares a tier), then its two combination rules, then the tier each verdict type
comes out with.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from apedp.evidence import (
    CLEARANCE_CEILING,
    EvidenceLedger,
    EvidenceTier,
    clearance_tier,
    strongest,
    weakest,
)
from apedp.falsification import (
    AXIS_EVIDENCE_TIER,
    PROBE_EVIDENCE_TIER,
    Counterexample,
    FailureClass,
    FalsificationAxis,
    FalsificationReport,
    IndependentAdversarialFalsifier,
)

FALSIFICATION_SOURCE = Path(__file__).resolve().parents[1] / "apedp" / "falsification.py"


def _probe_names_in_source() -> set[str]:
    source = FALSIFICATION_SOURCE.read_text()
    return set(re.findall(r'probe_name="([A-Za-z]+)"', source))


def test_every_probe_declares_an_evidence_tier():
    """A probe with no tier would report UNRESOLVED, so the table must be complete."""
    declared = _probe_names_in_source()
    missing = sorted(declared - set(PROBE_EVIDENCE_TIER))
    assert not missing, f"probes without a declared evidence tier: {missing}"
    assert len(declared) >= 20


def test_the_tier_table_has_no_stale_entries():
    """A tier for a probe that no longer exists is a claim about dead code."""
    declared = _probe_names_in_source()
    stale = sorted(set(PROBE_EVIDENCE_TIER) - declared)
    assert not stale, f"tier declared for probes that do not exist: {stale}"


def test_every_axis_declares_a_tier():
    assert set(AXIS_EVIDENCE_TIER) == set(FalsificationAxis)


def test_combination_rules():
    """Refutation takes the best evidence; clearance takes the worst, then caps."""
    assert strongest([EvidenceTier.HEURISTIC, EvidenceTier.PROVED]) is EvidenceTier.PROVED
    assert strongest([EvidenceTier.HEURISTIC, EvidenceTier.EMPIRICAL]) is EvidenceTier.EMPIRICAL
    # An unresolved side-probe must not weaken an established refutation.
    assert strongest([EvidenceTier.UNRESOLVED, EvidenceTier.EMPIRICAL]) is EvidenceTier.EMPIRICAL
    assert strongest([]) is EvidenceTier.UNRESOLVED

    assert weakest([EvidenceTier.PROVED, EvidenceTier.HEURISTIC]) is EvidenceTier.HEURISTIC
    # ...but a single hole collapses a clearance.
    assert weakest([EvidenceTier.PROVED, EvidenceTier.UNRESOLVED]) is EvidenceTier.UNRESOLVED
    assert weakest([]) is EvidenceTier.UNRESOLVED


def test_a_clearance_can_never_be_proved():
    """No finite set of evaluations establishes a universally quantified claim."""
    assert clearance_tier([EvidenceTier.PROVED, EvidenceTier.PROVED]) is CLEARANCE_CEILING
    assert CLEARANCE_CEILING is EvidenceTier.EMPIRICAL
    assert clearance_tier([EvidenceTier.HEURISTIC]) is EvidenceTier.HEURISTIC
    assert clearance_tier([EvidenceTier.EMPIRICAL, EvidenceTier.UNRESOLVED]) is EvidenceTier.UNRESOLVED


def test_ledger_publishes_the_contributors_it_used():
    ledger = EvidenceLedger()
    ledger.add(EvidenceTier.PROVED, "DenominatorRootProbe")
    ledger.add(EvidenceTier.UNRESOLVED, "ThirdLawHeatCapacityProbe")
    refutation = ledger.for_refutation()
    assert refutation.tier is EvidenceTier.PROVED
    assert "DenominatorRootProbe" in refutation.contributors
    clearance = ledger.for_clearance()
    assert clearance.tier is EvidenceTier.UNRESOLVED
    assert "weakest link governs" in clearance.basis


def test_probe_tiers_match_the_method_they_use():
    """Spot-check the classification of the two ends of the scale."""
    assert PROBE_EVIDENCE_TIER["ContradictoryConstraintProbe"] is EvidenceTier.PROVED
    assert PROBE_EVIDENCE_TIER["DenominatorRootProbe"] is EvidenceTier.PROVED
    assert PROBE_EVIDENCE_TIER["AsymptoticDivergenceProbe"] is EvidenceTier.PROVED
    assert PROBE_EVIDENCE_TIER["DimensionalHomogeneityProbe"] is EvidenceTier.PROVED
    # Threshold crossings, where editing a constant would flip the verdict.
    assert PROBE_EVIDENCE_TIER["GradientExplosionProbe"] is EvidenceTier.HEURISTIC
    assert PROBE_EVIDENCE_TIER["EmpiricalResidualProbe"] is EvidenceTier.HEURISTIC


# ---------------------------------------------------------------------------
# Tiers on real verdicts
# ---------------------------------------------------------------------------


def test_exact_pole_refutation_is_proved():
    report = IndependentAdversarialFalsifier().falsify(
        formula="1.0 / (x - 1.0)", variable_domains={"x": (0.5, 5.0)}, critical_boundaries={}
    )
    assert report.verdict == "FALSIFIED"
    assert report.tier is EvidenceTier.PROVED
    assert "DenominatorRootProbe" in report.evidence_basis
    assert report.to_dict()["evidence_tier"] == "PROVED"


def test_survival_without_declared_checks_is_heuristic_not_proved():
    report = IndependentAdversarialFalsifier().falsify(
        formula="2.0 * x + 1.0", variable_domains={"x": (1.0, 10.0)}, critical_boundaries={}
    )
    assert report.verdict == "SURVIVED_PROVISIONAL"
    assert report.tier is EvidenceTier.HEURISTIC
    assert report.tier is not EvidenceTier.PROVED
    assert "never a proof" in report.evidence_basis


def test_an_axis_with_nothing_to_examine_stays_silent():
    """Neutral is not the same as a hole: an axis with no work publishes no tier."""
    report = IndependentAdversarialFalsifier().falsify(
        formula="2.0 * x + 1.0", variable_domains={"x": (1.0, 10.0)}, critical_boundaries={}
    )
    metrics = report.detailed_metrics
    # No denominators, no roots, no declared invariants, no empirical data.
    for axis in ("mathematical", "boundary", "invariants"):
        assert "evidence_tier" not in metrics[axis], f"{axis} claimed evidence it did not gather"
    assert metrics["numerical"]["evidence_tier"] == EvidenceTier.HEURISTIC.value


def test_declared_but_unevaluated_check_collapses_the_clearance():
    """A hole in the evidence dominates the remaining measurements."""
    metrics = {
        "mathematical": {"evidence_tier": EvidenceTier.PROVED.value},
        "invariants": {"evidence_tier": EvidenceTier.UNRESOLVED.value},
    }
    report = FalsificationReport(
        formula_str="candidate",
        verdict="SURVIVED_PROVISIONAL",
        survived=True,
        cannot_repair=False,
        detailed_metrics=metrics,
    )
    assert report.tier is EvidenceTier.UNRESOLVED
    assert report.to_dict()["evidence_basis"]


def test_inconclusive_probe_error_has_no_tier():
    report = IndependentAdversarialFalsifier().falsify(
        formula="5.0", variable_domains={"x": (1.0, 10.0)}, critical_boundaries={}
    )
    assert report.verdict == "INCONCLUSIVE_PROBE_ERROR"
    assert report.tier is EvidenceTier.UNRESOLVED
    assert not report.tier.is_resolved
    assert "PROBE_ERROR_NOT_EVALUABLE" in report.refusal_reason


def test_constraint_certificate_refutation_is_proved():
    report = FalsificationReport(
        formula_str="candidate",
        verdict="REFUSED_INCOMPATIBLE_CONSTRAINTS",
        survived=False,
        cannot_repair=True,
        refusal_reason="no admissible parameter set",
    )
    assert report.tier is EvidenceTier.PROVED
    assert "infeasibility certificate" in report.evidence_basis


def test_counterexamples_serialise_their_tier_and_never_borrow_credibility():
    known = Counterexample(
        axis=FalsificationAxis.MATHEMATICAL,
        probe_name="DenominatorRootProbe",
        counterexample_point={"x": 1.0},
        violation_description="pole",
        failure_class=FailureClass.DENOMINATOR_POLE,
    )
    assert known.evidence_tier is EvidenceTier.PROVED
    assert known.to_dict()["evidence_tier"] == "PROVED"

    unknown = Counterexample(
        axis=FalsificationAxis.MATHEMATICAL,
        probe_name="SomeProbeAddedLater",
        counterexample_point={},
        violation_description="?",
        failure_class=known.failure_class,
    )
    assert unknown.evidence_tier is EvidenceTier.UNRESOLVED


@pytest.mark.parametrize(
    "tier", [EvidenceTier.PROVED, EvidenceTier.EMPIRICAL, EvidenceTier.HEURISTIC]
)
def test_resolved_tiers_expose_their_strength_ordering(tier):
    assert tier.is_resolved
    assert tier.rank < EvidenceTier.UNRESOLVED.rank


# ---------------------------------------------------------------------------
# The gate's adjudication verdicts and the benchmark summary carry tiers too
# ---------------------------------------------------------------------------


def _adjudication_report(verdict):
    from apedp.gated_assurance import AdjudicationReport

    return AdjudicationReport(
        verdict=verdict,
        confidence=0.9,
        boundary_concentration_ratio=0.0,
        noise_to_signal_ratio=0.0,
        reproducibility_independence=1.0,
        recommended_lifecycle_action="none",
        rationale="test",
    )


def test_every_adjudication_verdict_declares_a_tier_and_a_basis():
    from apedp.gated_assurance import AdjudicationVerdict

    for verdict in AdjudicationVerdict:
        report = _adjudication_report(verdict)
        assert report.evidence_tier in set(EvidenceTier)
        assert report.evidence_basis
        assert report.to_dict()["evidence_tier"] == report.evidence_tier.value


def test_a_thresholded_adjudication_is_never_tiered_as_proved():
    """Every classification verdict is decided by a constant in the module, so
    editing that constant flips it without any new measurement."""
    from apedp.gated_assurance import AdjudicationVerdict

    for verdict in AdjudicationVerdict:
        if verdict is AdjudicationVerdict.IMPLEMENTATION_FAILURE:
            # Not a classification: it observes the candidate's own output.
            continue
        assert _adjudication_report(verdict).evidence_tier is not EvidenceTier.PROVED


def test_the_benchmark_summary_admits_its_labels_are_authored():
    """A 98% conformance score must not read as a proof."""
    from apedp.benchmarks import BenchmarkSummary

    summary = BenchmarkSummary(
        total_cases=25,
        solvable_cases=13,
        pathological_cases=12,
        defect_detection_rate=1.0,
        correct_diagnosis_rate=1.0,
        solvable_repair_success_rate=0.92,
        pathological_refusal_accuracy=0.58,
        pathological_detection_rate=1.0,
        false_repair_rate=0.0,
        system_trust_score=98.08,
    )
    assert summary.evidence_tier is EvidenceTier.EMPIRICAL
    assert "authoring assertion" in summary.evidence_basis
    assert summary.to_dict()["evidence_tier"] == "EMPIRICAL"
