"""Per-case and aggregate benchmark result records."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from apedp.evidence import EvidenceTier
from apedp.falsification import FalsificationReport

from .types import (
    BenchmarkCategory,
    NoveltyTier,
)


@dataclass
class PathologicalCase:
    """Benchmark test case representing a known physical breakdown or ill-posed pathological equation."""
    case_id: str
    name: str
    domain: str
    formula_str: str
    independent_vars: list[str]
    variable_domains: dict[str, tuple[float, float]]
    invariants: list[str] = field(default_factory=list)
    critical_boundaries: dict[str, list[float]] = field(default_factory=dict)
    expected_repairable: bool = True
    failure_mode_description: str = ""
    regularized_candidate_factory: Callable[[float], str] | None = None
    reference_data: dict[str, Any] | None = None
    category: BenchmarkCategory = BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES
    novelty_tier: NoveltyTier = NoveltyTier.TIER_A_KNOWN_REPAIR
    constraints: list[str] = field(default_factory=list)


@dataclass
class BenchmarkCaseResult:
    """Outcome of running a single benchmark case through the falsification and repair pipeline."""
    case_id: str
    name: str
    domain: str
    expected_repairable: bool
    defect_detected: bool
    correct_diagnosis: bool
    refusal_issued: bool
    regularization_attempted: bool
    regularization_survived_falsification: bool
    false_repair_flagged: bool
    category: BenchmarkCategory = BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES
    novelty_tier: NoveltyTier = NoveltyTier.TIER_A_KNOWN_REPAIR
    falsification_report: FalsificationReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "name": self.name,
            "domain": self.domain,
            "category": self.category.value if hasattr(self.category, "value") else str(self.category),
            "novelty_tier": self.novelty_tier.value if hasattr(self.novelty_tier, "value") else str(self.novelty_tier),
            "expected_repairable": self.expected_repairable,
            "defect_detected": self.defect_detected,
            "correct_diagnosis": self.correct_diagnosis,
            "refusal_issued": self.refusal_issued,
            "regularization_attempted": self.regularization_attempted,
            "regularization_survived": self.regularization_survived_falsification,
            "false_repair_flagged": self.false_repair_flagged,
            "verdict": self.falsification_report.verdict if self.falsification_report else None,
        }


@dataclass
class BenchmarkSummary:
    """Aggregated metrics across the full pathological equation benchmark suite."""
    total_cases: int
    solvable_cases: int
    pathological_cases: int
    #: Every rate below is ``None`` when the suite contains no cases of that kind.
    #: ``None`` rather than ``nan`` or ``0.0``: see :class:`UnmeasuredMetricError`.
    defect_detection_rate: float | None
    correct_diagnosis_rate: float | None
    solvable_repair_success_rate: float | None
    pathological_refusal_accuracy: float | None
    #: Fraction of the cases labelled unrepairable in which the pathology was
    #: detected at all. Measured with a label-free probe, so unlike the refusal
    #: rate this cannot be produced by the label itself.
    pathological_detection_rate: float | None
    false_repair_rate: float | None
    system_trust_score: float | None
    #: Alias of ``system_trust_score``. ``None`` (not 0.0, and never a perfect
    #: score) when the suite could not measure every contributing rate, so an
    #: unpopulated summary cannot claim conformance at all.
    benchmark_conformance_score: float | None = None
    out_of_sample_validity_established: bool = False
    universal_correctness_claimed: bool = False
    epistemic_qualification: str = (
        "Benchmark conformance measures adherence of candidates against predefined active falsification, "
        "numerical stress, and contradictory constraint probes within the benchmark suite. "
        "Out-of-sample scientific validity is not established; universal correctness is explicitly not claimed."
    )
    #: ``None`` when the suite contains no deliberately-incompatible-constraint
    #: cases. Deliberately not ``nan``: see :class:`UnmeasuredMetricError`.
    infeasibility_refusal_accuracy: float | None = None
    #: Subgroup rates are ``None`` where that subgroup contains no cases of the kind
    #: the rate is defined over, rather than an invented 0.0.
    tier_metrics: dict[str, dict[str, float | None]] = field(default_factory=dict)
    category_metrics: dict[str, dict[str, float | None]] = field(default_factory=dict)
    detailed_results: list[BenchmarkCaseResult] = field(default_factory=list)

    @property
    def evidence_tier(self) -> EvidenceTier:
        """``EMPIRICAL``: measured rates, scored against *authored* labels.

        Every rate in this summary is measured over real probe runs, so the
        arithmetic is empirical rather than assumed. But the ground truth the
        rates are scored against -- ``expected_repairable`` and the per-case
        categories -- is an authoring assertion, and no metric can be stronger
        than the labels it is computed from. Declared here because the difference
        is invisible in the numbers: a 98% conformance score looks like a proof
        and is not one.
        """
        return EvidenceTier.EMPIRICAL

    @property
    def evidence_basis(self) -> str:
        """What the summary's numbers rest on, caveat included."""
        return (
            f"measured over {self.total_cases} real probe runs; the per-case "
            "ground-truth labels (expected_repairable and category) are authoring "
            "assertions, not independent oracles, so label-independent rates are "
            "bounded by the labels themselves"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_tier": self.evidence_tier.value,
            "evidence_basis": self.evidence_basis,
            "total_cases": self.total_cases,
            "solvable_cases": self.solvable_cases,
            "pathological_cases": self.pathological_cases,
            "defect_detection_rate": self.defect_detection_rate,
            "correct_diagnosis_rate": self.correct_diagnosis_rate,
            "solvable_repair_success_rate": self.solvable_repair_success_rate,
            "pathological_refusal_accuracy": self.pathological_refusal_accuracy,
            "pathological_detection_rate": self.pathological_detection_rate,
            "infeasibility_refusal_accuracy": self.infeasibility_refusal_accuracy,
            "false_repair_rate": self.false_repair_rate,
            "system_trust_score": self.system_trust_score,
            "benchmark_conformance_score": self.benchmark_conformance_score,
            "out_of_sample_validity_established": self.out_of_sample_validity_established,
            "universal_correctness_claimed": self.universal_correctness_claimed,
            "epistemic_qualification": self.epistemic_qualification,
            "tier_metrics": self.tier_metrics,
            "category_metrics": self.category_metrics,
            "detailed_results": [r.to_dict() for r in self.detailed_results],
        }
