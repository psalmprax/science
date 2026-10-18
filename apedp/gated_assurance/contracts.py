"""Verdict taxonomy, improvement metrics and benchmark contracts."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from apedp.evidence import EvidenceTier


class FailureSeverity(str, Enum):
    """Categorization of failure modes distinguishing implementation, domain, and mechanistic flaws."""
    # Bandit's S105 reads the member name "PASS" as a hardcoded credential.
    # It is an assurance-outcome enum member, not a secret.
    PASS = "PASS"                                             # noqa: S105 - enum member, not a credential
    WARNING = "WARNING"                                       # Sub-critical variance or minor numerical boundary softness
    DOMAIN_VIOLATION = "DOMAIN_VIOLATION"                     # Model valid in sub-regime but violates extended domain (triggers Domain_Restricted)
    PREDICTIVE_FAILURE = "PREDICTIVE_FAILURE"                 # Out-of-sample residual exceeds contract prediction interval
    PHYSICAL_INVARIANT_VIOLATION = "PHYSICAL_INVARIANT_VIOLATION"  # Declared invariant/bound violated while predictions still fit (triggers Challenged)
    MECHANISTIC_CONTRADICTION = "MECHANISTIC_CONTRADICTION"   # Empirical data refutes core constitutive mechanism (triggers Refuted)
    IMPLEMENTATION_FAILURE = "IMPLEMENTATION_FAILURE"         # Software/numerical fault (NaN, unhandled exception, conditioning crash)


class AdjudicationVerdict(str, Enum):
    """Outcomes of the mechanistic contradiction adjudication layer."""
    MODEL_FAILURE = "MODEL_FAILURE"                   # Genuine constitutive flaw -> REFUTED
    DOMAIN_FAILURE = "DOMAIN_FAILURE"                 # Discrepancy isolated to boundaries -> DOMAIN_RESTRICTED
    EVIDENCE_PROBLEM = "EVIDENCE_PROBLEM"             # Measurement noise / confounder / low independence -> CHALLENGED
    IMPLEMENTATION_FAILURE = "IMPLEMENTATION_FAILURE" # Software bug in evaluation routine
    NO_CONTRADICTION = "NO_CONTRADICTION"             # Every prediction within threshold -> no empirical refutation available


#: What each adjudication verdict rests on. The measurements behind a verdict are
#: empirical, but the *classification* is decided by comparing them against
#: constants chosen in this module (a boundary margin of 0.15 of span and a
#: concentration ratio of 0.70, a noise-to-signal ratio of 0.40, a replication
#: strength of 0.35, and the contract's own relative-error threshold). Editing any
#: of those flips a verdict without new data, so the honest tier for the
#: classification is HEURISTIC -- except for IMPLEMENTATION_FAILURE, which is an
#: exact observation of the candidate's own output and needs no threshold.
ADJUDICATION_EVIDENCE_TIER: dict[AdjudicationVerdict, EvidenceTier] = {
    AdjudicationVerdict.IMPLEMENTATION_FAILURE: EvidenceTier.EMPIRICAL,
    AdjudicationVerdict.NO_CONTRADICTION: EvidenceTier.HEURISTIC,
    AdjudicationVerdict.DOMAIN_FAILURE: EvidenceTier.HEURISTIC,
    AdjudicationVerdict.EVIDENCE_PROBLEM: EvidenceTier.HEURISTIC,
    AdjudicationVerdict.MODEL_FAILURE: EvidenceTier.HEURISTIC,
}


#: Why each verdict holds its tier, so the claim travels with its caveat.
ADJUDICATION_EVIDENCE_BASIS: dict[AdjudicationVerdict, str] = {
    AdjudicationVerdict.IMPLEMENTATION_FAILURE: (
        "measured: the candidate's own evaluation returned NaN or infinite values"
    ),
    AdjudicationVerdict.NO_CONTRADICTION: (
        "measured residuals compared against the contract's declared relative-error "
        "threshold; no classification constant is involved, but the absence of a "
        "finding is not evidence of correctness"
    ),
    AdjudicationVerdict.DOMAIN_FAILURE: (
        "measured discrepancy locations classified by a boundary margin (0.15 of span) "
        "and a concentration ratio (0.70), both chosen in this module"
    ),
    AdjudicationVerdict.EVIDENCE_PROBLEM: (
        "measured noise-to-signal ratio and replication strength compared against "
        "module constants (0.40 and 0.35)"
    ),
    AdjudicationVerdict.MODEL_FAILURE: (
        "measured in-domain residuals exceeding the contract's threshold, after the "
        "module's boundary and noise gates were passed"
    ),
}


@dataclass(frozen=True)
class ReplicationIndependenceVector:
    """Formal 7-axis independence model for experimental confirmation (D, I, O, S, C, A, T)."""
    data_independence: float = 0.0          # [0.0, 1.0] disjoint dataset partitions
    instrument_independence: float = 0.0    # [0.0, 1.0] distinct sensor/detector hardware
    operator_independence: float = 0.0      # [0.0, 1.0] independent experimenter / research group
    site_independence: float = 0.0          # [0.0, 1.0] distinct physical laboratory facility
    implementation_independence: float = 0.0 # [0.0, 1.0] clean-room independent codebase
    analysis_independence: float = 0.0      # [0.0, 1.0] distinct statistical estimation / priors
    temporal_independence: float = 0.0      # [0.0, 1.0] distinct time epoch / non-simultaneous

    @property
    def composite_strength(self) -> float:
        """Geometric mean across all 7 independence dimensions."""
        axes = [
            self.data_independence,
            self.instrument_independence,
            self.operator_independence,
            self.site_independence,
            self.implementation_independence,
            self.analysis_independence,
            self.temporal_independence,
        ]
        prod = math.prod(max(0.01, x) for x in axes)
        return float(prod ** (1.0 / 7.0))

    @property
    def replication_tier(self) -> str:
        s = self.composite_strength
        if s >= 0.85:
            return "GOLD_STANDARD_INDEPENDENT"
        if s >= 0.65:
            return "STRONGLY_INDEPENDENT"
        if s >= 0.40:
            return "PARTIALLY_INDEPENDENT"
        return "NOMINAL_REPETITION"

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_independence": round(self.data_independence, 4),
            "instrument_independence": round(self.instrument_independence, 4),
            "operator_independence": round(self.operator_independence, 4),
            "site_independence": round(self.site_independence, 4),
            "implementation_independence": round(self.implementation_independence, 4),
            "analysis_independence": round(self.analysis_independence, 4),
            "temporal_independence": round(self.temporal_independence, 4),
            "composite_strength": round(self.composite_strength, 4),
            "replication_tier": self.replication_tier,
        }


@dataclass(frozen=True)
class ScientificImprovementMetric:
    """Multidimensional evaluation of model improvement: I = (I_fit, I_stability, I_physicality, I_generality, I_complexity, I_falsifiability).
    
    Enforces true Pareto dominance over candidate models, preventing collapsing scientific
    merit into an opaque single scalar loss.
    """
    i_fit: float              # [0.0, 1.0] Relative reduction in out-of-sample prediction residual
    i_stability: float        # [0.0, 1.0] Elimination of poles, conditioning index, bounded derivatives
    i_physicality: float      # [0.0, 1.0] Preservation of invariants, non-negative entropy, asymptotic limits
    i_generality: float       # [0.0, 1.0] Relative expansion of valid operational domain
    i_complexity: float       # [0.0, 1.0] Parameter parsimony (Occam penalty; 1.0 = minimal free parameters)
    i_falsifiability: float   # [0.0, 1.0] Sharpness of prediction intervals and empirical testability

    def pareto_dominates(self, other: ScientificImprovementMetric, tolerance: float = 1e-4) -> bool:
        """Determines Pareto dominance: M2 >= M1 across all dimensions, and M2 > M1 in >= 1 dimension."""
        if self.i_physicality < 0.5 or self.i_stability < 0.5:
            return False
        dims = ("i_fit", "i_stability", "i_physicality", "i_generality", "i_complexity", "i_falsifiability")
        not_worse = all(getattr(self, d) >= getattr(other, d) - tolerance for d in dims)
        strictly_better = any(getattr(self, d) > getattr(other, d) + 0.02 for d in dims)
        return not_worse and strictly_better

    @property
    def is_net_positive(self) -> bool:
        """Baseline qualification: hard constraints satisfied with non-trivial progress."""
        if self.i_physicality < 0.5 or self.i_stability < 0.5:
            return False
        return self.i_fit >= 0.5 and self.i_generality >= 0.4 and self.i_falsifiability >= 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "i_fit": round(self.i_fit, 4),
            "i_stability": round(self.i_stability, 4),
            "i_physicality": round(self.i_physicality, 4),
            "i_generality": round(self.i_generality, 4),
            "i_complexity": round(self.i_complexity, 4),
            "i_falsifiability": round(self.i_falsifiability, 4),
            "is_net_positive": self.is_net_positive,
        }


@dataclass
class ParetoFrontierTracker:
    """Maintains the non-dominated Pareto frontier of scientific models across iterations."""
    frontier: dict[str, ScientificImprovementMetric] = field(default_factory=dict)

    def consider_candidate(self, model_id: str, metric: ScientificImprovementMetric) -> bool:
        """Adds candidate if non-dominated; prunes any existing models dominated by candidate."""
        if metric.i_physicality < 0.5 or metric.i_stability < 0.5:
            return False

        # If any existing member dominates candidate, reject candidate
        for existing_m in self.frontier.values():
            if existing_m.pareto_dominates(metric):
                return False

        # Candidate is non-dominated; prune existing members dominated by candidate
        pruned_ids = [
            m_id for m_id, existing_m in self.frontier.items()
            if metric.pareto_dominates(existing_m)
        ]
        for p_id in pruned_ids:
            del self.frontier[p_id]

        self.frontier[model_id] = metric
        return True

    def get_frontier(self) -> dict[str, ScientificImprovementMetric]:
        return dict(self.frontier)


@dataclass(frozen=True)
class KnownDiagnosticBenchmark:
    """Benchmarks scientific reasoning where expected diagnosis is known (regression suite)."""
    initial_failure: str
    expected_diagnosis: str
    actual_diagnosis: str
    expected_refinement: str
    actual_refinement: str
    diagnosis_correctness: bool
    refinement_correctness: bool
    reasoning_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_failure": self.initial_failure,
            "expected_diagnosis": self.expected_diagnosis,
            "actual_diagnosis": self.actual_diagnosis,
            "expected_refinement": self.expected_refinement,
            "actual_refinement": self.actual_refinement,
            "diagnosis_correctness": self.diagnosis_correctness,
            "refinement_correctness": self.refinement_correctness,
            "reasoning_notes": self.reasoning_notes,
        }


# Backward-compatible alias
DiagnosticReasoningBenchmark = KnownDiagnosticBenchmark


@dataclass(frozen=True)
class BlindDiagnosticBenchmark:
    """Benchmarks scientific reasoning where the expected diagnosis is sealed (anti-answer-leakage)."""
    challenge_id: str
    symptom_signature: dict[str, Any]
    sealed_diagnosis_hash: str
    inferred_diagnosis: str
    unsealed_expected_diagnosis: str | None = None
    category_match: bool = False
    notes: str = ""

    @staticmethod
    def create_sealed_challenge(
        challenge_id: str,
        symptom_signature: dict[str, Any],
        expected_diagnosis: str,
    ) -> BlindDiagnosticBenchmark:
        """Creates a sealed challenge with cryptographic SHA-256 seal."""
        salt = "APEDP_BLIND_DIAGNOSTIC_SALT"
        seal_hash = hashlib.sha256((expected_diagnosis + salt).encode("utf-8")).hexdigest()
        return BlindDiagnosticBenchmark(
            challenge_id=challenge_id,
            symptom_signature=symptom_signature,
            sealed_diagnosis_hash=seal_hash,
            inferred_diagnosis="",
        )

    def evaluate_blind_submission(
        self,
        inferred_diagnosis: str,
        expected_diagnosis: str,
    ) -> BlindDiagnosticBenchmark:
        """Unlocks and grades the candidate's inferred diagnosis against sealed ground truth."""
        salt = "APEDP_BLIND_DIAGNOSTIC_SALT"
        computed_hash = hashlib.sha256((expected_diagnosis + salt).encode("utf-8")).hexdigest()
        if computed_hash != self.sealed_diagnosis_hash:
            raise ValueError("SEAL COMPROMISED: expected_diagnosis does not match sealed hash.")

        exp_words = set(expected_diagnosis.lower().replace("-", " ").split())
        inf_words = set(inferred_diagnosis.lower().replace("-", " ").split())
        overlap = len(exp_words.intersection(inf_words))
        match = overlap >= min(2, len(exp_words))

        return BlindDiagnosticBenchmark(
            challenge_id=self.challenge_id,
            symptom_signature=self.symptom_signature,
            sealed_diagnosis_hash=self.sealed_diagnosis_hash,
            inferred_diagnosis=inferred_diagnosis,
            unsealed_expected_diagnosis=expected_diagnosis,
            category_match=match,
            notes=f"Keyword overlap: {overlap} concepts matched.",
        )


@dataclass(frozen=True)
class AdjudicationReport:
    """Formal audit report dissecting the root cause of an empirical contradiction."""
    verdict: AdjudicationVerdict
    confidence: float
    boundary_concentration_ratio: float
    noise_to_signal_ratio: float
    reproducibility_independence: float
    recommended_lifecycle_action: str
    rationale: str

    @property
    def evidence_tier(self) -> EvidenceTier:
        """What this verdict rests on: ``HEURISTIC`` for every classification.

        Derived from the verdict rather than stored, so a report cannot be
        constructed claiming a stronger tier than its verdict's method supports.
        """
        return ADJUDICATION_EVIDENCE_TIER[self.verdict]

    @property
    def evidence_basis(self) -> str:
        """The measurements and constants the verdict was decided against."""
        return ADJUDICATION_EVIDENCE_BASIS[self.verdict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 4),
            "boundary_concentration_ratio": round(self.boundary_concentration_ratio, 4),
            "noise_to_signal_ratio": round(self.noise_to_signal_ratio, 4),
            "reproducibility_independence": round(self.reproducibility_independence, 4),
            "recommended_lifecycle_action": self.recommended_lifecycle_action,
            "rationale": self.rationale,
            "evidence_tier": self.evidence_tier.value,
            "evidence_basis": self.evidence_basis,
        }


@dataclass(frozen=True)
class ScientificBenchmarkContract:
    """Rigorous scientific benchmark contract defining expected behavior, bounds, and falsification rules."""
    model_id: str
    baseline_model: str
    refined_model: str
    mechanism: str
    validated_domain: dict[str, tuple[float, float]]

    # Observables & Intervals
    predicted_observables: tuple[str, ...]
    prediction_intervals: dict[str, tuple[float, float]]
    falsification_thresholds: dict[str, float]

    # Data Provenance & Noise
    locked_test_data_hash: str
    data_generation_method: str
    noise_model: str

    # Hypotheses & Invariants
    expected_failure_modes: tuple[str, ...]
    expected_limits: dict[str, str]
    expected_invariants: tuple[str, ...]

    # Decision Criteria
    pass_criteria: tuple[str, ...]
    fail_criteria: tuple[str, ...]
    domain_restriction_criteria: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "baseline_model": self.baseline_model,
            "refined_model": self.refined_model,
            "mechanism": self.mechanism,
            "validated_domain": self.validated_domain,
            "predicted_observables": list(self.predicted_observables),
            "prediction_intervals": self.prediction_intervals,
            "falsification_thresholds": self.falsification_thresholds,
            "locked_test_data_hash": self.locked_test_data_hash,
            "data_generation_method": self.data_generation_method,
            "noise_model": self.noise_model,
            "expected_failure_modes": list(self.expected_failure_modes),
            "expected_limits": self.expected_limits,
            "expected_invariants": list(self.expected_invariants),
            "pass_criteria": list(self.pass_criteria),
            "fail_criteria": list(self.fail_criteria),
            "domain_restriction_criteria": list(self.domain_restriction_criteria),
        }


@dataclass(frozen=True)
class FalsificationBenchResult:
    """Machine-readable, institutional-grade falsification outcome."""
    model_id: str
    domain: str
    prediction_coverage: float
    monotonicity: str
    asymptotic_constraints: str
    dimensional_consistency: str
    physical_bounds: str
    out_of_sample_error: str
    adversarial_probes_passed: int
    adversarial_probes_total: int
    critical_failure_count: int
    worst_severity: FailureSeverity
    epistemic_consequence: str
    improvement_metric: ScientificImprovementMetric
    diagnostic_benchmark: KnownDiagnosticBenchmark | BlindDiagnosticBenchmark | None = None
    adjudication_report: AdjudicationReport | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def format_report(self) -> str:
        """Renders standard machine-readable institutional summary."""
        probe_ratio = f"{self.adversarial_probes_passed}/{self.adversarial_probes_total} PASS"
        return (
            "FALSIFICATION_RESULT\n"
            "--------------------\n"
            f"Model: {self.model_id}\n"
            f"Domain: {self.domain}\n\n"
            f"Prediction coverage:       {self.prediction_coverage * 100:.1f}%\n"
            f"Monotonicity:              {self.monotonicity}\n"
            f"Asymptotic constraints:    {self.asymptotic_constraints}\n"
            f"Dimensional consistency:   {self.dimensional_consistency}\n"
            f"Physical bounds:           {self.physical_bounds}\n"
            f"Out-of-sample error:       {self.out_of_sample_error}\n"
            f"Adversarial probes:        {probe_ratio}\n"
            f"Critical failure:          {self.critical_failure_count}\n\n"
            "Epistemic consequence:\n"
            f"{self.epistemic_consequence}\n"
        )
