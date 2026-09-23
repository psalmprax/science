"""Typed contracts for the decision layer (dataclasses, repo convention)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Decision(str, Enum):
    ACCEPT_FOR_BENCHMARK = "accept_for_benchmark"
    RUN_MORE_TESTS = "run_more_tests"
    SEEK_EMPIRICAL_VALIDATION = "seek_empirical_validation"
    REQUEST_DERIVATION = "request_derivation"
    HUMAN_REVIEW = "human_review"
    REJECT = "reject"
    ABSTAIN = "abstain"


class CandidateClass(str, Enum):
    LAW = "law"
    DERIVATION = "derivation"
    MODEL = "model"
    EMPIRICAL = "empirical"
    HEURISTIC = "heuristic"
    CANDIDATE = "candidate"


class EvidenceStrength(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    score: float | None = None
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: str
    candidate_class: CandidateClass
    mathematical: ValidationResult
    physics: ValidationResult
    numerical: ValidationResult
    empirical: ValidationResult
    evidence_strength: EvidenceStrength = EvidenceStrength.LOW
    independent_datasets: int = 0
    benchmark_datasets: int = 0
    domain_valid: bool = True
    evidence_ids: tuple = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["candidate_class"] = self.candidate_class.value
        d["evidence_strength"] = self.evidence_strength.value
        return d


@dataclass(frozen=True)
class DecisionResult:
    decision: Decision
    probabilities: dict
    confidence: float
    reasons: tuple = ()
    evidence_ids: tuple = ()
    policy_version: str = "apedp-policy-v1"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


@dataclass(frozen=True)
class CalibrationPolicy:
    """Maps backend confidence to actions; uncalibrated confidences abstain."""
    accept_threshold: float = 0.95
    review_threshold: float = 0.70

    def apply(self, probabilities: dict) -> str:
        if not probabilities:
            return Decision.ABSTAIN.value
        best = max(probabilities, key=lambda k: probabilities[k])
        conf = float(probabilities[best])
        if conf < self.review_threshold:
            return Decision.ABSTAIN.value
        if conf < self.accept_threshold:
            return Decision.HUMAN_REVIEW.value
        return str(best)
