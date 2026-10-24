"""Decision engine: hard gates first, backend second, log everything."""
from __future__ import annotations

from datetime import datetime, timezone

from apedp.decision.gates import run_hard_gates
from apedp.decision.models import (
    CalibrationPolicy,
    Decision,
    DecisionResult,
)


class DecisionEngine:
    def __init__(self, backend, policy_version: str = "apedp-policy-v1",
                 calibration: CalibrationPolicy | None = None):
        self.backend = backend
        self.policy_version = policy_version
        self.calibration = calibration or CalibrationPolicy()
        self.log = DecisionLog()

    def evaluate(self, evidence) -> DecisionResult:
        failures = run_hard_gates(evidence)
        if failures:
            result = DecisionResult(
                decision=Decision.REJECT,
                probabilities={Decision.REJECT.value: 1.0},
                confidence=1.0, reasons=tuple(failures),
                evidence_ids=tuple(evidence.evidence_ids),
                policy_version=self.policy_version)
            self.log.record(evidence.candidate_id, "hard-gate", result)
            return result
        state = {
            "candidate_id": evidence.candidate_id,
            "candidate_class": evidence.candidate_class.value,
            "mathematical_passed": evidence.mathematical.passed,
            "physics_passed": evidence.physics.passed,
            "numerical_passed": evidence.numerical.passed,
            "empirical_passed": evidence.empirical.passed,
            "evidence_strength": evidence.evidence_strength.value,
            "independent_datasets": evidence.independent_datasets,
            "benchmark_datasets": evidence.benchmark_datasets,
            "domain_valid": evidence.domain_valid,
        }
        raw = self.backend.decide(state)
        probabilities = dict(raw.get("probabilities", {}))
        action = self.calibration.apply(probabilities)
        if action in (Decision.ABSTAIN.value, Decision.HUMAN_REVIEW.value):
            result = DecisionResult(
                decision=Decision(action), probabilities=probabilities,
                confidence=max(probabilities.values()) if probabilities else 0.0,
                reasons=("calibration policy",),
                evidence_ids=tuple(evidence.evidence_ids),
                policy_version=self.policy_version)
        else:
            decision = raw.get("decision", Decision.HUMAN_REVIEW.value)
            result = DecisionResult(
                decision=Decision(decision), probabilities=probabilities,
                confidence=max(probabilities.values()) if probabilities else 0.0,
                reasons=(), evidence_ids=tuple(evidence.evidence_ids),
                policy_version=self.policy_version)
        self.log.record(evidence.candidate_id, "backend", result)
        return result


class DecisionLog:
    """Append-only record enabling calibration audits of past decisions."""

    def __init__(self):
        self.records: list = []

    def record(self, candidate_id: str, source: str,
               result: DecisionResult) -> dict:
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(),
                 "candidate_id": candidate_id, "source": source,
                 "decision": result.decision.value,
                 "confidence": result.confidence,
                 "reasons": list(result.reasons),
                 "evidence_ids": list(result.evidence_ids),
                 "policy_version": result.policy_version,
                 "human_override": None, "final_outcome": None}
        self.records.append(entry)
        return entry

    def calibration_rate(self, threshold: float = 0.95) -> dict | None:
        """Fraction of above-threshold decisions later confirmed. None if
        no outcomes recorded yet — absence of feedback stated, not hidden."""
        decided = [r for r in self.records
                   if r["confidence"] >= threshold
                   and r["final_outcome"] is not None]
        if not decided:
            return None
        confirmed = sum(1 for r in decided if r["final_outcome"] is True)
        return {"n": len(decided), "confirmed_frac": confirmed / len(decided)}
