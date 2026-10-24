"""Counterexample and infeasibility-certificate records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apedp.evidence import EvidenceTier

from .taxonomy import (
    PROBE_EVIDENCE_TIER,
    UNREPAIRABLE_FAILURE_CLASSES,
    FailureClass,
    FalsificationAxis,
)


@dataclass
class Counterexample:
    """Concrete evidence falsifying a proposed model or hypothesis."""
    axis: FalsificationAxis
    probe_name: str
    counterexample_point: Any
    violation_description: str
    failure_class: FailureClass
    severity: str = "FATAL"  # FATAL, HIGH, MODERATE, LOW
    metric_value: float | None = None

    @property
    def evidence_tier(self) -> EvidenceTier:
        """What this counterexample rests on, from the probe that produced it.

        A probe with no declared tier reports ``UNRESOLVED`` rather than borrowing
        credibility; the coverage test keeps the table complete.
        """
        return PROBE_EVIDENCE_TIER.get(self.probe_name, EvidenceTier.UNRESOLVED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis.value,
            "probe_name": self.probe_name,
            "counterexample_point": self.counterexample_point,
            "violation_description": self.violation_description,
            "failure_class": self.failure_class.value,
            "severity": self.severity,
            "metric_value": self.metric_value,
            "evidence_tier": self.evidence_tier.value,
        }


def unrepairable_counterexamples(counterexamples: list[Counterexample]) -> list[Counterexample]:
    """Counterexamples whose failure class admits no repair.

    Classification reads ``failure_class`` only. The description is carried along
    for the refusal message but never consulted, so a verdict cannot change because
    someone reworded a message.
    """
    return [c for c in counterexamples if c.failure_class in UNREPAIRABLE_FAILURE_CLASSES]


@dataclass
class InfeasibilityCertificate:
    """Formal mathematical proof that a given set of constraints is mutually contradictory and unrepairable."""
    is_infeasible: bool
    conflicting_constraints: list[str]
    incompatibility_proof: str
    mathematical_argument: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_infeasible": self.is_infeasible,
            "conflicting_constraints": self.conflicting_constraints,
            "incompatibility_proof": self.incompatibility_proof,
            "mathematical_argument": self.mathematical_argument,
        }
