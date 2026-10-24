"""Aggregated falsification report."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apedp.evidence import EvidenceAssessment, EvidenceLedger, EvidenceTier

from .counterexamples import (
    Counterexample,
    unrepairable_counterexamples,
)


@dataclass
class FalsificationReport:
    """Comprehensive verdict and counterexample dossier produced by the Independent Falsifier."""
    formula_str: str
    verdict: str  # SURVIVED_PROVISIONAL, FALSIFIED, REFUSED_CANNOT_REPAIR, REFUSED_INCOMPATIBLE_CONSTRAINTS
    survived: bool
    cannot_repair: bool
    counterexamples: list[Counterexample] = field(default_factory=list)
    refusal_reason: str | None = None
    axes_tested: list[str] = field(default_factory=list)
    detailed_metrics: dict[str, Any] = field(default_factory=dict)
    epistemic_qualification: str = ""
    #: What the verdict rests on. Derived from the counterexamples and the axes
    #: that actually ran unless the constructor names it explicitly, so a verdict
    #: cannot be presented as stronger evidence than the probes produced.
    evidence_tier: EvidenceTier | None = None
    evidence_basis: str = ""

    def __post_init__(self) -> None:
        if self.evidence_tier is not None and self.evidence_basis:
            return
        assessment = self._assess_evidence()
        if self.evidence_tier is None:
            self.evidence_tier = assessment.tier
        if not self.evidence_basis:
            self.evidence_basis = assessment.basis

    def _assess_evidence(self) -> EvidenceAssessment:
        """Derive the tier from the verdict's own support."""
        ledger = EvidenceLedger()
        if self.verdict == "SURVIVED_PROVISIONAL":
            for axis, axis_metrics in self.detailed_metrics.items():
                if not isinstance(axis_metrics, dict) or "evidence_tier" not in axis_metrics:
                    continue
                ledger.add(EvidenceTier(axis_metrics["evidence_tier"]), axis)
            return ledger.for_clearance(
                basis=(
                    "clearance within the audited envelope: bounded by the weakest "
                    "axis that ran, and a clearance is never a proof"
                )
            )
        if self.verdict == "INCONCLUSIVE_PROBE_ERROR":
            return EvidenceLedger().for_absence(
                self.refusal_reason or "one or more probes could not decide"
            )
        if self.verdict == "REFUSED_INCOMPATIBLE_CONSTRAINTS":
            # The certificate is the evidence, and it is exact: it proves no
            # admissible parameter set exists, rather than failing to find one.
            ledger.add(EvidenceTier.PROVED, "InfeasibilityCertificate")
            return ledger.for_refutation(
                basis="infeasibility certificate: the admissible function space is empty"
            )

        decisive = unrepairable_counterexamples(self.counterexamples) or self.counterexamples
        for counterexample in decisive:
            ledger.add(counterexample.evidence_tier, counterexample.probe_name)
        return ledger.for_refutation()

    @property
    def fatal_count(self) -> int:
        return sum(1 for c in self.counterexamples if c.severity == "FATAL")

    @property
    def tier(self) -> EvidenceTier:
        """The tier as a non-optional value (set in ``__post_init__``)."""
        return self.evidence_tier or EvidenceTier.UNRESOLVED

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula_str": self.formula_str,
            "verdict": self.verdict,
            "survived": self.survived,
            "cannot_repair": self.cannot_repair,
            "fatal_counterexamples": self.fatal_count,
            "counterexamples": [c.to_dict() for c in self.counterexamples],
            "refusal_reason": self.refusal_reason,
            "axes_tested": self.axes_tested,
            "detailed_metrics": self.detailed_metrics,
            "epistemic_qualification": self.epistemic_qualification,
            "evidence_tier": self.tier.value,
            "evidence_basis": self.evidence_basis,
        }
