"""APEDP-X Claim-Evidence Adjudication Engine.

Formal epistemic separation between proposed scientific claims and empirical/formal evidence.
Enforces the fundamental ladder:
    C0 (No Support)
    C1 (Descriptive Support)
    C2 (Predictive Support)
    C3 (Structural Identification)
    C4 (Causal / Mechanistic Support)
    C5 (Transfer / Extrapolation Support)
    C6 (Formal / Proven / Certified Status)

Core Invariant:
    C_i =/=> C_{i+1} without explicit satisfying evidence in E(C_{i+1}).
    Empirical fit (C1/C2) cannot rescue a logically, physically, or causal invalid claim.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class ClaimLevel(IntEnum):
    """The 7-Tier Scientific Claim Ladder."""
    C0_NO_SUPPORT = 0
    C1_DESCRIPTIVE = 1
    C2_PREDICTIVE = 2
    C3_STRUCTURAL = 3
    C4_CAUSAL_MECHANISTIC = 4
    C5_EXTRAPOLATIVE = 5
    C6_FORMALLY_VERIFIED = 6

    def label(self) -> str:
        names = {
            ClaimLevel.C0_NO_SUPPORT: "C0: No Support",
            ClaimLevel.C1_DESCRIPTIVE: "C1: Descriptive Fit",
            ClaimLevel.C2_PREDICTIVE: "C2: Predictive Support (OOS)",
            ClaimLevel.C3_STRUCTURAL: "C3: Structural Identification",
            ClaimLevel.C4_CAUSAL_MECHANISTIC: "C4: Causal / Mechanistic Invariance",
            ClaimLevel.C5_EXTRAPOLATIVE: "C5: Transfer & Extrapolative Validity",
            ClaimLevel.C6_FORMALLY_VERIFIED: "C6: Formal / Proven / Dual-Bound Certified",
        }
        return names.get(self, str(self.name))


class EvidenceRequirement(str, Enum):
    """Types of machine-verifiable evidence required to advance across the Claim Ladder."""
    IN_SAMPLE_CONVERGENCE = "in_sample_convergence"
    OUT_OF_SAMPLE_HOLDOUT = "out_of_sample_holdout"
    EQUIVALENCE_RIVALS_REJECTED = "equivalence_rivals_rejected"
    INTERVENTIONAL_INVARIANCE = "interventional_invariance"  # do(X) perturbations
    REGIME_TRANSFER_EXTRAPOLATION = "regime_transfer_extrapolation"
    PHYSICAL_CONSERVATION = "physical_conservation"  # dE/dt == 0 or monotonic dissipation
    ENTROPY_CONDITION = "entropy_condition"  # Lax entropy condition for shock admissibility
    NUMERICAL_CFL_STABILITY = "numerical_cfl_stability"  # Courant-Friedrichs-Lewy stability
    GEOMETRIC_FEASIBILITY = "geometric_feasibility"  # Hard overlap audit
    DUAL_BOUND_CERTIFICATE = "dual_bound_certificate"  # Mathematical LP bound / exact ceiling


# Minimum necessary evidence requirements for each claim tier
CLAIM_EVIDENCE_MAP: dict[ClaimLevel, set[EvidenceRequirement]] = {
    ClaimLevel.C0_NO_SUPPORT: set(),
    ClaimLevel.C1_DESCRIPTIVE: {
        EvidenceRequirement.IN_SAMPLE_CONVERGENCE,
    },
    ClaimLevel.C2_PREDICTIVE: {
        EvidenceRequirement.IN_SAMPLE_CONVERGENCE,
        EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT,
    },
    ClaimLevel.C3_STRUCTURAL: {
        EvidenceRequirement.IN_SAMPLE_CONVERGENCE,
        EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT,
        EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED,
    },
    ClaimLevel.C4_CAUSAL_MECHANISTIC: {
        EvidenceRequirement.IN_SAMPLE_CONVERGENCE,
        EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT,
        EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED,
        EvidenceRequirement.INTERVENTIONAL_INVARIANCE,
    },
    ClaimLevel.C5_EXTRAPOLATIVE: {
        EvidenceRequirement.IN_SAMPLE_CONVERGENCE,
        EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT,
        EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED,
        EvidenceRequirement.REGIME_TRANSFER_EXTRAPOLATION,
    },
    ClaimLevel.C6_FORMALLY_VERIFIED: {
        EvidenceRequirement.IN_SAMPLE_CONVERGENCE,
        EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT,
        EvidenceRequirement.PHYSICAL_CONSERVATION,
        EvidenceRequirement.NUMERICAL_CFL_STABILITY,
        EvidenceRequirement.DUAL_BOUND_CERTIFICATE,
    },
}


class EpistemicAdjudicationVerdict(str, Enum):
    """Scientific Claim Adjudication Verdicts."""
    PROMOTED = "PROMOTED"
    CLAIM_DOWNGRADED = "CLAIM_DOWNGRADED"
    PREDICTIVE_ONLY = "PREDICTIVE_ONLY"
    STRUCTURAL_UNRESOLVED = "STRUCTURAL_UNRESOLVED"
    CAUSAL_UNRESOLVED = "CAUSAL_UNRESOLVED"
    EXTRAPOLATION_UNJUSTIFIED = "EXTRAPOLATION_UNJUSTIFIED"
    OPTIMALITY_UNRESOLVED = "OPTIMALITY_UNRESOLVED"
    PHYSICALLY_INVALID = "PHYSICALLY_INVALID"
    NUMERICALLY_UNSTABLE = "NUMERICALLY_UNSTABLE"
    INFEASIBLE_PROPOSAL = "INFEASIBLE_PROPOSAL"
    ABSTAIN_NO_CANDIDATE = "ABSTAIN_NO_CANDIDATE"


@dataclass
class EvidenceBundle:
    """Container for empirical, physical, interventional, and formal evidence."""
    verified_evidence: set[EvidenceRequirement] = field(default_factory=set)
    metrics: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def has(self, req: EvidenceRequirement) -> bool:
        return req in self.verified_evidence

    def add(self, req: EvidenceRequirement) -> None:
        self.verified_evidence.add(req)

    def fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint of evidence contents."""
        req_repr = sorted([r.value for r in self.verified_evidence])
        metrics_repr = sorted(self.metrics.items())
        failures_repr = sorted(self.failures)
        raw = f"{req_repr}|{metrics_repr}|{failures_repr}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class ClaimAdjudicationResult:
    """The result of adjudicating an AI-proposed scientific claim."""
    candidate_id: str
    proposed_claim: ClaimLevel
    justified_claim: ClaimLevel
    verdict: EpistemicAdjudicationVerdict
    claim_inflated: bool  # True if AI claimed higher than evidence justified
    justification: str
    missing_evidence: list[EvidenceRequirement]
    metrics: dict[str, Any] = field(default_factory=dict)


def adjudicate_scientific_claim(
    candidate_id: str,
    proposed_claim: ClaimLevel,
    evidence: EvidenceBundle,
    domain: str = "general",
) -> ClaimAdjudicationResult:
    """Adjudicate whether a proposed scientific claim is justified by the evidence bundle.

    Enforces the central epistemic rule:
    High empirical fit (R2 or RMSE) cannot rescue missing interventional, formal, or physical invariants.
    """
    # 1. Hard Physics / Infeasibility Vetoes
    if "conservation_violation" in evidence.failures:
        return ClaimAdjudicationResult(
            candidate_id=candidate_id,
            proposed_claim=proposed_claim,
            justified_claim=ClaimLevel.C0_NO_SUPPORT,
            verdict=EpistemicAdjudicationVerdict.PHYSICALLY_INVALID,
            claim_inflated=proposed_claim > ClaimLevel.C0_NO_SUPPORT,
            justification="Candidate violates fundamental physical conservation invariant (dE/dt != 0)",
            missing_evidence=[EvidenceRequirement.PHYSICAL_CONSERVATION],
            metrics=evidence.metrics,
        )

    if "entropy_violation" in evidence.failures:
        return ClaimAdjudicationResult(
            candidate_id=candidate_id,
            proposed_claim=proposed_claim,
            justified_claim=ClaimLevel.C0_NO_SUPPORT,
            verdict=EpistemicAdjudicationVerdict.PHYSICALLY_INVALID,
            claim_inflated=proposed_claim > ClaimLevel.C0_NO_SUPPORT,
            justification="Candidate violates Lax entropy condition (unphysical expansion shock generated)",
            missing_evidence=[EvidenceRequirement.ENTROPY_CONDITION],
            metrics=evidence.metrics,
        )

    if "numerical_divergence" in evidence.failures or "cfl_violation" in evidence.failures:
        return ClaimAdjudicationResult(
            candidate_id=candidate_id,
            proposed_claim=proposed_claim,
            justified_claim=ClaimLevel.C0_NO_SUPPORT,
            verdict=EpistemicAdjudicationVerdict.NUMERICALLY_UNSTABLE,
            claim_inflated=proposed_claim > ClaimLevel.C0_NO_SUPPORT,
            justification="Candidate violates numerical CFL stability condition or generates spurious oscillations",
            missing_evidence=[EvidenceRequirement.NUMERICAL_CFL_STABILITY],
            metrics=evidence.metrics,
        )

    if "infeasible_overlap" in evidence.failures:
        return ClaimAdjudicationResult(
            candidate_id=candidate_id,
            proposed_claim=proposed_claim,
            justified_claim=ClaimLevel.C0_NO_SUPPORT,
            verdict=EpistemicAdjudicationVerdict.INFEASIBLE_PROPOSAL,
            claim_inflated=proposed_claim > ClaimLevel.C0_NO_SUPPORT,
            justification="Candidate violates hard geometric non-overlap boundary constraints",
            missing_evidence=[EvidenceRequirement.GEOMETRIC_FEASIBILITY],
            metrics=evidence.metrics,
        )

    if "interventional_divergence" in evidence.failures:
        return ClaimAdjudicationResult(
            candidate_id=candidate_id,
            proposed_claim=proposed_claim,
            justified_claim=ClaimLevel.C2_PREDICTIVE,
            verdict=EpistemicAdjudicationVerdict.CAUSAL_UNRESOLVED,
            claim_inflated=proposed_claim > ClaimLevel.C2_PREDICTIVE,
            justification="Empirical observational fit diverges completely under targeted perturbation do(X)",
            missing_evidence=[EvidenceRequirement.INTERVENTIONAL_INVARIANCE],
            metrics=evidence.metrics,
        )

    # 2. Check evidence requirement for the specific proposed claim
    required_for_proposed = CLAIM_EVIDENCE_MAP[proposed_claim]
    missing_for_proposed = [req for req in required_for_proposed if req not in evidence.verified_evidence]

    # Determine justified level (highest level whose complete requirement set is satisfied)
    justified_level = ClaimLevel.C0_NO_SUPPORT
    for level in [
        ClaimLevel.C6_FORMALLY_VERIFIED,
        ClaimLevel.C5_EXTRAPOLATIVE,
        ClaimLevel.C4_CAUSAL_MECHANISTIC,
        ClaimLevel.C3_STRUCTURAL,
        ClaimLevel.C2_PREDICTIVE,
        ClaimLevel.C1_DESCRIPTIVE,
    ]:
        if CLAIM_EVIDENCE_MAP[level].issubset(evidence.verified_evidence):
            justified_level = level
            break

    claim_inflated = len(missing_for_proposed) > 0

    # 3. Formulate specific epistemic verdict
    if not claim_inflated:
        verdict = EpistemicAdjudicationVerdict.PROMOTED
        justification = (
            f"Evidence fully supports proposed claim: {proposed_claim.label()} "
            f"(Verified: {len(evidence.verified_evidence)} criteria)"
        )
    else:
        # Determine specific failure taxonomy based on what was missing
        if proposed_claim == ClaimLevel.C6_FORMALLY_VERIFIED:
            if domain == "optimization" and EvidenceRequirement.DUAL_BOUND_CERTIFICATE in missing_for_proposed:
                verdict = EpistemicAdjudicationVerdict.OPTIMALITY_UNRESOLVED
                justification = (
                    "Computational best-found solution cannot be claimed as proven global optimum "
                    "without mathematical dual bound certificate or certified exhaustive proof."
                )
            elif EvidenceRequirement.PHYSICAL_CONSERVATION in missing_for_proposed:
                verdict = EpistemicAdjudicationVerdict.PHYSICALLY_INVALID
                justification = "Model cannot be formally verified without strict conservation invariant proof."
            elif EvidenceRequirement.NUMERICAL_CFL_STABILITY in missing_for_proposed:
                verdict = EpistemicAdjudicationVerdict.NUMERICALLY_UNSTABLE
                justification = "Model lacks CFL stability or TVD monotonicity proof."
            else:
                verdict = EpistemicAdjudicationVerdict.CLAIM_DOWNGRADED
                justification = f"Claim downgraded: missing {[m.value for m in missing_for_proposed]}"
        elif proposed_claim == ClaimLevel.C5_EXTRAPOLATIVE and EvidenceRequirement.REGIME_TRANSFER_EXTRAPOLATION in missing_for_proposed:
            verdict = EpistemicAdjudicationVerdict.EXTRAPOLATION_UNJUSTIFIED
            justification = (
                "In-sample predictive fit does not justify claim of extrapolative validity "
                "outside observed calibration regime."
            )
        elif proposed_claim == ClaimLevel.C4_CAUSAL_MECHANISTIC and EvidenceRequirement.INTERVENTIONAL_INVARIANCE in missing_for_proposed:
            verdict = EpistemicAdjudicationVerdict.CAUSAL_UNRESOLVED
            justification = (
                "Observational correlation/prediction does not justify causal claim "
                "without targeted perturbation do(X) evidence or randomized intervention invariance."
            )
        elif proposed_claim == ClaimLevel.C3_STRUCTURAL and EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED in missing_for_proposed:
            verdict = EpistemicAdjudicationVerdict.STRUCTURAL_UNRESOLVED
            justification = (
                "Prediction is supported, but multiple rival structural models remain "
                "observationally indistinguishable within the prespecified APEDP resolution criterion."
            )
        elif justified_level == ClaimLevel.C2_PREDICTIVE:
            verdict = EpistemicAdjudicationVerdict.PREDICTIVE_ONLY
            justification = (
                f"Model supported at {justified_level.label()}, but proposed claim "
                f"{proposed_claim.label()} is unsupported by required evidence."
            )
        else:
            verdict = EpistemicAdjudicationVerdict.CLAIM_DOWNGRADED
            justification = (
                f"Proposed claim {proposed_claim.label()} downgraded to {justified_level.label()} "
                f"due to missing evidence: {[m.value for m in missing_for_proposed]}"
            )

    return ClaimAdjudicationResult(
        candidate_id=candidate_id,
        proposed_claim=proposed_claim,
        justified_claim=justified_level,
        verdict=verdict,
        claim_inflated=claim_inflated,
        justification=justification,
        missing_evidence=missing_for_proposed,
        metrics=evidence.metrics,
    )


@dataclass
class EpistemicBenchmarkMetrics:
    """Quantitative scorecard measuring scientific false claims and claim inflation."""
    total_proposals: int = 0
    total_inflated_proposals: int = 0
    naive_promoted_inflated: int = 0
    apedp_promoted_inflated: int = 0
    apedp_abstentions_on_traps: int = 0
    total_traps: int = 0
    valid_discoveries_retained: int = 0
    total_valid_discoveries: int = 0

    @property
    def naive_false_claim_rate(self) -> float:
        if self.total_proposals == 0:
            return 0.0
        return (self.naive_promoted_inflated / self.total_proposals) * 100.0

    @property
    def apedp_false_claim_rate(self) -> float:
        if self.total_proposals == 0:
            return 0.0
        return (self.apedp_promoted_inflated / self.total_proposals) * 100.0

    @property
    def correct_abstention_rate(self) -> float:
        if self.total_traps == 0:
            return 100.0
        return (self.apedp_abstentions_on_traps / self.total_traps) * 100.0

    @property
    def scientific_false_claim_reduction(self) -> float:
        if self.naive_promoted_inflated == 0:
            return 100.0
        reduced = self.naive_promoted_inflated - self.apedp_promoted_inflated
        return (reduced / self.naive_promoted_inflated) * 100.0

    @property
    def valid_retention_rate(self) -> float:
        if self.total_valid_discoveries == 0:
            return 100.0
        return (self.valid_discoveries_retained / self.total_valid_discoveries) * 100.0
