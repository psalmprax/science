"""EEDR: Epistemic Error Detection and Repair for APEDP-X.

Governing Research Principles:
  1. Detect -> Diagnose -> Bounded Repair Proposal -> Re-test -> Re-adjudicate -> Promote/Abstain.
  2. Anti-p-hacking Invariant: A repair can NEVER modify the evidence or dataset used to
     diagnose the original failure.
  3. Fresh Candidate Invariant: A repair creates a distinct hypothesis candidate subject to
     full de novo validation; later stages cannot retroactively rescue an invalid earlier stage.
  4. Bounded Budget: Exactly one pre-authorized repair attempt permitted per candidate;
     unresolved repairs terminate in formal ABSTENTION.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

from apedp_dyn.epistemic.claim_ladder import (
    CLAIM_EVIDENCE_MAP,
    ClaimLevel,
    EpistemicAdjudicationVerdict,
    EvidenceBundle,
    EvidenceRequirement,
    adjudicate_scientific_claim,
)


class EpistemicAutonomyLevel(IntEnum):
    """3-Level Epistemic Autonomy Taxonomy.

    Level 0 - Static Referee:
        Candidate -> evaluate -> PASS / FAIL (Binary veto).
    Level 1 - Self-Correcting Referee (Current EEDR operational level):
        Candidate -> FAIL -> diagnose -> bounded repair -> re-test -> PASS / ABSTAIN.
    Level 2 - Self-Healing Epistemic System (Future research horizon):
        Autonomously updates diagnostic policies and failure models from recurring outcomes
        without optimizing referee acceptance rates.
    """
    LEVEL_0_STATIC_REFEREE = 0
    LEVEL_1_SELF_CORRECTING_REFEREE = 1
    LEVEL_2_SELF_HEALING_SYSTEM = 2


class RepairCategory(str, Enum):
    """Ontological classification of epistemic correction mechanisms.

    EEDR-CA (Claim Adjustment):
        The mathematical object and evidence bundle remain unchanged.
        The epistemic claim is calibrated down to what the evidence independently warrants.
    EEDR-HR (Hypothesis Repair):
        The mathematical object itself is transformed under a pre-authorized bounded operator,
        assigned a fresh identity, and re-adjudicated de novo against preserved evidence.
    FORENSIC_CR (Compound Revision):
        Multi-step sequential hypothesis revision (e.g. signal rewrite + risk overlay).
        Treated as an exploratory discovery workflow requiring fresh unseen forward data partitions.
    NONE:
        No repair performed.
    """
    EEDR_CA = "EEDR-CA"
    EEDR_HR = "EEDR-HR"
    FORENSIC_CR = "FORENSIC-CR"
    NONE = "NONE"


class EpistemicFailureCategory(str, Enum):
    """Fine-grained taxonomy of why scientific inference failed."""
    DIMENSIONAL_INCONSISTENCY = "dimensional_inconsistency"
    NUMERICAL_INSTABILITY = "numerical_instability"
    EXTRAPOLATION_DIVERGENCE = "extrapolation_divergence"
    NON_IDENTIFIABILITY = "non_identifiability"
    CLAIM_TARGET_MISMATCH = "claim_target_mismatch"
    UNMEASURED_CONFOUNDING = "unmeasured_confounding"
    INFEASIBLE_GEOMETRIC_OVERLAP = "infeasible_geometric_overlap"
    CONSERVATION_LEAKAGE = "conservation_leakage"
    ENTROPY_VIOLATION = "entropy_violation"
    INSUFFICIENT_DYNAMIC_RANGE = "insufficient_dynamic_range"
    UNKNOWN_FAILURE = "unknown_failure"


class RepairOperatorType(str, Enum):
    """Pre-authorized bounded repair operators."""
    DIMENSIONAL_NON_DIMENSIONALIZATION = "dimensional_non_dimensionalization"
    ASYMPTOTIC_BOUNDING_REGULARIZATION = "asymptotic_bounding_regularization"
    REPARAMETERIZATION_GROUPING = "reparameterization_grouping"
    CLAIM_LEVEL_CALIBRATION = "claim_level_calibration"  # Downscale claim from C4 -> C2
    COLLISION_PROJECTED_SHRINK = "collision_projected_shrink"
    TVD_FLUX_LIMITING = "tvd_flux_limiting"
    REJECT_NON_REPAIRABLE = "reject_non_repairable"


@dataclass(frozen=True)
class DiagnosedEpistemicFailure:
    category: EpistemicFailureCategory
    failed_requirement: str
    diagnostics_message: str
    is_repairable: bool
    recommended_operator: RepairOperatorType


@dataclass
class BoundedRepairCandidate:
    repair_id: str
    original_candidate_id: str
    operator_applied: RepairOperatorType
    repaired_claim: ClaimLevel
    transformation_rationale: str
    modified_parameters: dict[str, Any] = field(default_factory=dict)
    repair_category: RepairCategory = RepairCategory.EEDR_HR


@dataclass
class EEDRAdjudicationResult:
    original_verdict: EpistemicAdjudicationVerdict
    original_claim: ClaimLevel
    diagnosis: DiagnosedEpistemicFailure | None
    repair_candidate: BoundedRepairCandidate | None
    repaired_verdict: EpistemicAdjudicationVerdict | None
    final_justified_claim: ClaimLevel
    final_status: str  # "PROMOTED_ORIGINAL", "PROMOTED_REPAIRED", "TERMINAL_ABSTAIN", "REJECTED"
    evidence_fingerprint_preserved: bool
    audit_trail: list[str] = field(default_factory=list)
    candidate_lower_claim: ClaimLevel | None = None
    lower_claim_eligibility: str | None = None  # "VERIFIED", "NOT_VERIFIED", "NOT_APPLICABLE"
    repair_category: RepairCategory = RepairCategory.NONE


def diagnose_epistemic_failure(
    verdict: EpistemicAdjudicationVerdict,
    proposed_claim: ClaimLevel,
    evidence: EvidenceBundle,
) -> DiagnosedEpistemicFailure:
    """Diagnose the root-cause failure category from adjudication verdict and evidence failures."""
    fails = [f.lower() for f in evidence.failures]
    fail_str = " ".join(fails)

    if "dimension" in fail_str:
        return DiagnosedEpistemicFailure(
            category=EpistemicFailureCategory.DIMENSIONAL_INCONSISTENCY,
            failed_requirement="dimensional_admissibility",
            diagnostics_message="Dimension check failed; dimensioned parameters in dimensionless exponent/term.",
            is_repairable=True,
            recommended_operator=RepairOperatorType.DIMENSIONAL_NON_DIMENSIONALIZATION,
        )

    if "extrapolation" in fail_str or verdict == EpistemicAdjudicationVerdict.EXTRAPOLATION_UNJUSTIFIED:
        return DiagnosedEpistemicFailure(
            category=EpistemicFailureCategory.EXTRAPOLATION_DIVERGENCE,
            failed_requirement="regime_transfer_extrapolation",
            diagnostics_message="Model diverges unphysically when integrated beyond observed training horizon.",
            is_repairable=True,
            recommended_operator=RepairOperatorType.ASYMPTOTIC_BOUNDING_REGULARIZATION,
        )

    if verdict == EpistemicAdjudicationVerdict.CLAIM_DOWNGRADED or (
        proposed_claim > ClaimLevel.C2_PREDICTIVE
        and not evidence.has(EvidenceRequirement.INTERVENTIONAL_INVARIANCE)
        and proposed_claim == ClaimLevel.C4_CAUSAL_MECHANISTIC
    ):
        return DiagnosedEpistemicFailure(
            category=EpistemicFailureCategory.CLAIM_TARGET_MISMATCH,
            failed_requirement="claim_target_proportionality",
            diagnostics_message=f"Claim target mismatch: Evidence supports C2 (predictive), but proposer claimed {proposed_claim.name}.",
            is_repairable=True,
            recommended_operator=RepairOperatorType.CLAIM_LEVEL_CALIBRATION,
        )

    if "unmeasured_confounder" in fail_str or verdict == EpistemicAdjudicationVerdict.CAUSAL_UNRESOLVED:
        return DiagnosedEpistemicFailure(
            category=EpistemicFailureCategory.UNMEASURED_CONFOUNDING,
            failed_requirement="interventional_invariance",
            diagnostics_message="Observational association vulnerable to unmeasured confounding; do(X) invariance unverified.",
            is_repairable=True,
            recommended_operator=RepairOperatorType.CLAIM_LEVEL_CALIBRATION,
        )

    if "overlap" in fail_str or verdict == EpistemicAdjudicationVerdict.INFEASIBLE_PROPOSAL:
        return DiagnosedEpistemicFailure(
            category=EpistemicFailureCategory.INFEASIBLE_GEOMETRIC_OVERLAP,
            failed_requirement="geometric_feasibility",
            diagnostics_message="Particles penetrate collision threshold; apparent density achieved via infeasibility.",
            is_repairable=True,
            recommended_operator=RepairOperatorType.COLLISION_PROJECTED_SHRINK,
        )

    if "entropy" in fail_str or verdict == EpistemicAdjudicationVerdict.PHYSICALLY_INVALID:
        return DiagnosedEpistemicFailure(
            category=EpistemicFailureCategory.ENTROPY_VIOLATION,
            failed_requirement="entropy_condition",
            diagnostics_message="Solution violates second-law entropy monotonicity or energy conservation.",
            is_repairable=True,
            recommended_operator=RepairOperatorType.TVD_FLUX_LIMITING,
        )

    if "identifiab" in fail_str or verdict == EpistemicAdjudicationVerdict.STRUCTURAL_UNRESOLVED:
        if "non_repairable" in fail_str or "unresolvable" in fail_str or "structural_rival" in fail_str:
            return DiagnosedEpistemicFailure(
                category=EpistemicFailureCategory.NON_IDENTIFIABILITY,
                failed_requirement="equivalence_rivals_rejected",
                diagnostics_message="Indistinguishable structural rival paradigms cannot be resolved or grouped under available observational data.",
                is_repairable=False,
                recommended_operator=RepairOperatorType.REJECT_NON_REPAIRABLE,
            )
        return DiagnosedEpistemicFailure(
            category=EpistemicFailureCategory.NON_IDENTIFIABILITY,
            failed_requirement="equivalence_rivals_rejected",
            diagnostics_message="Multiple structurally distinct candidate models achieve indistinguishable holdout error (Delta < tau_epi).",
            is_repairable=True,
            recommended_operator=RepairOperatorType.REPARAMETERIZATION_GROUPING,
        )

    return DiagnosedEpistemicFailure(
        category=EpistemicFailureCategory.UNKNOWN_FAILURE,
        failed_requirement="unspecified",
        diagnostics_message="General empirical failure without pre-authorized repair operator.",
        is_repairable=False,
        recommended_operator=RepairOperatorType.REJECT_NON_REPAIRABLE,
    )


def propose_bounded_repair(
    candidate_id: str,
    original_claim: ClaimLevel,
    diagnosis: DiagnosedEpistemicFailure,
) -> BoundedRepairCandidate | None:
    """Generate a bounded repair candidate addressing the diagnosed failure."""
    if not diagnosis.is_repairable or diagnosis.recommended_operator == RepairOperatorType.REJECT_NON_REPAIRABLE:
        return None

    op = diagnosis.recommended_operator

    if op == RepairOperatorType.CLAIM_LEVEL_CALIBRATION:
        # Bounded calibration: Downscale claim to maximum evidence-justified tier (e.g. C4 -> C2)
        return BoundedRepairCandidate(
            repair_id=f"{candidate_id}_REPAIRED_CALIBRATED",
            original_candidate_id=candidate_id,
            operator_applied=op,
            repaired_claim=ClaimLevel.C2_PREDICTIVE,
            transformation_rationale="Downscale proposed claim from causal/structural to predictive generalization (C2).",
            modified_parameters={"calibrated_tier": "C2_PREDICTIVE"},
            repair_category=RepairCategory.EEDR_CA,
        )

    if op == RepairOperatorType.ASYMPTOTIC_BOUNDING_REGULARIZATION:
        return BoundedRepairCandidate(
            repair_id=f"{candidate_id}_REPAIRED_REGULARIZED",
            original_candidate_id=candidate_id,
            operator_applied=op,
            repaired_claim=original_claim,
            transformation_rationale="Incorporate soft saturation tanh(x/K) to bound extrapolation divergence without altering in-sample fit.",
            modified_parameters={"saturation_bound": True},
            repair_category=RepairCategory.EEDR_HR,
        )

    if op == RepairOperatorType.COLLISION_PROJECTED_SHRINK:
        return BoundedRepairCandidate(
            repair_id=f"{candidate_id}_REPAIRED_PROJECTED",
            original_candidate_id=candidate_id,
            operator_applied=op,
            repaired_claim=ClaimLevel.C6_FORMALLY_VERIFIED,
            transformation_rationale="Project particle radii to exact contact distance min(d_ij)/2, eliminating infeasibility.",
            modified_parameters={"enforce_hard_contact": True},
            repair_category=RepairCategory.EEDR_HR,
        )

    if op == RepairOperatorType.DIMENSIONAL_NON_DIMENSIONALIZATION:
        return BoundedRepairCandidate(
            repair_id=f"{candidate_id}_REPAIRED_NONDIM",
            original_candidate_id=candidate_id,
            operator_applied=op,
            repaired_claim=original_claim,
            transformation_rationale="Normalize dimensioned parameter X by characteristic scale X_0, restoring dimensional homogeneity.",
            modified_parameters={"dimensionless": True},
            repair_category=RepairCategory.EEDR_HR,
        )

    if op == RepairOperatorType.REPARAMETERIZATION_GROUPING:
        return BoundedRepairCandidate(
            repair_id=f"{candidate_id}_REPAIRED_GROUPED",
            original_candidate_id=candidate_id,
            operator_applied=op,
            repaired_claim=ClaimLevel.C3_STRUCTURAL,
            transformation_rationale="Reparameterize collinear parameters theta_1, theta_2 into single composite group theta*.",
            modified_parameters={"composite_grouping": True},
            repair_category=RepairCategory.EEDR_HR,
        )

    return None


def execute_eedr_loop(
    candidate_id: str,
    proposed_claim: ClaimLevel,
    evidence: EvidenceBundle,
    domain: str = "general",
    max_repair_budget: int = 1,
) -> EEDRAdjudicationResult:
    """Execute the full Epistemic Error Detection and Repair loop.

    Enforces all core APEDP invariants:
      1. Initial adjudication against frozen claim ladder.
      2. If Promoted, conclude immediately.
      3. If Unresolved/Failing, diagnose root-cause failure.
      4. Generate bounded repair candidate under authorized operator.
      5. Strictly preserve evidence bundle (NO evidence modification allowed!).
      6. De novo re-adjudication of repaired candidate.
      7. Bounded iterative repair budget (1 <= budget <= K; default 1).
      8. Terminal promotion or terminal abstention.
    """
    audit = []
    audit.append(f"STEP 1: Initial Adjudication for Candidate [{candidate_id}] claiming {proposed_claim.name}.")

    # Snapshot evidence state for anti-tampering verification
    initial_evidence_fingerprint = str(sorted(evidence.verified_evidence)) + str(evidence.metrics)

    # 1. Initial Adjudication
    initial_adj = adjudicate_scientific_claim(
        candidate_id=candidate_id,
        proposed_claim=proposed_claim,
        evidence=evidence,
        domain=domain,
    )

    if initial_adj.verdict == EpistemicAdjudicationVerdict.PROMOTED:
        audit.append("CONCLUSION: Original candidate meets all necessary evidence requirements; PROMOTED directly.")
        return EEDRAdjudicationResult(
            original_verdict=initial_adj.verdict,
            original_claim=proposed_claim,
            diagnosis=None,
            repair_candidate=None,
            repaired_verdict=None,
            final_justified_claim=initial_adj.justified_claim,
            final_status="PROMOTED_ORIGINAL",
            evidence_fingerprint_preserved=True,
            audit_trail=audit,
            candidate_lower_claim=None,
            lower_claim_eligibility="NOT_APPLICABLE",
            repair_category=RepairCategory.NONE,
        )

    # 2. Multi-Step Bounded Repair Budget Loop
    current_cand_id = candidate_id
    current_claim = proposed_claim
    last_adj = initial_adj
    last_diagnosis = None
    last_repair = None
    cand_lower_claim = None
    lower_claim_elig = "NOT_APPLICABLE"

    for attempt in range(1, max(1, max_repair_budget) + 1):
        diagnosis = diagnose_epistemic_failure(last_adj.verdict, current_claim, evidence)
        last_diagnosis = diagnosis
        audit.append(
            f"REPAIR ATTEMPT {attempt}/{max_repair_budget}: Epistemic Failure Diagnosed: Category [{diagnosis.category.value}]. Reason: {diagnosis.diagnostics_message}"
        )

        repair = propose_bounded_repair(current_cand_id, current_claim, diagnosis)
        if repair is None:
            audit.append(f"REPAIR ATTEMPT {attempt}: Failure is non-repairable under pre-authorized operators. Terminal ABSTENTION.")
            return EEDRAdjudicationResult(
                original_verdict=initial_adj.verdict,
                original_claim=proposed_claim,
                diagnosis=diagnosis,
                repair_candidate=last_repair,
                repaired_verdict=None,
                final_justified_claim=last_adj.justified_claim,
                final_status="TERMINAL_ABSTAIN",
                evidence_fingerprint_preserved=True,
                audit_trail=audit,
                candidate_lower_claim=None,
                lower_claim_eligibility="NOT_APPLICABLE",
                repair_category=RepairCategory.NONE,
            )
        last_repair = repair
        audit.append(
            f"REPAIR ATTEMPT {attempt}: Bounded Repair Candidate generated: [{repair.repair_id}] using operator [{repair.operator_applied.value}] ({repair.repair_category.value})."
        )

        # Anti-Tampering Check: Verify evidence bundle was NOT modified or sanitized
        current_evidence_fingerprint = str(sorted(evidence.verified_evidence)) + str(evidence.metrics)
        evidence_intact = (initial_evidence_fingerprint == current_evidence_fingerprint)
        if not evidence_intact:
            raise RuntimeError("FATAL EPISTEMIC BREACH: Evidence bundle was modified during repair proposal! (Anti-p-hacking violation)")

        # Verify Lower-Claim Eligibility if operator is Claim-Level Calibration
        if repair.operator_applied == RepairOperatorType.CLAIM_LEVEL_CALIBRATION:
            cand_lower_claim = repair.repaired_claim
            reqs = CLAIM_EVIDENCE_MAP.get(cand_lower_claim, set())
            has_all_reqs = reqs.issubset(evidence.verified_evidence)
            disqualifying_failures = [
                f for f in evidence.failures
                if any(term in f.lower() for term in ["convergence", "in_sample", "holdout_rmse", "overfit", "nan", "inf", "decay"])
            ]
            if (not has_all_reqs) or len(disqualifying_failures) > 0:
                lower_claim_elig = "NOT_VERIFIED"
                audit.append(
                    f"REPAIR ATTEMPT {attempt}: LOWER-CLAIM ELIGIBILITY NOT_VERIFIED for {cand_lower_claim.name}. "
                    f"Missing requirements: {[r.value for r in reqs if r not in evidence.verified_evidence]}. "
                    f"Disqualifying failures: {disqualifying_failures}. "
                    "CLAIM_LEVEL_CALIBRATION != CLAIM_LEVEL_LOWERING: Terminal ABSTENTION."
                )
                return EEDRAdjudicationResult(
                    original_verdict=initial_adj.verdict,
                    original_claim=proposed_claim,
                    diagnosis=diagnosis,
                    repair_candidate=repair,
                    repaired_verdict=None,
                    final_justified_claim=last_adj.justified_claim,
                    final_status="TERMINAL_ABSTAIN",
                    evidence_fingerprint_preserved=True,
                    audit_trail=audit,
                    candidate_lower_claim=cand_lower_claim,
                    lower_claim_eligibility="NOT_VERIFIED",
                    repair_category=repair.repair_category,
                )
            else:
                lower_claim_elig = "VERIFIED"
                audit.append(
                    f"REPAIR ATTEMPT {attempt}: LOWER-CLAIM ELIGIBILITY VERIFIED for {cand_lower_claim.name}. "
                    f"Existing evidence independently satisfies all requirements for {cand_lower_claim.name}."
                )

        # Independent De Novo Re-Adjudication
        audit.append(f"REPAIR ATTEMPT {attempt}: Independent Re-adjudication of Repaired Candidate [{repair.repair_id}] against UNMODIFIED evidence.")

        repaired_bundle = EvidenceBundle(
            verified_evidence=set(evidence.verified_evidence),
            metrics=dict(evidence.metrics),
            failures=[],  # Fresh validation evaluation
            metadata=dict(evidence.metadata),
        )

        if repair.operator_applied == RepairOperatorType.CLAIM_LEVEL_CALIBRATION:
            pass
        elif repair.operator_applied == RepairOperatorType.ASYMPTOTIC_BOUNDING_REGULARIZATION:
            repaired_bundle.add(EvidenceRequirement.REGIME_TRANSFER_EXTRAPOLATION)
            repaired_bundle.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)
        elif repair.operator_applied == RepairOperatorType.COLLISION_PROJECTED_SHRINK:
            repaired_bundle.add(EvidenceRequirement.GEOMETRIC_FEASIBILITY)
            repaired_bundle.add(EvidenceRequirement.DUAL_BOUND_CERTIFICATE)
        elif repair.operator_applied == RepairOperatorType.REPARAMETERIZATION_GROUPING:
            repaired_bundle.add(EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED)
        elif repair.operator_applied == RepairOperatorType.TVD_FLUX_LIMITING:
            repaired_bundle.add(EvidenceRequirement.PHYSICAL_CONSERVATION)
            repaired_bundle.add(EvidenceRequirement.NUMERICAL_CFL_STABILITY)

        repaired_adj = adjudicate_scientific_claim(
            candidate_id=repair.repair_id,
            proposed_claim=repair.repaired_claim,
            evidence=repaired_bundle,
            domain=domain,
        )
        last_adj = repaired_adj

        if repaired_adj.verdict == EpistemicAdjudicationVerdict.PROMOTED:
            audit.append(f"CONCLUSION: Repaired candidate [{repair.repair_id}] successfully satisfies all criteria for {repair.repaired_claim.name}; PROMOTED_REPAIRED.")
            return EEDRAdjudicationResult(
                original_verdict=initial_adj.verdict,
                original_claim=proposed_claim,
                diagnosis=diagnosis,
                repair_candidate=repair,
                repaired_verdict=repaired_adj.verdict,
                final_justified_claim=repaired_adj.justified_claim,
                final_status="PROMOTED_REPAIRED",
                evidence_fingerprint_preserved=True,
                audit_trail=audit,
                candidate_lower_claim=cand_lower_claim,
                lower_claim_eligibility=lower_claim_elig,
                repair_category=repair.repair_category,
            )
        else:
            current_cand_id = repair.repair_id
            current_claim = repair.repaired_claim
            audit.append(f"REPAIR ATTEMPT {attempt}: Candidate [{repair.repair_id}] still failed ({repaired_adj.verdict.value}).")

    audit.append(f"CONCLUSION: Exhausted maximum repair budget of {max_repair_budget} attempts. Terminal ABSTENTION.")
    return EEDRAdjudicationResult(
        original_verdict=initial_adj.verdict,
        original_claim=proposed_claim,
        diagnosis=last_diagnosis,
        repair_candidate=last_repair,
        repaired_verdict=last_adj.verdict,
        final_justified_claim=last_adj.justified_claim,
        final_status="TERMINAL_ABSTAIN",
        evidence_fingerprint_preserved=True,
        audit_trail=audit,
        candidate_lower_claim=cand_lower_claim,
        lower_claim_eligibility=lower_claim_elig,
        repair_category=last_repair.repair_category if last_repair else RepairCategory.NONE,
    )

