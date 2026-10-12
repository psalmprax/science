"""Unit tests for APEDP-X Claim-Evidence Adjudication Engine (claim_ladder.py)."""


from apedp_dyn.epistemic.claim_ladder import (
    ClaimLevel,
    EpistemicAdjudicationVerdict,
    EvidenceBundle,
    EvidenceRequirement,
    adjudicate_scientific_claim,
)


def test_claim_ladder_hierarchy_ordering():
    assert ClaimLevel.C0_NO_SUPPORT < ClaimLevel.C1_DESCRIPTIVE
    assert ClaimLevel.C1_DESCRIPTIVE < ClaimLevel.C2_PREDICTIVE
    assert ClaimLevel.C2_PREDICTIVE < ClaimLevel.C3_STRUCTURAL
    assert ClaimLevel.C3_STRUCTURAL < ClaimLevel.C4_CAUSAL_MECHANISTIC
    assert ClaimLevel.C4_CAUSAL_MECHANISTIC < ClaimLevel.C5_EXTRAPOLATIVE
    assert ClaimLevel.C5_EXTRAPOLATIVE < ClaimLevel.C6_FORMALLY_VERIFIED


def test_claim_ladder_downgrades_without_holdout():
    bundle = EvidenceBundle()
    bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)

    # Propose C2 predictive claim with only in-sample fit
    adj = adjudicate_scientific_claim(
        candidate_id="model_overfit",
        proposed_claim=ClaimLevel.C2_PREDICTIVE,
        evidence=bundle,
    )
    assert adj.claim_inflated is True
    assert adj.justified_claim == ClaimLevel.C1_DESCRIPTIVE
    assert adj.verdict == EpistemicAdjudicationVerdict.CLAIM_DOWNGRADED
    assert EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT in adj.missing_evidence


def test_claim_ladder_causal_requires_interventions():
    bundle = EvidenceBundle()
    bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
    bundle.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)
    bundle.add(EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED)

    # Propose C4 causal claim with only passive observational data
    adj = adjudicate_scientific_claim(
        candidate_id="model_passive_corr",
        proposed_claim=ClaimLevel.C4_CAUSAL_MECHANISTIC,
        evidence=bundle,
    )
    assert adj.claim_inflated is True
    assert adj.justified_claim == ClaimLevel.C3_STRUCTURAL
    assert adj.verdict == EpistemicAdjudicationVerdict.CAUSAL_UNRESOLVED
    assert EvidenceRequirement.INTERVENTIONAL_INVARIANCE in adj.missing_evidence


def test_claim_ladder_physical_veto_blocks_high_fit():
    bundle = EvidenceBundle(metrics={"apparent_r2": 0.999})
    bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
    bundle.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)
    bundle.failures.append("conservation_violation")

    adj = adjudicate_scientific_claim(
        candidate_id="non_conservative_pde",
        proposed_claim=ClaimLevel.C6_FORMALLY_VERIFIED,
        evidence=bundle,
        domain="computational_physics",
    )
    assert adj.justified_claim == ClaimLevel.C0_NO_SUPPORT
    assert adj.verdict == EpistemicAdjudicationVerdict.PHYSICALLY_INVALID
    assert "conservation" in adj.justification.lower()


def test_claim_ladder_promotes_when_all_evidence_satisfied():
    bundle = EvidenceBundle()
    bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
    bundle.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)
    bundle.add(EvidenceRequirement.PHYSICAL_CONSERVATION)
    bundle.add(EvidenceRequirement.NUMERICAL_CFL_STABILITY)
    bundle.add(EvidenceRequirement.DUAL_BOUND_CERTIFICATE)

    adj = adjudicate_scientific_claim(
        candidate_id="godunov_solver_verified",
        proposed_claim=ClaimLevel.C6_FORMALLY_VERIFIED,
        evidence=bundle,
        domain="computational_physics",
    )
    assert adj.claim_inflated is False
    assert adj.justified_claim == ClaimLevel.C6_FORMALLY_VERIFIED
    assert adj.verdict == EpistemicAdjudicationVerdict.PROMOTED
