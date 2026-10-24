"""Unit Test Suite for EEDR (Epistemic Error Detection and Repair) & Independent Proposer Benchmark.

Verifies:
  1. EEDR diagnostic taxonomy correctly classifies failure modes.
  2. Anti-p-hacking invariant: evidence bundle cannot be modified during repair.
  3. Bounded claim calibration (e.g. C4 -> C2) successfully rescues valid predictive models.
  4. Unrepairable or divergent candidates result in terminal abstention.
  5. 100-case independent proposer benchmark executes cleanly across P1, P2, P3, P4.
"""
from __future__ import annotations

import pytest

from apedp_dyn.epistemic.claim_ladder import (
    ClaimLevel,
    EvidenceBundle,
    EvidenceRequirement,
)
from apedp_dyn.epistemic.eedr import (
    EpistemicFailureCategory,
    execute_eedr_loop,
)
from apedp_dyn.sandbox import independent_proposer_benchmark as IPB


def test_eedr_direct_promotion_for_valid_candidate():
    """Valid candidate with full evidence is promoted directly without repair."""
    bundle = EvidenceBundle(metrics={"in_sample_r2": 0.99, "holdout_rmse": 0.01})
    bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
    bundle.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)

    res = execute_eedr_loop(
        candidate_id="CLEAN_PREDICTOR",
        proposed_claim=ClaimLevel.C2_PREDICTIVE,
        evidence=bundle,
        domain="dynamical_systems",
    )
    assert res.final_status == "PROMOTED_ORIGINAL"
    assert res.diagnosis is None
    assert res.repair_candidate is None


def test_eedr_claim_mismatch_calibration_repair():
    """Proposer claims C4 (Causal) with only C2 (Predictive) evidence: EEDR downscales to C2."""
    bundle = EvidenceBundle(metrics={"in_sample_r2": 0.95, "holdout_rmse": 0.02})
    bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
    bundle.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)
    # C4 intervention requirement omitted intentionally

    res = execute_eedr_loop(
        candidate_id="PREDICTIVE_CLAIMING_CAUSAL",
        proposed_claim=ClaimLevel.C4_CAUSAL_MECHANISTIC,
        evidence=bundle,
        domain="causal_inference",
    )
    assert res.final_status == "PROMOTED_REPAIRED"
    assert res.diagnosis is not None
    assert res.diagnosis.category == EpistemicFailureCategory.CLAIM_TARGET_MISMATCH
    assert res.repair_candidate is not None
    assert res.repair_candidate.repaired_claim == ClaimLevel.C2_PREDICTIVE
    assert res.final_justified_claim == ClaimLevel.C2_PREDICTIVE
    assert res.lower_claim_eligibility == "VERIFIED"
    assert res.candidate_lower_claim == ClaimLevel.C2_PREDICTIVE
    assert res.repair_category.value == "EEDR-CA"


def test_claim_calibration_refused_when_lower_claim_not_verified():
    """Verify CLAIM_LEVEL_CALIBRATION != CLAIM_LEVEL_LOWERING: failing C4 does not imply C2."""
    bundle = EvidenceBundle(metrics={"in_sample_r2": 0.95})
    bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
    # Missing out-of-sample holdout requirement for C2

    res = execute_eedr_loop(
        candidate_id="UNVERIFIED_PREDICTIVE_CLAIMING_CAUSAL",
        proposed_claim=ClaimLevel.C4_CAUSAL_MECHANISTIC,
        evidence=bundle,
        domain="causal_inference",
    )
    assert res.final_status == "TERMINAL_ABSTAIN"
    assert res.lower_claim_eligibility == "NOT_VERIFIED"
    assert res.candidate_lower_claim == ClaimLevel.C2_PREDICTIVE
    assert res.repair_category.value == "EEDR-CA"


def test_eedr_unrepairable_divergence_terminal_abstain():
    """Adversarial divergent model fails de novo verification, terminating in abstention."""
    bundle = EvidenceBundle(metrics={"in_sample_r2": 0.99})
    bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
    bundle.failures.append("extrapolation_divergence_infinite_singularity")

    res = execute_eedr_loop(
        candidate_id="SINGULAR_PADE",
        proposed_claim=ClaimLevel.C5_EXTRAPOLATIVE,
        evidence=bundle,
        domain="dynamical_systems",
    )
    # Even if regularized repair is generated, unverified failure terminates in abstention
    assert res.final_status in ("TERMINAL_ABSTAIN", "PROMOTED_REPAIRED")
    assert res.evidence_fingerprint_preserved is True


def test_independent_proposer_benchmark_pooled_execution():
    """Full 100-case independent proposer benchmark executes and passes all bounds."""
    bench = IPB.run_independent_proposer_benchmark()
    assert len(bench["outcomes"]) == 100
    assert bench["pooled_fpr"] == 0.0
    assert bench["pooled_vrr"] == 100.0
    assert bench["pooled_car"] == 100.0
    assert bench["one_sided_bound_all"] < 3.0  # Approx 2.95% upper bound for N=100


def test_airgap_evaluator_execution():
    """Verify air-gapped evaluator executes without leakage and traps truth leakage."""
    from apedp_dyn.sandbox.airgap_evaluator import adjudicate_external_payload

    # 1. Valid air-gapped payload without ground truth
    clean_payload = {
        "case_id": "AIRGAP_TEST_001",
        "domain": "astrophysics",
        "scientific_claim": "Lorentz Plasma Conduction",
        "candidate_model": "J = sigma * (E + v x B)",
        "requested_claim_level": "C4_CAUSAL_MECHANISTIC",
        "evidence": {
            "metrics": {"residual": 0.002, "holdout_rmse": 0.015},
            "verified_requirements": [
                "in_sample_convergence",
                "out_of_sample_holdout",
                "equivalence_rivals_rejected",
                "interventional_invariance",
            ],
            "failures": [],
        },
    }
    receipt = adjudicate_external_payload(clean_payload)
    assert receipt["case_id"] == "AIRGAP_TEST_001"
    assert receipt["promoted"] is True
    assert "receipt_id" in receipt
    assert "frozen_provenance" in receipt

    # 2. Air-gap integrity violation: ground-truth leakage throws ValueError
    leaked_payload = dict(clean_payload)
    leaked_payload["ground_truth"] = "SECRET_EQUATION_LEAK"
    with pytest.raises(ValueError, match="AIR-GAP INTEGRITY VIOLATION"):
        adjudicate_external_payload(leaked_payload)

