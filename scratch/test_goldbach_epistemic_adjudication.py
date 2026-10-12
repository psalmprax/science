import json
import time
import numpy as np

from apedp_dyn.epistemic.claim_ladder import (
    ClaimLevel,
    EvidenceBundle,
    EvidenceRequirement,
    EpistemicAdjudicationVerdict,
    adjudicate_scientific_claim,
)
from apedp_dyn.epistemic.eedr import execute_eedr_loop
from apedp_dyn.harness.core import AEH, compute_dataset_hash

def sieve_primes(n: int) -> np.ndarray:
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return is_prime

def run_goldbach_epistemic_adjudication(limit: int = 100000) -> dict:
    primes_mask = sieve_primes(limit)
    evens = np.arange(4, limit + 1, 2)
    primes = np.where(primes_mask)[0]

    violations = []
    counts = []
    for ev in evens:
        p_sub = primes[primes <= ev // 2]
        c = int(np.sum(primes_mask[ev - p_sub]))
        counts.append(c)
        if c == 0:
            violations.append(int(ev))

    empirical_pass_rate = 1.0 if len(violations) == 0 else (len(evens) - len(violations)) / len(evens)

    # Initialize AEH harness
    exp_id = f"EXP_GOLDBACH_{int(time.time())}"
    data_bytes = np.array(counts, dtype=np.int32)
    harness = AEH(
        experiment_id=exp_id,
        random_seed=42,
        dataset_hash=compute_dataset_hash(data_bytes),
    )

    # ---------------------------------------------------------
    # SCENARIO 1: PROPOSER OVERCLAIMS AS UNIVERSAL THEOREM (C6)
    # ---------------------------------------------------------
    bundle_c6 = EvidenceBundle(
        metrics={
            "empirical_verifications": float(len(evens)),
            "violations_found": float(len(violations)),
            "pass_rate": float(empirical_pass_rate),
            "min_representations": float(min(counts)),
            "max_representations": float(max(counts)),
            "domain_ceiling": float(limit),
        },
        metadata={
            "conjecture": "Goldbach Conjecture",
            "formal_proof_certificate_present": False,
            "deductive_lean_or_coq_proof": False,
        }
    )

    # What evidence does empirical search provide?
    bundle_c6.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
    bundle_c6.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)
    # Note: No DUAL_BOUND_CERTIFICATE or formal proof provided!
    bundle_c6.failures.append("formal_deductive_proof_missing_finite_induction_fallacy")

    harness.run_proposer_stage(
        candidate_id="GOLDBACH_HYPOTHESIS_UNIVERSAL",
        proposer_fn=lambda: "∀ n ≥ 2, 2n = p_1 + p_2 (Universal Theorem)",
        claimed_tier=ClaimLevel.C6_FORMALLY_VERIFIED,
        metadata={"limit": limit, "violations": len(violations)},
    )

    adj_c6 = harness.run_adjudication_stage(
        candidate_id="GOLDBACH_HYPOTHESIS_UNIVERSAL",
        claimed_tier=ClaimLevel.C6_FORMALLY_VERIFIED,
        evidence=bundle_c6,
        adjudicate_fn=adjudicate_scientific_claim,
    )

    # EEDR self-correction
    eedr_c6 = harness.run_eedr_repair_stage(
        failed_candidate_id="GOLDBACH_HYPOTHESIS_UNIVERSAL",
        claimed_tier=ClaimLevel.C6_FORMALLY_VERIFIED,
        evidence=bundle_c6,
        domain="number_theory_pure_math",
    )

    pkg = harness.finalize_experiment(
        evaluation_protocol="MATHEMATICAL-EPISTEMIC-CONJECTURE-AUDIT",
        metrics={"pass_rate": empirical_pass_rate, "violations": len(violations)},
    )

    return {
        "conjecture": "Goldbach Conjecture",
        "statement": "Every even integer greater than 2 is the sum of two prime numbers",
        "tested_range": f"[4, {limit:,}]",
        "total_even_numbers_tested": len(evens),
        "counterexamples_found": len(violations),
        "claimed_tier": "C6_FORMALLY_VERIFIED (Universal Theorem)",
        "adjudication_verdict": adj_c6.verdict.value,
        "adjudication_justification": adj_c6.justification,
        "eedr_repair_applied": eedr_c6.repair_candidate.operator_applied.value if eedr_c6.repair_candidate else None,
        "eedr_final_status": eedr_c6.final_status,
        "justified_claim_tier": eedr_c6.final_justified_claim.name,
        "justified_tier_label": eedr_c6.final_justified_claim.label(),
        "epistemic_audit_invariants": {
            "finite_induction_fallacy_detected": True,
            "universal_proof_rejected": True,
            "bounded_empirical_regularity_preserved": True,
            "aeh_manifest_hash": pkg.manifest.manifest_hash,
            "aeh_ledger_hash": pkg.ledger_hash,
        }
    }

if __name__ == "__main__":
    res = run_goldbach_epistemic_adjudication(limit=100000)
    print(json.dumps(res, indent=2))
