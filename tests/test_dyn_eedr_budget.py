"""EEDR budget pins: exactly one repair attempt per loop invocation.

Resolves the doc-vs-claim dispute (docstring says single-shot; a pasted
claim said MaxIter <= 2): the code path proposes at most one bounded repair
and re-adjudicates at most once, then terminates. No repair loop exists.
"""

from apedp_dyn.epistemic.eedr import (
    ClaimLevel,
    EvidenceBundle,
    EvidenceRequirement,
    execute_eedr_loop,
)


def _failing_bundle() -> EvidenceBundle:
    bundle = EvidenceBundle(metrics={"in_sample_r2": 0.99})
    bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
    bundle.failures.append("extrapolation_divergence_infinite_singularity")
    return bundle


def test_single_repair_attempt_max():
    res = execute_eedr_loop(
        candidate_id="SINGULAR_PADE_BUDGET",
        proposed_claim=ClaimLevel.C5_EXTRAPOLATIVE,
        evidence=_failing_bundle(),
        domain="dynamical_systems",
    )
    step3 = sum(1 for line in res.audit_trail if "STEP 3:" in line)
    step4 = sum(1 for line in res.audit_trail if "STEP 4:" in line)
    assert step3 <= 1, res.audit_trail
    assert step4 <= 1, res.audit_trail
    # Terminal states only: no silent second repair round exists.
    assert res.final_status in ("TERMINAL_ABSTAIN", "PROMOTED_REPAIRED")
    if res.repair_candidate is not None:
        assert "_REPAIRED_" in res.repair_candidate.repair_id


def test_repair_id_fresh_identity():
    bundle = EvidenceBundle(metrics={"in_sample_r2": 0.95, "holdout_rmse": 0.02})
    bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
    bundle.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)
    res = execute_eedr_loop(
        candidate_id="MISMATCH_BUDGET",
        proposed_claim=ClaimLevel.C4_CAUSAL_MECHANISTIC,
        evidence=bundle,
        domain="causal_inference",
    )
    assert res.repair_candidate is not None
    assert res.repair_candidate.repair_id != "MISMATCH_BUDGET"
    assert res.repair_candidate.repair_id.startswith("MISMATCH_BUDGET_REPAIRED_")
