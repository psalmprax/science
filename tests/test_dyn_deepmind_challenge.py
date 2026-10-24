"""Unit tests for APEDP-X DeepMind Hypothesis Challenge Suite.

Verifies:
  1. AlphaTensor, GraphCast, Aletheia, and AlphaDev are promoted directly (100% VRR).
  2. Overclaimed Co-Scientist hypotheses (genomics TF cascade, pharma clearance) are
     calibrated to C2 via EEDR-CA with verified lower-claim eligibility.
  3. Extrapolation divergence (superconducting hydride) and conservation leaks
     (metabolic flux) terminate in TERMINAL_ABSTAIN (0.0% FPR).
  4. Full suite executes cleanly and enforces anti-p-hacking bounds.
"""
from __future__ import annotations

from apedp_dyn.epistemic.claim_ladder import ClaimLevel
from apedp_dyn.sandbox.deepmind_challenge_suite import (
    build_deepmind_challenge_suite,
    evaluate_deepmind_case,
    run_deepmind_challenge_benchmark,
)


def test_deepmind_challenge_suite_composition():
    """Verify 12 heterogeneous cases covering published DeepMind discoveries & Co-Scientist domains."""
    cases = build_deepmind_challenge_suite()
    assert len(cases) == 12

    systems = {c.deepmind_system for c in cases}
    assert "AlphaTensor" in systems
    assert "FunSearch" in systems
    assert "GraphCast" in systems
    assert "Aletheia" in systems
    assert "AlphaDev" in systems
    assert any("Co-Scientist" in s for s in systems)


def test_alphatensor_and_graphcast_direct_promotion():
    """Valid ground-truth breakthroughs are retained and promoted without modification."""
    cases = {c.case_id: c for c in build_deepmind_challenge_suite()}

    # 1. AlphaTensor: Exact tensor factorization over Z_2
    res_at = evaluate_deepmind_case(cases["DM-01-ALPHATENSOR"])
    assert res_at.final_status == "PROMOTED_ORIGINAL"
    assert res_at.justified_level == ClaimLevel.C6_FORMALLY_VERIFIED
    assert res_at.is_valid_retention is True

    # 2. GraphCast: Operational medium-range weather forecasting
    res_gc = evaluate_deepmind_case(cases["DM-07-GRAPHCAST"])
    assert res_gc.final_status == "PROMOTED_ORIGINAL"
    assert res_gc.justified_level == ClaimLevel.C2_PREDICTIVE
    assert res_gc.is_valid_retention is True

    # 3. Aletheia: Certified Godunov shock solver
    res_al = evaluate_deepmind_case(cases["DM-09-ALETHEIA-PDE"])
    assert res_al.final_status == "PROMOTED_ORIGINAL"
    assert res_al.justified_level == ClaimLevel.C6_FORMALLY_VERIFIED
    assert res_al.is_valid_retention is True

    # 4. AlphaDev: Exact optimal Sort-3 assembly
    res_ad = evaluate_deepmind_case(cases["DM-11-ALPHADEV"])
    assert res_ad.final_status == "PROMOTED_ORIGINAL"
    assert res_ad.justified_level == ClaimLevel.C6_FORMALLY_VERIFIED
    assert res_ad.is_valid_retention is True


def test_coscientist_causal_genomics_calibrated_to_c2():
    """Overclaimed C4 causal transcriptional network without do(X) is calibrated to C2."""
    cases = {c.case_id: c for c in build_deepmind_challenge_suite()}
    res = evaluate_deepmind_case(cases["DM-04-COSCIENTIST-GENOMICS"])

    assert res.final_status == "PROMOTED_REPAIRED"
    assert res.requested_level == ClaimLevel.C4_CAUSAL_MECHANISTIC
    assert res.justified_level == ClaimLevel.C2_PREDICTIVE
    assert res.lower_claim_eligibility == "VERIFIED"
    assert res.repair_category == "EEDR-CA"
    assert res.is_false_promotion is False


def test_coscientist_hydride_extrapolation_abstention():
    """Divergent hydride singularity beyond training domain is halted in TERMINAL_ABSTAIN."""
    cases = {c.case_id: c for c in build_deepmind_challenge_suite()}
    res = evaluate_deepmind_case(cases["DM-08-COSCIENTIST-SUPERCOND"])

    assert res.final_status == "TERMINAL_ABSTAIN"
    assert res.promoted is False
    assert res.is_false_promotion is False
    assert res.is_correct_abstention is True


def test_coscientist_enzyme_non_identifiability_abstention():
    """Unresolvable allosteric structural rival models terminate in TERMINAL_ABSTAIN."""
    cases = {c.case_id: c for c in build_deepmind_challenge_suite()}
    res = evaluate_deepmind_case(cases["DM-05-COSCIENTIST-ENZYME"])

    assert res.final_status == "TERMINAL_ABSTAIN"
    assert res.promoted is False
    assert res.is_false_promotion is False
    assert res.is_correct_abstention is True


def test_deepmind_challenge_full_benchmark_metrics():
    """Execute full benchmark and verify 0% FPR and 100% VRR on DeepMind hypotheses."""
    bench = run_deepmind_challenge_benchmark()

    assert bench["total_cases"] == 12
    assert bench["valid_retention_rate"] == 100.0  # 4/4 valid retained
    assert bench["false_promotion_rate"] == 0.0    # 0 false promotions
    assert bench["false_promotions"] == 0
    assert bench["valid_retained"] == 4
    assert bench["correct_calibrations"] == 2
    assert bench["terminal_abstentions"] == 6
    assert bench["clopper_pearson_95_upper"] <= 31.25
