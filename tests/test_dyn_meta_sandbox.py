"""Unit tests for Unified APEDP-X Cross-Domain Meta-Sandbox."""


from apedp_dyn.sandbox.apedp_x_meta_sandbox import (
    evaluate_domain_a_dynamical,
    evaluate_domain_b_causal,
    evaluate_domain_c_optimality,
    evaluate_domain_d_physics,
    run_apedpx_unified_meta_benchmark,
)


def test_meta_domain_a_evaluations():
    evals = evaluate_domain_a_dynamical()
    assert len(evals) >= 4
    for e in evals:
        if e.claim_inflated:
            assert e.apedp_promoted is False


def test_meta_domain_b_evaluations():
    evals = evaluate_domain_b_causal()
    assert len(evals) >= 4
    for e in evals:
        if e.claim_inflated:
            assert e.apedp_promoted is False


def test_meta_domain_c_evaluations():
    evals = evaluate_domain_c_optimality()
    assert len(evals) >= 4
    for e in evals:
        if e.claim_inflated:
            assert e.apedp_promoted is False


def test_meta_domain_d_evaluations():
    evals = evaluate_domain_d_physics()
    assert len(evals) >= 4
    for e in evals:
        if e.claim_inflated:
            assert e.apedp_promoted is False


def test_master_epistemic_scorecard_passes():
    scorecard = run_apedpx_unified_meta_benchmark()
    assert scorecard["naive_fpr"] > 60.0
    assert scorecard["apedp_fpr"] == 0.0
    assert scorecard["car"] == 100.0
    assert scorecard["sfcr"] == 100.0
    assert scorecard["valid_retention_rate"] == 100.0
