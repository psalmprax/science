"""Unit tests for APEDP-X Domain B: Causal Discovery Sandbox."""


from apedp_dyn.epistemic.claim_ladder import EpistemicAdjudicationVerdict
from apedp_dyn.sandbox.causal_discovery_sandbox import (
    audit_interventional_invariance,
    fit_observational_regression,
    generate_scm_data,
    run_causal_adjudication,
    run_causal_benchmark_matrix,
)


def test_scm_generator_creates_confounded_data():
    scm = generate_scm_data(graph_type="confounded", n_samples=500)
    assert len(scm.X) == 500
    assert len(scm.Y) == 500
    assert scm.true_causal_effect == 0.0
    reg = fit_observational_regression(scm.X, scm.Y)
    # Confounding induces high observational R2 despite zero true causal effect
    assert reg["r2"] > 0.50
    assert abs(reg["beta"]) > 0.50


def test_interventional_audit_detects_confounding():
    scm = generate_scm_data(graph_type="confounded", n_samples=500)
    reg = fit_observational_regression(scm.X, scm.Y)
    audit = audit_interventional_invariance(scm, claimed_beta=reg["beta"])
    assert audit["has_interventions"] is True
    assert audit["is_invariant"] is False
    assert abs(audit["interv_beta"]) < 0.25  # True do(X) effect is 0.0


def test_trap1_hidden_confounder_abstains():
    res = run_causal_adjudication(trap_id=1, provide_interventions=False)
    # Naive regression promotes due to high R2
    assert res["naive_promoted"] is True
    # APEDP referee refuses to promote to C4 Causal without do(X) evidence
    assert res["apedp_adjudication"]["claim_inflated"] is True
    assert res["apedp_adjudication"]["verdict"] in [
        EpistemicAdjudicationVerdict.CAUSAL_UNRESOLVED.value,
        EpistemicAdjudicationVerdict.PREDICTIVE_ONLY.value,
    ]


def test_trap6_direct_causation_promoted_with_interventions():
    res = run_causal_adjudication(trap_id=6, provide_interventions=True)
    assert res["interv_audit"]["is_invariant"] is True
    assert res["apedp_adjudication"]["claim_inflated"] is False
    assert res["apedp_adjudication"]["verdict"] == EpistemicAdjudicationVerdict.PROMOTED.value


def test_causal_benchmark_scorecard_passes():
    scorecard = run_causal_benchmark_matrix()
    assert scorecard["naive_fpr"] > 50.0
    assert scorecard["apedp_fpr"] == 0.0
    assert scorecard["car"] >= 80.0
    assert scorecard["sfcr"] == 100.0
    assert scorecard["valid_discoveries_retained"] == 1
