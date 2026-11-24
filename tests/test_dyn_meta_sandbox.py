"""Unit tests for Unified APEDP-X Cross-Domain Meta-Sandbox."""


import pytest

from apedp_dyn.sandbox import apedp_x_meta_sandbox as meta
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


# --- The clean packing case earns the apex tier by measurement --------------
#
# FLAW-07 made C6 require rival rejection and interventional invariance. The
# clean Kepler case was handed its evidence by case number and was never
# handed those two, so the scorecard above read 50 % retention (with the PDE
# case, fixed in the PDE sandbox) from the first commit in this history.

def _clean_kepler():
    return next(e for e in evaluate_domain_c_optimality() if e.scenario_id == "clean_kepler")


def test_the_kepler_support_is_measured_on_real_packings():
    support = meta.measure_kepler_claim_support()
    assert support["invariant"] is True and support["rivals_rejected"] is True
    assert len(support["interventions"]) == 9
    assert {r["lattice"]: round(r["density"], 4) for r in support["rivals"]} == {"hcp": 0.7405, "bcc": 0.6802}


def test_the_clean_kepler_case_is_promoted_at_the_apex():
    assert _clean_kepler().apedp_promoted is True


@pytest.mark.parametrize("withheld", ["invariant", "rivals_rejected"])
def test_the_apex_is_refused_when_a_measurement_fails(monkeypatch, withheld):
    real = meta.measure_kepler_claim_support()
    monkeypatch.setattr(meta, "measure_kepler_claim_support", lambda: {**real, withheld: False})
    case = _clean_kepler()
    assert case.apedp_promoted is False
    assert case.apedp_verdict == "CLAIM_DOWNGRADED"


def test_a_denser_feasible_rival_would_be_noticed(monkeypatch):
    """The rival check reads densities; it is not a constant."""
    fcc = meta.generate_fcc_3d(n_cells=3)

    class _Denser:
        density = fcc.density + 0.01

    monkeypatch.setattr(meta, "generate_bcc_3d", lambda n_cells=3: _Denser())
    monkeypatch.setattr(meta, "audit_feasibility", lambda packing: (True, 0.0, "ok"))
    assert meta.measure_kepler_claim_support()["rivals_rejected"] is False


def test_a_size_dependent_density_would_be_noticed(monkeypatch):
    real = meta.generate_fcc_3d

    class _Drifting:
        def __init__(self, n_cells):
            self.density = real(n_cells=3).density - 0.001 * (n_cells - 2)

    monkeypatch.setattr(meta, "generate_fcc_3d", lambda n_cells=3, radius=0.5: _Drifting(n_cells))
    monkeypatch.setattr(meta, "audit_feasibility", lambda packing: (True, 0.0, "ok"))
    assert meta.measure_kepler_claim_support()["invariant"] is False
