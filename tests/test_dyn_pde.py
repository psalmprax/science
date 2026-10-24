"""Unit tests for APEDP-X Domain D: Computational Physics Sandbox."""


from apedp_dyn.epistemic.claim_ladder import EpistemicAdjudicationVerdict
from apedp_dyn.sandbox.pde_verification_sandbox import (
    audit_physical_invariants,
    run_pde_adjudication,
    run_pde_benchmark_matrix,
    simulate_burgers_conservative,
)


def test_simulate_burgers_clean_preserves_conservation():
    sol = simulate_burgers_conservative(trap_id=6, n_t=40)
    audit = audit_physical_invariants(sol, trap_id=6)
    assert audit["mass_conserved"] is True
    assert audit["energy_dissipative"] is True
    assert audit["entropy_satisfied"] is True
    assert audit["cfl_stable"] is True


def test_trap1_conservation_leakage_detected():
    res = run_pde_adjudication(trap_id=1)
    assert res["audit"]["mass_conserved"] is False
    assert res["apedp_adjudication"]["claim_inflated"] is True
    assert res["apedp_adjudication"]["verdict"] in [
        EpistemicAdjudicationVerdict.PHYSICALLY_INVALID.value,
        EpistemicAdjudicationVerdict.CLAIM_DOWNGRADED.value,
    ]


def test_trap2_unphysical_expansion_shock_detected():
    res = run_pde_adjudication(trap_id=2)
    assert res["audit"]["entropy_satisfied"] is False
    assert res["apedp_adjudication"]["verdict"] == EpistemicAdjudicationVerdict.PHYSICALLY_INVALID.value
    assert "entropy" in res["apedp_adjudication"]["justification"].lower()


def test_trap4_cfl_instability_blowup_detected():
    res = run_pde_adjudication(trap_id=4)
    assert res["audit"]["cfl_stable"] is False
    assert res["apedp_adjudication"]["verdict"] in [
        EpistemicAdjudicationVerdict.NUMERICALLY_UNSTABLE.value,
        EpistemicAdjudicationVerdict.PHYSICALLY_INVALID.value,
    ]


def test_trap6_verified_godunov_promoted():
    res = run_pde_adjudication(trap_id=6)
    assert res["apedp_adjudication"]["claim_inflated"] is False
    assert res["apedp_adjudication"]["verdict"] == EpistemicAdjudicationVerdict.PROMOTED.value


def test_pde_benchmark_scorecard_passes():
    scorecard = run_pde_benchmark_matrix()
    assert scorecard["naive_fpr"] > 50.0
    assert scorecard["apedp_fpr"] == 0.0
    assert scorecard["car"] >= 80.0
    assert scorecard["sfcr"] == 100.0
    assert scorecard["valid_discoveries_retained"] == 1
