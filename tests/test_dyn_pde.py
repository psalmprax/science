"""Unit tests for APEDP-X Domain D: Computational Physics Sandbox."""


import pytest

from apedp_dyn.epistemic.claim_ladder import EpistemicAdjudicationVerdict
from apedp_dyn.sandbox import pde_verification_sandbox as pde
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


# --- Interventional evidence for the apex tier (FLAW-07 follow-through) -----
#
# FLAW-07 made C6 require interventional invariance. This sandbox never
# supplied it, so its one valid case was downgraded to C5 and the two tests
# above failed from the first commit in the repository's history. Supplying it
# by trap number would have been a fiat; measuring it showed the "verified"
# scheme was not invariant at all.

@pytest.mark.parametrize("name, change", pde.PDE_INTERVENTIONS)
def test_the_time_step_respects_advection_and_diffusion_together(name, change):
    """The explicit update is stable when u*dt/dx + 2*nu*dt/dx**2 <= 1.

    The step used to come from the advective term alone. The default run sat
    at 0.995 by coincidence, and `nu doubled`, `grid 200` and `cfl 0.9` each
    put it past 1.
    """
    nu = float(change.get("nu", 0.01))
    sol = simulate_burgers_conservative(trap_id=6, **change)
    u_max = float(abs(sol.u[0]).max())
    stability_number = u_max * sol.dt / sol.dx + 2.0 * nu * sol.dt / sol.dx**2
    assert stability_number == pytest.approx(float(change.get("cfl", 0.5)))
    assert stability_number <= 1.0


def test_the_verified_scheme_keeps_its_invariants_under_every_intervention():
    audit = pde.audit_interventional_invariance(trap_id=6)
    assert audit["failed"] == []
    assert audit["invariant"] is True
    assert len(audit["runs"]) == len(pde.PDE_INTERVENTIONS)


@pytest.mark.parametrize("trap_id", [1, 2, 3, 4])
def test_a_defective_scheme_does_not_earn_interventional_evidence(trap_id):
    assert pde.audit_interventional_invariance(trap_id=trap_id)["invariant"] is False


def test_the_apex_claim_is_refused_when_an_intervention_breaks_the_scheme(monkeypatch):
    """The evidence is measured: take the measurement away and C6 goes with it."""
    monkeypatch.setattr(
        pde, "audit_interventional_invariance",
        lambda trap_id: {"invariant": False, "failed": ["grid 200"], "runs": []},
    )
    adjudication = run_pde_adjudication(trap_id=6)["apedp_adjudication"]
    assert adjudication["verdict"] == EpistemicAdjudicationVerdict.CLAIM_DOWNGRADED.value
    assert adjudication["missing_evidence"] == ["interventional_invariance"]


def test_passing_the_interventions_alone_does_not_promote():
    """Trap 5 survives every intervention and is still not a C6 result."""
    res = run_pde_adjudication(trap_id=5)
    assert res["interventional_audit"]["invariant"] is True
    assert res["apedp_adjudication"]["verdict"] != EpistemicAdjudicationVerdict.PROMOTED.value
