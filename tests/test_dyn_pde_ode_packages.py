"""Tests for public PDE and ODE subpackages (apedp.pde, apedp.ode, apedp_dyn.pde, apedp_dyn.ode, apedp_continuum).

Verifies:
1. Public package import ergonomics across all namespaces.
2. PdeGridData & PdePointCloudData containers.
3. 2D & 20D continuous PDE discovery and PdeReferee causal validation.
4. PdeProposer compliance with ProposerBase.
5. Continuous coupled ODE discovery (Lotka-Volterra) and OdeReferee validation.
6. CoupledOdeProposer compliance with ProposerBase.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_pkg_dir = Path(__file__).resolve().parent.parent / "packages" / "apedp-continuum"
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

# Test public namespaces
import apedp_continuum as ac

import apedp.ode as apedp_ode
import apedp.pde as apedp_pde
import apedp_dyn.ode as apedp_dyn_ode
import apedp_dyn.pde as apedp_dyn_pde
from apedp_dyn.epistemic.coupled_discovery import (
    CoupledEquationInfo,
    CoupledSystemDiscoveryResult,
)


def test_public_package_import_parity():
    """Verify that apedp, apedp_dyn, and apedp_continuum export the unified API."""
    assert apedp_pde.discover_pde is apedp_dyn_pde.discover_pde
    assert apedp_ode.discover_coupled_ode is apedp_dyn_ode.discover_coupled_ode
    assert ac.discover_pde is apedp_dyn_pde.discover_pde
    assert ac.discover_coupled_ode is apedp_dyn_ode.discover_coupled_ode

    assert hasattr(apedp_pde, "PdeGridData")
    assert hasattr(apedp_pde, "PdePointCloudData")
    assert hasattr(apedp_pde, "PdeProposer")
    assert hasattr(apedp_pde, "PdeReferee")

    assert hasattr(apedp_ode, "OdeTrajectoryData")
    assert hasattr(apedp_ode, "CoupledOdeProposer")
    assert hasattr(apedp_ode, "OdeReferee")


def test_pde_grid_and_point_cloud_containers():
    """Verify typed container validation."""
    u_grid = np.zeros((10, 20, 20))
    grid = apedp_pde.PdeGridData.from_array(u_grid, dt=0.01, dxs=0.05)
    assert grid.dimension == 2
    assert grid.dxs == (0.05, 0.05)

    u_anc = np.zeros((10, 30))
    u_plus = np.zeros((10, 30, 20))
    u_minus = np.zeros((10, 30, 20))
    cloud = apedp_pde.PdePointCloudData.from_axis_star(
        u_anc, u_plus, u_minus, dt=0.005, h=0.04, dimension=20
    )
    assert cloud.dimension == 20
    assert cloud.dt == 0.005
    assert cloud.h == 0.04


def test_continuous_pde_grid_discovery():
    """Verify 2D continuous heat diffusion discovery via discover_pde."""
    nt, nx, ny = 25, 20, 20
    dt, dx, dy = 0.005, 0.1, 0.1
    diff = 0.25

    u = np.zeros((nt, nx, ny))
    for t in range(nt):
        decay = np.exp(-2.0 * diff * t * dt)
        for i in range(nx):
            px = i * dx
            for j in range(ny):
                py = j * dy
                u[t, i, j] = decay * np.sin(px) * np.sin(py)

    grid_data = apedp_pde.PdeGridData.from_array(u, dt=dt, dxs=(dx, dy))
    res = apedp_pde.discover_pde(grid_data)

    assert res.r2_dev > 0.99
    assert any("laplacian" in t for t in res.active_terms)



def test_continuous_pde_point_cloud_and_referee_adjudication():
    """Verify 20D continuous heat diffusion discovery and PdeReferee causal adjudication."""
    dim, nt, m = 20, 20, 60
    dt, h = 0.004, 0.03
    alpha = 0.5

    rng_k = np.random.default_rng(2026)
    k1 = rng_k.normal(0.0, 1.0, size=dim)
    k1 = k1 / np.linalg.norm(k1) * 1.2
    k2 = rng_k.normal(0.0, 1.0, size=dim)
    k2 = k2 / np.linalg.norm(k2) * 2.5

    def law(t_val, pts):
        return (1.0 * np.exp(-alpha * np.sum(k1**2) * t_val) * np.cos(pts @ k1 + 0.3)
              + 0.8 * np.exp(-alpha * np.sum(k2**2) * t_val) * np.cos(pts @ k2 - 0.7))

    def make_cloud(seed: int):
        rng = np.random.default_rng(seed)
        anc = rng.uniform(-0.25, 0.25, size=(m, dim))
        u_a = np.zeros((nt, m))
        u_p = np.zeros((nt, m, dim))
        u_m = np.zeros((nt, m, dim))
        for ti in range(nt):
            tv = 1.0 + ti * dt
            u_a[ti] = law(tv, anc)
            for k in range(dim):
                p = anc.copy()
                p[:, k] += h
                u_p[ti, :, k] = law(tv, p)
                q = anc.copy()
                q[:, k] -= h
                u_m[ti, :, k] = law(tv, q)
        return apedp_pde.PdePointCloudData.from_axis_star(u_a, u_p, u_m, dt=dt, h=h, dimension=dim)

    dev_cloud = make_cloud(100)
    oos_cloud = make_cloud(200)

    res = apedp_pde.discover_pde(dev_cloud)
    assert any("laplacian" in t for t in res.active_terms)

    # Epistemic Referee adjudication
    referee = apedp_pde.PdeReferee(min_holdout_r2=0.95, enforce_thermodynamic_arrow=True)
    adj = referee.adjudicate_result(res, oos_cloud)

    assert adj.verdict == apedp_pde.EpistemicPdeVerdict.IDENTIFIED_CAUSAL_LAW
    assert adj.frozen_oos_r2 > 0.999


def test_pde_proposer_conforms_to_proposer_base():
    """Verify PdeProposer generates standardized CandidateProposal without holdout snooping."""
    proposer = apedp_pde.PdeProposer(dimension=2, dt=0.005, dx=0.1)
    assert proposer.proposer_id == "pde_continuum_proposer"

    nt, nx, ny = 25, 16, 16
    dt, dx, dy = 0.005, 0.1, 0.1
    diff = 0.25
    t_idx = np.repeat(np.arange(nt), nx * ny)
    x_idx = np.tile(np.repeat(np.arange(nx) * dx, ny), nt)
    y_idx = np.tile(np.tile(np.arange(ny) * dy, nx), nt)
    u_vals = np.exp(-2.0 * diff * t_idx * dt) * np.sin(x_idx) * np.sin(y_idx)

    df = pd.DataFrame({"t": t_idx, "x": x_idx, "y": y_idx, "u": u_vals})
    proposals = proposer.fit_and_propose(df, target_col="u")

    assert len(proposals) == 1
    p = proposals[0]
    assert p.proposer_id == "pde_continuum_proposer"
    assert "pde_" in p.candidate_id
    assert p.declared_scale_invariant
    scores = p.compute_score(df)
    assert len(scores) == len(df)


def test_continuous_coupled_ode_discovery():
    """Verify coupled Lotka-Volterra predator-prey system discovery."""
    dt = 0.005
    n = 200
    x = np.zeros(n)
    y = np.zeros(n)
    x[0], y[0] = 2.0, 1.0

    for i in range(n - 1):
        def f(cx: float, cy: float) -> tuple[float, float]:
            return 1.5 * cx - 1.0 * cx * cy, -3.0 * cy + 1.0 * cx * cy

        k1x, k1y = f(x[i], y[i])
        k2x, k2y = f(x[i] + 0.5 * dt * k1x, y[i] + 0.5 * dt * k1y)
        k3x, k3y = f(x[i] + 0.5 * dt * k2x, y[i] + 0.5 * dt * k2y)
        k4x, k4y = f(x[i] + dt * k3x, y[i] + dt * k3y)

        x[i + 1] = x[i] + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y[i + 1] = y[i] + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

    data = {"prey": x, "predator": y}
    traj = apedp_ode.OdeTrajectoryData.from_dict(data, dt=dt)

    res = apedp_ode.discover_coupled_ode(traj, max_polynomial_degree=2)

    assert res.is_exact_recovery
    assert res.worst_state_r2 > 0.99
    assert len(res.equations) == 2

    eq_x = next(eq for eq in res.equations if eq.state_variable == "prey")
    eq_y = next(eq for eq in res.equations if eq.state_variable == "predator")

    coeff_x = dict(zip(eq_x.active_terms, eq_x.coefficients, strict=False))
    coeff_y = dict(zip(eq_y.active_terms, eq_y.coefficients, strict=False))

    assert abs(coeff_x.get("prey", 0.0) - 1.5) < 0.05
    assert abs(coeff_x.get("prey*predator", 0.0) - (-1.0)) < 0.05
    assert abs(coeff_y.get("predator", 0.0) - (-3.0)) < 0.05
    assert abs(coeff_y.get("prey*predator", 0.0) - 1.0) < 0.05

    # OdeReferee adjudication
    referee = apedp_ode.OdeReferee(min_mean_r2=0.95, min_worst_r2=0.85)
    adj = referee.adjudicate_result(res, traj)
    assert adj.verdict == apedp_ode.EpistemicOdeVerdict.IDENTIFIED_DYNAMICAL_LAW
    assert adj.is_stable
    assert adj.mean_holdout_r2 > 0.95
    assert adj.worst_state_r2 > 0.85


def _lotka_volterra_holdout(n: int = 200, dt: float = 0.005) -> apedp_ode.OdeTrajectoryData:
    x = np.zeros(n)
    y = np.zeros(n)
    x[0], y[0] = 2.0, 1.0
    for i in range(n - 1):
        def f(cx: float, cy: float) -> tuple[float, float]:
            return 1.5 * cx - 1.0 * cx * cy, -3.0 * cy + 1.0 * cx * cy

        k1x, k1y = f(x[i], y[i])
        k2x, k2y = f(x[i] + 0.5 * dt * k1x, y[i] + 0.5 * dt * k1y)
        k3x, k3y = f(x[i] + 0.5 * dt * k2x, y[i] + 0.5 * dt * k2y)
        k4x, k4y = f(x[i] + dt * k3x, y[i] + dt * k3y)
        x[i + 1] = x[i] + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y[i + 1] = y[i] + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    return apedp_ode.OdeTrajectoryData.from_dict({"prey": x, "predator": y}, dt=dt)


def _equation(state: str, terms: list[str], coeffs: list[float]) -> CoupledEquationInfo:
    return CoupledEquationInfo(
        state_variable=state,
        formula="test",
        active_terms=terms,
        coefficients=coeffs,
        r2=0.999,
        residual_rmse=0.001,
    )


def test_ode_referee_rejects_wrong_structure_despite_in_sample_claim():
    """Regression: the referee must adjudicate from the HOLD OUT rollout, never
    from the discovery-side in-sample fit statistic."""
    wrong = CoupledSystemDiscoveryResult(
        equations=[
            _equation("prey", ["prey", "prey*predator"], [1.5, -1.0]),
            _equation("predator", ["predator", "prey*predator"], [-1.0, 1.0]),  # true value -3.0
        ],
        mean_system_r2=0.999,
        worst_state_r2=0.999,
        is_exact_recovery=False,
    )
    holdout = _lotka_volterra_holdout()
    referee = apedp_ode.OdeReferee(min_mean_r2=0.95, min_worst_r2=0.85)
    adj = referee.adjudicate_result(wrong, holdout)
    assert adj.verdict == apedp_ode.EpistemicOdeVerdict.PREDICTIVE_WRONG_STRUCTURE
    assert adj.mean_holdout_r2 < 0.95


def test_ode_referee_flags_unstable_forward_simulation():
    """Regression: divergent rollouts get the dedicated unstable verdict."""
    blow_up = CoupledSystemDiscoveryResult(
        equations=[_equation("x", ["x"], [500.0])],
        mean_system_r2=0.999,
        worst_state_r2=0.999,
        is_exact_recovery=False,
    )
    t = np.arange(50) * 0.01
    holdout = apedp_ode.OdeTrajectoryData.from_dict({"x": np.exp(-t)}, dt=0.01)
    referee = apedp_ode.OdeReferee()
    adj = referee.adjudicate_result(blow_up, holdout)
    assert adj.verdict == apedp_ode.EpistemicOdeVerdict.UNSTABLE_FORWARD_SIMULATION
    assert adj.is_stable is False


def test_ode_referee_fails_closed_on_unknown_state_reference():
    """Regression: a TERM referencing a state absent from the holdout fails closed.

    The state sets match, so this reaches term translation: a factor with no
    holdout trajectory means the discovery result is internally inconsistent,
    which must crash loudly rather than be silently dropped from the rollout.
    """
    import pytest

    bogus = CoupledSystemDiscoveryResult(
        equations=[
            _equation("prey", ["prey", "prey*predator"], [1.5, -1.0]),
            _equation("predator", ["predator", "prey*ghost"], [-1.0, 1.0]),
        ],
        mean_system_r2=0.999,
        worst_state_r2=0.999,
        is_exact_recovery=False,
    )
    holdout = _lotka_volterra_holdout(n=20)
    referee = apedp_ode.OdeReferee()
    with pytest.raises(ValueError, match="unknown state"):
        referee.adjudicate_result(bogus, holdout)


def test_ode_referee_refuses_state_set_mismatch_as_insufficient_evidence():
    """A discovered state set that differs from the holdout is an epistemic refusal.

    Distinguishing this from the raise above: a missing/extra state is a normal
    incomplete-discovery outcome (refuse with INSUFFICIENT_EVIDENCE), while a
    term referencing a non-existent state is an inconsistent result (raise).
    """
    bogus = CoupledSystemDiscoveryResult(
        equations=[_equation("ghost", ["ghost"], [1.0])],
        mean_system_r2=0.999,
        worst_state_r2=0.999,
        is_exact_recovery=False,
    )
    holdout = _lotka_volterra_holdout(n=20)
    adj = apedp_ode.OdeReferee().adjudicate_result(bogus, holdout)
    assert adj.verdict == apedp_ode.EpistemicOdeVerdict.INSUFFICIENT_EVIDENCE
    assert adj.is_stable is False
    assert "does not match the holdout state set" in adj.rational_justification


def test_coupled_ode_proposer_conforms_to_proposer_base():
    """Verify CoupledOdeProposer generates standardized CandidateProposal without holdout snooping."""
    proposer = apedp_ode.CoupledOdeProposer(dt=0.05)
    assert proposer.proposer_id == "coupled_ode_proposer"

    nt = 100
    t = np.linspace(0, 5, nt)
    x1 = np.cos(t)
    x2 = np.sin(t)
    df = pd.DataFrame({"x1": x1, "x2": x2})

    proposals = proposer.fit_and_propose(df, target_col="x1")
    assert len(proposals) >= 1
    p = proposals[0]
    assert p.proposer_id == "coupled_ode_proposer"
    assert "coupled_ode_" in p.candidate_id
    assert p.declared_scale_invariant
