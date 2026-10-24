"""Regression Suite for 9/10 Architectural Frontier Capabilities.

Verifies:
1. Nonlinear Riccati Jet ODE Discovery: Recovers exact Planck Radiance ODE
   v^3 u' + (v^3 - 3v^2) u + u^2 = 0 at R2 > 0.9999.
2. Second-Order Jet ODE Discovery: Recovers Harmonic Oscillator y'' + 4y = 0.
3. Spatio-Temporal Field PDE Discovery: Recovers Heat/Diffusion and Viscous Burgers PDEs.
4. Heterogeneous Multi-Stage Composition: Chained TV Jump + Smooth Polynomial Pipeline.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import apedp_kernel as ak
from apedp_dyn.epistemic.pde_discovery import discover_pde
from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer


def test_planck_riccati_ode_discovery_in_python():
    """Verify nonlinear Riccati Jet relation discovery on Planck Radiance."""
    n = 80
    v = np.linspace(0.3, 5.0, n)
    u = (v ** 3) / (np.exp(v) - 1.0)

    cfg = {
        "max_x_degree": 3,
        "max_diff_order": 1,
        "smooth_window": 5,
        "sparsity_threshold": 0.02,
        "rcond_threshold": 1e-12,
        "include_riccati": True,
    }
    res = json.loads(
        ak.rime_diff_pade_discover_json(
            json.dumps(v.tolist()), json.dumps(u.tolist()), json.dumps(cfg)
        )
    )

    assert res["integrated_trajectory_r2"] > 0.9999, f"Traj R2: {res['integrated_trajectory_r2']}"
    assert res["jet_rmse"] < 0.01, f"Jet RMSE: {res['jet_rmse']}"
    assert "y^2" in res["formula"], f"Expected Riccati quadratic term in formula: {res['formula']}"


def test_harmonic_oscillator_second_order_ode_in_python():
    """Verify second-order Jet ODE discovery: y'' + 4y = 0."""
    n = 70
    t = np.linspace(0.1, 3.0, n)
    y = np.sin(2.0 * t)

    cfg = {
        "max_x_degree": 0,
        "max_diff_order": 2,
        "smooth_window": 5,
        "sparsity_threshold": 0.05,
        "rcond_threshold": 1e-12,
        "include_riccati": False,
    }
    res = json.loads(
        ak.rime_diff_pade_discover_json(
            json.dumps(t.tolist()), json.dumps(y.tolist()), json.dumps(cfg)
        )
    )

    assert "y''" in res["formula"], f"Expected 2nd order y'' in formula: {res['formula']}"
    assert res["integrated_trajectory_r2"] > 0.99, f"Traj R2: {res['integrated_trajectory_r2']}"
    assert res["is_exact_recovery"]


def test_heat_equation_spatio_temporal_pde_discovery():
    """Verify continuous field discovery on 1D Heat Equation: u_t = 0.5 * u_xx."""
    nx, nt = 50, 40
    dx, dt = 0.1, 0.005
    u = np.zeros((nt, nx))
    x = np.linspace(0, (nx - 1) * dx, nx)
    u[0, :] = np.sin(x) + 0.5 * np.sin(2.0 * x)
    D = 0.5
    for i in range(nt - 1):
        for j in range(1, nx - 1):
            u[i + 1, j] = u[i, j] + dt * D * (u[i, j + 1] - 2.0 * u[i, j] + u[i, j - 1]) / (dx ** 2)
        u[i + 1, 0] = u[i + 1, 1]
        u[i + 1, -1] = u[i + 1, -2]

    res = discover_pde(u, dt, dx, holdout_ratio=0.25)
    assert res.is_exact_recovery
    assert res.r2_dev > 0.995
    assert res.holdout_r2 > 0.995
    assert res.active_terms == ["u_xx"]
    assert abs(res.coefficients[0] - 0.5) < 0.05


def test_viscous_burgers_equation_spatio_temporal_pde_discovery():
    """Verify continuous field discovery on Viscous Burgers: u_t = -1.0 * u*u_x + 0.1 * u_xx."""
    nx, nt = 60, 50
    dx, dt = 0.1, 0.002
    u = np.zeros((nt, nx))
    x = np.linspace(0, (nx - 1) * dx, nx)
    u[0, :] = np.sin(x)
    nu = 0.1
    for i in range(nt - 1):
        for j in range(1, nx - 1):
            adv = -u[i, j] * (u[i, j + 1] - u[i, j - 1]) / (2.0 * dx)
            diff = nu * (u[i, j + 1] - 2.0 * u[i, j] + u[i, j - 1]) / (dx ** 2)
            u[i + 1, j] = u[i, j] + dt * (adv + diff)
        u[i + 1, 0] = u[i + 1, 1]
        u[i + 1, -1] = u[i + 1, -2]

    res = discover_pde(u, dt, dx, holdout_ratio=0.25)
    assert res.is_exact_recovery
    assert res.r2_dev > 0.999
    assert res.holdout_r2 > 0.999
    assert "u*u_x" in res.active_terms
    assert "u_xx" in res.active_terms
    adv_idx = res.active_terms.index("u*u_x")
    diff_idx = res.active_terms.index("u_xx")
    assert abs(res.coefficients[adv_idx] - (-1.0)) < 0.05
    assert abs(res.coefficients[diff_idx] - 0.1) < 0.05


def test_heterogeneous_tv_smooth_composite_pipeline():
    """Verify multi-stage pipeline: TV Staircase + Smooth Polynomial decomposition."""
    x = np.linspace(0.2, 8.0, 80)
    # Quantum steps: step size 2.0, width 1.5, offset 0.0
    y_jump = 2.0 * np.floor(x / 1.5)
    # Smooth background: 0.5 * x + 0.08 * x^2
    y_smooth = 0.5 * x + 0.08 * (x ** 2)
    y = y_jump + y_smooth

    df = pd.DataFrame({"x": x, "target": y})
    proposer = RimeAdProposer(target_col="target")
    candidates = proposer.fit_and_propose(df)

    comp_cands = [c for c in candidates if c.candidate_id == "tv_smooth_composite"]
    assert len(comp_cands) > 0, "Proposer must emit heterogeneous TV+Smooth composite candidate"
    best_comp = comp_cands[0]
    assert best_comp.metadata["r2"] > 0.99
    pred = best_comp.compute_score(df)
    r2_fit = 1.0 - (np.mean((y - pred) ** 2) / np.var(y))
    assert r2_fit > 0.99
