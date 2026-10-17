"""Tests for 2D Spatio-Temporal PDEs, Non-Uniform Meshes, and Coupled Multi-Variable Systems.

Verifies:
1. 2D spatial Heat/Diffusion discovery on u(t, x, y) fields.
2. Exact derivative recovery on non-uniform spatial meshes via Fornberg weights.
3. Multi-variable coupled ODE discovery (Lotka-Volterra predator-prey system).
"""

from __future__ import annotations

import numpy as np

from apedp_dyn.epistemic.coupled_discovery import discover_coupled_system
from apedp_dyn.epistemic.pde_discovery import (
    discover_pde_2d,
    discover_pde_nonuniform,
)


def test_2d_heat_diffusion_continuous_field():
    """Discovers 2D spatial diffusion u_t = D * (u_xx + u_yy) from u(t, x, y)."""
    nt, nx, ny = 25, 20, 20
    dt, dx, dy = 0.005, 0.1, 0.1
    diff = 0.25

    # Exact continuous analytical solution for 2D heat: u(t, x, y) = exp(-2 D t) * sin(x) * sin(y)
    u = np.zeros((nt, nx, ny))
    for t in range(nt):
        decay = np.exp(-2.0 * diff * t * dt)
        for i in range(nx):
            px = i * dx
            for j in range(ny):
                py = j * dy
                u[t, i, j] = decay * np.sin(px) * np.sin(py)

    res = discover_pde_2d(u, dt, dx, dy, sparsity_threshold=0.02)
    assert res.r2_dev > 0.99, f"2D Heat R2={res.r2_dev}"
    assert res.is_exact_recovery

    # Active terms must contain diffusion (laplacian or u_xx/u_yy)
    has_diff = any(t in res.active_terms for t in ("laplacian(u)", "u_xx", "u_yy"))
    assert has_diff, f"Active terms: {res.active_terms}"


def test_nonuniform_spatial_mesh_pde_discovery():
    """Recovers governing PDE on non-uniform, irregularly spaced spatial and temporal coordinates."""
    rng = np.random.RandomState(42)
    # Non-uniform spatial coordinates (e.g. boundary layer clustering)
    x_raw = np.concatenate([
        np.linspace(0.0, 1.0, 10),
        np.linspace(1.1, 4.0, 10),
        np.linspace(4.05, 5.0, 10),
    ])
    x_coords = np.sort(np.unique(x_raw))
    t_coords = np.sort(rng.uniform(0.0, 0.5, 25))

    # Planted decay-diffusion field: u(t, x) = exp(-0.5 * t) * sin(x)
    u_grid = np.zeros((len(t_coords), len(x_coords)))
    for ti, t in enumerate(t_coords):
        decay = np.exp(-0.5 * t)
        for xi, x in enumerate(x_coords):
            u_grid[ti, xi] = decay * np.sin(x)

    res = discover_pde_nonuniform(u_grid, t_coords, x_coords, sparsity_threshold=0.02)
    assert res.r2_dev > 0.98, f"Non-uniform PDE R2={res.r2_dev}"
    assert "u" in res.active_terms or "u_xx" in res.active_terms


def test_coupled_lotka_volterra_system_discovery():
    """Discovers coupled multi-variable non-linear ODE system (Predator-Prey)."""
    # dx/dt = 1.5 * x - 1.0 * x * y
    # dy/dt = -3.0 * y + 1.0 * x * y
    dt = 0.005
    n = 200
    x = np.zeros(n)
    y = np.zeros(n)
    x[0], y[0] = 2.0, 1.0

    for i in range(n - 1):
        # Forward RK4 integration
        def f(cx: float, cy: float) -> tuple[float, float]:
            return 1.5 * cx - 1.0 * cx * cy, -3.0 * cy + 1.0 * cx * cy

        k1x, k1y = f(x[i], y[i])
        k2x, k2y = f(x[i] + 0.5 * dt * k1x, y[i] + 0.5 * dt * k1y)
        k3x, k3y = f(x[i] + 0.5 * dt * k2x, y[i] + 0.5 * dt * k2y)
        k4x, k4y = f(x[i] + dt * k3x, y[i] + dt * k3y)

        x[i + 1] = x[i] + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y[i + 1] = y[i] + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

    data = {"prey": x, "predator": y}
    res = discover_coupled_system(data, dt)

    assert res.is_exact_recovery
    assert res.worst_state_r2 > 0.99
    assert len(res.equations) == 2

    # Verify Equation 1 (prey): dprey/dt = 1.5*prey - 1.0*prey*predator
    eq1 = res.equations[0]
    assert "prey" in eq1.active_terms
    assert "prey*predator" in eq1.active_terms
    p_idx = eq1.active_terms.index("prey")
    inter_idx = eq1.active_terms.index("prey*predator")
    assert abs(eq1.coefficients[p_idx] - 1.5) < 0.05
    assert abs(eq1.coefficients[inter_idx] - (-1.0)) < 0.05

    # Verify Equation 2 (predator): dpredator/dt = -3.0*predator + 1.0*prey*predator
    eq2 = res.equations[1]
    assert "predator" in eq2.active_terms
    assert "prey*predator" in eq2.active_terms
    pred_idx = eq2.active_terms.index("predator")
    inter2_idx = eq2.active_terms.index("prey*predator")
    assert abs(eq2.coefficients[pred_idx] - (-3.0)) < 0.05
    assert abs(eq2.coefficients[inter2_idx] - 1.0) < 0.05
