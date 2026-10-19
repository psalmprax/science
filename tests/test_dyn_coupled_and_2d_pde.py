"""Tests for 2D Spatio-Temporal PDEs, Non-Uniform Meshes, and Coupled Multi-Variable Systems.

Verifies:
1. 2D spatial Heat/Diffusion discovery on u(t, x, y) fields.
2. Exact derivative recovery on non-uniform spatial meshes via Fornberg weights.
3. Multi-variable coupled ODE discovery (Lotka-Volterra predator-prey system).
4. The Python Fornberg mirror of the kernel's stencil library.
5. ODE referee adjudication of monomials with integer power terms.
"""

from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.epistemic.coupled_discovery import discover_coupled_system
from apedp_dyn.epistemic.pde_discovery import (
    _fornberg_centered_weights,
    discover_pde_2d,
    discover_pde_nonuniform,
)
from apedp_dyn.ode.data import OdeTrajectoryData
from apedp_dyn.ode.referee import (
    EpistemicOdeVerdict,
    OdeReferee,
    _parse_monomial_factors,
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


def test_nonuniform_discovery_never_fits_holdout_rows():
    """Discovery must run on the develop partition only.

    The planted field changes its governing decay rate inside the holdout
    window (0.5 -> 3.0). Discovery run on the develop partition must recover
    the dev law, and the holdout must only ever be SCORED: corrupting the
    holdout rows cannot change the discovered formula or its coefficients,
    while the holdout R^2 must move (it is computed, not copied from r2_dev).
    """
    rng = np.random.default_rng(7)
    n_x = 24
    x_coords = np.sort(rng.uniform(0.0, 2.0 * np.pi, n_x))
    t_coords = np.sort(rng.uniform(0.0, 1.0, 40))
    n_dev = max(5, int(len(t_coords) * 0.75))

    u = np.zeros((len(t_coords), n_x))
    for ti, t in enumerate(t_coords):
        rate = 0.5 if ti < n_dev else 3.0
        u[ti, :] = np.exp(-rate * t) * np.sin(x_coords)

    res = discover_pde_nonuniform(u, t_coords, x_coords, sparsity_threshold=0.02)
    assert res.r2_dev > 0.98, f"dev R2={res.r2_dev}"
    assert np.isfinite(res.holdout_r2)

    # The frozen operator generalizes worse on the faster-decay holdout, so the
    # holdout score is demonstrably NOT a copy of the in-sample score.
    assert res.holdout_r2 < res.r2_dev - 0.05, (res.holdout_r2, res.r2_dev)

    # Anti-snooping: destroying the holdout rows leaves discovery untouched.
    u_corrupt = u.copy()
    u_corrupt[n_dev:, :] *= 100.0
    res2 = discover_pde_nonuniform(u_corrupt, t_coords, x_coords, sparsity_threshold=0.02)
    assert res2.formula == res.formula
    assert res2.coefficients == pytest.approx(res.coefficients)
    assert res2.active_terms == res.active_terms
    # ...while the holdout score does move (it really is scored on those rows).
    assert res2.holdout_r2 != pytest.approx(res.holdout_r2)


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


# ---------------------------------------------------------------------------
# Python Fornberg mirror (apes the kernel's fornberg_weights for frozen
# holdout scoring). Guarded against the Vandermonde-orientation bug: solving
# vander @ w = rhs (the interpolation problem) instead of vander.T @ w = rhs
# produced garbage weights with magnitudes ~1/dt**2.
# ---------------------------------------------------------------------------


def test_fornberg_centered_weights_uniform_classic_stencils():
    """On a uniform grid the mirror must reproduce the classic centered stencils."""
    dt = 0.02
    w_t = _fornberg_centered_weights([-dt, 0.0, dt], 0.0, 1)
    # Centered first difference: (f(+dt) - f(-dt)) / (2 dt), zero weight at center.
    assert w_t[1] == pytest.approx([-1.0 / (2.0 * dt), 0.0, 1.0 / (2.0 * dt)])

    dx = 0.1
    w_x = _fornberg_centered_weights([-2 * dx, -dx, 0.0, dx, 2 * dx], 0.0, 2)
    # 4th-order 5-point first derivative: (f(-2) - 8f(-1) + 8f(+1) - f(+2)) / (12 dx).
    assert w_x[1] == pytest.approx(
        [1.0 / (12.0 * dx), -8.0 / (12.0 * dx), 0.0, 8.0 / (12.0 * dx), -1.0 / (12.0 * dx)]
    )
    # 4th-order 5-point second derivative: (-f(-2) + 16f(-1) - 30f(0) + 16f(+1) - f(+2)) / (12 dx^2).
    assert w_x[2] == pytest.approx(
        [
            -1.0 / (12.0 * dx**2),
            16.0 / (12.0 * dx**2),
            -30.0 / (12.0 * dx**2),
            16.0 / (12.0 * dx**2),
            -1.0 / (12.0 * dx**2),
        ]
    )


def test_fornberg_centered_weights_nonuniform_polynomial_exactness():
    """Moment-system exactness: degree <= 4 polynomials must be differentiated exactly
    on arbitrarily spaced nodes (this is what makes the frozen holdout score
    trustworthy on genuinely irregular meshes)."""
    rng = np.random.default_rng(3)
    x_sub = np.sort(rng.uniform(-1.5, 1.5, 5))
    z = 0.37
    w = _fornberg_centered_weights(x_sub, z, 2)

    # f(x) = x^4 -> f'(x) = 4x^3, f''(x) = 12x^2.
    vals = x_sub**4
    assert float(w[1] @ vals) == pytest.approx(4.0 * z**3, abs=1e-10)
    assert float(w[2] @ vals) == pytest.approx(12.0 * z**2, abs=1e-10)

    # Mixed-degree polynomial: f(x) = x^2 - 3x -> f' = 2x - 3, f'' = 2.
    vals = x_sub**2 - 3.0 * x_sub
    assert float(w[1] @ vals) == pytest.approx(2.0 * z - 3.0, abs=1e-10)
    assert float(w[2] @ vals) == pytest.approx(2.0, abs=1e-10)


def test_fornberg_centered_weights_nonuniform_smooth_function_accuracy():
    """On a smooth non-polynomial the non-uniform 5-point mirror must hit the
    analytic derivative to 4th-order accuracy (far better than a naive
    finite difference on the same spacing)."""
    rng = np.random.default_rng(11)
    # Nodes spread over a small window so the 4th-order truncation error of a
    # 5-point stencil stays below 1e-4 relative (h_effective ~ 0.05).
    x_sub = np.sort(1.3 + rng.uniform(-0.1, 0.1, 5))
    z = 1.3
    w = _fornberg_centered_weights(x_sub, z, 2)

    vals = np.sin(3.0 * x_sub)
    numeric_first = float(w[1] @ vals)
    numeric_second = float(w[2] @ vals)
    assert numeric_first == pytest.approx(3.0 * np.cos(3.0 * z), rel=1e-4)
    assert numeric_second == pytest.approx(-9.0 * np.sin(3.0 * z), rel=1e-4)

    # The Vandermonde-orientation bug produced weights with magnitudes that
    # scale like 1/dx^2 even for the FIRST derivative; with dx ~ 0.05, a wrong
    # orientation blows the first derivative up by orders of magnitude.
    assert abs(numeric_first) < 10.0, "first-derivative weights are not O(1/dx)"


def test_frozen_nonuniform_holdout_exact_recovery():
    """End-to-end: a field whose law holds everywhere must score holdout R^2 ~ 1
    through the Python mirror. Under the stencil-orientation bug the frozen
    holdout score collapsed to negative R^2 for the same data."""
    nt, nx = 60, 24
    dt, dx = 0.01, 0.05
    t_coords = np.arange(nt) * dt
    x_coords = np.arange(nx) * dx
    u = 2.0 * np.exp(-0.5 * t_coords)[:, None] * np.sin(2.0 * np.pi * x_coords)[None, :]

    res = discover_pde_nonuniform(u, t_coords, x_coords, sparsity_threshold=0.02)
    assert res.r2_dev > 0.999
    assert "u" in res.active_terms or "u_xx" in res.active_terms
    assert res.holdout_r2 > 0.99, f"frozen holdout R2={res.holdout_r2}"
    assert res.holdout_passed


# ---------------------------------------------------------------------------
# ODE referee: monomial power terms. The discovered system library includes
# degree-2 terms ("x^2", "x*x1", ...); the referee must evaluate them, refuse
# malformed powers fail-closed, and never crash on a well-formed candidate.
# ---------------------------------------------------------------------------


def test_parse_monomial_factors_accepts_powers_and_products():
    name_to_index = {"x": 0, "y": 1}
    assert _parse_monomial_factors("1", name_to_index, "x") == []
    assert _parse_monomial_factors("x", name_to_index, "x") == [("x", 1)]
    assert _parse_monomial_factors("x^2", name_to_index, "x") == [("x", 2)]
    assert _parse_monomial_factors("x^2*y", name_to_index, "x") == [("x", 2), ("y", 1)]
    assert _parse_monomial_factors("x*y^3", name_to_index, "x") == [("x", 1), ("y", 3)]


def test_parse_monomial_factors_fail_closed_on_malformed_terms():
    name_to_index = {"x": 0, "y": 1}
    with pytest.raises(ValueError, match="non-integer power"):
        _parse_monomial_factors("x^1.5", name_to_index, "x")
    with pytest.raises(ValueError, match="non-positive power"):
        _parse_monomial_factors("x^0", name_to_index, "x")
    with pytest.raises(ValueError, match="unknown state"):
        _parse_monomial_factors("z^2", name_to_index, "x")


def _power_term_trajectory(n: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """RK4 trajectory of the coupled system with genuine power terms:
    dx/dt = 0.6 x - 0.3 x^2,  dy/dt = -0.5 y + 0.4 x^2."""
    xs, ys = np.empty(n), np.empty(n)
    xs[0], ys[0] = 0.5, 0.2

    def f(x: float, y: float) -> tuple[float, float]:
        return 0.6 * x - 0.3 * x * x, -0.5 * y + 0.4 * x * x

    for k in range(n - 1):
        x, y = xs[k], ys[k]
        k1x, k1y = f(x, y)
        k2x, k2y = f(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
        k3x, k3y = f(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
        k4x, k4y = f(x + dt * k3x, y + dt * k3y)
        xs[k + 1] = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        ys[k + 1] = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    return xs, ys


def test_ode_referee_adjudicates_power_term_system():
    """End-to-end on a genuinely quadratic coupled law: discovery emits 'x^2',
    the referee must roll it out on the holdout and promote (previously it
    crashed on the unhandled '^' syntax before reaching any verdict)."""
    dt, n_total = 0.02, 600
    xs, ys = _power_term_trajectory(n_total, dt)
    n_dev = 400
    names = ("x", "y")
    dev = OdeTrajectoryData(
        t=np.arange(n_dev) * dt,
        states={"x": xs[:n_dev], "y": ys[:n_dev]},
        dt=dt,
    )
    hold = OdeTrajectoryData(
        t=np.arange(n_total - n_dev) * dt,
        states={"x": xs[n_dev:], "y": ys[n_dev:]},
        dt=dt,
    )

    res = discover_coupled_system({n: dev.states[n] for n in names}, dt, labels=names)
    eq_x = next(e for e in res.equations if e.state_variable == "x")
    eq_y = next(e for e in res.equations if e.state_variable == "y")
    assert "x^2" in eq_x.active_terms, eq_x.active_terms
    assert "x^2" in eq_y.active_terms, eq_y.active_terms
    coeff_x = dict(zip(eq_x.active_terms, eq_x.coefficients, strict=False))
    coeff_y = dict(zip(eq_y.active_terms, eq_y.coefficients, strict=False))
    assert coeff_x["x"] == pytest.approx(0.6, abs=0.05)
    assert coeff_x["x^2"] == pytest.approx(-0.3, abs=0.05)
    assert coeff_y["y"] == pytest.approx(-0.5, abs=0.05)
    assert coeff_y["x^2"] == pytest.approx(0.4, abs=0.05)

    adj = OdeReferee(min_mean_r2=0.95, min_worst_r2=0.85).adjudicate_result(res, hold)
    assert adj.verdict == EpistemicOdeVerdict.IDENTIFIED_DYNAMICAL_LAW, adj.rational_justification
    assert adj.is_stable
    assert adj.mean_holdout_r2 > 0.95


def test_ode_referee_fails_closed_on_malformed_power_term():
    """A non-integer power is a malformed candidate: fail closed with a verdict,
    never crash with an unhandled exception (referee contract)."""
    from apedp_dyn.epistemic.coupled_discovery import (
        CoupledEquationInfo,
        CoupledSystemDiscoveryResult,
    )

    bad = CoupledSystemDiscoveryResult(
        equations=[
            CoupledEquationInfo(
                state_variable="x",
                active_terms=("x^1.5",),
                coefficients=(1.0,),
                formula="dx/dt = x^1.5",
                residual_rmse=0.0,
                r2=1.0,
            )
        ],
        mean_system_r2=1.0,
        worst_state_r2=1.0,
        is_exact_recovery=True,
    )
    hold = OdeTrajectoryData(
        t=np.array([0.0, 1.0]), states={"x": np.array([1.0, 1.1])}, dt=1.0
    )
    with pytest.raises(ValueError, match="non-integer power"):
        OdeReferee().adjudicate_result(bad, hold)
