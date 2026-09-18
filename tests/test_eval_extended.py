"""Tests for extended evaluation: implicit, quadrature, ODE IVP, PDE residual, heat."""
from __future__ import annotations

import numpy as np
import pytest

from apedp.eval_extended import (
    evaluate_implicit,
    evaluate_ode_ivp,
    evaluate_pde_residual,
    evaluate_quadrature,
    solve_heat_1d_explicit,
)


def test_implicit_sqrt2():
    assert evaluate_implicit("x**2 - 2", "x", 1.0) == pytest.approx(1.41421356, rel=1e-6)


def test_implicit_no_real_root_raises():
    with pytest.raises(ArithmeticError):
        evaluate_implicit("x**2 + 1", "x", 1.0)


def test_implicit_missing_var_raises():
    with pytest.raises(ValueError, match="not in expression"):
        evaluate_implicit("x + 1", "y", 0.0)


def test_quadrature_x_squared():
    assert evaluate_quadrature("x**2", "x", 0.0, 1.0) == pytest.approx(1 / 3, rel=1e-4)


def test_quadrature_bad_bounds_raises():
    with pytest.raises(ValueError):
        evaluate_quadrature("x", "x", 1.0, 0.0)


def test_ode_exponential_decay():
    ys = evaluate_ode_ivp("-y", 1.0, np.linspace(0, 1, 101))
    assert ys[-1] == pytest.approx(np.exp(-1), rel=1e-4)


def test_ode_nonfinite_rhs_raises():
    with pytest.raises((ArithmeticError, ValueError)):
        evaluate_ode_ivp("1/(x - 0.5)", 1.0, np.linspace(0, 1, 101))


def test_pde_residual_exact_heat_solution():
    r = evaluate_pde_residual(
        "Derivative(T(x,t),t) - D*Derivative(T(x,t),x,2)",
        "exp(-D*t)*sin(x)",
        {"x": np.linspace(0.1, 0.9, 5).tolist(), "t": np.linspace(0.1, 0.5, 4).tolist()},
        params={"D": 0.1},
    )
    assert r["max_abs_residual"] < 1e-9
    assert r["num_points"] == 20


def test_pde_residual_wrong_candidate_is_large():
    r = evaluate_pde_residual(
        "Derivative(T(x,t),t) - D*Derivative(T(x,t),x,2)",
        "x*t",
        {"x": [0.5, 1.0], "t": [0.5, 1.0]},
        params={"D": 0.1},
    )
    assert r["max_abs_residual"] > 0.1


def test_pde_residual_needs_derivative():
    with pytest.raises(ValueError, match="No Derivative"):
        evaluate_pde_residual("x + y", "x + y", {"x": [0.0]})


def test_heat_solver_decays_and_stays_finite():
    h = solve_heat_1d_explicit(1.0, 0.01, nx=21, nt=50)
    assert bool(np.all(np.isfinite(h["u"])))
    assert float(np.max(h["u"])) < 1.0
    assert h["r"] <= 0.5


def test_heat_solver_refuses_unstable_grid():
    with pytest.raises(ValueError, match="Unstable grid"):
        solve_heat_1d_explicit(1.0, 1.0, nx=6, nt=10, diffusivity=1.0)


def test_repair_routes_pde_to_template():
    from apedp.repair import AgnosticClosedLoopAssuranceSystem

    s = AgnosticClosedLoopAssuranceSystem()
    r = s.autonomous_stress_test_and_repair("Derivative(T(x,t),t) - D*Derivative(T(x,t),x,2)")
    assert r["status"] == "OUT_OF_SCOPE"
    assert "DIFFUSION" in r["validity_domain"]
    r2 = s.autonomous_stress_test_and_repair("1/x")
    assert r2["status"] == "REPAIRED"
