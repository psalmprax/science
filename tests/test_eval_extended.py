"""Tests for extended evaluation: implicit, quadrature, ODE IVP, PDE residual, heat."""
from __future__ import annotations

import numpy as np
import pytest

from apedp.eval_extended import (
    evaluate_einsum,
    evaluate_implicit,
    evaluate_implicit_system,
    evaluate_indexed,
    evaluate_ode_ivp,
    evaluate_ode_system,
    evaluate_pde_residual,
    evaluate_quadrature,
    simulate_sde_euler_maruyama,
    solve_heat_1d_explicit,
    symbolic_contract,
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


def test_implicit_system_two_by_two():
    r = evaluate_implicit_system(["x + y - 3", "x - y - 1"], ["x", "y"], [1.0, 1.0])
    assert r["x"] == pytest.approx(2.0)
    assert r["y"] == pytest.approx(1.0)
    assert r["residual"] < 1e-9


def test_implicit_system_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="len\\(formulas\\)"):
        evaluate_implicit_system(["x - 1"], ["x", "y"], [1.0, 1.0])


def test_ode_rk45_matches_exp_decay():
    ys = evaluate_ode_ivp("-y", 1.0, np.linspace(0, 1, 11), method="rk45")
    assert ys[-1] == pytest.approx(np.exp(-1), rel=1e-4)


def test_ode_bad_method_raises():
    with pytest.raises(ValueError, match="method"):
        evaluate_ode_ivp("-y", 1.0, [0.0, 1.0], method="euler")


def test_einsum_trace():
    assert float(evaluate_einsum("ii->", np.ones((3, 3)))) == pytest.approx(3.0)


def test_einsum_mismatch_raises():
    with pytest.raises(ValueError, match="einsum failed"):
        evaluate_einsum("ii->", np.ones((2, 3)))


def test_sde_wiener_statistics():
    s = simulate_sde_euler_maruyama("0.0", "1.0", 0.0, 1.0, 0.01, n_paths=2000, seed=1)
    assert s["terminal_mean"] == pytest.approx(0.0, abs=0.1)
    assert s["terminal_std"] == pytest.approx(1.0, rel=0.1)


def test_sde_reproducible_seed():
    a = simulate_sde_euler_maruyama("0.0", "1.0", 0.0, 0.1, 0.01, n_paths=50, seed=7)
    b = simulate_sde_euler_maruyama("0.0", "1.0", 0.0, 0.1, 0.01, n_paths=50, seed=7)
    assert np.array_equal(a["paths"], b["paths"])


def test_ode_system_stiff_pair_backward_euler():
    xs = np.linspace(0, 1, 11)
    r = evaluate_ode_system(["-0.5*y1", "-50*y2"], [1.0, 1.0], xs, method="backward_euler")
    assert r.shape == (11, 2)
    assert bool(np.all(np.abs(r) <= 1.0))
    assert r[-1, 0] == pytest.approx(np.exp(-0.5), rel=0.05)


def test_ode_system_rk4_vector_matches_scalar():
    xs = np.linspace(0, 1, 11)
    vec = evaluate_ode_system(["-0.5*y1", "-0.5*y2"], [1.0, 2.0], xs, method="rk4")
    assert vec[-1, 0] == pytest.approx(np.exp(-0.5), rel=1e-3)
    assert vec[-1, 1] == pytest.approx(2 * np.exp(-0.5), rel=1e-3)


def test_indexed_matmul_affine_trace():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert bool((evaluate_indexed("C[i,j] = A[i,k] * B[k,j]", {"A": a, "B": b}) == a @ b).all())
    assert list(evaluate_indexed("y[i] = 2*A[i,j]*x[j] - b[i]", {"A": a, "x": [1.0, 1.0], "b": [1.0, 1.0]})) == pytest.approx([5.0, 13.0])
    assert float(evaluate_indexed("t = A[i,i]", {"A": a})) == pytest.approx(5.0)


def test_indexed_bad_contraction_raises():
    a = np.ones((2, 2))
    with pytest.raises(ValueError, match="exactly once in each term"):
        evaluate_indexed("C[i,j] = A[i,k] * B[k,l]", {"A": a, "B": a})


def test_symbolic_delta_identity():
    import sympy as sp

    # y[i] - x[i]: the delta is eliminated, no sums or deltas remain.
    res = symbolic_contract("y[i] = delta[i,j]*x[j]", {"i": 3, "j": 3})
    _y = sp.IndexedBase(sp.Symbol("y"))
    _x = sp.IndexedBase(sp.Symbol("x"))
    assert res == _y[sp.Symbol("i")] - _x[sp.Symbol("i")]
    assert not res.has(sp.Sum) and not res.has(sp.KroneckerDelta)


def test_symbolic_trace_expands():
    import sympy as sp

    res = symbolic_contract("t = A[i,i]", {"i": 2})
    _A = sp.IndexedBase(sp.Symbol("A"))
    assert res == sp.Symbol("t") - (_A[0, 0] + _A[1, 1])
