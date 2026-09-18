"""Tests for full-PDE finite-difference solvers (numpy only)."""
from __future__ import annotations

import numpy as np
import pytest

from apedp.pde_solvers import (
    solve_advection_1d,
    solve_heat_2d,
    solve_heat_3d,
    solve_heat_nd_feynman_kac,
    solve_laplace_2d,
    solve_laplace_3d,
    solve_laplace_nd_walk_on_spheres,
    solve_poisson_2d,
    solve_poisson_fem,
    solve_wave_1d,
)


def test_wave_standing_mode():
    w = solve_wave_1d(1.0, 0.1, nx=101, nt=100, speed=1.0)
    exact = np.sin(np.pi * w["x"]) * np.cos(np.pi * 1.0 * 0.1)
    assert np.max(np.abs(w["u"] - exact)) < 1e-3
    assert w["courant"] <= 1.0


def test_wave_refuses_cfl_violation():
    with pytest.raises(ValueError, match="C="):
        solve_wave_1d(1.0, 1.0, nx=11, nt=5, speed=1.0)


def test_advection_stays_finite_and_bounded():
    a = solve_advection_1d(1.0, 0.2, nx=201, nt=200, velocity=1.0)
    assert bool(np.all(np.isfinite(a["u"])))
    assert float(np.max(np.abs(a["u"]))) <= 1.0 + 1e-9


def test_advection_refuses_cfl_violation():
    with pytest.raises(ValueError, match="\\|C\\|"):
        solve_advection_1d(1.0, 1.0, nx=11, nt=5, velocity=1.0)


def test_heat_2d_decays():
    h = solve_heat_2d(1.0, 1.0, 0.005, nx=21, ny=21, nt=50)
    assert bool(np.all(np.isfinite(h["u"])))
    assert 0.0 < float(np.max(h["u"])) < 1.0
    assert h["r"] <= 0.25


def test_heat_2d_refuses_unstable():
    with pytest.raises(ValueError, match="r="):
        solve_heat_2d(1.0, 1.0, 1.0, nx=11, ny=11, nt=10)


def test_laplace_linear_is_harmonic():
    sol = solve_laplace_2d(nx=21, ny=21, bc="linear", tol=1e-6)
    xx, yy = np.meshgrid(sol["x"], sol["y"], indexing="ij")
    assert np.max(np.abs(sol["u"] - (xx + yy))) < 1e-3


def test_poisson_manufactured_sine():
    p = solve_poisson_2d(
        "-2*3.141592653589793**2*sin(3.141592653589793*x)*sin(3.141592653589793*y)",
        nx=21, ny=21, tol=1e-5, maxiter=20000,
    )
    xx, yy = np.meshgrid(p["x"], p["y"], indexing="ij")
    exact = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    assert np.max(np.abs(p["u"] - exact)) < 5e-3


def test_poisson_nonconvergence_raises():
    with pytest.raises(ArithmeticError, match="did not converge"):
        solve_poisson_2d("-sin(x)*sin(y)", nx=21, ny=21, tol=1e-15, maxiter=3)


def test_heat_3d_decays():
    h = solve_heat_3d(1.0, 1.0, 1.0, 0.002, nx=11, ny=11, nz=11, nt=20)
    assert bool(np.all(np.isfinite(h["u"])))
    assert 0.0 < float(np.max(h["u"])) < 1.0
    assert h["r"] <= 1 / 6


def test_heat_3d_refuses_unstable_and_oversize():
    with pytest.raises(ValueError, match="r="):
        solve_heat_3d(1.0, 1.0, 1.0, 1.0, nx=11, ny=11, nz=11, nt=5)
    with pytest.raises(ValueError, match="budget"):
        solve_heat_3d(1.0, 1.0, 1.0, 0.001, nx=41, ny=41, nz=41, nt=5)


def test_laplace_3d_linear_is_harmonic():
    sol = solve_laplace_3d(nx=11, ny=11, nz=11, bc="linear", tol=1e-5)
    xx, yy, zz = np.meshgrid(sol["x"], sol["y"], sol["z"], indexing="ij")
    assert np.max(np.abs(sol["u"] - (xx + yy + zz))) < 1e-3


def test_feynman_kac_5d_linear_is_exact_in_expectation():
    r = solve_heat_nd_feynman_kac([0.5] * 5, 0.3, "x0+x1+x2+x3+x4", n_paths=20000, seed=3)
    assert r["dims"] == 5
    assert abs(r["value"] - 2.5) < 5 * r["stderr"]


def test_feynman_kac_10d_gaussian_matches_closed_form():
    import math

    d = 10
    s2, D, t = 1.0, 0.2, 0.5
    x0 = [0.1] * d
    v = s2 / (s2 + 2 * D * t)
    exact = v ** (d / 2) * math.exp(-sum(v * v for v in x0) / (2 * (s2 + 2 * D * t)))
    r = solve_heat_nd_feynman_kac(
        x0, t, "exp(-(" + "+".join(f"x{i}**2" for i in range(d)) + ")/2)",
        diffusivity=D, n_paths=30000, seed=5,
    )
    assert abs(r["value"] - exact) < 5 * r["stderr"]


def test_walk_on_spheres_5d_linear():
    r = solve_laplace_nd_walk_on_spheres([0.2] * 5, "x0+x1+x2+x3+x4", n_paths=10000, seed=7)
    assert abs(r["value"] - 1.0) < 5 * r["stderr"]


def test_walk_on_spheres_20d():
    r = solve_laplace_nd_walk_on_spheres([0.1] * 20, "x0", n_paths=5000, seed=9)
    assert r["dims"] == 20
    assert abs(r["value"] - 0.1) < 5 * r["stderr"]


def test_walk_on_spheres_rejects_outside_ball():
    with pytest.raises(ValueError, match="strictly inside"):
        solve_laplace_nd_walk_on_spheres([2.0, 0.0], "x0")


def test_poisson_fem_manufactured_sine():
    n = 9
    xs = np.linspace(0, 1, n)
    nodes = [[float(x), float(y)] for y in xs for x in xs]
    tris = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1
            tris += [[a, b, c], [b, d, c]]
    r = solve_poisson_fem(
        nodes, tris,
        source="2*9.869604401089358*sin(3.141592653589793*x)*sin(3.141592653589793*y)",
    )
    pts = np.array(nodes)
    exact = np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
    assert np.max(np.abs(r["u"] - exact)) < 5e-2


def test_poisson_fem_refuses_degenerate_element():
    with pytest.raises(ValueError, match="Degenerate triangle"):
        solve_poisson_fem([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], [[0, 1, 2]])
