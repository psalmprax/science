"""Tests for full-PDE finite-difference solvers (numpy only)."""
from __future__ import annotations

import numpy as np
import pytest

from apedp.pde_solvers import (
    solve_advection_1d,
    solve_heat_2d,
    solve_heat_3d,
    solve_laplace_2d,
    solve_laplace_3d,
    solve_poisson_2d,
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
