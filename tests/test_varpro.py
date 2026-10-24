"""Tests for Variable Projection (VarPro) separable nonlinear solver."""
from __future__ import annotations

import numpy as np
import sympy as sp

from apedp.varpro import IdentifiabilityReport, VarProOptimizer, VarProResult


def test_varpro_purely_linear_model():
    x_sym = sp.Symbol("x")
    p0 = sp.Symbol("p0")
    p1 = sp.Symbol("p1")

    x_vals = np.linspace(-2.0, 2.0, 40)
    y_true = 3.5 * (x_vals**2) - 1.8

    skeleton = p0 * (x_sym**2) + p1
    solver = VarProOptimizer()
    res = solver.fit(skeleton, [p0, p1], [x_sym], {"x": x_vals}, y_true)

    assert isinstance(res, VarProResult)
    assert res.mse < 1e-10
    assert abs(res.linear_params[p0] - 3.5) < 1e-4
    assert abs(res.linear_params[p1] - (-1.8)) < 1e-4
    assert isinstance(res.identifiability, IdentifiabilityReport)
    assert res.identifiability.status == "IDENTIFIABLE"
    assert res.identifiability.rank == 2


def test_varpro_separable_nonlinear_model():
    x1_sym = sp.Symbol("x1")
    x2_sym = sp.Symbol("x2")
    p0 = sp.Symbol("p0")
    p1 = sp.Symbol("p1")
    a0 = sp.Symbol("a0")

    # True model: y = 2.0 * x1 + 5.0 * exp(-2.5 * x2)
    x1_vals = np.linspace(0.5, 3.0, 40)
    x2_vals = np.linspace(0.2, 2.0, 40)
    y_true = 2.0 * x1_vals + 5.0 * np.exp(-2.5 * x2_vals)

    skeleton = p0 * x1_sym + p1 * sp.exp(-a0 * x2_sym)
    solver = VarProOptimizer()
    res = solver.fit(
        skeleton,
        [p0, p1, a0],
        [x1_sym, x2_sym],
        {"x1": x1_vals, "x2": x2_vals},
        y_true,
        alpha_bounds={a0: (0.5, 5.0)},
    )

    assert isinstance(res, VarProResult)
    assert res.mse < 1e-6
    assert abs(res.linear_params[p0] - 2.0) < 1e-2
    assert abs(res.linear_params[p1] - 5.0) < 1e-2
    assert abs(res.nonlinear_params[a0] - 2.5) < 1e-2
    assert res.identifiability.status == "IDENTIFIABLE"


def test_varpro_collinear_basis_detects_rank_deficiency():
    x_sym = sp.Symbol("x")
    p0 = sp.Symbol("p0")
    p1 = sp.Symbol("p1")

    x_vals = np.linspace(1.0, 5.0, 30)
    y_true = 2.0 * x_vals

    # Degenerate collinear basis: p0 * x + p1 * (2*x)
    skeleton = p0 * x_sym + p1 * (2 * x_sym)
    solver = VarProOptimizer()
    res = solver.fit(skeleton, [p0, p1], [x_sym], {"x": x_vals}, y_true)

    assert res.identifiability.status == "RANK_DEFICIENT"
    assert res.identifiability.rank == 1
    assert res.identifiability.num_bases == 2
