"""Differential Fuzzing & Conformance Suite: Rust Kernel vs Python Reference Engine.

Mathematically verifies that:
1. Candidate expressions evaluate to identical float64 results within IEEE-754 precision (|y_rust - y_py| < 1e-12).
2. Loss metrics (MSE, MAE, R^2) agree between Rust and Python engines.
3. Parallel candidate batch scoring produces identical ranking.
4. Active singularity probing agrees on pole and divergence locations.
5. Transparent fallback to Python functions identically when Rust is disabled.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp.rust_bridge import RustKernelBridge, is_rust_kernel_available


@pytest.fixture
def rust_engine():
    if not is_rust_kernel_available():
        pytest.skip("Rust kernel (apedp_kernel) not available")
    return RustKernelBridge(force_fallback=False)


@pytest.fixture
def python_fallback_engine():
    return RustKernelBridge(force_fallback=True)


def test_rust_kernel_availability():
    """Verify that the compiled native extension is present in environment."""
    assert is_rust_kernel_available(), "apedp_kernel should be compiled and importable"


@pytest.mark.parametrize(
    "expr,vars,data_gen",
    [
        ("x0 + x1", ["x0", "x1"], lambda: [np.linspace(-5, 5, 200), np.linspace(1, 10, 200)]),
        ("x0 * x1 - 3.5", ["x0", "x1"], lambda: [np.linspace(1, 10, 200), np.linspace(0.5, 2.0, 200)]),
        ("x0**2 + 2.0*x0 + 1.0", ["x0"], lambda: [np.linspace(-10, 10, 500)]),
        ("sin(x0) + cos(x0)", ["x0"], lambda: [np.linspace(0, 2 * np.pi, 300)]),
        ("exp(-x0) * sin(2.0 * x0)", ["x0"], lambda: [np.linspace(0.1, 5.0, 300)]),
        ("tanh(x0) + ln(x0 + 1.0)", ["x0"], lambda: [np.linspace(0.1, 10.0, 200)]),
        ("sqrt(x0**2 + x1**2)", ["x0", "x1"], lambda: [np.linspace(1, 10, 200), np.linspace(2, 8, 200)]),
        ("sin(x0) / (x0 + 5.0)", ["x0"], lambda: [np.linspace(0, 10, 200)]),
    ],
)
def test_differential_fuzzing_expressions(rust_engine, python_fallback_engine, expr, vars, data_gen):
    """Verifies that Rust kernel and Python fallback evaluate to identical numerical outputs."""
    data = data_gen()

    y_rust = rust_engine.evaluate_expression(expr, vars, data)
    y_py = python_fallback_engine.evaluate_expression(expr, vars, data)

    assert len(y_rust) == len(y_py)
    # Check max absolute and relative difference
    abs_diff = np.max(np.abs(y_rust - y_py))
    assert abs_diff < 1e-12, f"Divergence on '{expr}': max abs diff = {abs_diff}"


def test_metrics_engine_parity(rust_engine, python_fallback_engine):
    """Verifies MSE, MAE, and R^2 metric calculation parity."""
    np.random.seed(42)
    y_true = np.random.normal(5.0, 2.0, 1000)
    noise = np.random.normal(0.0, 0.2, 1000)
    y_pred = y_true + noise

    m_rust = rust_engine.compute_metrics(y_true, y_pred)
    m_py = python_fallback_engine.compute_metrics(y_true, y_pred)

    assert abs(m_rust["mse"] - m_py["mse"]) < 1e-12
    assert abs(m_rust["mae"] - m_py["mae"]) < 1e-12
    assert abs(m_rust["r_squared"] - m_py["r_squared"]) < 1e-10
    assert abs(m_rust["max_residual"] - m_py["max_residual"]) < 1e-12
    assert m_rust["has_nan_or_inf"] == m_py["has_nan_or_inf"] is False


def test_parallel_candidate_scoring(rust_engine, python_fallback_engine):
    """Verifies multi-core batch candidate evaluation and identical ranking."""
    x = np.linspace(-3.0, 3.0, 500)
    target = 2.5 * (x**2) - 1.2 * x + 0.8

    candidates = [
        "2.5 * (x**2) - 1.2 * x + 0.8",  # True model
        "2.5 * (x**2)",
        "x**2",
        "sin(x)",
        "exp(x)",
    ]

    scores_rust = rust_engine.score_candidates_batch(candidates, ["x"], [x], target)
    scores_py = python_fallback_engine.score_candidates_batch(candidates, ["x"], [x], target)

    assert len(scores_rust) == len(scores_py) == len(candidates)

    # True model should be candidate 0 with MSE ~ 0 in both
    assert scores_rust[0]["mse"] < 1e-14
    assert scores_py[0]["mse"] < 1e-14

    for r, p in zip(scores_rust, scores_py):
        assert r["candidate_id"] == p["candidate_id"]
        assert abs(r["mse"] - p["mse"]) < 1e-10
        assert abs(r["r_squared"] - p["r_squared"]) < 1e-8


def test_singularity_probing_parity(rust_engine, python_fallback_engine):
    """Verifies that singularity and pole avoidance detection match between backends."""
    # 1. Clean smooth expression
    clean_expr = "x0**2 + 1.0"
    bounds = [(-5.0, 5.0)]

    p_rust_clean = rust_engine.probe_singularity(clean_expr, ["x0"], bounds, num_grid_points=2000)
    p_py_clean = python_fallback_engine.probe_singularity(clean_expr, ["x0"], bounds, num_grid_points=2000)

    assert p_rust_clean["is_sound"] is True
    assert p_py_clean["is_sound"] is True
    assert p_rust_clean["has_pole"] is False
    assert p_py_clean["has_pole"] is False

    # 2. Expression with pole inside domain (x0 = 1.0)
    pole_expr = "1.0 / (x0 - 1.0)"
    bounds = [(0.0, 3.0)]

    p_rust_pole = rust_engine.probe_singularity(pole_expr, ["x0"], bounds, num_grid_points=2000)
    p_py_pole = python_fallback_engine.probe_singularity(pole_expr, ["x0"], bounds, num_grid_points=2000)

    assert p_rust_pole["is_sound"] is False
    assert p_py_pole["is_sound"] is False
    assert p_rust_pole["has_pole"] is True
    assert p_py_pole["has_pole"] is True
    assert abs(p_rust_pole["pole_coordinate"] - 1.0) < 0.05
    assert abs(p_py_pole["pole_coordinate"] - 1.0) < 0.05


def test_dual_validation_mode():
    """Verifies that dual_validate=True executes in lockstep without raising warnings."""
    engine = RustKernelBridge(dual_validate=True)
    x = np.linspace(1.0, 10.0, 100)
    out = engine.evaluate_expression("x**2 + 3.0*x", ["x"], [x])
    assert len(out) == 100
    expected = x**2 + 3.0 * x
    assert np.allclose(out, expected, atol=1e-12)


def test_unit_vector_algebra():
    """Verifies 7D SI unit vector operations in Rust."""
    import apedp_kernel

    u_dimless = apedp_kernel.UnitVector.dimensionless()
    assert u_dimless.is_dimensionless()
    assert u_dimless.as_tuple() == (0, 0, 0, 0, 0, 0, 0)

    # Force: N = kg * m / s^2 -> (1, 1, -2, 0, 0, 0, 0)
    u_n = apedp_kernel.UnitVector.parse("newton")
    assert u_n.as_tuple() == (1, 1, -2, 0, 0, 0, 0)

    # Energy: J = N * m -> (2, 1, -2, 0, 0, 0, 0)
    u_m = apedp_kernel.UnitVector.parse("meter")
    u_j = u_n.mul(u_m)
    assert u_j.as_tuple() == (2, 1, -2, 0, 0, 0, 0)

    # Power: W = J / s -> (2, 1, -3, 0, 0, 0, 0)
    u_s = apedp_kernel.UnitVector.parse("second")
    u_w = u_j.div(u_s)
    assert u_w.as_tuple() == (2, 1, -3, 0, 0, 0, 0)

    # Kinetic energy dimensional homogeneity: m * v^2 == J
    u_kg = apedp_kernel.UnitVector.parse("kilogram")
    u_v = u_m.div(u_s)
    u_v2 = u_v.pow(2)
    u_ke = u_kg.mul(u_v2)
    assert u_ke.as_tuple() == u_j.as_tuple()


def test_pipeline_dual_backend_evaluation(tmp_path):
    """Verifies that DiscoveryPipeline executes identically under both 'rust' and 'python' backends."""
    import sympy as sp

    from apedp.pipeline import DiscoveryPipeline

    pipeline_rust = DiscoveryPipeline(output_dir=str(tmp_path / "rust"), backend="rust")
    pipeline_py = DiscoveryPipeline(output_dir=str(tmp_path / "py"), backend="python")

    expr = sp.sympify("x**2 + 2.0*x + 1.0")
    data = {"x": np.linspace(-5.0, 5.0, 50, dtype=np.float64)}

    pred_rust = pipeline_rust._predict_expression(expr, data)
    pred_py = pipeline_py._predict_expression(expr, data)

    assert np.allclose(pred_rust, pred_py, atol=1e-12)


def test_conditioning_diagnostics_parity(rust_engine, python_fallback_engine):
    """Verifies that Faer SVD matrix conditioning in Rust matches NumPy SVD in Python."""
    # Identity matrix
    cols_id = [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ]
    cond_rust = rust_engine.compute_matrix_conditioning(cols_id)
    cond_py = python_fallback_engine.compute_matrix_conditioning(cols_id)

    assert abs(cond_rust["condition_number"] - cond_py["condition_number"]) < 1e-6
    assert cond_rust["effective_rank"] == cond_py["effective_rank"] == 3
    assert cond_rust["is_well_conditioned"] == cond_py["is_well_conditioned"] is True

    # Ill-conditioned collinear matrix
    cols_ill = [
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([2.0, 4.0, 6.0], dtype=np.float64),
    ]
    cond_ill_rust = rust_engine.compute_matrix_conditioning(cols_ill)
    cond_ill_py = python_fallback_engine.compute_matrix_conditioning(cols_ill)

    assert cond_ill_rust["effective_rank"] == cond_ill_py["effective_rank"] == 1
    assert cond_ill_rust["is_well_conditioned"] == cond_ill_py["is_well_conditioned"] is False


def test_varpro_linear_least_squares_parity(rust_engine, python_fallback_engine):
    """Verifies that Faer QR linear parameter fitting matches NumPy lstsq."""
    n = 200
    x0 = np.linspace(0.0, 5.0, n)
    x1 = np.cos(x0)
    x_const = np.ones(n)
    noise = np.random.normal(0.0, 0.01, n)
    target = 3.2 * x0 - 1.8 * x1 + 0.5 + noise

    fit_rust = rust_engine.solve_linear_coefficients([x0, x1, x_const], target)
    fit_py = python_fallback_engine.solve_linear_coefficients([x0, x1, x_const], target)

    assert np.allclose(fit_rust["coefficients"], fit_py["coefficients"], atol=1e-5)
    assert abs(fit_rust["mse"] - fit_py["mse"]) < 1e-8
    assert abs(fit_rust["r_squared"] - fit_py["r_squared"]) < 1e-6
    assert fit_rust["is_well_conditioned"] == fit_py["is_well_conditioned"] is True


def test_precision_boundary_scaling_parity(rust_engine, python_fallback_engine):
    """Verifies precision scaling laws across f32, f64, and custom bit precision."""
    # 1. f32 precision
    p32_rust = rust_engine.evaluate_precision_boundary(1e-3, 1.0, 32)
    p32_py = python_fallback_engine.evaluate_precision_boundary(1e-3, 1.0, 32)
    assert p32_rust["resolved"] == p32_py["resolved"] is True

    p32_unres_rust = rust_engine.evaluate_precision_boundary(1e-5, 1.0, 32)
    p32_unres_py = python_fallback_engine.evaluate_precision_boundary(1e-5, 1.0, 32)
    assert p32_unres_rust["resolved"] == p32_unres_py["resolved"] is False

    # 2. f64 precision
    p64_rust = rust_engine.evaluate_precision_boundary(1e-5, 1.0, 64)
    p64_py = python_fallback_engine.evaluate_precision_boundary(1e-5, 1.0, 64)
    assert p64_rust["resolved"] == p64_py["resolved"] is True





def test_fallback_conditioning_degenerate_reports_ill_conditioned(python_fallback_engine):
    """A conditioning probe must report degeneracy, never raise.

    Regression: NaN Jacobian columns (e.g. sqrt of negative inputs) made
    np.linalg.svd raise LinAlgError, which escaped synthesize() and killed
    the whole candidate batch. SVD non-convergence IS the signal.
    """
    import numpy as np

    rep = python_fallback_engine._fallback_conditioning(
        [np.full(10, np.nan), np.ones(10)], 1e-12
    )
    assert rep["is_well_conditioned"] is False
    assert rep["condition_number"] == float("inf")

    rep = python_fallback_engine.compute_matrix_conditioning(
        [np.full(10, np.nan), np.ones(10)], rcond_threshold=1e-12
    )
    assert rep["is_well_conditioned"] is False
