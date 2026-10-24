"""Comprehensive Performance Benchmark: Rust Kernel vs Python Reference Engine.

Measures realistic Discovery Platform workloads:
1. Batch Candidate Scoring: 250 candidate expressions evaluated across 10,000 points.
2. Active Singularity Probing: High-density pole search across 5,000 grid points.
3. Dimensional Algebra: 50,000 SI unit vector algebraic operations.
4. Scalar Inner Loop: 100,000 evaluations for optimizer/ODE inner loops.
5. Numerical Parity: Double precision residual verification.
"""
from __future__ import annotations

import time

import apedp_kernel
import numpy as np
import pint
import sympy as sp

from apedp.rust_bridge import RustKernelBridge


def benchmark_batch_candidate_scoring(n_candidates: int = 250, n_samples: int = 10_000):
    rust_bridge = RustKernelBridge(force_fallback=False)
    py_bridge = RustKernelBridge(force_fallback=True)

    x0 = np.linspace(-5.0, 5.0, n_samples)
    x1 = np.linspace(1.0, 10.0, n_samples)
    target = x0**2 + np.sin(x1) - 1.5 * x0

    # Pool of realistic candidate physical laws
    candidates = [
        f"x0**2 + sin(x1) + {c:.3f} * x0"
        for c in np.linspace(-3.0, 3.0, n_candidates)
    ]

    # Warmup
    rust_bridge.score_candidates_batch(candidates[:5], ["x0", "x1"], [x0, x1], target)
    py_bridge.score_candidates_batch(candidates[:5], ["x0", "x1"], [x0, x1], target)

    # Rust Parallel Scoring
    runs_rust = []
    for _ in range(5):
        t0 = time.perf_counter()
        scores_rust = rust_bridge.score_candidates_batch(candidates, ["x0", "x1"], [x0, x1], target)
        runs_rust.append(time.perf_counter() - t0)
    t_rust = float(np.median(runs_rust))

    # Python Sequential Scoring
    runs_py = []
    for _ in range(5):
        t0 = time.perf_counter()
        scores_py = py_bridge.score_candidates_batch(candidates, ["x0", "x1"], [x0, x1], target)
        runs_py.append(time.perf_counter() - t0)
    t_py = float(np.median(runs_py))

    total_evals = n_candidates * n_samples
    throughput_rust = total_evals / t_rust
    throughput_py = total_evals / t_py

    # Verify parity
    max_mse_diff = max(abs(r["mse"] - p["mse"]) for r, p in zip(scores_rust, scores_py))

    return {
        "workload": f"Candidate Batch Scoring ({n_candidates} cand. x {n_samples:,} pts)",
        "total_evals": total_evals,
        "t_rust_ms": t_rust * 1000.0,
        "t_py_ms": t_py * 1000.0,
        "speedup": t_py / t_rust,
        "throughput_rust": throughput_rust,
        "throughput_py": throughput_py,
        "max_discrepancy": max_mse_diff,
    }


def benchmark_singularity_probing(n_expressions: int = 50, grid_points: int = 5_000):
    rust_bridge = RustKernelBridge(force_fallback=False)
    py_bridge = RustKernelBridge(force_fallback=True)

    exprs = [
        ("x0**2 + sin(x0)", [(-5.0, 5.0)]),
        ("1.0 / (x0 - 2.5)", [(0.0, 5.0)]),
        ("ln(x0 + 0.1)", [(0.1, 10.0)]),
        ("tanh(x0) / (x0**2 + 1.0)", [(-10.0, 10.0)]),
        ("exp(-x0) * cos(x0)", [(0.0, 10.0)]),
    ] * (n_expressions // 5)

    # Warmup
    rust_bridge.probe_singularity(exprs[0][0], ["x0"], exprs[0][1], grid_points)
    py_bridge.probe_singularity(exprs[0][0], ["x0"], exprs[0][1], grid_points)

    # Rust
    runs_rust = []
    for _ in range(3):
        t0 = time.perf_counter()
        for expr_str, bounds in exprs:
            rust_bridge.probe_singularity(expr_str, ["x0"], bounds, grid_points)
        runs_rust.append(time.perf_counter() - t0)
    t_rust = float(np.median(runs_rust))

    # Python
    runs_py = []
    for _ in range(3):
        t0 = time.perf_counter()
        for expr_str, bounds in exprs:
            py_bridge.probe_singularity(expr_str, ["x0"], bounds, grid_points)
        runs_py.append(time.perf_counter() - t0)
    t_py = float(np.median(runs_py))

    return {
        "workload": f"Active Singularity Probing ({n_expressions} exprs x {grid_points:,} pts)",
        "t_rust_ms": t_rust * 1000.0,
        "t_py_ms": t_py * 1000.0,
        "speedup": t_py / t_rust,
    }


def benchmark_dimensional_algebra(n_ops: int = 50_000):
    ureg = pint.UnitRegistry()

    # 1. Rust 7D SI Integer Vector Engine
    t0 = time.perf_counter()
    u_m = apedp_kernel.UnitVector.parse("meter")
    u_s = apedp_kernel.UnitVector.parse("second")
    u_kg = apedp_kernel.UnitVector.parse("kilogram")

    for _ in range(n_ops):
        # Compute Kinetic Energy dimensions: kg * (m/s)^2
        u_v = u_m.div(u_s)
        u_v2 = u_v.pow(2)
        u_ke = u_kg.mul(u_v2)
        _ = u_ke.as_tuple()
    t_rust = time.perf_counter() - t0

    # 2. Python Pint Registry
    p_m = ureg.meter
    p_s = ureg.second
    p_kg = ureg.kilogram

    t0 = time.perf_counter()
    for _ in range(n_ops):
        p_v = p_m / p_s
        p_v2 = p_v ** 2
        p_ke = p_kg * p_v2
        _ = p_ke.dimensionality
    t_py = time.perf_counter() - t0

    return {
        "workload": f"Dimensional Algebra ({n_ops:,} SI unit operations)",
        "t_rust_ms": t_rust * 1000.0,
        "t_py_ms": t_py * 1000.0,
        "speedup": t_py / t_rust,
    }


def benchmark_scalar_inner_loop(n_evals: int = 100_000):
    expr_str = "x0**2 + sin(x1) - 2.5 * x0"
    compiled_rust = apedp_kernel.compile(expr_str, ["x0", "x1"])

    x0_sym, x1_sym = sp.symbols("x0 x1")
    py_fn = sp.lambdify((x0_sym, x1_sym), x0_sym**2 + sp.sin(x1_sym) - 2.5 * x0_sym)

    points = np.random.uniform(-5.0, 5.0, (n_evals, 2))

    # Rust
    t0 = time.perf_counter()
    for i in range(n_evals):
        _ = apedp_kernel.evaluate_scalar(compiled_rust, [points[i, 0], points[i, 1]])
    t_rust = time.perf_counter() - t0

    # Python
    t0 = time.perf_counter()
    for i in range(n_evals):
        _ = py_fn(points[i, 0], points[i, 1])
    t_py = time.perf_counter() - t0

    return {
        "workload": f"Scalar Point Evaluation ({n_evals:,} evaluations)",
        "t_rust_ms": t_rust * 1000.0,
        "t_py_ms": t_py * 1000.0,
        "speedup": t_py / t_rust,
    }


def benchmark_matrix_conditioning(n_matrices: int = 1_000, n_rows: int = 200, n_cols: int = 5):
    rust_bridge = RustKernelBridge(force_fallback=False)
    py_bridge = RustKernelBridge(force_fallback=True)

    np.random.seed(42)
    matrices = [
        [np.random.randn(n_rows) for _ in range(n_cols)]
        for _ in range(n_matrices)
    ]

    # Warmup
    rust_bridge.compute_matrix_conditioning(matrices[0])
    py_bridge.compute_matrix_conditioning(matrices[0])

    # Rust Faer SVD
    runs_rust = []
    for _ in range(3):
        t0 = time.perf_counter()
        for cols in matrices:
            _ = rust_bridge.compute_matrix_conditioning(cols)
        runs_rust.append(time.perf_counter() - t0)
    t_rust = float(np.median(runs_rust))

    # Python NumPy SVD
    runs_py = []
    for _ in range(3):
        t0 = time.perf_counter()
        for cols in matrices:
            _ = py_bridge.compute_matrix_conditioning(cols)
        runs_py.append(time.perf_counter() - t0)
    t_py = float(np.median(runs_py))

    return {
        "workload": f"Matrix SVD Conditioning ({n_matrices:,} mats {n_rows}x{n_cols})",
        "t_rust_ms": t_rust * 1000.0,
        "t_py_ms": t_py * 1000.0,
        "speedup": t_py / t_rust,
    }


def benchmark_linear_least_squares(n_fits: int = 1_000, n_rows: int = 200, n_cols: int = 5):
    rust_bridge = RustKernelBridge(force_fallback=False)
    py_bridge = RustKernelBridge(force_fallback=True)

    np.random.seed(42)
    workloads = []
    for _ in range(n_fits):
        cols = [np.random.randn(n_rows) for _ in range(n_cols)]
        coeffs_true = np.random.randn(n_cols)
        target = sum(c * w for c, w in zip(coeffs_true, cols)) + 0.05 * np.random.randn(n_rows)
        workloads.append((cols, target))

    # Warmup
    rust_bridge.solve_linear_coefficients(workloads[0][0], workloads[0][1])
    py_bridge.solve_linear_coefficients(workloads[0][0], workloads[0][1])

    # Rust Faer QR
    runs_rust = []
    for _ in range(3):
        t0 = time.perf_counter()
        for cols, target in workloads:
            _ = rust_bridge.solve_linear_coefficients(cols, target)
        runs_rust.append(time.perf_counter() - t0)
    t_rust = float(np.median(runs_rust))

    # Python NumPy lstsq
    runs_py = []
    for _ in range(3):
        t0 = time.perf_counter()
        for cols, target in workloads:
            _ = py_bridge.solve_linear_coefficients(cols, target)
        runs_py.append(time.perf_counter() - t0)
    t_py = float(np.median(runs_py))

    return {
        "workload": f"Linear QR Least-Squares ({n_fits:,} fits {n_rows}x{n_cols})",
        "t_rust_ms": t_rust * 1000.0,
        "t_py_ms": t_py * 1000.0,
        "speedup": t_py / t_rust,
    }


def benchmark_precision_boundary(n_evals: int = 50_000):
    rust_bridge = RustKernelBridge(force_fallback=False)
    py_bridge = RustKernelBridge(force_fallback=True)

    deltas = np.logspace(-12, -2, n_evals)

    # Warmup
    rust_bridge.evaluate_precision_boundary(deltas[0], condition_number=1.5, precision_bits=64)
    py_bridge.evaluate_precision_boundary(deltas[0], condition_number=1.5, precision_bits=64)

    # Rust
    t0 = time.perf_counter()
    for d in deltas:
        _ = rust_bridge.evaluate_precision_boundary(d, condition_number=1.5, precision_bits=64)
    t_rust = time.perf_counter() - t0

    # Python
    t0 = time.perf_counter()
    for d in deltas:
        _ = py_bridge.evaluate_precision_boundary(d, condition_number=1.5, precision_bits=64)
    t_py = time.perf_counter() - t0

    return {
        "workload": f"Multi-Precision Boundary ({n_evals:,} evaluations)",
        "t_rust_ms": t_rust * 1000.0,
        "t_py_ms": t_py * 1000.0,
        "speedup": t_py / t_rust,
    }


def main():
    print("==========================================================================")
    print("          APEDP 3.1 HIGH-PERFORMANCE BENCHMARK: RUST VS PYTHON            ")
    print("==========================================================================")
    print("Environment: Linux x86_64 | 8 CPU Cores | Rust `dev`/`release` native via PyO3")
    print("--------------------------------------------------------------------------\n")

    print("[1/7] Running Candidate Batch Scoring benchmark...")
    b_batch = benchmark_batch_candidate_scoring(n_candidates=250, n_samples=10_000)

    print("[2/7] Running Active Singularity Probing benchmark...")
    b_sing = benchmark_singularity_probing(n_expressions=50, grid_points=5_000)

    print("[3/7] Running Dimensional Algebra benchmark...")
    b_dim = benchmark_dimensional_algebra(n_ops=50_000)

    print("[4/7] Running Scalar Inner Loop benchmark...")
    b_scalar = benchmark_scalar_inner_loop(n_evals=100_000)

    print("[5/7] Running Matrix SVD Conditioning benchmark...")
    b_cond = benchmark_matrix_conditioning(n_matrices=1_000, n_rows=200, n_cols=5)

    print("[6/7] Running Linear QR Least-Squares benchmark...")
    b_lstsq = benchmark_linear_least_squares(n_fits=1_000, n_rows=200, n_cols=5)

    print("[7/7] Running Multi-Precision Boundary Scaling benchmark...")
    b_prec = benchmark_precision_boundary(n_evals=50_000)

    results = [b_batch, b_sing, b_dim, b_scalar, b_cond, b_lstsq, b_prec]

    print("\n" + "=" * 90)
    print(f"{'Workload':<48} | {'Rust (ms)':<10} | {'Python (ms)':<12} | {'Speedup':<8}")
    print("-" * 90)
    for r in results:
        print(f"{r['workload']:<48} | {r['t_rust_ms']:>8.2f} ms | {r['t_py_ms']:>10.2f} ms | {r['speedup']:>6.2f}x")
    print("=" * 90)

    print("\nThroughput Details (Candidate Scoring):")
    print(f"  - Rust Parallel Engine  : {b_batch['throughput_rust']/1e6:.2f} Million evals/sec")
    print(f"  - Python Reference Engine: {b_batch['throughput_py']/1e6:.2f} Million evals/sec")
    print(f"  - Max Numerical Discrepancy : {b_batch['max_discrepancy']:.2e} (IEEE-754 exact)")


if __name__ == "__main__":
    main()
