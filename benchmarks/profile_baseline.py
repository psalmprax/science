"""Baseline Profiler for APEDP 3.0 Inner Loops.

Measures wall-clock time spent in:
1. Expression evaluation across N points
2. Dense grid singularity probing
3. Dimensional validation
"""
import time

import numpy as np
import sympy as sp

from apedp.pipeline import DiscoveryPipeline, safe_parse_expr
from apedp.singularity_probing import ActiveSingularityProbeEngine


def run_benchmark():
    print("=== Running Baseline Profiling (Python Reference) ===")
    
    expr_strs = [
        "a**1.5",
        "10.0 / (r**2)",
        "2.5e-4 * (T**4)",
        "8.314 * T / V",
        "sqrt(p**2 + 1.0)",
        "30.0 + 15.0*T - 5.0*T**2 + 0.5*T**3",
        "5.0 * exp(-0.8*r) / r",
        "2.0 * nu**3 / (exp(0.5*nu) - 1.0)",
        "1.0 / (1.0 + 25.0 * x**2)",
        "0.05 + 0.95 / ((1.0 + 0.25*gamma_dot**2)**0.3)",
    ]
    
    N = 10_000
    n_iters = 100
    x_data = np.linspace(0.1, 10.0, N)
    
    print(f"1. Expression Evaluation: {len(expr_strs)} expressions x {N} points x {n_iters} iterations = {len(expr_strs) * N * n_iters:,} evaluations...")
    
    parsed = [safe_parse_expr(s) for s in expr_strs]
    
    # Pre-lambdify to measure pure numerical array evaluation vs overhead
    lambdified = []
    for ep in parsed:
        syms = sorted(ep.free_symbols, key=lambda s: s.name)
        fn = sp.lambdify(syms, ep, modules=["numpy"])
        lambdified.append((syms, fn))
        
    t0 = time.perf_counter()
    for _ in range(n_iters):
        for syms, fn in lambdified:
            if len(syms) == 1:
                _ = fn(x_data)
            elif len(syms) == 2:
                _ = fn(x_data, x_data)
    dt_eval = time.perf_counter() - t0
    total_evals = len(expr_strs) * N * n_iters
    print(f"   -> Wall-clock time: {dt_eval:.3f}s ({total_evals / dt_eval:,.0f} point-evals/s, {len(expr_strs) * n_iters / dt_eval:.1f} batch-evals/s)")
    
    # 2. Singularity Probing
    prober = ActiveSingularityProbeEngine()
    n_sing_iters = 30
    print(f"2. Singularity Probing: {len(expr_strs)} expressions x {n_sing_iters} iterations...")
    obs_data = {"a": x_data, "r": x_data, "T": x_data, "V": x_data, "p": x_data, "nu": x_data, "x": x_data, "gamma_dot": x_data, "y": x_data}
    t0 = time.perf_counter()
    for _ in range(n_sing_iters):
        for s in expr_strs:
            prober.probe_candidate_singularity(s, [], obs_data)
    dt_sing = time.perf_counter() - t0
    print(f"   -> Wall-clock time: {dt_sing:.3f}s ({len(expr_strs) * n_sing_iters / dt_sing:.1f} probes/s)")
    
    # 3. Dimensional Analysis
    pipeline = DiscoveryPipeline()
    n_dim_iters = 50
    print(f"3. Dimensional Analysis: {len(parsed)} expressions x {n_dim_iters} iterations...")
    t0 = time.perf_counter()
    for _ in range(n_dim_iters):
        for ep in parsed:
            pipeline.validate_dimensions(ep, "dimensionless")
    dt_dim = time.perf_counter() - t0
    print(f"   -> Wall-clock time: {dt_dim:.3f}s ({len(parsed) * n_dim_iters / dt_dim:.1f} checks/s)")
    
    print("\n=== Profiling Complete ===")


if __name__ == "__main__":
    run_benchmark()
