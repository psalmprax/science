#!/usr/bin/env python3
"""Audit ``size_positions`` convergence on ill-conditioned covariances.

The sizing rule solves a capped, long-only mean-variance problem with an L1
turnover penalty by proximal gradient.  Its step is ``1 / (gamma * lambda_max)``,
so the convergence rate degrades as the covariance condition number grows.  This
script measures that degradation instead of asserting it is absent:

* the KKT residual of the returned weights,
* whether the loop met its own tolerance,
* the objective gap and max weight delta against a 200k-iteration run,
* gross exposure and whether the cap bound.

Covariances are correlated (random orthogonal eigenbasis), not diagonal, and the
audit covers n = 2, 4, 8 with and without previous weights.  The record is written
to ``benchmarks/sizing_solver_convergence.json``; the script always exits 0 so the
measurement can be read even when a case does not converge.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apedp_dyn.sandbox.finance_l2_demo import size_positions  # noqa: E402

OUT = ROOT / "benchmarks" / "sizing_solver_convergence.json"
CONDITION_NUMBERS = (1e2, 1e4, 1e6, 1e8, 1e10, 1e12)
SIZES = (2, 4, 8)
FEE_BPS = 5.0


def _spd(n: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """A correlated SPD matrix with condition number ``kappa`` (max eigenvalue 1)."""
    basis, _ = np.linalg.qr(rng.normal(size=(n, n)))
    matrix = basis @ np.diag(np.logspace(0.0, -np.log10(kappa), n)) @ basis.T
    return 0.5 * (matrix + matrix.T)


def _objective(weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
               previous: np.ndarray, cost_rate: float) -> float:
    return float(mu @ weights - 0.5 * weights @ sigma @ weights
                 - cost_rate * np.abs(weights - previous).sum())


def _case(n: int, kappa: float, previous: str, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    sigma = _spd(n, kappa, rng)
    mu = rng.uniform(0.005, 0.02, size=n)
    previous_weights = (np.zeros(n) if previous == "zero"
                        else rng.dirichlet(np.ones(n)))
    cost_rate = FEE_BPS / 10_000.0

    short = size_positions(mu, sigma, fee_bps=FEE_BPS,
                           previous_weights=previous_weights)
    long = size_positions(mu, sigma, fee_bps=FEE_BPS,
                          previous_weights=previous_weights,
                          max_iterations=200_000, tolerance=1e-15)
    w_short = np.asarray(short.weights)
    w_long = np.asarray(long.weights)
    return {
        "n": n,
        "kappa": kappa,
        "previous": previous,
        "iterations": short.iterations,
        "converged": short.converged,
        "kkt_residual": short.kkt_residual,
        "objective": _objective(w_short, mu, sigma, previous_weights, cost_rate),
        "objective_gap_vs_long_run": (
            _objective(w_short, mu, sigma, previous_weights, cost_rate)
            - _objective(w_long, mu, sigma, previous_weights, cost_rate)
        ),
        "max_weight_delta_vs_long_run": float(np.max(np.abs(w_short - w_long))),
        "gross_exposure": short.gross_exposure,
        "cap_binding": short.cap_binding,
    }


def main() -> int:
    cases: list[dict[str, object]] = []
    seed = 20261107
    for n in SIZES:
        for kappa in CONDITION_NUMBERS:
            for previous in ("zero", "nonzero"):
                cases.append(_case(n, kappa, previous, seed))
                seed += 1

    worst = max(cases, key=lambda case: float(case["kkt_residual"] or 0.0))
    unconverged = [case for case in cases if not case["converged"]]
    record = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "solver": "apedp_dyn.sandbox.finance_l2_demo.size_positions",
        "method": "proximal gradient, step=1/(gamma*lambda_max), capped simplex",
        "fee_bps": FEE_BPS,
        "condition_numbers": list(CONDITION_NUMBERS),
        "sizes": list(SIZES),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "commit": subprocess.run(  # noqa: S603 - fixed argv, no shell
                [shutil.which("git") or "git", "rev-parse", "HEAD"], cwd=ROOT,
                capture_output=True, text=True, check=False,
            ).stdout.strip() or None,
        },
        "cases": cases,
        "summary": {
            "n_cases": len(cases),
            "n_unconverged": len(unconverged),
            "worst_kkt_residual": worst["kkt_residual"],
            "worst_case": {k: worst[k] for k in ("n", "kappa", "previous")},
            "max_weight_delta_vs_long_run": max(
                float(case["max_weight_delta_vs_long_run"]) for case in cases),
            "max_abs_objective_gap": max(
                abs(float(case["objective_gap_vs_long_run"])) for case in cases),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {OUT.relative_to(ROOT)} ({len(cases)} cases)")
    print(f"{'n':>2} {'kappa':>7} {'prev':>7} {'iters':>6} {'conv':>5} "
          f"{'kkt':>10} {'dW_long':>10}")
    for case in cases:
        print(f"{case['n']:>2} {case['kappa']:>7.0e} {case['previous']:>7} "
              f"{case['iterations']:>6} {case['converged']!s:>5} "
              f"{float(case['kkt_residual']):>10.2e} "
              f"{float(case['max_weight_delta_vs_long_run']):>10.2e}")
    print(f"unconverged: {len(unconverged)}/{len(cases)} | worst KKT "
          f"{float(record['summary']['worst_kkt_residual']):.2e} at "
          f"{record['summary']['worst_case']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
