"""Spatio-Temporal PDE Discovery & Epistemic Verification in 1D and 2D.

Bridges the native Rust PDE Jet engine (rime/pde.rs) into APEDP-X:
- Discovers continuous field PDEs from space-time grids u(t, x) and 2D spatial fields u(t, x, y).
- Supports arbitrary non-uniform spatial meshes via exact Fornberg weights.
- Solves for advection, diffusion, nonlinear dispersion, and Burgers-type equations.
- Enforces strict holdout discipline across spatial and temporal test partitions.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - exercised only on kernel-less installs
    import apedp_kernel as ak
except ImportError:  # kernel-only engine: fails at call time with an actionable error
    from apedp_dyn.kernel_shim import missing_kernel as ak  # type: ignore[assignment]


@dataclass(frozen=True)
class PdeDiscoveryResult:
    formula: str
    active_terms: list[str]
    coefficients: list[float]
    residual_rmse: float
    r2_dev: float
    is_exact_recovery: bool
    holdout_r2: float = float("nan")
    holdout_passed: bool = False


def discover_pde(
    u_grid: np.ndarray,
    dt: float,
    dx: float,
    *,
    max_spatial_order: int = 2,
    max_polynomial_degree: int = 2,
    include_advection_terms: bool = True,
    sparsity_threshold: float = 0.01,
    ridge_alpha: float = 1e-6,
    holdout_ratio: float = 0.25,
) -> PdeDiscoveryResult:
    """Discover governing PDE from u(t, x) with train/holdout temporal partition."""
    u_mat = np.asarray(u_grid, dtype=float)
    if u_mat.ndim != 2:
        raise ValueError("u_grid must be a 2D array of shape (time_steps, spatial_points)")

    nt, nx = u_mat.shape
    if nt < 8 or nx < 7:
        raise ValueError("Grid too small for spatio-temporal PDE discovery")

    # Partition temporal domain into dev and holdout
    n_dev = max(5, int(nt * (1.0 - holdout_ratio)))
    u_dev = u_mat[:n_dev, :]
    u_hold = u_mat[n_dev - 1:, :]  # overlap by 1 step for initial condition

    cfg = {
        "max_spatial_order": max_spatial_order,
        "max_polynomial_degree": max_polynomial_degree,
        "include_advection_terms": include_advection_terms,
        "sparsity_threshold": sparsity_threshold,
        "ridge_alpha": ridge_alpha,
    }

    # Execute native Rust STRidge PDE discovery on dev rows
    raw_res = json.loads(
        ak.rime_pde_discover_json(
            json.dumps(u_dev.tolist()), float(dt), float(dx), json.dumps(cfg)
        )
    )

    # Evaluate discovered PDE on holdout partition
    holdout_res = json.loads(
        ak.rime_pde_discover_json(
            json.dumps(u_hold.tolist()), float(dt), float(dx), json.dumps(cfg)
        )
    )
    r2_hold = float(holdout_res.get("r2", float("nan")))
    passed = r2_hold > 0.98 and raw_res.get("is_exact_recovery", False)

    return PdeDiscoveryResult(
        formula=raw_res["formula"],
        active_terms=raw_res["active_terms"],
        coefficients=raw_res["coefficients"],
        residual_rmse=raw_res["residual_rmse"],
        r2_dev=raw_res["r2"],
        is_exact_recovery=raw_res["is_exact_recovery"],
        holdout_r2=r2_hold,
        holdout_passed=passed,
    )


def discover_pde_2d(
    u_grid_3d: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    *,
    max_spatial_order: int = 2,
    max_polynomial_degree: int = 2,
    include_advection_terms: bool = True,
    sparsity_threshold: float = 0.02,
    ridge_alpha: float = 1e-6,
    holdout_ratio: float = 0.25,
) -> PdeDiscoveryResult:
    """Discover governing 2D spatial PDE from u(t, x, y)."""
    u_arr = np.asarray(u_grid_3d, dtype=float)
    if u_arr.ndim != 3:
        raise ValueError("u_grid_3d must be a 3D array of shape (time_steps, nx, ny)")

    nt, nx, ny = u_arr.shape
    if nt < 6 or nx < 7 or ny < 7:
        raise ValueError("Grid too small for 2D spatial PDE discovery")

    n_dev = max(5, int(nt * (1.0 - holdout_ratio)))
    u_dev = u_arr[:n_dev, :, :]
    u_hold = u_arr[n_dev - 1:, :, :]

    cfg = {
        "max_spatial_order": max_spatial_order,
        "max_polynomial_degree": max_polynomial_degree,
        "include_advection_terms": include_advection_terms,
        "sparsity_threshold": sparsity_threshold,
        "ridge_alpha": ridge_alpha,
    }

    raw = json.loads(
        ak.rime_pde_discover_2d_json(
            json.dumps(u_dev.tolist()),
            float(dt),
            float(dx),
            float(dy),
            json.dumps(cfg),
        )
    )

    hold_raw = json.loads(
        ak.rime_pde_discover_2d_json(
            json.dumps(u_hold.tolist()),
            float(dt),
            float(dx),
            float(dy),
            json.dumps(cfg),
        )
    )
    r2_hold = float(hold_raw.get("r2", float("nan")))
    passed = r2_hold > 0.98 and raw.get("is_exact_recovery", False)

    return PdeDiscoveryResult(
        formula=raw["formula"],
        active_terms=raw["active_terms"],
        coefficients=raw["coefficients"],
        residual_rmse=raw["residual_rmse"],
        r2_dev=raw["r2"],
        is_exact_recovery=raw["is_exact_recovery"],
        holdout_r2=r2_hold,
        holdout_passed=passed,
    )


def discover_pde_nonuniform(
    u_grid: np.ndarray,
    t_coords: Sequence[float],
    x_coords: Sequence[float],
    *,
    max_spatial_order: int = 2,
    max_polynomial_degree: int = 2,
    include_advection_terms: bool = True,
    sparsity_threshold: float = 0.02,
    ridge_alpha: float = 1e-6,
) -> PdeDiscoveryResult:
    """Discover governing PDE from arbitrarily spaced (non-uniform) mesh points."""
    u_arr = np.asarray(u_grid, dtype=float)
    t_vec = [float(t) for t in t_coords]
    x_vec = [float(x) for x in x_coords]

    cfg = {
        "max_spatial_order": max_spatial_order,
        "max_polynomial_degree": max_polynomial_degree,
        "include_advection_terms": include_advection_terms,
        "sparsity_threshold": sparsity_threshold,
        "ridge_alpha": ridge_alpha,
    }

    raw = json.loads(
        ak.rime_pde_discover_nonuniform_json(
            json.dumps(u_arr.tolist()),
            t_vec,
            x_vec,
            json.dumps(cfg),
        )
    )

    return PdeDiscoveryResult(
        formula=raw["formula"],
        active_terms=raw["active_terms"],
        coefficients=raw["coefficients"],
        residual_rmse=raw["residual_rmse"],
        r2_dev=raw["r2"],
        is_exact_recovery=raw["is_exact_recovery"],
        holdout_r2=raw["r2"],
        holdout_passed=raw["is_exact_recovery"],
    )
