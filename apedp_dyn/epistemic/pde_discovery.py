"""Spatio-Temporal PDE Discovery & Epistemic Verification in 1D and 2D.

Bridges the native Rust PDE Jet engine (rime/pde.rs) into APEDP-X:
- Discovers continuous field PDEs from space-time grids u(t, x) and 2D spatial fields u(t, x, y).
- Supports arbitrary non-uniform spatial meshes via exact Fornberg weights.
- Solves for advection, diffusion, nonlinear dispersion, and Burgers-type equations.
- Enforces strict holdout discipline across spatial and temporal test partitions.
"""
from __future__ import annotations

import json
import math
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


# ── Frozen-operator holdout scoring ────────────────────────────────────────
# The holdout stages previously called rime_pde_discover_* on the holdout
# partition itself: that re-FITS a second model to the holdout data, so dev and
# holdout could each report a high in-sample fit for entirely different
# operators. Holdout validation must FREEZE the coefficients discovered on the
# develop partition and score the SAME operator's residuals on the holdout
# rows. The finite-difference stencils below mirror the Rust kernel exactly
# (4th-order central spatial stencils, centered temporal differences).


def _fornberg_centered_weights(x_sub: Sequence[float], z: float, m_max: int) -> np.ndarray:
    """Fornberg stencil weights for derivatives up to ``m_max`` at ``z``.

    Equivalent to the kernel's ``fornberg_weights``: with centered offsets
    ``xi_k = x_k - z``, row m solves ``sum_k w_k * xi_k**p = m! * delta_{m,p}``.
    """
    xi = np.subtract(np.asarray(x_sub, dtype=float), float(z))
    n = xi.size
    vander = np.vander(xi, n, increasing=True)  # rows k: [1, xi_k, ..., xi_k^(n-1)]
    weights = np.zeros((m_max + 1, n), dtype=float)
    for m_i in range(m_max + 1):
        rhs = np.zeros(n, dtype=float)
        rhs[m_i] = math.factorial(m_i)
        # Exactness is sum_k w_k xi_k^p = m! delta_{m,p}, i.e. vander^T @ w = rhs
        # (the transposed solve; solving vander @ w instead solves the
        # interpolation problem and produces garbage weights).
        weights[m_i] = np.linalg.solve(vander.T, rhs)
    return weights


def _frozen_residual_r2(
    ut: np.ndarray,
    columns_by_name: dict[str, np.ndarray],
    active_terms: list[str],
    coefficients: list[float],
) -> float:
    """R^2 of the FROZEN operator (dev-discovered coefficients) against u_t."""
    coeff_by_term = dict(zip(active_terms, coefficients, strict=True))
    prediction = np.zeros(ut.shape[0], dtype=float)
    for term, coeff in coeff_by_term.items():
        column = columns_by_name.get(term)
        if column is not None:
            prediction += coeff * column
    residual = ut - prediction
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((ut - ut.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0


def _eval_frozen_1d_library(
    u_grid: np.ndarray,
    dt: float,
    dx: float,
    *,
    max_spatial_order: int,
    max_polynomial_degree: int,
    include_advection_terms: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Mirror the Rust ``discover_pde_1d`` library build for frozen scoring."""
    m, n = u_grid.shape
    ut_a: list[float] = []
    u_vals: list[float] = []
    ux: list[float] = []
    uxx: list[float] = []
    for i in range(1, m - 1):
        for j in range(2, n - 2):
            ut_a.append((u_grid[i + 1][j] - u_grid[i - 1][j]) / (2.0 * dt))
            u_vals.append(u_grid[i][j])
            ux.append((-u_grid[i][j + 2] + 8.0 * u_grid[i][j + 1] - 8.0 * u_grid[i][j - 1] + u_grid[i][j - 2]) / (12.0 * dx))
            uxx.append((-u_grid[i][j + 2] + 16.0 * u_grid[i][j + 1] - 30.0 * u_grid[i][j]
                        + 16.0 * u_grid[i][j - 1] - u_grid[i][j - 2]) / (12.0 * dx * dx))
    ut = np.asarray(ut_a, dtype=float)
    columns: dict[str, np.ndarray] = {}
    columns["u"] = np.asarray(u_vals, dtype=float)
    if max_polynomial_degree >= 2:
        columns["u^2"] = columns["u"] ** 2
    if max_polynomial_degree >= 3:
        columns["u^3"] = columns["u"] ** 3
    columns["u_x"] = np.asarray(ux, dtype=float)
    if include_advection_terms:
        columns["u*u_x"] = columns["u"] * columns["u_x"]
        if max_polynomial_degree >= 2:
            columns["u^2*u_x"] = columns["u"] ** 2 * columns["u_x"]
    if max_spatial_order >= 2:
        columns["u_xx"] = np.asarray(uxx, dtype=float)
        if include_advection_terms:
            columns["u*u_xx"] = columns["u"] * columns["u_xx"]
    if max_spatial_order >= 3:
        ux3: list[float] = []
        for i in range(1, m - 1):
            for j in range(2, n - 2):
                ux3.append((u_grid[i][j + 2] - 2.0 * u_grid[i][j + 1] + 2.0 * u_grid[i][j - 1] - u_grid[i][j - 2]) / (2.0 * dx ** 3))
        columns["u_xxx"] = np.asarray(ux3, dtype=float)
    return ut, columns


def _eval_frozen_2d_library(
    u_arr: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    *,
    max_polynomial_degree: int,
    include_advection_terms: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Mirror the Rust ``discover_pde_2d`` library build for frozen scoring."""
    nt, nx, ny = u_arr.shape
    ut_a: list[float] = []
    u_vals: list[float] = []
    ux_a: list[float] = []
    uy_a: list[float] = []
    uxx_a: list[float] = []
    uyy_a: list[float] = []
    for t in range(1, nt - 1):
        for x in range(2, nx - 2):
            for y in range(2, ny - 2):
                ut_a.append((u_arr[t + 1][x][y] - u_arr[t - 1][x][y]) / (2.0 * dt))
                u_vals.append(u_arr[t][x][y])
                ux_a.append((-u_arr[t][x + 2][y] + 8.0 * u_arr[t][x + 1][y] - 8.0 * u_arr[t][x - 1][y] + u_arr[t][x - 2][y]) / (12.0 * dx))
                uy_a.append((-u_arr[t][x][y + 2] + 8.0 * u_arr[t][x][y + 1] - 8.0 * u_arr[t][x][y - 1] + u_arr[t][x][y - 2]) / (12.0 * dy))
                uxx_a.append((-u_arr[t][x + 2][y] + 16.0 * u_arr[t][x + 1][y] - 30.0 * u_arr[t][x][y]
                              + 16.0 * u_arr[t][x - 1][y] - u_arr[t][x - 2][y]) / (12.0 * dx * dx))
                uyy_a.append((-u_arr[t][x][y + 2] + 16.0 * u_arr[t][x][y + 1] - 30.0 * u_arr[t][x][y]
                              + 16.0 * u_arr[t][x][y - 1] - u_arr[t][x][y - 2]) / (12.0 * dy * dy))
    ut = np.asarray(ut_a, dtype=float)
    columns: dict[str, np.ndarray] = {}
    columns["u"] = np.asarray(u_vals, dtype=float)
    if max_polynomial_degree >= 2:
        columns["u^2"] = columns["u"] ** 2
    columns["u_x"] = np.asarray(ux_a, dtype=float)
    columns["u_y"] = np.asarray(uy_a, dtype=float)
    u_xx = np.asarray(uxx_a, dtype=float)
    u_yy = np.asarray(uyy_a, dtype=float)
    columns["laplacian(u)"] = u_xx + u_yy
    columns["u_xx"] = u_xx
    columns["u_yy"] = u_yy
    if include_advection_terms:
        columns["u*u_x"] = columns["u"] * columns["u_x"]
        columns["u*u_y"] = columns["u"] * columns["u_y"]
    return ut, columns


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

    # FROZEN-operator holdout validation: the coefficients discovered on dev
    # are fixed and residuals of the SAME operator are scored on the holdout
    # rows. Calling rime_pde_discover_json on the holdout again would re-fit a
    # second model to holdout data, letting train and holdout certify entirely
    # different operators on in-sample fits alone.
    ut_hold, cols_hold = _eval_frozen_1d_library(
        u_hold,
        float(dt),
        float(dx),
        max_spatial_order=max_spatial_order,
        max_polynomial_degree=max_polynomial_degree,
        include_advection_terms=include_advection_terms,
    )
    r2_hold = _frozen_residual_r2(
        ut_hold, cols_hold, raw_res["active_terms"], raw_res["coefficients"]
    )
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

    # FROZEN-operator holdout validation (see discover_pde): dev coefficients
    # are frozen and the same operator is scored on holdout residuals, instead
    # of re-fitting a fresh model on the holdout partition.
    ut_hold, cols_hold = _eval_frozen_2d_library(
        u_hold,
        float(dt),
        float(dx),
        float(dy),
        max_polynomial_degree=max_polynomial_degree,
        include_advection_terms=include_advection_terms,
    )
    r2_hold = _frozen_residual_r2(
        ut_hold, cols_hold, raw["active_terms"], raw["coefficients"]
    )
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
    holdout_ratio: float = 0.25,
) -> PdeDiscoveryResult:
    """Discover governing PDE from arbitrarily spaced (non-uniform) mesh points.

    The grid is partitioned temporally into develop/holdout parts (sharing one
    boundary step for the initial condition). Discovery runs on the develop
    part only; holdout validation scores the FROZEN operator — evaluated with
    the same Fornberg stencils — against holdout residuals. The previous
    implementation never validated a holdout at all: it copied the in-sample
    ``r2`` / ``is_exact_recovery`` straight into the holdout fields.
    """
    u_arr = np.asarray(u_grid, dtype=float)
    t_vec = [float(t) for t in t_coords]
    x_vec = [float(x) for x in x_coords]
    if u_arr.ndim != 2 or u_arr.shape[0] != len(t_vec):
        raise ValueError("u_grid must be (n_time, n_space) matching len(t_coords)")

    nt = len(t_vec)
    if nt < 8:
        raise ValueError("Non-uniform PDE discovery requires at least 8 time steps for holdout partitioning")
    n_dev = max(5, int(nt * (1.0 - holdout_ratio)))
    dev_idx = list(range(n_dev))
    hold_idx = list(range(n_dev - 1, nt))  # overlap by 1 step for initial condition

    u_dev = u_arr[dev_idx, :]
    t_dev = [t_vec[i] for i in dev_idx]
    u_hold = u_arr[hold_idx, :]
    t_hold = [t_vec[i] for i in hold_idx]

    cfg = {
        "max_spatial_order": max_spatial_order,
        "max_polynomial_degree": max_polynomial_degree,
        "include_advection_terms": include_advection_terms,
        "sparsity_threshold": sparsity_threshold,
        "ridge_alpha": ridge_alpha,
    }

    raw = json.loads(
        ak.rime_pde_discover_nonuniform_json(
            json.dumps(u_dev.tolist()),
            t_dev,
            x_vec,
            json.dumps(cfg),
        )
    )

    # FROZEN-operator holdout validation (see discover_pde): the Fornberg
    # library built on the holdout part is scored against the dev coefficients,
    # never against a second fit on holdout data.
    ut_hold: list[float] = []
    col_u: list[float] = []
    col_ux: list[float] = []
    col_uxx: list[float] = []
    n_sp = len(x_vec)
    if n_sp < 5:
        raise ValueError("Non-uniform PDE discovery requires at least 5 spatial points")
    for i in range(1, len(hold_idx) - 1):
        t_sub = [t_hold[i - 1], t_hold[i], t_hold[i + 1]]
        w_t = _fornberg_centered_weights(t_sub, t_hold[i], 1)
        for j in range(2, n_sp - 2):
            x_sub = x_vec[j - 2: j + 3]
            w_x = _fornberg_centered_weights(x_sub, x_vec[j], 2)
            vals = [u_hold[i - 1][j], u_hold[i][j], u_hold[i + 1][j]]
            ut_hold.append(w_t[1] @ vals)
            col_u.append(u_hold[i][j])
            col_ux.append(w_x[1] @ u_hold[i][j - 2: j + 3])
            col_uxx.append(w_x[2] @ u_hold[i][j - 2: j + 3])
    columns_by_name = {"u": np.asarray(col_u, dtype=float), "u_x": np.asarray(col_ux, dtype=float),
                       "u_xx": np.asarray(col_uxx, dtype=float)}
    if max_polynomial_degree >= 2:
        columns_by_name["u^2"] = columns_by_name["u"] ** 2
    if include_advection_terms:
        columns_by_name["u*u_x"] = columns_by_name["u"] * columns_by_name["u_x"]
    r2_hold = _frozen_residual_r2(
        np.asarray(ut_hold, dtype=float),
        columns_by_name,
        raw["active_terms"],
        raw["coefficients"],
    )
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
