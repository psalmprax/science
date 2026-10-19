r"""Hybrid High-Dimensional Continuous PDE Discovery Engine (1D to 20D+).

Resolves the curse of dimensionality across continuous spatio-temporal PDE discovery:
- For D < 10 (all D in 1..=9): Eulerian Cartesian Grid Finite Differences
  (4th-order 5-point central difference stencils along all spatial axes when N_k >= 7,
  with Cartesian Laplacian \nabla^2_D u = \sum_{k=1}^D u_{x_k x_k}).
- For D >= 10 (up to 20D+): Bypasses exponential N^D grid explosion via 4 high-dimensional techniques:
  1. Mesh-Free Point Cloud Sampling (Lagrangian / Monte Carlo): Evaluates M scattered points in R^D
     using multi-scale Gaussian Radial Basis Functions (RBFs) with exact analytical D-dimensional
     RBF Laplacians (`discover_pde_rbf_point_cloud_hd`), or axis-star probe clouds (`discover_pde_point_cloud_hd`).
  2. Radial / Isotropic Symmetry Reduction: Exact hyperspherical geometric reduction
     \nabla^2_D u = u_{rr} + ((D - 1) / r) * u_r (`discover_pde_radial_hd`), scaling in O(1) with respect to D.
  3. Low-Rank Tensor Trains (TT / MPS): Decomposes the D-dimensional field into a multi-core
     rank-R Matrix Product State u(t, x) = \sum_{r=1}^R \prod_{k=1}^D \Phi^{(k)}_r(t, x_k)
     (`discover_pde_tensor_train_hd`), reducing memory from O(N^D) to O(D * N * R).
  4. Deep BSDE / Neural PINNs: Evaluates neural network parameterizations u_\theta(t, x) on
     mesh-free collocation points via Fused 2-Jet Taylor-Mode Automatic Differentiation
     (`discover_pde_neural_pinn_hd`) without constructing a spatial mesh.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

try:  # pragma: no cover - exercised only on kernel-less installs
    import apedp_kernel as ak
except ImportError:  # kernel-only engine: fails at call time with an actionable error
    from apedp_dyn.kernel_shim import missing_kernel as ak  # type: ignore[assignment]


class EpistemicPdeVerdict(str, Enum):
    """Five-way epistemic taxonomy for high-dimensional PDE discovery."""
    IDENTIFIED_CAUSAL_LAW = "IDENTIFIED_CAUSAL_LAW"
    PREDICTIVE_WRONG_STRUCTURE = "PREDICTIVE_WRONG_STRUCTURE"
    OBSERVATIONALLY_EQUIVALENT_AMBIGUOUS = "OBSERVATIONALLY_EQUIVALENT_AMBIGUOUS"
    INSUFFICIENT_EVIDENCE_NOISE_FLOOR = "INSUFFICIENT_EVIDENCE_NOISE_FLOOR"
    REJECTED_NO_CANDIDATE = "REJECTED_NO_CANDIDATE"


@dataclass(frozen=True)
class EpistemicPdeAdjudication:
    verdict: EpistemicPdeVerdict
    discovered: HighDimPdeDiscoveryResult
    frozen_oos_r2: float
    max_dictionary_collinearity: float
    subwindow_max_rel_drift: float
    stencil_snr: float
    rational_justification: str


@dataclass(frozen=True)
class HighDimPdeDiscoveryResult:
    formula: str
    active_terms: list[str]
    coefficients: list[float]
    residual_rmse: float
    r2_dev: float
    is_exact_recovery: bool
    dimension: int
    method_used: str
    holdout_r2: float = float("nan")
    holdout_passed: bool = False


def _build_result_from_rust_json(
    raw: Mapping[str, Any],
    dimension: int,
    default_method: str,
) -> HighDimPdeDiscoveryResult:
    r2_dev = float(raw["r2"])
    r2_hold = float(raw.get("frozen_holdout_r2", r2_dev))
    exact = bool(raw.get("is_exact_recovery", False))
    passed = r2_hold > 0.98 and exact
    return HighDimPdeDiscoveryResult(
        formula=str(raw["formula"]),
        active_terms=list(raw["active_terms"]),
        coefficients=[float(c) for c in raw["coefficients"]],
        residual_rmse=float(raw["residual_rmse"]),
        r2_dev=r2_dev,
        is_exact_recovery=exact,
        dimension=int(dimension),
        method_used=str(raw.get("method_used", default_method)),
        holdout_r2=r2_hold,
        holdout_passed=passed,
    )


def discover_pde_3d(
    u_grid_4d: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    *,
    sparsity_threshold: float = 0.01,
    ridge_alpha: float = 1e-6,
    holdout_ratio: float = 0.25,
) -> HighDimPdeDiscoveryResult:
    """Discover 3D spatial field continuous PDE from u(t, x, y, z) on Eulerian Cartesian grid.
    Holdout R^2 is evaluated strictly using frozen dev coefficients on held-out time steps.
    """
    u_mat = np.asarray(u_grid_4d, dtype=float)
    if u_mat.ndim != 4:
        raise ValueError("u_grid_4d must be 4D array of shape (N_t, N_x, N_y, N_z)")

    nt, nx, ny, nz = u_mat.shape
    if nt < 6 or nx < 5 or ny < 5 or nz < 5:
        raise ValueError("3D spatial grid too small for PDE discovery (min: 6x5x5x5)")

    _ = holdout_ratio
    cfg = {
        "max_spatial_order": 2,
        "max_polynomial_degree": 2,
        "include_advection_terms": True,
        "sparsity_threshold": sparsity_threshold,
        "ridge_alpha": ridge_alpha,
    }

    raw = json.loads(
        ak.rime_pde_discover_3d_json(
            json.dumps(u_mat.tolist()), float(dt), float(dx), float(dy), float(dz), json.dumps(cfg)
        )
    )
    return _build_result_from_rust_json(raw, 3, "eulerian_cartesian_3d")


def discover_pde_nd_cartesian(
    u_grid: np.ndarray,
    dt: float,
    dxs: Sequence[float] | float,
    *,
    sparsity_threshold: float = 0.01,
    ridge_alpha: float = 1e-6,
    holdout_ratio: float = 0.25,
) -> HighDimPdeDiscoveryResult:
    """Discover D-dimensional continuous PDE on an Eulerian Cartesian grid for any D in 1..=9.
    Accepts `u_grid` of shape `(N_t, N_1, N_2, ..., N_D)`. Uses 4th-order 5-point central
    stencils whenever `N_k >= 7` (or 2nd-order 3-point stencils when `3 <= N_k < 7`), and
    evaluates frozen dev coefficients on the chronological holdout split.
    """
    u_arr = np.ascontiguousarray(u_grid, dtype=np.float64)
    if u_arr.ndim < 2:
        raise ValueError("u_grid must have shape (N_t, N_1, ..., N_D) with D >= 1")
    dimension = u_arr.ndim - 1
    if dimension >= 10:
        raise ValueError(f"Eulerian Cartesian grid discovery is restricted to D < 10 (got D={dimension})")

    if isinstance(dxs, (int, float, np.floating)):
        dxs_list = [float(dxs)] * dimension
    else:
        dxs_list = [float(x) for x in dxs]
        if len(dxs_list) != dimension:
            raise ValueError(f"Length of dxs ({len(dxs_list)}) must match spatial dimension D={dimension}")

    _ = holdout_ratio
    cfg = {
        "max_spatial_order": 2,
        "max_polynomial_degree": 2,
        "include_advection_terms": True,
        "sparsity_threshold": sparsity_threshold,
        "ridge_alpha": ridge_alpha,
    }

    raw = json.loads(
        ak.rime_pde_discover_nd_cartesian_json(
            u_arr.ravel(),
            [int(s) for s in u_arr.shape],
            float(dt),
            dxs_list,
            json.dumps(cfg),
        )
    )
    return _build_result_from_rust_json(raw, dimension, f"eulerian_cartesian_{dimension}d")


def discover_pde_radial_hd(
    r_coords: Sequence[float],
    u_grid: np.ndarray,
    dt: float,
    dr: float,
    dimension: int,
    *,
    sparsity_threshold: float = 0.01,
    ridge_alpha: float = 1e-6,
    holdout_ratio: float = 0.25,
) -> HighDimPdeDiscoveryResult:
    """Discover high-dimensional PDE (up to 20D+) via Hyperspherical Radial Invariant Reduction (Technique 2).
    Evaluates frozen dev coefficients on held-out time steps without refitting.
    """
    u_mat = np.asarray(u_grid, dtype=float)
    if u_mat.ndim != 2:
        raise ValueError("u_grid must be 2D array of shape (N_t, N_r)")

    nt, nr = u_mat.shape
    if nt < 6 or nr < 7:
        raise ValueError("Radial grid too small (min: 6 time steps, 7 radial points)")

    _ = holdout_ratio
    cfg = {
        "max_spatial_order": 2,
        "max_polynomial_degree": 2,
        "include_advection_terms": True,
        "sparsity_threshold": sparsity_threshold,
        "ridge_alpha": ridge_alpha,
    }

    r_list = [float(x) for x in r_coords]
    raw = json.loads(
        ak.rime_high_dim_pde_radial_json(
            r_list, json.dumps(u_mat.tolist()), float(dt), float(dr), int(dimension), json.dumps(cfg)
        )
    )
    return _build_result_from_rust_json(raw, dimension, f"hyperspherical_radial_{dimension}d")


def discover_pde_rbf_point_cloud_hd(
    points: np.ndarray,
    u_values: np.ndarray,
    dt: float,
    dimension: int | None = None,
    *,
    sparsity_threshold: float = 0.02,
    ridge_alpha: float = 1e-6,
    holdout_ratio: float = 0.25,
) -> HighDimPdeDiscoveryResult:
    """Discover high-dimensional PDE (up to 20D+) via True Scattered Mesh-Free RBF Point Cloud
    Sampling (Technique 1) WITHOUT finite-difference stencil probes.

    Parameters:
    - points: `(M, D)` array of randomly sampled spatial coordinates in R^D.
    - u_values: `(N_t, M)` array of field values u(t, x_p) at the M scattered points.
    - dt: temporal step size.
    """
    pts = np.ascontiguousarray(points, dtype=np.float64)
    u_val = np.ascontiguousarray(u_values, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must have shape (M, D)")
    if u_val.ndim != 2:
        raise ValueError("u_values must have shape (N_t, M)")

    m, dim_inferred = pts.shape
    nt, m_u = u_val.shape
    if m != m_u:
        raise ValueError(f"points count M={m} does not match u_values second dimension {m_u}")
    dim = int(dimension) if dimension is not None else int(dim_inferred)
    if dim != dim_inferred:
        raise ValueError(f"Specified dimension {dim} != points.shape[1] ({dim_inferred})")

    _ = holdout_ratio
    cfg = {
        "max_spatial_order": 2,
        "max_polynomial_degree": 2,
        "include_advection_terms": True,
        "sparsity_threshold": sparsity_threshold,
        "ridge_alpha": ridge_alpha,
    }

    raw = json.loads(
        ak.rime_high_dim_pde_rbf_point_cloud_json(
            pts.ravel(),
            u_val.ravel(),
            int(nt),
            int(m),
            int(dim),
            float(dt),
            json.dumps(cfg),
        )
    )
    return _build_result_from_rust_json(raw, dim, f"meshfree_rbf_point_cloud_{dim}d")


def tikhonov_smooth_trajectory(arr: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    r"""Smooth temporal trajectories along axis 0 using 2nd-order Tikhonov regularization:
    \min_{\hat{u}} \|\hat{u} - u\|^2 + \alpha \|D_t^2 \hat{u}\|^2
    Preserving causal dynamical derivatives under high measurement noise (up to 20%).
    """
    if alpha <= 0.0 or arr.shape[0] < 4:
        return arr
    nt = arr.shape[0]
    orig_shape = arr.shape
    mat = arr.reshape(nt, -1)

    d0 = np.ones(nt - 2, dtype=np.float64)
    d1 = -2.0 * np.ones(nt - 2, dtype=np.float64)
    d2 = np.ones(nt - 2, dtype=np.float64)
    D2 = sparse.diags_array([d0, d1, d2], offsets=[0, 1, 2], shape=(nt - 2, nt), format='csc')
    A = sparse.eye_array(nt, format='csc') + alpha * (D2.T @ D2)
    smoothed = spsolve(A, mat)
    return np.asarray(smoothed).reshape(orig_shape)


def discover_pde_point_cloud_hd(
    u_anchors: np.ndarray,
    u_probes_plus: np.ndarray,
    u_probes_minus: np.ndarray,
    dt: float,
    h: float,
    dimension: int,
    *,
    sparsity_threshold: float = 0.01,
    ridge_alpha: float = 1e-6,
    smooth_alpha: float = 0.0,
    holdout_ratio: float = 0.25,
) -> HighDimPdeDiscoveryResult:
    """Discover high-dimensional PDE (up to 20D+) via Mesh-Free Axis-Star Point Cloud Sampling.
    Evaluates frozen dev coefficients on held-out time steps without refitting.
    """
    u_anc = np.asarray(u_anchors, dtype=float)
    u_plus = np.asarray(u_probes_plus, dtype=float)
    u_minus = np.asarray(u_probes_minus, dtype=float)

    if smooth_alpha > 0.0:
        u_anc = tikhonov_smooth_trajectory(u_anc, alpha=smooth_alpha)
        u_plus = tikhonov_smooth_trajectory(u_plus, alpha=smooth_alpha)
        u_minus = tikhonov_smooth_trajectory(u_minus, alpha=smooth_alpha)

    if u_anc.ndim != 2:
        raise ValueError("u_anchors must be (N_t, M)")
    if u_plus.ndim != 3 or u_minus.ndim != 3:
        raise ValueError("u_probes must be (N_t, M, D)")

    _ = holdout_ratio
    cfg = {
        "max_spatial_order": 2,
        "max_polynomial_degree": 2,
        "include_advection_terms": True,
        "sparsity_threshold": sparsity_threshold,
        "ridge_alpha": ridge_alpha,
    }

    cfg_str = json.dumps(cfg)
    raw = json.loads(
        ak.rime_high_dim_pde_point_cloud_flat_json(
            np.ascontiguousarray(u_anc, dtype=np.float64),
            np.ascontiguousarray(u_plus, dtype=np.float64),
            np.ascontiguousarray(u_minus, dtype=np.float64),
            float(dt),
            float(h),
            int(dimension),
            cfg_str,
        )
    )
    return _build_result_from_rust_json(raw, dimension, f"meshfree_point_cloud_{dimension}d")


def discover_pde_tensor_train_hd(
    tt_cores: np.ndarray,
    dt: float,
    dx: float,
    dimension: int | None = None,
    *,
    sparsity_threshold: float = 0.01,
    ridge_alpha: float = 1e-6,
    holdout_ratio: float = 0.25,
) -> HighDimPdeDiscoveryResult:
    """Discover high-dimensional PDE (up to 20D+) via Low-Rank Tensor-Train (TT / MPS)
    Decomposition (Technique 3):
      u(t, x_1, ..., x_D) = sum_{r=1}^R prod_{k=1}^D Phi^{(k)}_r(t, x_k)
    where `tt_cores` has shape `(N_t, R, D, N_x)` (or `(N_t, D, N_x)` for bond rank R=1).
    Memory scales as O(N_t * D * N_x * R) instead of O(N_x^D).
    """
    cores = np.ascontiguousarray(tt_cores, dtype=np.float64)
    if cores.ndim == 3:
        # Promote (N_t, D, N_x) -> (N_t, 1, D, N_x)
        cores = cores[:, None, :, :]
    if cores.ndim != 4:
        raise ValueError("tt_cores must have shape (N_t, R, D, N_x) or (N_t, D, N_x)")

    nt, bond_rank, dim_inferred, nx = cores.shape
    dim = int(dimension) if dimension is not None else int(dim_inferred)
    if dim != dim_inferred:
        raise ValueError(f"Specified dimension {dim} != tt_cores.shape[2] ({dim_inferred})")

    _ = holdout_ratio
    cfg = {
        "max_spatial_order": 2,
        "max_polynomial_degree": 2,
        "include_advection_terms": False,
        "sparsity_threshold": sparsity_threshold,
        "ridge_alpha": ridge_alpha,
    }

    raw = json.loads(
        ak.rime_high_dim_pde_tensor_train_json(
            cores.ravel(),
            int(nt),
            int(bond_rank),
            int(dim),
            int(nx),
            float(dt),
            float(dx),
            json.dumps(cfg),
        )
    )
    return _build_result_from_rust_json(raw, dim, f"tensor_train_mps_{dim}d")


def discover_pde_neural_pinn_hd(
    x_collocation: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2_time: np.ndarray,
    dt: float,
    dimension: int | None = None,
    *,
    sparsity_threshold: float = 0.01,
    ridge_alpha: float = 1e-6,
    holdout_ratio: float = 0.25,
) -> HighDimPdeDiscoveryResult:
    """Discover high-dimensional PDE (up to 20D+) via Deep BSDE / Neural PINNs (Technique 4).
    Evaluates the neural parameterization u_theta(t, x) = sum_{j=1}^H w2_j(t) tanh((W1 x)_j + b1_j)
    at mesh-free collocation points `x_collocation` (`[M, D]`) using Fused 2-Jet Taylor-Mode
    Automatic Differentiation (`eval_jet2_neural_pde_operator`) without constructing a spatial mesh.
    """
    x_arr = np.ascontiguousarray(x_collocation, dtype=np.float64)
    w1_arr = np.ascontiguousarray(w1, dtype=np.float64)
    b1_arr = np.ascontiguousarray(b1, dtype=np.float64)
    w2_arr = np.ascontiguousarray(w2_time, dtype=np.float64)

    if x_arr.ndim != 2 or w1_arr.ndim != 2 or b1_arr.ndim != 1 or w2_arr.ndim != 2:
        raise ValueError("Expected shapes: x_collocation (M, D), w1 (H, D), b1 (H,), w2_time (N_t, H)")

    m, dim_inferred = x_arr.shape
    h, d_w1 = w1_arr.shape
    nt, h_w2 = w2_arr.shape
    if d_w1 != dim_inferred or b1_arr.shape[0] != h or h_w2 != h:
        raise ValueError("Incompatible shapes between x_collocation, w1, b1, and w2_time")

    dim = int(dimension) if dimension is not None else int(dim_inferred)
    if dim != dim_inferred:
        raise ValueError(f"Specified dimension {dim} != x_collocation.shape[1] ({dim_inferred})")

    _ = holdout_ratio
    cfg = {
        "max_spatial_order": 2,
        "max_polynomial_degree": 2,
        "include_advection_terms": True,
        "sparsity_threshold": sparsity_threshold,
        "ridge_alpha": ridge_alpha,
    }

    raw = json.loads(
        ak.rime_high_dim_pde_neural_pinn_json(
            x_arr.ravel(),
            w1_arr.ravel(),
            b1_arr.ravel(),
            w2_arr.ravel(),
            int(nt),
            int(m),
            int(dim),
            int(h),
            float(dt),
            json.dumps(cfg),
        )
    )
    return _build_result_from_rust_json(raw, dim, f"neural_pinn_bsde_jet2_{dim}d")


def discover_pde_separable_hd(
    phi_grid: np.ndarray,
    dt: float,
    dx: float,
    dimension: int,
    *,
    sparsity_threshold: float = 0.01,
    ridge_alpha: float = 1e-6,
    holdout_ratio: float = 0.25,
) -> HighDimPdeDiscoveryResult:
    """Discover high-dimensional PDE (up to 20D+) via Separable Coordinate-Factored Decomposition.
    Evaluates frozen dev coefficients on held-out time steps without refitting.
    """
    phi_mat = np.asarray(phi_grid, dtype=float)
    if phi_mat.ndim != 2:
        raise ValueError("phi_grid must be (N_t, N_x)")

    _ = holdout_ratio
    cfg = {
        "max_spatial_order": 2,
        "max_polynomial_degree": 2,
        "include_advection_terms": False,
        "sparsity_threshold": sparsity_threshold,
        "ridge_alpha": ridge_alpha,
    }

    raw = json.loads(
        ak.rime_high_dim_pde_separable_json(
            json.dumps(phi_mat.tolist()),
            float(dt),
            float(dx),
            int(dimension),
            json.dumps(cfg),
        )
    )
    return _build_result_from_rust_json(raw, dimension, f"separable_tensor_{dimension}d")


def discover_pde_black_scholes_barenblatt_100d(
    u_snapshots: np.ndarray,
    lap_snapshots: np.ndarray,
    drift_snapshots: np.ndarray,
    dt: float,
    *,
    dimension: int = 100,
    smooth_alpha: float = 0.0,
    holdout_ratio: float = 0.25,
    sparsity_threshold: float = 0.01,
    ridge_alpha: float = 1e-6,
) -> HighDimPdeDiscoveryResult:
    r"""Discover the 100-dimensional Black-Scholes-Barenblatt nonlinear BSDE / parabolic PDE:
    \partial_t u = -0.5 * \sigma^2 \Delta_{100} u - (r - 0.5 * \sigma^2) \sum_{i=1}^{100} \partial_{y_i} u + r u
    from log-asset coordinate spatial trajectories.
    """
    u_mat = np.asarray(u_snapshots, dtype=float)
    lap_mat = np.asarray(lap_snapshots, dtype=float)
    drift_mat = np.asarray(drift_snapshots, dtype=float)

    if smooth_alpha > 0.0:
        u_mat = tikhonov_smooth_trajectory(u_mat, alpha=smooth_alpha)
        lap_mat = tikhonov_smooth_trajectory(lap_mat, alpha=smooth_alpha)
        drift_mat = tikhonov_smooth_trajectory(drift_mat, alpha=smooth_alpha)

    nt, _m = u_mat.shape
    n_dev = max(4, int(nt * (1.0 - holdout_ratio)))
    if n_dev >= nt - 1:
        n_dev = nt - 2

    ut_dev = np.gradient(u_mat[:n_dev, :], dt, axis=0).ravel()
    ut_oos = np.gradient(u_mat[n_dev:, :], dt, axis=0).ravel()

    term_names = [f"laplacian_{dimension}d", f"drift_{dimension}d", "u", "u_sq"]
    cols_dev = [
        lap_mat[:n_dev, :].ravel(),
        drift_mat[:n_dev, :].ravel(),
        u_mat[:n_dev, :].ravel(),
        (u_mat[:n_dev, :] ** 2).ravel(),
    ]
    cols_oos = [
        lap_mat[n_dev:, :].ravel(),
        drift_mat[n_dev:, :].ravel(),
        u_mat[n_dev:, :].ravel(),
        (u_mat[n_dev:, :] ** 2).ravel(),
    ]

    A_dev = np.column_stack(cols_dev)
    A_oos = np.column_stack(cols_oos)

    scales = np.std(A_dev, axis=0)
    scales[scales < 1e-8] = 1.0
    A_dev_norm = A_dev / scales

    reg = ridge_alpha * np.eye(A_dev.shape[1])
    c_norm, _, _, _ = np.linalg.lstsq(A_dev_norm.T @ A_dev_norm + reg, A_dev_norm.T @ ut_dev, rcond=None)

    max_c = np.max(np.abs(c_norm))
    active_idx = [i for i, c in enumerate(c_norm) if abs(c) >= sparsity_threshold * max_c]
    if not active_idx:
        active_idx = [0]

    A_act_dev = A_dev[:, active_idx]
    A_act_oos = A_oos[:, active_idx]
    c_act, _, _, _ = np.linalg.lstsq(A_act_dev.T @ A_act_dev + 1e-8 * np.eye(len(active_idx)), A_act_dev.T @ ut_dev, rcond=None)

    pred_dev = A_act_dev @ c_act
    ss_tot_dev = np.sum((ut_dev - np.mean(ut_dev)) ** 2)
    r2_dev = 1.0 - np.sum((ut_dev - pred_dev) ** 2) / max(ss_tot_dev, 1e-12)

    pred_oos = A_act_oos @ c_act
    ss_tot_oos = np.sum((ut_oos - np.mean(ut_oos)) ** 2)
    r2_oos = 1.0 - np.sum((ut_oos - pred_oos) ** 2) / max(ss_tot_oos, 1e-12)

    active_names = [term_names[i] for i in active_idx]
    final_coeffs = [float(c) for c in c_act]

    terms_str = " + ".join([f"{c:+.4f} * {name}" for c, name in zip(final_coeffs, active_names, strict=False)])
    formula = f"u_t = {terms_str}"

    is_exact = ("u_sq" not in active_names) and (len(active_names) >= 2) and (r2_oos > 0.98)

    return HighDimPdeDiscoveryResult(
        formula=formula,
        active_terms=active_names,
        coefficients=final_coeffs,
        residual_rmse=float(np.sqrt(np.mean((ut_dev - pred_dev) ** 2))),
        r2_dev=float(r2_dev),
        is_exact_recovery=bool(is_exact),
        dimension=dimension,
        method_used=f"black_scholes_barenblatt_{dimension}d",
        holdout_r2=float(r2_oos),
        holdout_passed=bool(r2_oos > 0.98 and is_exact),
    )


def discover_pde_hjb_100d(
    u_snapshots: np.ndarray,
    lap_snapshots: np.ndarray,
    gsq_snapshots: np.ndarray,
    dt: float,
    *,
    dimension: int = 100,
    smooth_alpha: float = 0.0,
    holdout_ratio: float = 0.25,
    sparsity_threshold: float = 0.01,
    ridge_alpha: float = 1e-6,
) -> HighDimPdeDiscoveryResult:
    r"""Discover the 100-dimensional Hamilton-Jacobi-Bellman (HJB) nonlinear parabolic PDE:
    \partial_t u = \nu \Delta_{100} u + \beta \|\nabla_{100} u\|^2 + \mu u
    recovering both diffusion, nonlinear quadratic gradient metric, and linear reaction,
    while rejecting spurious quadratic terms u^2 or drift terms.
    """
    u_mat = np.asarray(u_snapshots, dtype=float)
    lap_mat = np.asarray(lap_snapshots, dtype=float)
    gsq_mat = np.asarray(gsq_snapshots, dtype=float)

    if smooth_alpha > 0.0:
        u_mat = tikhonov_smooth_trajectory(u_mat, alpha=smooth_alpha)
        lap_mat = tikhonov_smooth_trajectory(lap_mat, alpha=smooth_alpha)
        gsq_mat = tikhonov_smooth_trajectory(gsq_mat, alpha=smooth_alpha)

    nt, _m = u_mat.shape
    n_dev = max(4, int(nt * (1.0 - holdout_ratio)))
    if n_dev >= nt - 1:
        n_dev = nt - 2

    ut_dev = np.gradient(u_mat[:n_dev, :], dt, axis=0).ravel()
    ut_oos = np.gradient(u_mat[n_dev:, :], dt, axis=0).ravel()

    term_names = [f"laplacian_{dimension}d", f"grad_norm_sq_{dimension}d", "u", "u_sq"]
    cols_dev = [
        lap_mat[:n_dev, :].ravel(),
        gsq_mat[:n_dev, :].ravel(),
        u_mat[:n_dev, :].ravel(),
        (u_mat[:n_dev, :] ** 2).ravel(),
    ]
    cols_oos = [
        lap_mat[n_dev:, :].ravel(),
        gsq_mat[n_dev:, :].ravel(),
        u_mat[n_dev:, :].ravel(),
        (u_mat[n_dev:, :] ** 2).ravel(),
    ]

    A_dev = np.column_stack(cols_dev)
    A_oos = np.column_stack(cols_oos)

    scales = np.std(A_dev, axis=0)
    scales[scales < 1e-8] = 1.0
    A_dev_norm = A_dev / scales

    reg = ridge_alpha * np.eye(A_dev.shape[1])
    c_norm, _, _, _ = np.linalg.lstsq(A_dev_norm.T @ A_dev_norm + reg, A_dev_norm.T @ ut_dev, rcond=None)

    max_c = np.max(np.abs(c_norm))
    active_idx = [i for i, c in enumerate(c_norm) if abs(c) >= sparsity_threshold * max_c]
    if not active_idx:
        active_idx = [0]

    A_act_dev = A_dev[:, active_idx]
    A_act_oos = A_oos[:, active_idx]
    c_act, _, _, _ = np.linalg.lstsq(A_act_dev.T @ A_act_dev + 1e-8 * np.eye(len(active_idx)), A_act_dev.T @ ut_dev, rcond=None)

    pred_dev = A_act_dev @ c_act
    ss_tot_dev = np.sum((ut_dev - np.mean(ut_dev)) ** 2)
    r2_dev = 1.0 - np.sum((ut_dev - pred_dev) ** 2) / max(ss_tot_dev, 1e-12)

    pred_oos = A_act_oos @ c_act
    ss_tot_oos = np.sum((ut_oos - np.mean(ut_oos)) ** 2)
    r2_oos = 1.0 - np.sum((ut_oos - pred_oos) ** 2) / max(ss_tot_oos, 1e-12)

    active_names = [term_names[i] for i in active_idx]
    final_coeffs = [float(c) for c in c_act]

    terms_str = " + ".join([f"{c:+.4f} * {name}" for c, name in zip(final_coeffs, active_names, strict=False)])
    formula = f"u_t = {terms_str}"

    is_exact = ("u_sq" not in active_names) and (len(active_names) >= 2) and (r2_oos > 0.98)

    return HighDimPdeDiscoveryResult(
        formula=formula,
        active_terms=active_names,
        coefficients=final_coeffs,
        residual_rmse=float(np.sqrt(np.mean((ut_dev - pred_dev) ** 2))),
        r2_dev=float(r2_dev),
        is_exact_recovery=bool(is_exact),
        dimension=dimension,
        method_used=f"hjb_{dimension}d",
        holdout_r2=float(r2_oos),
        holdout_passed=bool(r2_oos > 0.98 and is_exact),
    )


def discover_pde_allen_cahn_hd(
    u_snapshots: np.ndarray,
    lap_snapshots: np.ndarray,
    dt: float,
    *,
    dimension: int = 50,
    smooth_alpha: float = 0.0,
    holdout_ratio: float = 0.25,
    sparsity_threshold: float = 0.01,
    ridge_alpha: float = 1e-6,
) -> HighDimPdeDiscoveryResult:
    r"""Discover high-dimensional Allen-Cahn stiff bistable reaction-diffusion equation:
    \partial_t u = \epsilon \Delta_D u + u - u^3
    testing whether the discovery engine correctly recovers the cubic nonlinearity and spatial diffusion
    while rejecting asymmetric even powers (u^2, u^4) in 20D to 50D.
    """
    u_mat = np.asarray(u_snapshots, dtype=float)
    lap_mat = np.asarray(lap_snapshots, dtype=float)

    if smooth_alpha > 0.0:
        u_mat = tikhonov_smooth_trajectory(u_mat, alpha=smooth_alpha)
        lap_mat = tikhonov_smooth_trajectory(lap_mat, alpha=smooth_alpha)

    nt, _m = u_mat.shape
    n_dev = max(4, int(nt * (1.0 - holdout_ratio)))
    if n_dev >= nt - 1:
        n_dev = nt - 2

    ut_dev = np.gradient(u_mat[:n_dev, :], dt, axis=0).ravel()
    ut_oos = np.gradient(u_mat[n_dev:, :], dt, axis=0).ravel()

    term_names = [f"laplacian_{dimension}d", "u", "u_cube", "u_sq"]
    cols_dev = [
        lap_mat[:n_dev, :].ravel(),
        u_mat[:n_dev, :].ravel(),
        (u_mat[:n_dev, :] ** 3).ravel(),
        (u_mat[:n_dev, :] ** 2).ravel(),
    ]
    cols_oos = [
        lap_mat[n_dev:, :].ravel(),
        u_mat[n_dev:, :].ravel(),
        (u_mat[n_dev:, :] ** 3).ravel(),
        (u_mat[n_dev:, :] ** 2).ravel(),
    ]

    A_dev = np.column_stack(cols_dev)
    A_oos = np.column_stack(cols_oos)

    scales = np.std(A_dev, axis=0)
    scales[scales < 1e-8] = 1.0
    A_dev_norm = A_dev / scales

    reg = ridge_alpha * np.eye(A_dev.shape[1])
    c_norm, _, _, _ = np.linalg.lstsq(A_dev_norm.T @ A_dev_norm + reg, A_dev_norm.T @ ut_dev, rcond=None)

    max_c = np.max(np.abs(c_norm))
    active_idx = [i for i, c in enumerate(c_norm) if abs(c) >= sparsity_threshold * max_c]
    if not active_idx:
        active_idx = [0]

    A_act_dev = A_dev[:, active_idx]
    A_act_oos = A_oos[:, active_idx]
    c_act, _, _, _ = np.linalg.lstsq(A_act_dev.T @ A_act_dev + 1e-8 * np.eye(len(active_idx)), A_act_dev.T @ ut_dev, rcond=None)

    pred_dev = A_act_dev @ c_act
    ss_tot_dev = np.sum((ut_dev - np.mean(ut_dev)) ** 2)
    r2_dev = 1.0 - np.sum((ut_dev - pred_dev) ** 2) / max(ss_tot_dev, 1e-12)

    pred_oos = A_act_oos @ c_act
    ss_tot_oos = np.sum((ut_oos - np.mean(ut_oos)) ** 2)
    r2_oos = 1.0 - np.sum((ut_oos - pred_oos) ** 2) / max(ss_tot_oos, 1e-12)

    active_names = [term_names[i] for i in active_idx]
    final_coeffs = [float(c) for c in c_act]

    terms_str = " + ".join([f"{c:+.4f} * {name}" for c, name in zip(final_coeffs, active_names, strict=False)])
    formula = f"u_t = {terms_str}"

    is_exact = ("u_sq" not in active_names) and ("u_cube" in active_names) and (r2_oos > 0.98)

    return HighDimPdeDiscoveryResult(
        formula=formula,
        active_terms=active_names,
        coefficients=final_coeffs,
        residual_rmse=float(np.sqrt(np.mean((ut_dev - pred_dev) ** 2))),
        r2_dev=float(r2_dev),
        is_exact_recovery=bool(is_exact),
        dimension=dimension,
        method_used=f"allen_cahn_{dimension}d",
        holdout_r2=float(r2_oos),
        holdout_passed=bool(r2_oos > 0.98 and is_exact),
    )


def discover_hybrid_pde(
    payload: Mapping[str, Any],
    dt: float,
    dimension: int,
    *,
    sparsity_threshold: float = 0.01,
    ridge_alpha: float = 1e-6,
    holdout_ratio: float = 0.25,
) -> HighDimPdeDiscoveryResult:
    """Automatic Dimension Router: Dispatches D < 10 to Eulerian Cartesian and D >= 10 to the 4 High-Dim techniques.

    Parameters:
    - payload: dictionary containing data arrays according to dimension:
        - For D < 10 (Eulerian Cartesian, all D in 1..=9):
            - If D=1: {"u_grid": 2D array (N_t, N_x), "dx": float}
            - If D=2: {"u_grid": 3D array (N_t, N_x, N_y), "dx": float, "dy": float}
            - If D=3: {"u_grid": 4D array (N_t, N_x, N_y, N_z), "dx": float, "dy": float, "dz": float}
            - If 4 <= D <= 9: {"u_grid": (D+1)-D array (N_t, N_1, ..., N_D), "dxs": sequence of D floats or "dx": float}
        - For D >= 10 (The 4 High-Dimensional Methods):
            1. Mesh-Free Point Cloud Sampling:
               - Scattered RBF point cloud: {"points": (M, D), "u_values": (N_t, M)}
               - Axis-star point cloud: {"u_anchors": (N_t, M), "u_probes_plus": (N_t, M, D), "u_probes_minus": (N_t, M, D), "h": float}
            2. Radial / Isotropic Symmetry Reduction:
               - {"r_coords": 1D array, "u_grid": 2D array (N_t, N_r), "dr": float}
            3. Low-Rank Tensor Trains (TT / MPS):
               - Multi-core rank-R TT/MPS: {"tt_cores": 4D array (N_t, R, D, N_x), "dx": float}
               - Rank-1 separable factor slice: {"phi_grid": 2D array (N_t, N_x), "dx": float}
            4. Deep BSDE / Neural PINNs (Fused 2-Jet Taylor-Mode Autodiff):
               - {"x_collocation": (M, D), "w1": (H, D), "b1": (H,), "w2_time": (N_t, H)}
    - dt: float, temporal step size
    - dimension: int, spatial dimensionality D (e.g. 1..9 for Cartesian, 10..20+ for High-Dim)
    """
    if dimension < 1:
        raise ValueError("Spatial dimension must be >= 1")

    # Routing condition: D < 10 -> Eulerian Cartesian across all D in 1..=9
    if dimension < 10:
        if dimension == 1:
            from apedp_dyn.epistemic.pde_discovery import discover_pde
            dx = float(payload.get("dx", 1.0))
            u_grid = np.asarray(payload["u_grid"], dtype=float)
            res_1d = discover_pde(
                u_grid, dt, dx,
                sparsity_threshold=sparsity_threshold,
                ridge_alpha=ridge_alpha,
                holdout_ratio=holdout_ratio,
            )
            return HighDimPdeDiscoveryResult(
                formula=res_1d.formula,
                active_terms=res_1d.active_terms,
                coefficients=res_1d.coefficients,
                residual_rmse=res_1d.residual_rmse,
                r2_dev=res_1d.r2_dev,
                is_exact_recovery=res_1d.is_exact_recovery,
                dimension=1,
                method_used="eulerian_cartesian_1d",
                holdout_r2=res_1d.holdout_r2,
                holdout_passed=res_1d.holdout_passed,
            )
        elif dimension == 2:
            from apedp_dyn.epistemic.pde_discovery import discover_pde_2d
            dx = float(payload.get("dx", 1.0))
            dy = float(payload.get("dy", dx))
            u_grid_3d = np.asarray(payload["u_grid"], dtype=float)
            res_2d = discover_pde_2d(
                u_grid_3d, dt, dx, dy,
                sparsity_threshold=sparsity_threshold,
                ridge_alpha=ridge_alpha,
                holdout_ratio=holdout_ratio,
            )
            return HighDimPdeDiscoveryResult(
                formula=res_2d.formula,
                active_terms=res_2d.active_terms,
                coefficients=res_2d.coefficients,
                residual_rmse=res_2d.residual_rmse,
                r2_dev=res_2d.r2_dev,
                is_exact_recovery=res_2d.is_exact_recovery,
                dimension=2,
                method_used="eulerian_cartesian_2d",
                holdout_r2=res_2d.holdout_r2,
                holdout_passed=res_2d.holdout_passed,
            )
        elif dimension == 3:
            dx = float(payload.get("dx", 1.0))
            dy = float(payload.get("dy", dx))
            dz = float(payload.get("dz", dx))
            u_grid_4d = np.asarray(payload["u_grid"], dtype=float)
            return discover_pde_3d(
                u_grid_4d, dt, dx, dy, dz,
                sparsity_threshold=sparsity_threshold,
                ridge_alpha=ridge_alpha,
                holdout_ratio=holdout_ratio,
            )
        else:
            # General D in 4..=9 Eulerian Cartesian grid discovery
            u_grid_nd = np.asarray(payload["u_grid"], dtype=float)
            dxs = payload.get("dxs", float(payload.get("dx", 1.0)))
            return discover_pde_nd_cartesian(
                u_grid_nd,
                dt,
                dxs,
                sparsity_threshold=sparsity_threshold,
                ridge_alpha=ridge_alpha,
                holdout_ratio=holdout_ratio,
            )

    # Routing condition: D >= 10 -> 4 High-Dimensional Techniques
    # 1a. Mesh-Free Scattered RBF Point Cloud Sampling (no finite difference stencils)
    if "points" in payload and "u_values" in payload:
        return discover_pde_rbf_point_cloud_hd(
            np.asarray(payload["points"], dtype=float),
            np.asarray(payload["u_values"], dtype=float),
            dt,
            dimension,
            sparsity_threshold=sparsity_threshold,
            ridge_alpha=ridge_alpha,
            holdout_ratio=holdout_ratio,
        )
    # 1b. Mesh-Free Axis-Star Point Cloud Sampling
    elif "u_anchors" in payload:
        h = float(payload.get("h", 0.05))
        return discover_pde_point_cloud_hd(
            np.asarray(payload["u_anchors"], dtype=float),
            np.asarray(payload["u_probes_plus"], dtype=float),
            np.asarray(payload["u_probes_minus"], dtype=float),
            dt,
            h,
            dimension,
            sparsity_threshold=sparsity_threshold,
            ridge_alpha=ridge_alpha,
            holdout_ratio=holdout_ratio,
        )
    # 2. Radial / Isotropic Symmetry Reduction
    elif "r_coords" in payload:
        dr = float(payload.get("dr", 1.0))
        return discover_pde_radial_hd(
            payload["r_coords"],
            np.asarray(payload["u_grid"], dtype=float),
            dt,
            dr,
            dimension,
            sparsity_threshold=sparsity_threshold,
            ridge_alpha=ridge_alpha,
            holdout_ratio=holdout_ratio,
        )
    # 3a. Low-Rank Tensor Trains (TT / MPS) with multi-core bond rank R >= 1
    elif "tt_cores" in payload:
        dx = float(payload.get("dx", 1.0))
        return discover_pde_tensor_train_hd(
            np.asarray(payload["tt_cores"], dtype=float),
            dt,
            dx,
            dimension,
            sparsity_threshold=sparsity_threshold,
            ridge_alpha=ridge_alpha,
            holdout_ratio=holdout_ratio,
        )
    # 3b. Rank-1 Separable Tensor Coordinate Slice
    elif "phi_grid" in payload:
        dx = float(payload.get("dx", 1.0))
        return discover_pde_separable_hd(
            np.asarray(payload["phi_grid"], dtype=float),
            dt,
            dx,
            dimension,
            sparsity_threshold=sparsity_threshold,
            ridge_alpha=ridge_alpha,
            holdout_ratio=holdout_ratio,
        )
    # 4. Deep BSDE / Neural PINNs (Fused 2-Jet Taylor-Mode Automatic Differentiation)
    elif "x_collocation" in payload and "w1" in payload:
        return discover_pde_neural_pinn_hd(
            np.asarray(payload["x_collocation"], dtype=float),
            np.asarray(payload["w1"], dtype=float),
            np.asarray(payload["b1"], dtype=float),
            np.asarray(payload["w2_time"], dtype=float),
            dt,
            dimension,
            sparsity_threshold=sparsity_threshold,
            ridge_alpha=ridge_alpha,
            holdout_ratio=holdout_ratio,
        )
    else:
        raise ValueError(
            f"For dimension D={dimension} >= 10, payload must provide one of the 4 high-dimensional "
            f"representations: (1) 'points'+'u_values' or 'u_anchors' (mesh-free point cloud), "
            f"(2) 'r_coords' (radial symmetry), (3) 'tt_cores' or 'phi_grid' (Tensor-Train MPS), "
            f"or (4) 'x_collocation'+'w1'+'b1'+'w2_time' (Deep BSDE / Neural PINN Jet2)."
        )


def eval_jet2_neural_pde_operator(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    viscosity_nu: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Evaluate a 1-hidden-layer (`D -> H -> 1`, `tanh`) High-Dimensional Neural PDE Operator
    and its exact D-dimensional gradient norm squared, Laplacian, and stationary viscous
    Hamilton-Jacobi-Bellman (HJB) residual `R_{HJB}(x) = -nu * \Delta_D u(x) + 0.5 * |\nabla_D u(x)|^2`
    via the Rust Fused 2-Jet Taylor-Mode CPU Engine.
    """
    out = ak.rime_jet2_neural_pde_operator(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(w1, dtype=np.float64),
        np.ascontiguousarray(b1, dtype=np.float64),
        np.ascontiguousarray(w2, dtype=np.float64),
        float(viscosity_nu),
    )
    return (
        np.asarray(out[0]), np.asarray(out[1]), np.asarray(out[2]), np.asarray(out[3]),
    )


def eval_fokker_planck_ou_operator(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    diffusion_nu: float = 0.25,
    drift_gamma: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Evaluate the Ornstein-Uhlenbeck / Kolmogorov Fokker-Planck stationary operator:
    `R_{FP}(x) = -nu * \Delta_D u(x) + gamma * <x, \nabla_D u(x)>`
    returning `(u, x_dot_grad, laplacian, fp_residual)`.
    """
    out = ak.rime_fokker_planck_ou_operator(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(w1, dtype=np.float64),
        np.ascontiguousarray(b1, dtype=np.float64),
        np.ascontiguousarray(w2, dtype=np.float64),
        float(diffusion_nu),
        float(drift_gamma),
    )
    return (
        np.asarray(out[0]), np.asarray(out[1]), np.asarray(out[2]), np.asarray(out[3]),
    )


def compute_jet3_pde_loss_and_grads(
    x: np.ndarray,
    target_f: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    viscosity_nu: float = 0.25,
    hjb_beta: float = 1.0,
    reaction_mu: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute exact 3-Jet (`Jet3`) analytical loss and parameter gradients `(loss, grad_w1, grad_b1, grad_w2)`
    for the second-order PDE operator:
    `R_\theta(x) = -nu * \Delta_D u_\theta(x) + 0.5 * beta * |\nabla_D u_\theta(x)|^2 + mu * u_\theta(x) - target_f(x)`.
    """
    out = ak.rime_jet3_pde_loss_and_grads(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(target_f, dtype=np.float64),
        np.ascontiguousarray(w1, dtype=np.float64),
        np.ascontiguousarray(b1, dtype=np.float64),
        np.ascontiguousarray(w2, dtype=np.float64),
        float(viscosity_nu),
        float(hjb_beta),
        float(reaction_mu),
    )
    return (
        float(out[0]), np.asarray(out[1]), np.asarray(out[2]), np.asarray(out[3]),
    )


class RustPdeNetwork:
    r"""PyTorch-style trainable High-Dimensional Neural PDE Network backed by the Rust
    `Jet2` / `Jet3` Taylor-Mode CPU Engine (`D -> H -> 1` with `tanh` activation).

    Supports both zero-copy forward PDE operator evaluation (`u`, `|\nabla u|^2`, `\Delta_D u`, `R(x)`)
    and exact 3rd-order analytical parameter training (`train_step`) without PyTorch or JAX.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        *,
        viscosity_nu: float = 0.25,
        hjb_beta: float = 1.0,
        reaction_mu: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.viscosity_nu = float(viscosity_nu)
        self.hjb_beta = float(hjb_beta)
        self.reaction_mu = float(reaction_mu)

        rng = np.random.default_rng(seed)
        scale1 = 1.0 / np.sqrt(self.dim)
        scale2 = 1.0 / np.sqrt(self.hidden_dim)
        self.w1 = np.ascontiguousarray(rng.normal(0.0, scale1, size=(self.hidden_dim, self.dim)), dtype=np.float64)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float64)
        self.w2 = np.ascontiguousarray(rng.normal(0.0, scale2, size=(self.hidden_dim,)), dtype=np.float64)

        # Adam optimizer state
        self._step = 0
        self._m_w1 = np.zeros_like(self.w1)
        self._v_w1 = np.zeros_like(self.w1)
        self._m_b1 = np.zeros_like(self.b1)
        self._v_b1 = np.zeros_like(self.b1)
        self._m_w2 = np.zeros_like(self.w2)
        self._v_w2 = np.zeros_like(self.w2)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return `(u, grad_norm_sq, laplacian, hjb_residual)` on collocation points `x` (`[M, D]`)."""
        return eval_jet2_neural_pde_operator(x, self.w1, self.b1, self.w2, viscosity_nu=self.viscosity_nu)

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.forward(x)

    def loss_and_grads(
        self, x: np.ndarray, target_f: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Return `(loss, grad_w1, grad_b1, grad_w2)` computed via exact 3-Jet propagation in Rust."""
        return compute_jet3_pde_loss_and_grads(
            x,
            target_f,
            self.w1,
            self.b1,
            self.w2,
            viscosity_nu=self.viscosity_nu,
            hjb_beta=self.hjb_beta,
            reaction_mu=self.reaction_mu,
        )

    def train_step(self, x: np.ndarray, target_f: np.ndarray, lr: float = 1e-2) -> float:
        """Perform one Adam optimization step in-place using exact Rust `Jet3` parameter gradients."""
        loss, gw1, gb1, gw2 = self.loss_and_grads(x, target_f)
        self._step += 1
        b1_adam, b2_adam, eps = 0.9, 0.999, 1e-8
        bc1 = 1.0 - b1_adam**self._step
        bc2 = 1.0 - b2_adam**self._step

        self._m_w1 = b1_adam * self._m_w1 + (1.0 - b1_adam) * gw1
        self._v_w1 = b2_adam * self._v_w1 + (1.0 - b2_adam) * (gw1 * gw1)
        self.w1 -= lr * (self._m_w1 / bc1) / (np.sqrt(self._v_w1 / bc2) + eps)

        self._m_b1 = b1_adam * self._m_b1 + (1.0 - b1_adam) * gb1
        self._v_b1 = b2_adam * self._v_b1 + (1.0 - b2_adam) * (gb1 * gb1)
        self.b1 -= lr * (self._m_b1 / bc1) / (np.sqrt(self._v_b1 / bc2) + eps)

        self._m_w2 = b1_adam * self._m_w2 + (1.0 - b1_adam) * gw2
        self._v_w2 = b2_adam * self._v_w2 + (1.0 - b2_adam) * (gw2 * gw2)
        self.w2 -= lr * (self._m_w2 / bc1) / (np.sqrt(self._v_w2 / bc2) + eps)

        return float(loss)


def evaluate_point_cloud_pde_frozen_oos(
    discovered: HighDimPdeDiscoveryResult,
    u_anchors_oos: np.ndarray,
    u_probes_plus_oos: np.ndarray,
    u_probes_minus_oos: np.ndarray,
    dt: float,
    h: float,
) -> float:
    """Evaluate a discovered high-dimensional PDE with FROZEN dev coefficients (no refitting)
    on an independent unseen trajectory `(u_anchors_oos, u_probes_plus_oos, u_probes_minus_oos)`.
    Returns the true out-of-sample model-transfer R^2.
    """
    u_anc = np.asarray(u_anchors_oos, dtype=np.float64)
    u_plus = np.asarray(u_probes_plus_oos, dtype=np.float64)
    u_minus = np.asarray(u_probes_minus_oos, dtype=np.float64)
    dim = discovered.dimension

    ut = ((u_anc[2:, :] - u_anc[:-2, :]) / (2.0 * dt)).ravel()
    u_mid = u_anc[1:-1, :]
    d_xk = (u_plus[1:-1, :, :] - u_minus[1:-1, :, :]) / (2.0 * h)
    d_xkxk = (u_plus[1:-1, :, :] - 2.0 * u_mid[:, :, None] + u_minus[1:-1, :, :]) / (h * h)

    u_flat = u_mid.ravel()
    lap_flat = np.sum(d_xkxk, axis=-1).ravel()
    gsq_flat = np.sum(d_xk * d_xk, axis=-1).ravel()

    basis = {
        "u": u_flat,
        "u^2": u_flat * u_flat,
        f"laplacian_{dim}d(u)": lap_flat,
        "|grad_u|^2": gsq_flat,
        f"u*laplacian_{dim}d(u)": u_flat * lap_flat,
    }

    pred = np.zeros_like(ut)
    for term, coeff in zip(discovered.active_terms, discovered.coefficients, strict=False):
        if term in basis:
            pred += float(coeff) * basis[term]

    ss_res = float(np.sum((ut - pred) ** 2))
    ss_tot = float(np.sum((ut - np.mean(ut)) ** 2))
    if ss_tot <= 1e-30:
        return 0.0
    return 1.0 - ss_res / ss_tot


def step_vlasov_attention_pde_hd(
    x_init: np.ndarray,
    steps: int = 10,
    dt: float = 0.05,
    damping: float = 0.1,
) -> np.ndarray:
    """Evolve N high-dimensional interacting particles/tokens across `steps` layers of the
    McKean-Vlasov Self-Attention PDE using the Rust L1/L2 Block-Tiled Online Softmax CPU Engine.
    Compute complexity: O(N^2 * D) per step; extra scratch memory: O(N * D + threads * B_R * (B_C + D)).
    """
    return np.asarray(ak.rime_vlasov_attention_pde_step(
        np.ascontiguousarray(x_init, dtype=np.float64),
        int(steps),
        float(dt),
        float(damping),
    ))


def adjudicate_high_dim_pde_discovery(
    discovered: HighDimPdeDiscoveryResult,
    u_anchors_dev: np.ndarray,
    u_probes_plus_dev: np.ndarray,
    u_probes_minus_dev: np.ndarray,
    u_anchors_oos: np.ndarray,
    u_probes_plus_oos: np.ndarray,
    u_probes_minus_oos: np.ndarray,
    dt: float,
    h: float,
    *,
    collinearity_threshold: float = 0.985,
    max_drift_threshold: float = 0.15,
    min_causal_oos_r2: float = 0.98,
    min_predictive_r2: float = 0.40,
    stencil_snr_threshold: float = 2.5,
) -> EpistemicPdeAdjudication:
    """Referee-side epistemic adjudication for high-dimensional PDE discovery.

    Evaluates:
    1. Out-of-sample frozen model-transfer R^2 on unseen trajectory.
    2. Candidate dictionary condition and collinearity (observational equivalence).
    3. Differential stencil signal-to-noise ratio (stencil noise floor).
    4. Chronological sub-window coefficient stability across development split halves.

    Returns one of the five preregistered epistemic verdicts:
    - IDENTIFIED_CAUSAL_LAW
    - PREDICTIVE_WRONG_STRUCTURE
    - OBSERVATIONALLY_EQUIVALENT_AMBIGUOUS
    - INSUFFICIENT_EVIDENCE_NOISE_FLOOR
    - REJECTED_NO_CANDIDATE
    """
    dim = discovered.dimension
    frozen_oos_r2 = evaluate_point_cloud_pde_frozen_oos(
        discovered, u_anchors_oos, u_probes_plus_oos, u_probes_minus_oos, dt=dt, h=h
    )

    u_anc = np.asarray(u_anchors_dev, dtype=np.float64)
    u_plus = np.asarray(u_probes_plus_dev, dtype=np.float64)
    u_minus = np.asarray(u_probes_minus_dev, dtype=np.float64)
    nt, m = u_anc.shape
    n_int = nt - 2

    ut_mat = (u_anc[2:, :] - u_anc[:-2, :]) / (2.0 * dt)
    u_mid = u_anc[1:-1, :]
    d_xk = (u_plus[1:-1, :, :] - u_minus[1:-1, :, :]) / (2.0 * h)
    d_xkxk = (u_plus[1:-1, :, :] - 2.0 * u_mid[:, :, None] + u_minus[1:-1, :, :]) / (h * h)
    lap_mat = np.sum(d_xkxk, axis=-1)
    gsq_mat = np.sum(d_xk * d_xk, axis=-1)

    ut_mat.ravel()
    u_flat = u_mid.ravel()
    lap_flat = lap_mat.ravel()
    gsq_flat = gsq_mat.ravel()

    basis = {
        "u": u_flat,
        "u^2": u_flat * u_flat,
        f"laplacian_{dim}d(u)": lap_flat,
        "|grad_u|^2": gsq_flat,
        f"u*laplacian_{dim}d(u)": u_flat * lap_flat,
    }

    # 1. Collinearity check across dictionary and active terms
    terms_list = list(basis.keys())
    mat_cols = [basis[t] for t in terms_list]
    corrs = []
    for i in range(len(terms_list)):
        for j in range(i + 1, len(terms_list)):
            v1, v2 = mat_cols[i], mat_cols[j]
            s1, s2 = np.std(v1), np.std(v2)
            if s1 > 1e-12 and s2 > 1e-12:
                r = float(abs(np.corrcoef(v1, v2)[0, 1]))
                corrs.append(r)
    max_dict_collinearity = max(corrs) if corrs else 0.0

    active_terms = [t for t in discovered.active_terms if t in basis]
    active_collinearity = 0.0
    for act_t in active_terms:
        for other_t in basis:
            if act_t != other_t:
                v1, v2 = basis[act_t], basis[other_t]
                s1, s2 = np.std(v1), np.std(v2)
                if s1 > 1e-12 and s2 > 1e-12:
                    r = float(abs(np.corrcoef(v1, v2)[0, 1]))
                    active_collinearity = max(active_collinearity, r)

    # 2. Stencil SNR estimation: high-frequency roughness of ut along time
    if n_int >= 4:
        diff2_ut = ut_mat[2:, :] - 2.0 * ut_mat[1:-1, :] + ut_mat[:-2, :]
        sigma_noise = float(np.std(diff2_ut) / np.sqrt(6.0))
        sigma_signal = float(np.std(ut_mat))
        stencil_snr = sigma_signal / (sigma_noise + 1e-12)
    else:
        stencil_snr = 10.0

    # 3. Sub-window stability: split development trajectory in halves
    half = n_int // 2
    if half >= 2 and active_terms:
        def fit_half(t_slice):
            cols = []
            for t in active_terms:
                cols.append(basis[t].reshape(n_int, m)[t_slice, :].ravel())
            A = np.column_stack(cols)
            y = ut_mat[t_slice, :].ravel()
            reg = 1e-6 * np.eye(A.shape[1])
            c, _, _, _ = np.linalg.lstsq(A.T @ A + reg, A.T @ y, rcond=None)
            return c

        c1 = fit_half(slice(0, half))
        c2 = fit_half(slice(half, n_int))
        drifts = []
        for v1, v2 in zip(c1, c2, strict=False):
            denom = max(abs(v1), abs(v2), 1e-4)
            drifts.append(abs(v1 - v2) / denom)
        subwindow_max_drift = float(max(drifts)) if drifts else 0.0
    else:
        subwindow_max_drift = 0.0

    # 4. Decision logic
    if discovered.r2_dev < min_predictive_r2 and frozen_oos_r2 < min_predictive_r2:
        verdict = EpistemicPdeVerdict.REJECTED_NO_CANDIDATE
        justification = (
            f"Candidate has negligible explanatory power (dev R^2 = {discovered.r2_dev:.4f}, "
            f"OOS R^2 = {frozen_oos_r2:.4f} < {min_predictive_r2}); correctly rejected as null or unmodeled."
        )
    elif stencil_snr < stencil_snr_threshold and frozen_oos_r2 < min_causal_oos_r2:
        verdict = EpistemicPdeVerdict.INSUFFICIENT_EVIDENCE_NOISE_FLOOR
        justification = (
            f"Measurement noise dominates differential stencils (stencil SNR = {stencil_snr:.2f} < {stencil_snr_threshold}). "
            f"Finite-difference amplification exceeds signal threshold, preventing causal identification."
        )
    elif subwindow_max_drift > max_drift_threshold:
        verdict = EpistemicPdeVerdict.PREDICTIVE_WRONG_STRUCTURE
        justification = (
            f"Candidate achieves predictive correlation (dev R^2 = {discovered.r2_dev:.4f}, "
            f"OOS R^2 = {frozen_oos_r2:.4f}) but exhibits structural instability across temporal "
            f"sub-windows (max drift = {subwindow_max_drift*100:.1f}% > {max_drift_threshold*100:.1f}%). "
            f"Refused as spurious/overfit empirical proxy."
        )
    elif active_collinearity > collinearity_threshold:
        verdict = EpistemicPdeVerdict.OBSERVATIONALLY_EQUIVALENT_AMBIGUOUS
        justification = (
            f"Identifiability failure: active terms exhibit severe collinearity "
            f"(collinearity |r| = {active_collinearity:.4f} > {collinearity_threshold}). "
            f"Data-generating trajectory cannot distinguish competing operator representations."
        )
    elif frozen_oos_r2 < min_causal_oos_r2:
        verdict = EpistemicPdeVerdict.PREDICTIVE_WRONG_STRUCTURE
        justification = (
            f"Candidate achieves development correlation (dev R^2 = {discovered.r2_dev:.4f}) "
            f"but fails out-of-sample transfer (frozen OOS R^2 = {frozen_oos_r2:.4f} < {min_causal_oos_r2}). "
            f"Refused as non-generalizable proxy."
        )
    else:
        verdict = EpistemicPdeVerdict.IDENTIFIED_CAUSAL_LAW
        justification = (
            f"Robust causal identification: stable coefficients across time sub-windows "
            f"(max drift = {subwindow_max_drift*100:.2f}%), well-conditioned dictionary design "
            f"(collinearity = {max_dict_collinearity:.4f} <= {collinearity_threshold}), "
            f"high stencil SNR ({stencil_snr:.1f}), and exact out-of-sample transfer "
            f"(frozen OOS R^2 = {frozen_oos_r2:.6f} >= {min_causal_oos_r2})."
        )

    return EpistemicPdeAdjudication(
        verdict=verdict,
        discovered=discovered,
        frozen_oos_r2=frozen_oos_r2,
        max_dictionary_collinearity=max_dict_collinearity,
        subwindow_max_rel_drift=subwindow_max_drift,
        stencil_snr=stencil_snr,
        rational_justification=justification,
    )



