"""Unified dispatch and functional API for continuous PDE discovery across all dimensions."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from apedp_dyn.epistemic.high_dim_pde import (
    EpistemicPdeAdjudication as EpistemicPdeAdjudication,  # explicit re-export surface
)
from apedp_dyn.epistemic.high_dim_pde import (
    EpistemicPdeVerdict as EpistemicPdeVerdict,  # explicit re-export surface
)
from apedp_dyn.epistemic.high_dim_pde import (
    HighDimPdeDiscoveryResult,
    discover_pde_nd_cartesian,
    discover_pde_point_cloud_hd,
    discover_pde_rbf_point_cloud_hd,
)
from apedp_dyn.epistemic.high_dim_pde import (
    adjudicate_high_dim_pde_discovery as adjudicate_high_dim_pde_discovery,  # re-export surface
)
from apedp_dyn.epistemic.high_dim_pde import (
    discover_pde_allen_cahn_hd as discover_pde_allen_cahn_hd,  # re-export surface
)
from apedp_dyn.epistemic.high_dim_pde import (
    discover_pde_black_scholes_barenblatt_100d as discover_pde_black_scholes_barenblatt_100d,  # re-export surface
)
from apedp_dyn.epistemic.high_dim_pde import (
    discover_pde_hjb_100d as discover_pde_hjb_100d,  # re-export surface
)
from apedp_dyn.epistemic.high_dim_pde import (
    discover_pde_neural_pinn_hd as discover_pde_neural_pinn_hd,  # re-export surface
)
from apedp_dyn.epistemic.high_dim_pde import (
    discover_pde_radial_hd as discover_pde_radial_hd,  # re-export surface
)
from apedp_dyn.epistemic.high_dim_pde import (
    discover_pde_separable_hd as discover_pde_separable_hd,  # re-export surface
)
from apedp_dyn.epistemic.high_dim_pde import (
    discover_pde_tensor_train_hd as discover_pde_tensor_train_hd,  # re-export surface
)
from apedp_dyn.epistemic.high_dim_pde import (
    evaluate_point_cloud_pde_frozen_oos as evaluate_point_cloud_pde_frozen_oos,  # re-export surface
)
from apedp_dyn.epistemic.high_dim_pde import (
    tikhonov_smooth_trajectory as tikhonov_smooth_trajectory,  # re-export surface
)
from apedp_dyn.epistemic.pde_discovery import (
    discover_pde as _discover_pde_1d_engine,
)
from apedp_dyn.epistemic.pde_discovery import (
    discover_pde_2d as discover_pde_2d,  # re-export surface
)
from apedp_dyn.epistemic.pde_discovery import (
    discover_pde_nonuniform as discover_pde_nonuniform,  # re-export surface
)
from apedp_dyn.pde.data import PdeGridData, PdePointCloudData

# 1D public API alias: the epistemic 1D engine (which wraps the kernel's
# `rime::pde::discover_pde_1d` via `rime_pde_discover_json`). The unified
# `discover_pde` definition below intentionally shadows the engine's own name.
discover_pde_1d = _discover_pde_1d_engine


def discover_pde(
    data: PdeGridData | PdePointCloudData | np.ndarray,
    dt: float | None = None,
    dxs: float | Sequence[float] | None = None,
    *,
    dimension: int | None = None,
    method: str = "auto",
    sparsity_threshold: float = 0.01,
    ridge_alpha: float = 1e-6,
    smooth_alpha: float = 0.0,
    **kwargs: Any,
) -> HighDimPdeDiscoveryResult:
    """Unified autonomous discovery interface for continuous PDEs from 1D to 100D+.

    Parameters:
        data: Either `PdeGridData`, `PdePointCloudData`, or a raw numpy array:
            - If grid array shape is `(N_t, N_x)` or `(N_t, N_1, ..., N_D)`.
            - If point cloud, pass `PdePointCloudData` or appropriate kwargs.
        dt: Temporal step size (required if data is raw array).
        dxs: Spatial grid spacing (required if data is raw Eulerian grid array).
        dimension: Spatial dimension D (inferred if None).
        method: Discovery method ('auto', 'cartesian', 'point_cloud', 'rbf', 'radial', 'tensor_train', 'pinn').
        sparsity_threshold: L0 ST-Ridge sparsity cutoff.
        ridge_alpha: L2 Tikhonov regularization.
        smooth_alpha: Temporal smoothing coefficient for noisy observations.
    """
    # Case 1: PdeGridData container
    if isinstance(data, PdeGridData):
        dim = data.dimension
        if dim < 10 and method in ("auto", "cartesian"):
            return discover_pde_nd_cartesian(
                data.u_grid,
                dt=data.dt,
                dxs=data.dxs,
                sparsity_threshold=sparsity_threshold,
                ridge_alpha=ridge_alpha,
            )
        raise ValueError(f"Cartesian grid discovery unsupported for D={dim} >= 10. Use point cloud representation.")

    # Case 2: PdePointCloudData container
    if isinstance(data, PdePointCloudData):
        if data.scattered_points is not None and data.u_values is not None:
            return discover_pde_rbf_point_cloud_hd(
                data.scattered_points,
                data.u_values,
                dt=data.dt,
                dimension=data.dimension,
                sparsity_threshold=sparsity_threshold,
                ridge_alpha=ridge_alpha,
            )
        if data.u_anchors is not None and data.u_probes_plus is not None and data.u_probes_minus is not None:
            h_val = data.h if data.h is not None else 0.04
            return discover_pde_point_cloud_hd(
                data.u_anchors,
                data.u_probes_plus,
                data.u_probes_minus,
                dt=data.dt,
                h=h_val,
                dimension=data.dimension,
                sparsity_threshold=sparsity_threshold,
                ridge_alpha=ridge_alpha,
                smooth_alpha=smooth_alpha,
            )
        raise ValueError("Invalid PdePointCloudData: missing anchor/probe arrays or scattered points")

    # Case 3: Raw numpy array
    arr = np.ascontiguousarray(data, dtype=np.float64)
    if dt is None:
        raise ValueError("dt must be specified when passing raw numpy array")

    if arr.ndim < 2:
        raise ValueError(f"Array must have at least 2 dimensions, got shape {arr.shape}")

    dim = dimension if dimension is not None else (arr.ndim - 1)

    if dim < 10 and method in ("auto", "cartesian"):
        if dxs is None:
            raise ValueError("dxs must be specified for Eulerian Cartesian grid discovery")
        dxs_seq = [float(dxs)] * dim if isinstance(dxs, (int, float, np.floating)) else [float(x) for x in dxs]
        return discover_pde_nd_cartesian(
            arr,
            dt=float(dt),
            dxs=dxs_seq,
            sparsity_threshold=sparsity_threshold,
            ridge_alpha=ridge_alpha,
        )

    raise ValueError(f"For D={dim} >= 10, provide PdePointCloudData or specify method='point_cloud'/'rbf'")
