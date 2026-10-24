"""Data structures and typed containers for continuous PDE discovery."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PdeGridData:
    """Eulerian Cartesian grid field data u(t, x_1, ..., x_D).

    Attributes:
        u_grid: Dense numpy array of shape (N_t, N_1, ..., N_D) with D >= 1.
        dt: Temporal step size.
        dxs: Spatial grid spacing along each spatial dimension.
        dimension: Spatial dimension D.
        coords: Optional list of 1D spatial coordinate grids.
    """
    u_grid: np.ndarray
    dt: float
    dxs: tuple[float, ...]
    dimension: int

    @classmethod
    def from_array(
        cls,
        u_grid: np.ndarray,
        dt: float,
        dxs: float | Sequence[float],
    ) -> PdeGridData:
        arr = np.ascontiguousarray(u_grid, dtype=np.float64)
        if arr.ndim < 2:
            raise ValueError(f"u_grid must have at least 2 dimensions (N_t, N_1, ...), got shape {arr.shape}")
        dim = arr.ndim - 1
        if isinstance(dxs, (int, float, np.floating)):
            dxs_tuple = tuple([float(dxs)] * dim)
        else:
            dxs_tuple = tuple(float(x) for x in dxs)
            if len(dxs_tuple) != dim:
                raise ValueError(f"Length of dxs ({len(dxs_tuple)}) must match spatial dimension D={dim}")
        return cls(u_grid=arr, dt=float(dt), dxs=dxs_tuple, dimension=dim)


@dataclass(frozen=True)
class PdePointCloudData:
    """Mesh-free Lagrangian / scattered point cloud PDE data.

    Supports two representations:
    1. Axis-star probe clouds: u_anchors of shape (N_t, M) with u_probes_plus and u_probes_minus (N_t, M, D).
    2. True scattered RBF points: scattered_points of shape (M, D) with u_values (N_t, M).
    """
    dt: float
    dimension: int
    u_anchors: np.ndarray | None = None
    u_probes_plus: np.ndarray | None = None
    u_probes_minus: np.ndarray | None = None
    h: float | None = None
    scattered_points: np.ndarray | None = None
    u_values: np.ndarray | None = None

    @classmethod
    def from_axis_star(
        cls,
        u_anchors: np.ndarray,
        u_probes_plus: np.ndarray,
        u_probes_minus: np.ndarray,
        dt: float,
        h: float,
        dimension: int,
    ) -> PdePointCloudData:
        anc = np.ascontiguousarray(u_anchors, dtype=np.float64)
        plus = np.ascontiguousarray(u_probes_plus, dtype=np.float64)
        minus = np.ascontiguousarray(u_probes_minus, dtype=np.float64)
        if anc.ndim != 2:
            raise ValueError(f"u_anchors must have shape (N_t, M), got {anc.shape}")
        if plus.ndim != 3 or minus.ndim != 3:
            raise ValueError(f"u_probes must have shape (N_t, M, D), got {plus.shape}")
        if plus.shape[-1] != dimension or minus.shape[-1] != dimension:
            raise ValueError(f"Probe spatial dimension ({plus.shape[-1]}) != specified dimension {dimension}")
        return cls(
            dt=float(dt),
            dimension=int(dimension),
            u_anchors=anc,
            u_probes_plus=plus,
            u_probes_minus=minus,
            h=float(h),
        )

    @classmethod
    def from_scattered_rbf(
        cls,
        points: np.ndarray,
        u_values: np.ndarray,
        dt: float,
        dimension: int | None = None,
    ) -> PdePointCloudData:
        pts = np.ascontiguousarray(points, dtype=np.float64)
        u_val = np.ascontiguousarray(u_values, dtype=np.float64)
        if pts.ndim != 2:
            raise ValueError(f"points must have shape (M, D), got {pts.shape}")
        if u_val.ndim != 2:
            raise ValueError(f"u_values must have shape (N_t, M), got {u_val.shape}")
        m, dim_inferred = pts.shape
        if u_val.shape[1] != m:
            raise ValueError(f"Point count M={m} does not match u_values second dimension {u_val.shape[1]}")
        dim = int(dimension) if dimension is not None else int(dim_inferred)
        return cls(
            dt=float(dt),
            dimension=dim,
            scattered_points=pts,
            u_values=u_val,
        )
