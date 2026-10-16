"""Frozen-coefficient PDE holdout transfer evaluation.

The dev/holdout protocol in `pde_discovery.discover_pde` re-runs Rust STRidge
on the holdout partition, so its `holdout_r2` measures a *re-fit*, not
transfer of the discovered law (documented limitation #1 in
`docs/frontier_9_of_10_architectural_resolution.md`).

This module evaluates dev-discovered coefficients FROZEN on the holdout
window with the identical finite-difference stencils (2nd-order central in
time, 4th-order central in space), the identical interior mask, and the
identical R^2 definition (vs mean of u_t) as `rime/pde.rs::discover_pde_1d`,
giving a true forward-transfer metric. No coefficient is ever adjusted here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrozenTransferResult:
    r2: float
    residual_rmse: float
    n_samples: int
    mean_ut: float


def _interior_stencils(
    u: np.ndarray, dt: float, dx: float
) -> dict[str, np.ndarray]:
    """Finite-difference columns over the pde.rs interior mask.

    Interior: time rows 1..m-2 (inclusive), space cols 2..n-3 (inclusive),
    matching `discover_pde_1d`.
    """
    u = np.asarray(u, dtype=float)
    m, n = u.shape
    if m < 5 or n < 7:
        raise ValueError("holdout grid too small for PDE transfer evaluation")
    core = u[1:-1, :]
    ut = ((u[2:, :] - u[:-2, :]) / (2.0 * dt))[:, 2:-2]
    ux = (-u[1:-1, 4:] + 8.0 * u[1:-1, 3:-1] - 8.0 * u[1:-1, 1:-3] + u[1:-1, :-4]) / (
        12.0 * dx
    )
    uxx = (
        -u[1:-1, 4:]
        + 16.0 * u[1:-1, 3:-1]
        - 30.0 * u[1:-1, 2:-2]
        + 16.0 * u[1:-1, 1:-3]
        - u[1:-1, :-4]
    ) / (12.0 * dx * dx)
    uxxx = (
        u[1:-1, 4:] - 2.0 * u[1:-1, 3:-1] + 2.0 * u[1:-1, 1:-3] - u[1:-1, :-4]
    ) / (2.0 * dx**3)
    ucol = core[:, 2:-2]
    return {
        "u": ucol,
        "u^2": ucol * ucol,
        "u_x": ux,
        "u*u_x": ucol * ux,
        "u^2*u_x": ucol * ucol * ux,
        "u_xx": uxx,
        "u*u_xx": ucol * uxx,
        "u_xxx": uxxx,
        "u_t": ut,
    }


def evaluate_frozen_transfer(
    u_hold: np.ndarray,
    dt: float,
    dx: float,
    active_terms: list[str],
    coefficients: list[float],
) -> FrozenTransferResult:
    """R^2 of u_t = sum(c_i * T_i) on the holdout window with FROZEN c_i.

    `active_terms`/`coefficients` must come from discovery on the dev
    partition only. Unknown term names raise (never silently dropped).
    """
    if len(active_terms) != len(coefficients):
        raise ValueError("active_terms and coefficients length mismatch")
    cols = _interior_stencils(u_hold, dt, dx)
    ut = cols["u_t"].reshape(-1)
    pred = np.zeros_like(ut)
    for name, c in zip(active_terms, coefficients):
        if name not in cols:
            raise ValueError(f"unknown PDE term: {name!r}")
        pred += float(c) * cols[name].reshape(-1)
    diff = ut - pred
    ss_res = float(np.sum(diff * diff))
    mean_ut = float(np.mean(ut))
    ss_tot = float(np.sum((ut - mean_ut) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
    rmse = float(np.sqrt(ss_res / ut.size))
    return FrozenTransferResult(
        r2=float(r2),
        residual_rmse=rmse,
        n_samples=int(ut.size),
        mean_ut=mean_ut,
    )
