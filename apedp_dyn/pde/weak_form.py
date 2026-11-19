"""Weak-form identification of 1-D PDEs: the data are integrated, never differentiated.

Status: experimental

For u_t = sum_k w_k d^a_k/dx^a_k f_k(u), multiply by a smooth test function
phi(x, t) with compact support and integrate by parts until every derivative
sits on phi:

    -integral(phi_t u) = sum_k w_k (-1)^a_k integral(d^a_k phi/dx^a_k f_k(u)).

One row per test-function window (Messenger & Bortz, Weak SINDy for PDEs,
2021). Second derivatives of noisy data, which the pointwise route needs for a
diffusion term, never appear.

The library is the divergence-form one the weak form can express:
``u, u^2, u^3, u_x, (u^2)_x, u_xx, (u^2)_xx``. Terms such as ``|grad u|^2`` or
``u * laplacian(u)``, which the pointwise kernel offers, are not derivatives of
a function of u and are not in it (``u u_xx = (u^2)_xx / 2 - u_x^2``).
"""
from __future__ import annotations

import numpy as np

#: name, power of u, order of the x-derivative
LIBRARY: tuple[tuple[str, int, int], ...] = (
    ("u", 1, 0), ("u^2", 2, 0), ("u^3", 3, 0), ("u_x", 1, 1), ("(u^2)_x", 2, 1), ("u_xx", 1, 2), ("(u^2)_xx", 2, 2),
)


def _bump(points: int, order: int, half_span: float) -> list[np.ndarray]:
    """(1 - s^2)^order on [-1, 1] and its first two derivatives in physical units."""
    s = np.linspace(-1.0, 1.0, points)
    base = 1.0 - s ** 2
    d0 = base ** order
    d1 = -2.0 * order * s * base ** (order - 1) / half_span
    d2 = (-2.0 * order * base ** (order - 1) + 4.0 * order * (order - 1) * s ** 2 * base ** (order - 2)) / half_span ** 2
    return [d0, d1, d2]


def weak_system(u: np.ndarray, dt: float, dx: float, *, window_t: int = 41, window_x: int = 21,
                stride_t: int = 4, stride_x: int = 2, order: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """``G w = b`` over LIBRARY for a field ``u`` of shape (times, positions) on a uniform grid."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 2:
        raise ValueError("u must be (times, positions)")
    window_t = min(window_t, u.shape[0] if u.shape[0] % 2 else u.shape[0] - 1)
    window_x = min(window_x, u.shape[1] if u.shape[1] % 2 else u.shape[1] - 1)
    if window_t < 5 or window_x < 5:
        raise ValueError("the field is too small for a test function")
    pt = _bump(window_t, order, 0.5 * (window_t - 1) * dt)
    px = _bump(window_x, order, 0.5 * (window_x - 1) * dx)
    powers: dict[int, np.ndarray] = {p: u ** p for p in {p for _n, p, _a in LIBRARY}}
    rows_g, rows_b = [], []
    for i in range(0, u.shape[0] - window_t + 1, stride_t):
        for j in range(0, u.shape[1] - window_x + 1, stride_x):
            block = (slice(i, i + window_t), slice(j, j + window_x))
            rows_b.append(-float(pt[1] @ u[block] @ px[0]) * dt * dx)
            rows_g.append([(-1.0) ** a * float(pt[0] @ powers[p][block] @ px[a]) * dt * dx for _n, p, a in LIBRARY])
    return np.array(rows_g), np.array(rows_b)


def identify(u: np.ndarray, dt: float, dx: float, threshold: float = 0.05, max_iter: int = 20,
             **window: int) -> dict[str, float]:
    """Sparse coefficients of u_t on LIBRARY.

    A term is dropped when its contribution ``|w_k| * |G_k|`` is below
    ``threshold`` times the size of the left-hand side, which does not depend
    on the units of u, x or t; the survivors are refitted until nothing changes.
    """
    G, b = weak_system(u, dt, dx, **window)
    norms = np.linalg.norm(G, axis=0)
    scale = float(np.linalg.norm(b)) or 1.0
    keep = norms > 0
    coef = np.zeros(G.shape[1])
    for _ in range(max_iter):
        coef[:] = 0.0
        if not keep.any():
            break
        coef[keep] = np.linalg.lstsq(G[:, keep], b, rcond=None)[0]
        new_keep = keep & (np.abs(coef) * norms >= threshold * scale)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return {name: float(c) for (name, _p, _a), c in zip(LIBRARY, coef, strict=True) if c != 0.0}


def weak_fit_quality(u: np.ndarray, dt: float, dx: float, coefficients: dict[str, float], **window: int) -> dict[str, float]:
    """How well the coefficients satisfy the weak system: residual RMS and R^2 over the test-function rows."""
    G, b = weak_system(u, dt, dx, **window)
    w = np.array([coefficients.get(name, 0.0) for name, _p, _a in LIBRARY])
    resid = b - G @ w
    sst = float(np.sum((b - b.mean()) ** 2))
    return {"residual_rmse": float(np.sqrt(np.mean(resid ** 2))),
            "r2": 1.0 - float(np.sum(resid ** 2)) / sst if sst > 0 else 0.0}
