"""Weak-form identification of PDEs in two space dimensions: the data are integrated, never differentiated.

Status: experimental

The one-dimensional module (:mod:`apedp_dyn.pde.weak_form`) recovers the heat
equation exactly through 10 % noise; this is the same method for a field
``u(t, x, y)``. For

    u_t = sum_k w_k d^a/dx^a d^b/dy^b f_k(u),

multiply by a smooth test function ``phi(t, x, y)`` with compact support and
integrate by parts until every derivative sits on ``phi``:

    -integral(phi_t u) = sum_k w_k (-1)^(a+b) integral(d^a_x d^b_y phi * f_k(u)).

One row per test-function window (Messenger & Bortz, Weak SINDy for PDEs,
2021). ``phi`` is a product of one-dimensional bumps, so each integral is three
one-dimensional correlations, computed for all windows at once.

The library is the divergence-form one: reaction terms ``u, u^2, u^3``; first
derivatives ``u_x, u_y, (u^2)_x, (u^2)_y``; second derivatives
``u_xx, u_yy, (u^2)_xx, (u^2)_yy``. An isotropic diffusion ``D * laplacian(u)``
is recovered as ``u_xx`` and ``u_yy`` with equal coefficients; nothing forces
them to be equal. Mixed derivatives and terms that are not derivatives of a
function of u (``|grad u|^2``, ``u * laplacian(u)``) are not in it.
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.pde.weak_form import _bump

#: name, power of u, order of the x-derivative, order of the y-derivative
LIBRARY_2D: tuple[tuple[str, int, int, int], ...] = (
    ("u", 1, 0, 0), ("u^2", 2, 0, 0), ("u^3", 3, 0, 0),
    ("u_x", 1, 1, 0), ("u_y", 1, 0, 1), ("(u^2)_x", 2, 1, 0), ("(u^2)_y", 2, 0, 1),
    ("u_xx", 1, 2, 0), ("u_yy", 1, 0, 2), ("(u^2)_xx", 2, 2, 0), ("(u^2)_yy", 2, 0, 2),
)


def _correlate(field: np.ndarray, kernel: np.ndarray, axis: int, stride: int) -> np.ndarray:
    """Slide ``kernel`` along ``axis`` in steps of ``stride`` and integrate against it."""
    view = np.lib.stride_tricks.sliding_window_view(field, len(kernel), axis=axis)
    index: list[slice] = [slice(None)] * field.ndim
    index[axis] = slice(None, None, stride)
    out: np.ndarray = view[tuple(index)] @ kernel
    return out


def weak_system(u: np.ndarray, dt: float, dx: float, dy: float, *, window_t: int = 21, window_xy: int = 15,
                stride_t: int = 3, stride_xy: int = 3, order: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """``G w = b`` over LIBRARY_2D for a field ``u`` of shape (times, nx, ny) on a uniform grid."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 3:
        raise ValueError("u must be (times, nx, ny)")

    def odd(limit: int, wanted: int) -> int:
        return min(wanted, limit if limit % 2 else limit - 1)

    wt, wx, wy = odd(u.shape[0], window_t), odd(u.shape[1], window_xy), odd(u.shape[2], window_xy)
    if min(wt, wx, wy) < 5:
        raise ValueError("the field is too small for a test function")
    pt = _bump(wt, order, 0.5 * (wt - 1) * dt)
    px = _bump(wx, order, 0.5 * (wx - 1) * dx)
    py = _bump(wy, order, 0.5 * (wy - 1) * dy)
    volume = dt * dx * dy

    def integrate(field: np.ndarray, kt: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
        out = _correlate(_correlate(_correlate(field, kt, 0, stride_t), kx, 1, stride_xy), ky, 2, stride_xy)
        flat: np.ndarray = out.ravel() * volume
        return flat

    b = -integrate(u, pt[1], px[0], py[0])
    powers: dict[int, np.ndarray] = {p: u ** p for p in {p for _n, p, _a, _b in LIBRARY_2D}}
    G = np.column_stack([(-1.0) ** (a + c) * integrate(powers[p], pt[0], px[a], py[c]) for _n, p, a, c in LIBRARY_2D])
    return G, b


def _select_by_threshold(G: np.ndarray, b: np.ndarray, threshold: float, max_iter: int) -> np.ndarray:
    norms = np.linalg.norm(G, axis=0)
    scale = float(np.linalg.norm(b)) or 1.0
    keep = norms > 0
    coef: np.ndarray = np.zeros(G.shape[1])
    for _ in range(max_iter):
        coef[:] = 0.0
        if not keep.any():
            break
        coef[keep] = np.linalg.lstsq(G[:, keep], b, rcond=None)[0]
        new_keep = keep & (np.abs(coef) * norms >= threshold * scale)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return coef


def _select_by_evidence(G: np.ndarray, b: np.ndarray, effective_rows: float, floor: float) -> np.ndarray:
    """Backward elimination: drop the term whose loss costs least, as long as the information criterion improves."""
    norms = np.linalg.norm(G, axis=0)
    scale = float(np.linalg.norm(b)) or 1.0
    active = [k for k in range(G.shape[1]) if norms[k] > 0]

    def fit(columns: list[int]) -> tuple[float, np.ndarray]:
        if not columns:
            return float(b @ b), np.zeros(0)
        w = np.linalg.lstsq(G[:, columns], b, rcond=None)[0]
        r = b - G[:, columns] @ w
        return float(r @ r), w

    def criterion(rss: float, terms: int) -> float:
        return effective_rows * float(np.log(max(rss, 1e-300) / len(b))) + terms * float(np.log(effective_rows))

    current, _ = fit(active)
    while len(active) > 1:
        rss, worst = min((fit([k for k in active if k != drop])[0], drop) for drop in active)
        if criterion(rss, len(active) - 1) > criterion(current, len(active)):
            break
        active.remove(worst)
        current = rss
    _, w = fit(active)
    coef: np.ndarray = np.zeros(G.shape[1])
    for k, value in zip(active, w, strict=True):
        if abs(value) * norms[k] >= floor * scale:
            coef[k] = value
    return coef


def identify(u: np.ndarray, dt: float, dx: float, dy: float, select: str = "evidence", threshold: float = 0.05,
             floor: float = 0.005, max_iter: int = 20, **window: int) -> dict[str, float]:
    """Sparse coefficients of ``u_t`` on LIBRARY_2D.

    ``select="evidence"`` (default) starts from the whole library and removes
    terms one at a time while a Bayesian information criterion improves, then
    drops anything contributing less than ``floor`` of the left-hand side. The
    windows overlap, so the criterion counts the rows as the number of
    non-overlapping windows they amount to, not as independent samples.

    ``select="threshold"`` is the one-dimensional module's rule: a term is
    dropped when its contribution is below ``threshold`` times the left-hand
    side. It loses real terms that are small beside the others: in development
    it dropped the diffusion of a Fisher-KPP field, where reaction dominates,
    which the evidence rule keeps.
    """
    G, b = weak_system(u, dt, dx, dy, **window)
    if select == "threshold":
        coef = _select_by_threshold(G, b, threshold, max_iter)
    elif select == "evidence":
        wt, wxy = window.get("window_t", 21), window.get("window_xy", 15)
        st, sxy = window.get("stride_t", 3), window.get("stride_xy", 3)
        effective = max(20.0, len(b) * min(1.0, st / wt) * min(1.0, sxy / wxy) ** 2)
        coef = _select_by_evidence(G, b, effective, floor)
    else:
        raise ValueError("select must be 'evidence' or 'threshold'")
    return {name: float(c) for (name, _p, _a, _b), c in zip(LIBRARY_2D, coef, strict=True) if c != 0.0}


def weak_fit_quality(u: np.ndarray, dt: float, dx: float, dy: float, coefficients: dict[str, float], **window: int) -> dict[str, float]:
    """How well the identified equation explains the weak left-hand side: relative residual and R^2 over the windows."""
    G, b = weak_system(u, dt, dx, dy, **window)
    w = np.array([coefficients.get(name, 0.0) for name, _p, _a, _b in LIBRARY_2D])
    resid = b - G @ w
    total = float(np.sum((b - b.mean()) ** 2))
    return {"relative_residual": float(np.linalg.norm(resid) / (np.linalg.norm(b) or 1.0)),
            "r2": 1.0 - float(np.sum(resid ** 2)) / total if total > 0 else 0.0, "windows": float(len(b))}
