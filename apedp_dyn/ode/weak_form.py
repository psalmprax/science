"""Weak-form sparse identification of ODEs: no pointwise derivative of noisy data.

Status: experimental

The strong form regresses an estimated derivative on a library, dx/dt = Theta(x) w,
and the derivative estimate (smoothing plus finite differences) amplifies noise.
The weak form multiplies by a smooth test function phi that vanishes at both ends
of a window and integrates by parts,

    -integral(phi' x dt) = integral(phi Theta(x) dt) w,

so the derivative falls on phi, which is known exactly, and the data enter only
through integrals, which average noise down (Messenger & Bortz, Weak SINDy,
2021). One row per window; the sparse regression is the same sequentially
thresholded least squares either way.

Measured against the derivative route on the Lorenz system of
PREREG-LORENZ-PUB-V1 by ``scripts/study_weak_form_lorenz.py`` (a development
study): at up to 5 % noise the weak form's coefficients are several times to
tens of times more accurate, but with a twenty-term cubic library and a 10 s
trajectory it selects the exact terms no more often than the derivative route.
It fixes the coefficient estimate, not term selection.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def weak_system(
    t: np.ndarray, X: np.ndarray, library: Callable[[np.ndarray], np.ndarray],
    *, window: int = 201, stride: int = 10, order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """The weak-form regression ``G w = b`` for uniformly sampled states ``X`` (n, d).

    ``window`` samples per test function (odd), one test function every
    ``stride`` samples, phi(s) = (1 - s^2)^order on s in [-1, 1].
    Returns ``G`` (windows, library terms) and ``b`` (windows, d).
    """
    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(t) != len(X):
        raise ValueError("X must be (n, d) with one row per time")
    if window % 2 == 0 or window < 5 or window > len(t):
        raise ValueError("window must be odd, at least 5 and at most the series length")
    steps = np.diff(t)
    dt = float(steps.mean())
    if not np.allclose(steps, dt, rtol=1e-6, atol=0.0):
        raise ValueError("the weak form here needs uniformly sampled times")
    s = np.linspace(-1.0, 1.0, window)
    half_span = 0.5 * (window - 1) * dt
    phi = (1.0 - s ** 2) ** order
    dphi = -2.0 * order * s * (1.0 - s ** 2) ** (order - 1) / half_span
    theta = np.asarray(library(X), dtype=float)
    starts = np.arange(0, len(t) - window + 1, stride)
    G = np.empty((len(starts), theta.shape[1]))
    b = np.empty((len(starts), X.shape[1]))
    for row, k in enumerate(starts):
        G[row] = (phi[:, None] * theta[k:k + window]).sum(axis=0) * dt
        b[row] = -(dphi[:, None] * X[k:k + window]).sum(axis=0) * dt
    return G, b


def stlsq(A: np.ndarray, b: np.ndarray, threshold: float, max_iter: int = 20) -> np.ndarray:
    """Sequentially thresholded least squares; returns coefficients (terms, targets)."""
    b = np.asarray(b, dtype=float).reshape(len(A), -1)
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(max_iter):
        small = np.abs(coef) < threshold
        if not small.any() and _ > 0:
            break
        coef[small] = 0.0
        for j in range(b.shape[1]):
            keep = ~small[:, j]
            if keep.any():
                coef[keep, j] = np.linalg.lstsq(A[:, keep], b[:, j], rcond=None)[0]
    out: np.ndarray = coef
    return out


def identify(
    t: np.ndarray, X: np.ndarray, library: Callable[[np.ndarray], np.ndarray], threshold: float,
    *, window: int = 201, stride: int = 10, order: int = 4,
) -> np.ndarray:
    """Sparse coefficients (terms, d) of dX/dt = library(X) @ w from the weak form."""
    G, b = weak_system(t, X, library, window=window, stride=stride, order=order)
    return stlsq(G, b, threshold)


def term_column(term: str, series: dict[str, np.ndarray]) -> np.ndarray:
    """A library term in the coupled-system kernel's encoding: ``1``, ``a``, ``a^2``, ``a^3``, ``a*b``."""
    if term == "1":
        return np.ones(len(next(iter(series.values()))))
    if "*" in term:
        left, right = term.split("*", 1)
        product: np.ndarray = series[left] * series[right]
        return product
    if term.endswith(("^2", "^3")):
        out: np.ndarray = series[term[:-2]] ** int(term[-1])
        return out
    return series[term]


def refit_coefficients(
    series: dict[str, np.ndarray], state: str, terms: list[str], dt: float, *, window: int = 201, stride: int = 5,
) -> np.ndarray | None:
    """Weak-form coefficients of ``d state/dt`` on terms already selected; ``None`` if the series is too short.

    Selection stays with whoever chose ``terms``. Measured on the Lorenz system
    (scripts/study_weak_form_lorenz.py), the weak form improves the coefficient
    estimate, not which terms are chosen.
    """
    n = len(series[state])
    window = min(window, n if n % 2 else n - 1)
    if window < 5 or not terms:
        return None
    names = list(series)
    X = np.column_stack([series[k] for k in names])
    G, b = weak_system(np.arange(n) * dt, X,
                       lambda S: np.column_stack([term_column(t, dict(zip(names, S.T, strict=True))) for t in terms]),
                       window=window, stride=stride)
    if len(G) < len(terms):
        return None
    coef: np.ndarray = np.linalg.lstsq(G, b[:, names.index(state)], rcond=None)[0]
    return coef
