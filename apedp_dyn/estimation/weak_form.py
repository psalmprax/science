"""D2a — weak-form estimation for fixed-structure candidates.

Replaces pointwise derivative estimation with integral/weak representations:
for the acceleration equation dv/dt = sum_k c_k f_k(x, v, u) and a test
function φ,

    ∫ v̇ φ dt = [vφ] − ∫ v φ̇ dt.

The boundary term vanishes by construction: windows use compactly supported
C∞ bump functions strictly inside the observation span, so φ = φ̇ = 0 at both
window edges and [vφ] = 0 exactly (up to quadrature). If a caller supplies a
non-compact test function, the boundary contribution is retained explicitly
rather than silently dropped.

This module is an ESTIMATION mechanism, not a route around admissibility:
it sizes the terms of a grammar-generated, gate-1/2-cleared IR. Fitted
models still pass through gates 3-4 and the fitter like any other estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def bump(tau: np.ndarray) -> np.ndarray:
    """Compact C∞ bump: exp(-1/(1-tau^2)) on |tau|<1, else 0."""
    out = np.zeros_like(np.asarray(tau, dtype=float))
    inside = np.abs(tau) < 1.0
    s = tau[inside]
    out[inside] = np.exp(-1.0 / (1.0 - s ** 2))
    return out


def bump_deriv(tau: np.ndarray) -> np.ndarray:
    """Analytic derivative of the bump."""
    out = np.zeros_like(np.asarray(tau, dtype=float))
    inside = np.abs(tau) < 1.0
    s = tau[inside]
    out[inside] = np.exp(-1.0 / (1.0 - s ** 2)) * (-2.0 * s / (1.0 - s ** 2) ** 2)
    return out


@dataclass(frozen=True)
class WeakWindows:
    """Compact test-function windows strictly inside (t0, t1)."""

    centers: np.ndarray
    radius: float
    t0: float
    t1: float

    def phi(self, t: np.ndarray, j: int) -> np.ndarray:
        return bump((np.asarray(t, dtype=float) - self.centers[j]) / self.radius)

    def phi_dot(self, t: np.ndarray, j: int) -> np.ndarray:
        return bump_deriv((np.asarray(t, dtype=float) - self.centers[j]) / self.radius) / self.radius


def make_windows(t_span: tuple[float, float], n_windows: int = 32,
                 radius: float = 1.0) -> WeakWindows:
    """Windows with full support inside the span (boundary term vanishes)."""
    t0, t1 = t_span
    assert 2 * radius < (t1 - t0), "window diameter must fit inside the span"
    centers = np.linspace(t0 + radius, t1 - radius, n_windows)
    return WeakWindows(centers=centers, radius=radius, t0=t0, t1=t1)


def boundary_term(v_edge: tuple[float, float], phi_edge: tuple[float, float]) -> float:
    """Explicit [v φ] boundary contribution (zero for compact windows)."""
    (va, vb), (pa, pb) = v_edge, phi_edge
    return vb * pb - va * pa


def weak_coefficients(problem, candidate, t: np.ndarray, Y: np.ndarray, U: np.ndarray,
                      n_windows: int = 32, radius: float = 1.0) -> dict[str, float]:
    """Weak-form LS estimate of a candidate's parameters (structure fixed).

    G[j,k] = ∫ f_k(x_obs, u_obs) φ_j dt ; b[j] = −∫ v_obs φ̇_j dt (+ boundary,
    which is zero for compact windows). Solves min ||G c − b||.
    """
    import sympy as sp

    from apedp_dyn.grammar import materialize

    model = materialize(problem, candidate)
    order = [s.name for s in model.states]
    inames = [u.name for u in model.inputs]
    t = np.asarray(t, dtype=float)
    windows = make_windows((float(t[0]), float(t[-1])), n_windows, radius)
    vel = order[1]
    v_obs = Y[:, 1]
    accel_eq = next(e.sympy_expr for e in model.equations if e.state == vel)
    syms = [sp.Symbol(n) for n in order] + [sp.Symbol(n) for n in inames]
    args = [Y[:, i] for i in range(len(order))] + [U[:, i] for i in range(len(inames))]
    n_w = len(windows.centers)
    n_p = len(model.parameters)
    cols = []
    n = len(t)
    for p in model.parameters:
        term = sp.diff(accel_eq, sp.Symbol(p.name))
        f = sp.lambdify(syms, term, modules=["numpy"])
        col = np.asarray(f(*args), dtype=float).ravel()
        if col.size == 1 and n > 1:
            col = np.full(n, col[0])
        cols.append(col)
    cols = np.column_stack(cols)
    G = np.zeros((n_w, n_p))
    b = np.zeros(n_w)
    for j in range(n_w):
        phi = windows.phi(t, j)
        phid = windows.phi_dot(t, j)
        G[j, :] = np.trapezoid(cols * phi[:, None], t, axis=0)
        b[j] = -np.trapezoid(v_obs * phid, t)
        b[j] += boundary_term((float(v_obs[0]), float(v_obs[-1])),
                              (float(phi[0]), float(phi[-1])))
    coef, *_ = np.linalg.lstsq(G, b, rcond=None)
    return {p.name: float(coef[k]) for k, p in enumerate(model.parameters)}
