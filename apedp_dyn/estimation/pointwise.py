"""Pointwise derivative estimator (D1 baseline arm of the D2a A/B).

Spline-differentiated targets regressed onto a fixed candidate's term columns.
Structure comes solely from the IR; this sizes numbers only — the same honesty
contract as the weak-form arm. Compared against weak_coefficients() on identical
noise realizations in experiments/d2a_ab.py.
"""
from __future__ import annotations

import numpy as np


def spline_coefficients(problem, candidate, t: np.ndarray, Y: np.ndarray,
                        U: np.ndarray) -> dict[str, float]:
    """Spline-derivative LS estimate of a candidate's parameters."""
    import sympy as sp
    from scipy.interpolate import CubicSpline

    from apedp_dyn.grammar import materialize

    model = materialize(problem, candidate)
    order = [s.name for s in model.states]
    inames = [u.name for u in model.inputs]
    t = np.asarray(t, dtype=float)
    dvel = CubicSpline(t, Y[:, 1]).derivative()(t)
    vel = order[1]
    accel_eq = next(e.sympy_expr for e in model.equations if e.state == vel)
    syms = [sp.Symbol(n) for n in order] + [sp.Symbol(n) for n in inames]
    args = [Y[:, i] for i in range(len(order))] + [U[:, i] for i in range(len(inames))]
    cols = []
    n = len(t)
    for p in model.parameters:
        term = sp.diff(accel_eq, sp.Symbol(p.name))
        f = sp.lambdify(syms, term, modules=["numpy"])
        col = np.asarray(f(*args), dtype=float).ravel()
        if col.size == 1 and n > 1:
            col = np.full(n, col[0])
        cols.append(col)
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), dvel, rcond=None)
    return {p.name: float(coef[k]) for k, p in enumerate(model.parameters)}
