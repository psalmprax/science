"""Square-window weak arm (ported from apedp/dyn estimator, third A/B arm).

Integrates the acceleration equation once over sliding windows:

    Δv = v(t2) − v(t1) = Σ c_k ∫ f_k(x, v, u) dt.

Response = SG-prefiltered first-derivative differences (never differentiated
twice); regressors = trapezoidal integrals of consistently filtered channels,
which smooths noise instead of amplifying it. Inputs are NOT prefiltered
(assumed clean/known; filtering spectrally rich inputs biases the input gain).

Differs from Galerkin bumps (weak_form.py): square windows + endpoint
differences vs compact test functions + parts. Both size fixed IR structures
only and run the same gates. Conditioning is diagnosed, never hidden: an
ill-conditioned design returns inf cond (caller decides).
"""
from __future__ import annotations

import numpy as np


def square_window_coefficients(problem, candidate, t: np.ndarray, Y: np.ndarray,
                               U: np.ndarray, n_integral: int = 20,
                               decimate: int = 1,
                               prefilter_seconds: float = 0.4) -> dict[str, float]:
    """Square-window weak LS estimate of a candidate's parameters."""
    import sympy as sp
    from scipy.signal import savgol_filter

    from apedp_dyn.grammar import materialize

    model = materialize(problem, candidate)
    order = [s.name for s in model.states]
    inames = [u.name for u in model.inputs]
    t = np.asarray(t, dtype=float)
    dt = float(t[1] - t[0])
    n = len(t)

    # Prefilter spans a fixed time fraction (~0.4 s default), not a fixed
    # sample count: the upstream constant (61 @ dt=0.01) over-smooths coarser
    # grids and attenuates every weight uniformly (~0.75x measured at dt=0.025).
    w1 = max(15, round(prefilter_seconds / dt) | 1)
    w1 = min(w1, (n // 4) | 1)
    x_f = savgol_filter(Y[:, 0], w1, 2)
    v_f = savgol_filter(Y[:, 0], w1, 2, deriv=1, delta=dt)
    v_state = savgol_filter(Y[:, 1], w1, 2) if Y.shape[1] > 1 else v_f
    half = w1 // 2
    w = max(n_integral, w1)
    stride = max(1, w // 2)
    starts = np.arange(half, n - 1 - w - half, stride)
    if starts.size < len(model.parameters) + 1:
        raise ValueError(
            f"trajectory too short for square windows "
            f"({starts.size} windows, {len(model.parameters)} params)")

    vel = order[1]
    accel_eq = next(e.sympy_expr for e in model.equations if e.state == vel)
    syms = [sp.Symbol(nm) for nm in order] + [sp.Symbol(nm) for nm in inames]
    # Evaluate per-term on filtered channels (x_f, v_state, raw U, t).
    cols = {}
    state_cols = [x_f, v_state][:len(order)]
    for p in model.parameters:
        term = sp.diff(accel_eq, sp.Symbol(p.name))
        f = sp.lambdify(syms, term, modules=["numpy"])
        col = np.asarray(f(*state_cols, *[U[:, i] for i in range(len(inames))]),
                         dtype=float).ravel()
        if col.size == 1 and n > 1:
            col = np.full(n, col[0])
        cols[p.name] = col

    names = [p.name for p in model.parameters]
    rows_y = np.empty(starts.size)
    rows_phi = np.empty((starts.size, len(names)))
    for row, i in enumerate(starts):
        j = i + w
        rows_y[row] = v_f[j] - v_f[i]
        for cidx, nm in enumerate(names):
            rows_phi[row, cidx] = np.trapezoid(cols[nm][i:j + 1], dx=dt)
    weights, *_ = np.linalg.lstsq(rows_phi, rows_y, rcond=None)
    return {nm: float(weights[k]) for k, nm in enumerate(names)}
