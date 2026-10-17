"""Finance problem: second-order price dynamics on real market data.

Maps a price series onto the mechanical-grammar state space so the
dynamic-systems side (enumerate -> prune -> fit -> adjudicate) operates
genuinely, instead of the sandbox's hand-built 3-candidate sets:

    x = standardized log-price   (dimensionless)
    v = dx/dt                    (1/day, finite-difference of x)

Candidate RHS terms then read as real market hypotheses:
    v        momentum persistence      dv/dt ~ v
    x        price-level mean reversion dv/dt ~ x
    x3       nonlinear reversion        dv/dt ~ x^3
    abs_v_v  volatility damping        dv/dt ~ |v|*v
    sin_x    cyclical component        dv/dt ~ sin(x)
    one      constant drift

Standardization uses train-split statistics only (no holdout leakage).
The problem carries REAL trajectories as truth_y; finalize() is bypassed
because there is no closed-form truth — the market is the truth.
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.model import DynamicalEquation, InputVariable, StateVariable
from apedp_dyn.problems.base import DynProblem, make_signal

PROBLEM_ID = "FIN-PRICE"


def _estimate_velocity(x: np.ndarray, dt: float = 1.0, smooth: bool = True) -> np.ndarray:
    """Estimate velocity dx/dt with Savitzky-Golay polynomial smoothing when len >= 5.

    Mitigates O(1/dt) noise amplification from raw finite differences.
    """
    if not smooth or len(x) < 5:
        return np.asarray(np.gradient(x, dt))
    w = min(7, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if w < 5:
        return np.asarray(np.gradient(x, dt))
    try:
        from scipy.signal import savgol_filter
        return np.asarray(savgol_filter(
            x, window_length=w, polyorder=2, deriv=1, delta=dt, mode="interp"))
    except Exception:  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
        # failure (import, window/polyorder bounds, non-finite input) falls back
        # to the raw finite difference, which is always available.
        return np.asarray(np.gradient(x, dt))


def spec_from_series(prices: np.ndarray,
                     train_range: tuple[int, int],
                     holdout_range: tuple[int, int],
                     problem_id: str = PROBLEM_ID,
                     orderflow: np.ndarray | None = None,
                     smooth_derivatives: bool = False) -> DynProblem:
    """Build a DynProblem from a 1-D price series and index ranges.

    train_range / holdout_range are [start, end) index pairs into prices.
    Returns standardized states so optimizer bounds stay meaningful.
    Optional orderflow provides empirical driving signal u(t) (e.g. OFI).
    smooth_derivatives applies regularized polynomial derivative filtering.
    """
    prices = np.asarray(prices, dtype=float)
    tr = prices[train_range[0]:train_range[1]]
    ho = prices[holdout_range[0]:holdout_range[1]]
    if len(tr) < 10 or len(ho) < 5:
        raise ValueError("train/holdout windows too short")

    lp_tr = np.log(tr)
    mu, sigma = float(lp_tr.mean()), float(lp_tr.std())
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("degenerate train series")

    def standardize(p):
        x = (np.log(np.asarray(p, dtype=float)) - mu) / sigma
        v = _estimate_velocity(x, 1.0, smooth=smooth_derivatives)
        return np.column_stack([x, v])

    Ytr, Yho = standardize(tr), standardize(ho)
    t_tr = np.arange(len(tr), dtype=float)
    t_ho = np.arange(len(ho), dtype=float)

    states = (StateVariable("x", "dimensionless", float(Ytr[0, 0])),
              StateVariable("v", "1/day", float(Ytr[0, 1])))
    inputs = (InputVariable("u", "dimensionless"),)
    truth = (DynamicalEquation("x", "v"),
             DynamicalEquation("v", "0"))  # placeholder; never integrated

    if orderflow is not None:
        of_arr = np.asarray(orderflow, dtype=float)
        of_tr = of_arr[train_range[0]:train_range[1]]
        of_ho = of_arr[holdout_range[0]:holdout_range[1]]
        u_tr = make_signal({"kind": "interpolated", "t": t_tr, "u": of_tr})
        u_ho = make_signal({"kind": "interpolated", "t": t_ho, "u": of_ho})
    else:
        u_tr = make_signal({"kind": "zero", "n": 1})
        u_ho = make_signal({"kind": "zero", "n": 1})

    train = {"t_span": (0.0, float(len(tr) - 1)),
             "u_of_t": u_tr,
             "x0": (float(Ytr[0, 0]), float(Ytr[0, 1])),
             "truth_t": t_tr, "truth_y": Ytr}
    from apedp_dyn.gates.intervention import auto_tolerance
    tol_auto = auto_tolerance(Yho)
    tol_std = float(np.std(Yho[:, 0])) if len(Yho) > 1 else 0.0
    tol_val = max(tol_auto, tol_std)

    holdout = {"t_span": (0.0, float(len(ho) - 1)),
               "u_of_t": u_ho,
               "x0": (float(Yho[0, 0]), float(Yho[0, 1])),
               "truth_t": t_ho, "truth_y": Yho,
               "tolerance": tol_val}
    return DynProblem(problem_id, "finance-price", states, inputs,
                      (), truth, train, holdout)
