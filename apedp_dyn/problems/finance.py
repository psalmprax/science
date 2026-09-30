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


def spec_from_series(prices: np.ndarray,
                     train_range: tuple[int, int],
                     holdout_range: tuple[int, int],
                     problem_id: str = PROBLEM_ID) -> DynProblem:
    """Build a DynProblem from a 1-D price series and index ranges.

    train_range / holdout_range are [start, end) index pairs into prices.
    Returns standardized states so optimizer bounds stay meaningful.
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
        v = np.gradient(x, 1.0)  # dx/dt, dt = 1 day
        return np.column_stack([x, v])

    Ytr, Yho = standardize(tr), standardize(ho)
    t_tr = np.arange(len(tr), dtype=float)
    t_ho = np.arange(len(ho), dtype=float)

    states = (StateVariable("x", "dimensionless", float(Ytr[0, 0])),
              StateVariable("v", "1/day", float(Ytr[0, 1])))
    # Dummy zero drive: keeps u_of_t arity correct downstream; no candidate
    # in the finance slate uses the "u" term, so it never enters a model.
    inputs = (InputVariable("u", "dimensionless"),)
    truth = (DynamicalEquation("x", "v"),
             DynamicalEquation("v", "0"))  # placeholder; never integrated
    train = {"t_span": (0.0, float(len(tr) - 1)),
             "u_of_t": make_signal({"kind": "zero", "n": 1}),
             "x0": (float(Ytr[0, 0]), float(Ytr[0, 1])),
             "truth_t": t_tr, "truth_y": Ytr}
    holdout = {"t_span": (0.0, float(len(ho) - 1)),
               "u_of_t": make_signal({"kind": "zero", "n": 1}),
               "x0": (float(Yho[0, 0]), float(Yho[0, 1])),
               "truth_t": t_ho, "truth_y": Yho,
               "tolerance": "auto"}
    return DynProblem(problem_id, "finance-price", states, inputs,
                      (), truth, train, holdout)
