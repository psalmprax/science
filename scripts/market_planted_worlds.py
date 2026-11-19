#!/usr/bin/env python3
"""Return series with a planted law and market-like noise, for PREREG-MARKET-PLANTED-2026-11.

What the physics benchmarks do not have and return data does:

* the predictable part is small: a law explains a fraction of a percent to a few
  percent of the variance of the next return, not 99 %;
* the noise is heavy-tailed (Student t, four degrees of freedom) and its scale
  moves in clusters (GARCH(1,1), persistence 0.98);
* the inputs are slow series, so neighbouring rows are alike, and one of them
  is the current volatility itself, so the noise scale depends on an input;
* in most worlds there is nothing to find.

A world is 5 000 rows of four inputs and one target:

    x0  fast-decaying trend signal     AR(1), coefficient 0.95
    x1  slow spread                    AR(1), coefficient 0.99
    x2  order-flow imbalance           AR(1), coefficient 0.30, correlated 0.3 with x0's shocks
    x3  log volatility, standardised   known before the return it scales
    y   = strength * f(x) + sigma * e

with ``f`` standardised to unit variance and ``strength`` set so that the law
explains a stated share of the variance of ``y`` (zero in the null worlds).
Rows are in time order: the first 60 % are for development, the next 20 % are
the pool a confirmation may read, the last 20 % are for scoring.

Not here: regime changes, a law that fades, transaction costs, or real data.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import sympy as sp

ROWS = 5000
DEV_SHARE, POOL_SHARE = 0.6, 0.2
NAMES = ("x0", "x1", "x2", "x3")
LAWS = {  # in the inputs' names; each is standardised before it is scaled
    "trend": "x0",
    "trend_with_flow": "x0*x2",
    "saturating_trend": "tanh(2*x0)",
    "trend_fading_in_volatility": "x0*exp(-x3)",
    "spread_plus_interaction": "x1 + 0.5*x0*x2",
}
SHARES = (0.005, 0.02, 0.05, 0.2)  # share of the variance of y the law explains
GARCH_ALPHA, GARCH_BETA, T_DF = 0.08, 0.90, 4


def _rng(tag: str, *parts: Any) -> np.random.Generator:
    key = hashlib.sha256("|".join([tag, *map(str, parts)]).encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(key, "big"))


def _ar1(shocks: np.ndarray, rho: float) -> np.ndarray:
    out = np.empty_like(shocks)
    out[0] = shocks[0] / np.sqrt(1.0 - rho ** 2)
    for t in range(1, len(shocks)):
        out[t] = rho * out[t - 1] + shocks[t]
    scaled: np.ndarray = out * np.sqrt(1.0 - rho ** 2)
    return scaled


def world(law: str | None, share: float, seed: int, tag: str = "PREREG-MARKET-PLANTED-2026-11", rows: int = ROWS) -> dict[str, Any]:
    """One world; ``law=None`` (or ``share=0``) is a null world."""
    rng = _rng(tag, law, share, seed)
    burn = 500
    n = rows + burn
    z0, z1, z2 = rng.standard_normal((3, n))
    x0, x1 = _ar1(z0, 0.95), _ar1(z1, 0.99)
    x2 = _ar1(0.3 * z0 + np.sqrt(1.0 - 0.09) * z2, 0.30)
    innovations = rng.standard_t(T_DF, n) / np.sqrt(T_DF / (T_DF - 2.0))
    variance = np.empty(n)
    noise = np.empty(n)
    variance[0] = 1.0
    noise[0] = innovations[0]
    omega = 1.0 - GARCH_ALPHA - GARCH_BETA
    for t in range(1, n):
        variance[t] = omega + GARCH_ALPHA * noise[t - 1] ** 2 + GARCH_BETA * variance[t - 1]
        noise[t] = np.sqrt(variance[t]) * innovations[t]
    keep = slice(burn, n)
    log_vol = 0.5 * np.log(variance[keep])
    X = np.column_stack([x0[keep], x1[keep], x2[keep], (log_vol - log_vol.mean()) / log_vol.std()])
    signal = np.zeros(rows)
    truth: sp.Expr | None = None
    if law is not None and share > 0:
        truth = sp.sympify(LAWS[law])
        raw = np.asarray(sp.lambdify([sp.Symbol(n_) for n_ in NAMES], truth, "numpy")(*X.T), dtype=float)
        signal = np.sqrt(share / (1.0 - share)) * (raw - raw.mean()) / raw.std()
    y = signal + noise[keep]
    a, b = round(DEV_SHARE * rows), round((DEV_SHARE + POOL_SHARE) * rows)
    return {"X": X, "y": y, "signal": signal, "truth": truth, "dev": slice(0, a), "pool": slice(a, b), "test": slice(b, rows),
            "law": law, "share": share if truth is not None else 0.0, "seed": seed}
