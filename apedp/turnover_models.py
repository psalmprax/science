# Databricks notebook source
"""Scale-aware selection for non-monotonic one-variable relationships."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class TurnoverCandidate:
    expression: sp.Expr
    mse: float
    bic: float
    peak_x: float
    family: str = "log-polynomial-turnover"


def select_log_scale_turnover(x_data: np.ndarray, y_data: np.ndarray, *, variable: str = "x") -> TurnoverCandidate | None:
    """Fit BIC-ranked polynomials in ``log(x)`` with an interior maximum.

    This is a generic regime selector for positive data spanning orders of
    magnitude. It searches polynomial degrees two through four, requires a
    maximum strictly inside the measured log-domain, and returns the least-BIC
    valid turnover candidate. Training arrays only are used.
    """
    x = np.asarray(x_data, dtype=float).ravel()
    y = np.asarray(y_data, dtype=float).ravel()
    if x.shape != y.shape or len(x) < 8 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.all(x > 0):
        return None
    z = np.log(x)
    symbol = sp.Symbol(variable)
    probe = np.linspace(float(z.min()), float(z.max()), 2048)
    best: TurnoverCandidate | None = None
    for degree in (2, 3, 4):
        design = np.column_stack([z ** power for power in range(degree + 1)])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        prediction = design @ coefficients
        mse = float(np.mean((prediction - y) ** 2))
        bic = float(len(y) * np.log(max(mse, 1e-300)) + (degree + 1) * np.log(len(y)))
        curve = sum(coefficients[power] * probe ** power for power in range(degree + 1))
        peak_index = int(np.argmax(curve))
        if peak_index in (0, len(probe) - 1) or curve[peak_index] <= max(curve[0], curve[-1]):
            continue
        expression = sum(sp.Float(coefficients[power]) * sp.log(symbol) ** power for power in range(degree + 1))
        candidate = TurnoverCandidate(expression, mse, bic, float(np.exp(probe[peak_index])))
        if best is None or (candidate.bic, candidate.mse) < (best.bic, best.mse):
            best = candidate
    return best


__all__ = ["TurnoverCandidate", "select_log_scale_turnover"]
