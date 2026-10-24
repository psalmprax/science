# Databricks notebook source
"""Compact, deterministic priors for shifted power-law decay."""
from __future__ import annotations

import numpy as np
import sympy as sp


def fit_shifted_power_decay(
    x_data: np.ndarray,
    y_data: np.ndarray,
    *,
    variable: str = "x",
    shift_points: int = 64,
    exponent_points: int = 80,
) -> sp.Expr:
    """Fit ``A / (x + c)**p + B`` using a deterministic grid search.

    The shift and exponent are selected from the supplied training data only.
    For each pair, amplitude and offset are solved with linear least squares.
    This helper is intended for dimensionless or pre-normalized time axes.
    """
    x = np.asarray(x_data, dtype=float).ravel()
    y = np.asarray(y_data, dtype=float).ravel()
    if x.shape != y.shape or len(x) < 8:
        raise ValueError("x_data and y_data must be equal-length arrays with at least 8 rows")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.all(x > 0):
        raise ValueError("shifted power decay requires finite positive x_data and finite y_data")

    scale = max(float(np.median(x)), float(np.ptp(x)), 1e-6)
    best: tuple[float, float, float, float, float] | None = None
    for shift in np.geomspace(scale * 1e-4, scale * 2.0, shift_points):
        shifted = x + shift
        for exponent in np.linspace(0.1, 3.0, exponent_points):
            basis = shifted ** (-exponent)
            design = np.column_stack([basis, np.ones_like(basis)])
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            mse = float(np.mean((design @ coefficients - y) ** 2))
            candidate = (mse, shift, exponent, float(coefficients[0]), float(coefficients[1]))
            if best is None or candidate < best:
                best = candidate
    if best is None:  # unreachable while shift_points >= 1; a real guard, not an assert
        raise RuntimeError("shifted power-decay fit produced no candidate")
    # Fresh names: ``shift``/``exponent`` are already bound as np.float64 by the
    # search loops above, so reusing them here mixes numpy and Python scalars.
    _, fit_shift, fit_exponent, amplitude, offset = best
    symbol = sp.Symbol(variable)
    return (sp.Float(amplitude) / (symbol + sp.Float(fit_shift)) ** sp.Float(fit_exponent)
            + sp.Float(offset))


__all__ = ["fit_shifted_power_decay"]
