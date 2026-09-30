# Databricks notebook source
"""Data-driven proposal and selection of compact univariate equation families."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class FamilyCandidate:
    family: str
    expression: sp.Expr
    mse: float
    bic: float
    parameters: dict[str, float | list[float]]


def _bic(y: np.ndarray, prediction: np.ndarray, parameter_count: int) -> tuple[float, float]:
    mse = float(np.mean((prediction - y) ** 2))
    return mse, float(len(y) * np.log(max(mse, 1e-300)) + parameter_count * np.log(len(y)))


def _fit(terms: list[np.ndarray], y: np.ndarray, shape_parameter_count: int = 0) -> tuple[np.ndarray, float, float]:
    design = np.column_stack([*terms, np.ones_like(y)])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    mse, bic = _bic(y, design @ coefficients, len(coefficients) + shape_parameter_count)
    return coefficients, mse, bic


def _best_grid(
    y: np.ndarray,
    family: str,
    grid: list[tuple[dict[str, float], list[np.ndarray], Callable[..., sp.Expr]]],
) -> FamilyCandidate | None:
    best = None
    for parameters, terms, builder in grid:
        if not all(np.all(np.isfinite(term)) for term in terms):
            continue
        coefficients, mse, bic = _fit(terms, y, len(parameters))
        item = (bic, mse, parameters, builder, coefficients)
        if best is None or item[:2] < best[:2]:
            best = item
    if best is None:
        return None
    bic, mse, parameters, builder, coefficients = best
    return FamilyCandidate(family, builder(coefficients), mse, bic, {**parameters, "coefficients": [float(value) for value in coefficients]})


def propose_univariate_families(x_data: np.ndarray, y_data: np.ndarray, *, variable: str = "x", grid_size: int = 48) -> list[FamilyCandidate]:
    """Fit generic equation families and rank their training fits by BIC."""
    x = np.asarray(x_data, dtype=float).ravel()
    y = np.asarray(y_data, dtype=float).ravel()
    if x.shape != y.shape or len(x) < 8:
        raise ValueError("x_data and y_data must have equal length of at least 8")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x_data and y_data must be finite")
    symbol = sp.Symbol(variable)
    candidates: list[FamilyCandidate] = []

    for degree in (1, 2, 3, 4):
        coefficients, mse, bic = _fit([x ** power for power in range(1, degree + 1)], y)
        expression = sum(sp.Float(coefficients[power - 1]) * symbol ** power for power in range(1, degree + 1)) + sp.Float(coefficients[-1])
        candidates.append(FamilyCandidate(f"polynomial-{degree}", expression, mse, bic, {"degree": float(degree), "coefficients": [float(value) for value in coefficients]}))

    scale = max(float(np.median(np.abs(x))), float(np.ptp(x)), 1e-6)
    span = max(float(np.ptp(x)), 1e-6)
    if np.all(x > 0):
        shifts = np.geomspace(scale * 1e-4, scale * 2.0, grid_size)
        exponents = np.linspace(0.1, 3.0, grid_size)
        grids: dict[str, list] = {name: [] for name in ("shifted-power", "shifted-reciprocal", "logarithmic", "rational-saturation", "exponential-decay")}
        for shift in shifts:
            shifted = x + shift
            grids["shifted-reciprocal"].append(({"shift": float(shift)}, [1 / shifted], lambda c, shift=shift: c[0] / (symbol + sp.Float(shift)) + c[1]))
            grids["logarithmic"].append(({"shift": float(shift)}, [np.log(shifted)], lambda c, shift=shift: c[0] * sp.log(symbol + sp.Float(shift)) + c[1]))
            grids["rational-saturation"].append(({"shift": float(shift)}, [x / shifted], lambda c, shift=shift: c[0] * symbol / (symbol + sp.Float(shift)) + c[1]))
            for exponent in exponents:
                grids["shifted-power"].append(({"shift": float(shift), "exponent": float(exponent)}, [shifted ** (-exponent)], lambda c, shift=shift, exponent=exponent: c[0] / (symbol + sp.Float(shift)) ** sp.Float(exponent) + c[1]))
        for rate in np.geomspace(1e-4 / scale, 5 / scale, grid_size):
            grids["exponential-decay"].append(({"rate": float(rate)}, [np.exp(-rate * x)], lambda c, rate=rate: c[0] * sp.exp(-sp.Float(rate) * symbol) + c[1]))
        for family, grid in grids.items():
            candidate = _best_grid(y, family, grid)
            if candidate is not None:
                candidates.append(candidate)

    frequencies = np.geomspace(2 * np.pi / (span * 4), 2 * np.pi * 4 / span, grid_size)
    sinusoid_grid, damped_grid, logistic_grid, gaussian_grid = [], [], [], []
    for frequency in frequencies:
        sinusoid_grid.append(({"frequency": float(frequency)}, [np.sin(frequency * x), np.cos(frequency * x)], lambda c, frequency=frequency: c[0] * sp.sin(sp.Float(frequency) * symbol) + c[1] * sp.cos(sp.Float(frequency) * symbol) + c[2]))
        for rate in np.geomspace(1e-4 / span, 5 / span, grid_size):
            envelope = np.exp(-rate * (x - x.min()))
            damped_grid.append(({"frequency": float(frequency), "decay_rate": float(rate)}, [envelope * np.sin(frequency * x), envelope * np.cos(frequency * x)], lambda c, frequency=frequency, rate=rate: c[0] * sp.exp(-sp.Float(rate) * (symbol - sp.Float(x.min()))) * sp.sin(sp.Float(frequency) * symbol) + c[1] * sp.exp(-sp.Float(rate) * (symbol - sp.Float(x.min()))) * sp.cos(sp.Float(frequency) * symbol) + c[2]))
    for midpoint in np.linspace(float(x.min()), float(x.max()), grid_size):
        for rate in np.geomspace(1e-4 / span, 5 / span, grid_size):
            sigmoid = 1 / (1 + np.exp(-rate * (x - midpoint)))
            logistic_grid.append(({"midpoint": float(midpoint), "rate": float(rate)}, [sigmoid], lambda c, midpoint=midpoint, rate=rate: c[0] / (1 + sp.exp(-sp.Float(rate) * (symbol - sp.Float(midpoint)))) + c[1]))
        for width in np.geomspace(span / 100, span * 2, grid_size):
            gaussian = np.exp(-0.5 * ((x - midpoint) / width) ** 2)
            gaussian_grid.append(({"center": float(midpoint), "width": float(width)}, [gaussian], lambda c, midpoint=midpoint, width=width: c[0] * sp.exp(-sp.Rational(1, 2) * ((symbol - sp.Float(midpoint)) / sp.Float(width)) ** 2) + c[1]))
    for family, grid in (("sinusoid", sinusoid_grid), ("damped-sinusoid", damped_grid), ("logistic", logistic_grid), ("gaussian-envelope", gaussian_grid)):
        candidate = _best_grid(y, family, grid)
        if candidate is not None:
            candidates.append(candidate)
    return sorted(candidates, key=lambda candidate: (candidate.bic, candidate.mse, candidate.family))


def select_univariate_family(x_data: np.ndarray, y_data: np.ndarray, *, variable: str = "x") -> FamilyCandidate:
    return propose_univariate_families(x_data, y_data, variable=variable)[0]


def select_turnover_family(x_data: np.ndarray, y_data: np.ndarray, *, variable: str = "x") -> FamilyCandidate | None:
    """Select the simplest BIC candidate with a resolved interior maximum."""
    x = np.asarray(x_data, dtype=float).ravel()
    candidates = propose_univariate_families(x, y_data, variable=variable)
    symbol = sp.Symbol(variable)
    grid = np.linspace(float(x.min()), float(x.max()), 512)
    for candidate in candidates:
        prediction = np.asarray(sp.lambdify(symbol, candidate.expression, "numpy")(grid), dtype=float)
        if prediction.ndim == 0:
            prediction = np.full_like(grid, float(prediction))
        peak = int(np.argmax(prediction))
        if 0 < peak < len(grid) - 1 and prediction[peak] > max(prediction[0], prediction[-1]):
            return candidate
    return None


def discover_with_proposed_family(
    search: Any,
    x_data: dict[str, np.ndarray],
    y_data: np.ndarray,
    **run_kwargs: Any,
) -> tuple[sp.Expr | None, float, list[dict[str, Any]]]:
    """Route a single-variable dimensionless proposal into normal discovery.

    The selected expression is appended as a template seed before invoking the
    existing engine. It is fitted from the supplied training arrays only.
    """
    if len(x_data) != 1 or getattr(search, "target_unit_str", None) != "dimensionless":
        return cast("tuple[Any | None, float, list[dict[str, Any]]]",
                    search.run_discovery(x_data, y_data, **run_kwargs))
    name, values = next(iter(x_data.items()))
    candidate = select_univariate_family(values, y_data, variable=name)
    if candidate.expression not in search.template_exprs:
        search.template_exprs.insert(0, candidate.expression)
    expression, score, history = search.run_discovery(x_data, y_data, **run_kwargs)
    history.insert(0, {"proposed_family": candidate.family, "proposal_bic": candidate.bic, "proposal_expression": str(candidate.expression)})
    return expression, score, history


__all__ = ["FamilyCandidate", "discover_with_proposed_family", "propose_univariate_families", "select_turnover_family", "select_univariate_family"]


# COMMAND ----------

def select_univariate_family_validated(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    variable: str = "x",
    validation_uncertainty: np.ndarray | None = None,
) -> FamilyCandidate | None:
    """Choose a training-fitted family by untouched validation error.

    BIC orders ties only. When uncertainties are supplied, residuals are
    weighted by inverse variance so noisy observations cannot dominate family
    selection. The returned expression is fitted only on ``x_train, y_train``;
    callers may refit its family separately after the choice is locked.
    """
    x_val = np.asarray(x_validation, dtype=float).ravel()
    y_val = np.asarray(y_validation, dtype=float).ravel()
    if x_val.shape != y_val.shape or len(x_val) < 3:
        raise ValueError("validation arrays must have equal length of at least 3")
    if not np.all(np.isfinite(x_val)) or not np.all(np.isfinite(y_val)):
        raise ValueError("validation arrays must be finite")
    weights = np.ones_like(y_val)
    if validation_uncertainty is not None:
        uncertainty = np.asarray(validation_uncertainty, dtype=float).ravel()
        if uncertainty.shape != y_val.shape or not np.all(np.isfinite(uncertainty)) or np.any(uncertainty <= 0):
            raise ValueError("validation_uncertainty must be finite, positive, and match validation rows")
        weights = 1.0 / uncertainty ** 2
    symbol = sp.Symbol(variable)
    scored: list[tuple[float, float, FamilyCandidate]] = []
    for candidate in propose_univariate_families(x_train, y_train, variable=variable):
        try:
            prediction = np.asarray(sp.lambdify(symbol, candidate.expression, "numpy")(x_val), dtype=float)
            if prediction.ndim == 0:
                prediction = np.full_like(y_val, float(prediction))
            if prediction.shape != y_val.shape or not np.all(np.isfinite(prediction)):
                continue
            weighted_mse = float(np.sum(weights * (prediction - y_val) ** 2) / np.sum(weights))
            scored.append((weighted_mse, candidate.bic, candidate))
        except (TypeError, ValueError, ArithmeticError):
            continue
    return min(scored, default=(float("inf"), float("inf"), None), key=lambda item: (item[0], item[1]))[2]


__all__.append("select_univariate_family_validated")
