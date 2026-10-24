# Databricks notebook source
"""Rolling-origin diagnostics for forecast quality versus parameter identifiability."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RollingOriginPoint:
    """One origin's fitted parameter, fit score, and future forecast error."""

    endpoint: Any
    parameter: float
    fit_score: float
    forecast_error: float
    forecast_horizon: int

    def to_dict(self) -> dict[str, Any]:
        return {"endpoint": self.endpoint, "parameter": self.parameter, "fit_score": self.fit_score, "forecast_error": self.forecast_error, "forecast_horizon": self.forecast_horizon}


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Evidence separating predictive performance from parameter stability."""

    points: tuple[RollingOriginPoint, ...]
    status: str
    parameter_range: float | None
    parameter_relative_range: float | None
    forecast_error_trend: float | None
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {"points": [point.to_dict() for point in self.points], "status": self.status, "parameter_range": self.parameter_range, "parameter_relative_range": self.parameter_relative_range, "forecast_error_trend": self.forecast_error_trend, "interpretation": self.interpretation}


def rolling_origin_identifiability(
    x: Iterable[Any],
    y: Iterable[float],
    endpoints: Iterable[int],
    fit: Callable[[np.ndarray, np.ndarray], tuple[float, Callable[[np.ndarray], np.ndarray]]],
    *,
    horizon: int = 1,
    score: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> IdentifiabilityReport:
    """Measure parameter drift and forecast behavior without claiming formal uncertainty."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    x_values = np.asarray(list(x))
    y_values = np.asarray(list(y), dtype=float).ravel()
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same length")
    def default_score(observed, predicted):
        denominator = max(np.sum((observed - np.mean(observed)) ** 2), np.finfo(float).tiny)
        return float(1.0 - np.sum((observed - predicted) ** 2) / denominator)
    score_fn = score or default_score
    points: list[RollingOriginPoint] = []
    for endpoint in endpoints:
        endpoint_int = int(endpoint)
        if endpoint_int < 2 or endpoint_int + horizon > len(y_values):
            raise ValueError(f"endpoint {endpoint_int} leaves no complete forecast horizon")
        parameter, predictor = fit(x_values[:endpoint_int], y_values[:endpoint_int])
        train_prediction = np.asarray(predictor(x_values[:endpoint_int]), dtype=float).ravel()
        future_prediction = np.asarray(predictor(x_values[endpoint_int:endpoint_int + horizon]), dtype=float).ravel()
        if train_prediction.shape != (endpoint_int,) or future_prediction.shape != (horizon,):
            raise ValueError("fit predictor returned an unexpected shape")
        if not np.all(np.isfinite(train_prediction)) or not np.all(np.isfinite(future_prediction)):
            raise ValueError("fit predictor returned non-finite values")
        points.append(RollingOriginPoint(endpoint, float(parameter), float(score_fn(y_values[:endpoint_int], train_prediction)), float(np.mean(np.abs(future_prediction - y_values[endpoint_int:endpoint_int + horizon]))), horizon))
    parameters = np.asarray([point.parameter for point in points], dtype=float)
    errors = np.asarray([point.forecast_error for point in points], dtype=float)
    parameter_range = float(np.max(parameters) - np.min(parameters)) if parameters.size else None
    scale = max(float(np.median(np.abs(parameters))), np.finfo(float).tiny) if parameters.size else np.nan
    relative_range = float(parameter_range / scale) if parameter_range is not None else None
    error_trend = float(np.polyfit(np.arange(errors.size), errors, 1)[0]) if errors.size >= 2 else None
    if not points or not np.all(np.isfinite(parameters)):
        status = "UNRESOLVED"
        interpretation = "Parameter identifiability could not be measured."
    elif relative_range is not None and relative_range > 0.1:
        status = "POORLY_IDENTIFIED"
        interpretation = "Forecast quality and parameter identifiability must be reported separately; the parameter is origin-sensitive."
    else:
        status = "STABLE"
        interpretation = "Parameter variation is small across the evaluated origins; this is not a formal uniqueness proof."
    return IdentifiabilityReport(tuple(points), status, parameter_range, relative_range, error_trend, interpretation)


__all__ = ["IdentifiabilityReport", "RollingOriginPoint", "rolling_origin_identifiability"]