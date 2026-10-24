# Databricks notebook source
"""Leakage-safe feature construction for regularly sampled time series."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def build_lagged_supervised_dataset(
    features: Mapping[str, Sequence[float] | np.ndarray],
    target: Sequence[float] | np.ndarray,
    *,
    lags: Sequence[int],
    horizon: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Align past and present observations with a future target.

    Inputs must be equally sized observations sampled on a regular time grid.
    A feature with lag ``k`` at source index ``t`` is paired only with target
    value ``t + horizon``. Rows with a non-finite value anywhere in their
    input history or future target are excluded.
    """
    unique_lags = sorted({int(lag) for lag in lags})
    if not unique_lags or unique_lags[0] < 0:
        raise ValueError("lags must contain one or more non-negative integers")
    if horizon < 1:
        raise ValueError("horizon must be at least one sample")

    y = np.asarray(target, dtype=float).ravel()
    if not features:
        raise ValueError("at least one feature is required")
    columns = {
        name: np.asarray(values, dtype=float).ravel()
        for name, values in features.items()
    }
    if any(values.shape != y.shape for values in columns.values()):
        raise ValueError("every feature and target must have the same length")

    start = unique_lags[-1]
    stop = len(y) - horizon
    if stop <= start:
        raise ValueError("series is too short for the requested lags and horizon")
    indices = np.arange(start, stop)
    x_data = {
        f"{name}_lag_{lag}": values[indices - lag]
        for name, values in columns.items()
        for lag in unique_lags
    }
    y_data = y[indices + horizon]
    valid = np.isfinite(y_data)
    for values in x_data.values():
        valid &= np.isfinite(values)
    return (
        {name: values[valid] for name, values in x_data.items()},
        y_data[valid],
        indices[valid],
    )


__all__ = ["build_lagged_supervised_dataset"]
