"""Calibration checks for posterior-predictive uncertainty."""
from __future__ import annotations

import numpy as np


def posterior_predictive_calibration(
    predictive: np.ndarray,
    observed: np.ndarray,
    coverage: float = 0.95,
) -> dict[str, float]:
    """Measure empirical coverage and width of posterior-predictive intervals.

    ``predictive`` has shape ``(draws, observations)`` and ``observed`` has
    shape ``(observations,)``. This is a calibration diagnostic, not proof that
    the posterior is correctly specified. It refuses malformed/non-finite
    inputs instead of silently dropping observations.
    """
    draws = np.asarray(predictive, dtype=float)
    values = np.asarray(observed, dtype=float).reshape(-1)
    if draws.ndim != 2 or draws.shape[0] < 2 or draws.shape[1] != values.size:
        raise ValueError("predictive must be (draws >= 2, observations)")
    if not (0.0 < float(coverage) < 1.0):
        raise ValueError("coverage must be between 0 and 1")
    if not np.all(np.isfinite(draws)) or not np.all(np.isfinite(values)):
        raise ValueError("predictive and observed values must be finite")
    tail = (1.0 - float(coverage)) / 2.0
    lower = np.quantile(draws, tail, axis=0)
    upper = np.quantile(draws, 1.0 - tail, axis=0)
    inside = (values >= lower) & (values <= upper)
    width = upper - lower
    empirical = float(np.mean(inside))
    return {
        "nominal_coverage": float(coverage),
        "empirical_coverage": empirical,
        "coverage_error": empirical - float(coverage),
        "mean_interval_width": float(np.mean(width)),
        "n_observations": float(values.size),
    }
