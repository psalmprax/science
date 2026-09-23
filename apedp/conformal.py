# Databricks notebook source
"""Group-aware split-conformal prediction intervals.

Point extrapolation gets a quantified boundary: intervals are calibrated on
whole held-out groups (compounds, satellites, files -- never random rows),
so the reported width reflects generalization across the grouping that
matters. Coverage is reported per group, because pooled coverage can hide
systematic failure on one group.

Exchangeability caveat (documented): split-conformal coverage guarantees
assume calibration and test points are exchangeable. Grouped calibration
weakens row-level dependence but does not manufacture exchangeability across
regimes; the per-group coverage report exists precisely to expose violations.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConformalInterval:
    """Calibrated interval estimator from one calibration split."""

    halfwidth: float
    alpha: float
    n_calibration: int
    calibration_mae: float

    def band(self, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        point = np.asarray(prediction, dtype=float).ravel()
        width = np.full(point.shape, self.halfwidth)
        return point - width, point + width

    def to_dict(self) -> dict[str, Any]:
        return {
            "halfwidth": self.halfwidth,
            "alpha": self.alpha,
            "n_calibration": self.n_calibration,
            "calibration_mae": self.calibration_mae,
        }


def conformalize(
    y_calib_true: Iterable[float],
    y_calib_pred: Iterable[float],
    *,
    alpha: float = 0.1,
) -> ConformalInterval:
    """Split-conformal halfwidth from absolute residuals on calibration data.

    Uses the finite-sample corrected quantile ``ceil((n+1)(1-alpha))/n``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    truth = np.asarray(list(y_calib_true), dtype=float).ravel()
    pred = np.asarray(list(y_calib_pred), dtype=float).ravel()
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("calibration truth and predictions must be non-empty and aligned")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(pred)):
        raise ValueError("calibration data must be finite")
    scores = np.abs(truth - pred)
    level = float(np.ceil((scores.size + 1) * (1.0 - alpha)) / scores.size)
    halfwidth = float(np.quantile(scores, min(level, 1.0), method="higher"))
    return ConformalInterval(halfwidth, alpha, int(scores.size), float(np.mean(scores)))


PredictFn = Callable[[np.ndarray], np.ndarray]


def grouped_conformal(
    predict_calib: PredictFn,
    y_calib_true: Iterable[float],
    *,
    alpha: float = 0.1,
    calib_group_ids: Iterable[Any] | None = None,
    group_of_calib: Iterable[Any] | None = None,
) -> ConformalInterval:
    """Calibrate on whole held-out groups; optionally verify group sizes.

    ``predict_calib`` maps calibration features to point predictions;
    ``y_calib_true`` are the locked calibration targets. When
    ``group_of_calib`` is supplied, every listed ``calib_group_ids`` entry
    must actually occur in the calibration rows -- a guard against silent
    group leakage where a "held-out" group never appears.
    """
    interval = conformalize(y_calib_true, predict_calib, alpha=alpha)
    if group_of_calib is not None and calib_group_ids is not None:
        seen = set(group_of_calib)
        missing = [group for group in calib_group_ids if group not in seen]
        if missing:
            raise ValueError(f"calibration groups absent from calibration rows: {missing}")
    return interval


def coverage_report(
    y_true: Iterable[float],
    lower: Iterable[float],
    upper: Iterable[float],
    groups: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Empirical coverage overall and per group."""
    truth = np.asarray(list(y_true), dtype=float).ravel()
    low = np.asarray(list(lower), dtype=float).ravel()
    high = np.asarray(list(upper), dtype=float).ravel()
    if not (truth.shape == low.shape == high.shape) or truth.size == 0:
        raise ValueError("coverage inputs must be non-empty and aligned")
    covered = (truth >= low) & (truth <= high)
    report: dict[str, Any] = {
        "n": int(truth.size),
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(high - low)),
    }
    if groups is not None:
        group_list = list(groups)
        if len(group_list) != truth.size:
            raise ValueError("groups must align with observations")
        per_group = {}
        for group in sorted(set(group_list), key=str):
            mask = np.asarray([item == group for item in group_list])
            per_group[str(group)] = {
                "n": int(mask.sum()),
                "coverage": float(np.mean(covered[mask])) if mask.sum() else None,
            }
        report["per_group"] = per_group
    return report


__all__ = ["ConformalInterval", "conformalize", "coverage_report", "grouped_conformal"]
