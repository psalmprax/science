# Databricks notebook source
"""Fail-closed admission decisions for discovered equations."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp

from apedp.discovery import BayesianRefinementEngine
from apedp.safety import safe_parse_expr


class AdmissionStatus(str, Enum):
    PERSISTED = "PERSISTED"
    REFUSED = "REFUSED"


@dataclass(frozen=True)
class AdmissionDecision:
    """Auditable result of mandatory candidate checks."""

    status: AdmissionStatus
    reasons: tuple[str, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "reasons": list(self.reasons), "evidence": self.evidence}


def _finite_predictions(expression: sp.Expr, x_data: dict[str, np.ndarray]) -> np.ndarray:
    names = sorted(x_data)
    function = sp.lambdify([sp.Symbol(name) for name in names], expression, modules=["numpy"])
    inputs = [np.asarray(x_data[name], dtype=float) for name in names]
    prediction = np.asarray(function(*inputs), dtype=float)
    if prediction.ndim == 0:
        prediction = np.full(inputs[0].shape, float(prediction), dtype=float)
    return np.broadcast_to(prediction, inputs[0].shape).astype(float, copy=False)


def numeric_conditioning_report(
    expression: sp.Expr | str,
    x_data: dict[str, np.ndarray],
    *,
    coefficient_magnitude_limit: float | None = None,
    cancellation_ratio_limit: float = 1.0e8,
) -> dict[str, Any]:
    """Measure suspicious coefficient scale and destructive cancellation."""
    expr = safe_parse_expr(expression) if isinstance(expression, str) else expression
    coefficients = [abs(float(term)) for term in expr.atoms(sp.Float) if np.isfinite(float(term))]
    max_coefficient = max(coefficients, default=0.0)
    try:
        prediction = _finite_predictions(expr, x_data)
        finite = bool(np.all(np.isfinite(prediction)))
        terms = sp.Add.make_args(expr)
        if len(terms) < 2:
            cancellation_ratio = 1.0
        else:
            term_values = np.vstack([np.abs(_finite_predictions(term, x_data)) for term in terms])
            numerator = np.sum(term_values, axis=0)
            denominator = np.maximum(np.abs(prediction), np.finfo(float).tiny)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                cancellation_ratio = float(np.max(numerator / denominator))
    except (TypeError, ValueError, ArithmeticError, OverflowError):
        finite = False
        cancellation_ratio = float("inf")
    coefficient_within_limit = coefficient_magnitude_limit is None or max_coefficient <= coefficient_magnitude_limit
    cancellation_within_limit = finite and cancellation_ratio <= cancellation_ratio_limit
    return {
        "finite": finite,
        "max_coefficient_magnitude": max_coefficient,
        "coefficient_magnitude_limit": coefficient_magnitude_limit,
        "coefficient_within_limit": coefficient_within_limit,
        "max_cancellation_ratio": cancellation_ratio,
        "cancellation_ratio_limit": cancellation_ratio_limit,
        "cancellation_within_limit": cancellation_within_limit,
    }


def admit_candidate(
    expression: sp.Expr | str | None,
    *,
    x_holdout: dict[str, np.ndarray] | None = None,
    y_holdout: np.ndarray | None = None,
    train_r_squared: float | None = None,
    min_holdout_r_squared: float | None = None,
    max_r2_drop: float | None = None,
    dimensionally_consistent: bool = True,
    repair_status: str | None = "ACCEPTED",
    degenerate: bool = False,
    search_stagnated: bool = False,
    baseline_predictions: np.ndarray | None = None,
    require_baseline_improvement: bool = False,
    conditioning: dict[str, Any] | None = None,
) -> AdmissionDecision:
    """Return ``PERSISTED`` only when every supplied mandatory check passes."""
    reasons: list[str] = []
    evidence: dict[str, Any] = {
        "dimensionally_consistent": bool(dimensionally_consistent),
        "repair_status": repair_status,
        "degenerate": bool(degenerate),
        "search_stagnated": bool(search_stagnated),
    }
    try:
        expr = safe_parse_expr(expression) if isinstance(expression, str) else expression
        if expr is None:
            raise ValueError("missing expression")
    except (TypeError, ValueError) as exc:
        reasons.append(f"unparseable expression: {exc}")
        expr = None
    if not dimensionally_consistent:
        reasons.append("dimensional validation failed")
    if repair_status in {"REJECTED", "UNDETERMINED", None}:
        reasons.append(f"repair status is not admissible: {repair_status}")
    if degenerate:
        reasons.append("candidate is degenerate")
    if search_stagnated:
        reasons.append("search stagnated")
    if conditioning is not None:
        evidence["numeric_conditioning"] = conditioning
        if not conditioning.get("finite", False):
            reasons.append("numeric conditioning evaluation is non-finite")
        if not conditioning.get("coefficient_within_limit", True):
            reasons.append("coefficient magnitude exceeds declared scale limit")
        if not conditioning.get("cancellation_within_limit", True):
            reasons.append("destructive cancellation exceeds declared limit")
    if x_holdout is not None or y_holdout is not None:
        if expr is None or x_holdout is None or y_holdout is None:
            reasons.append("locked holdout evidence is incomplete")
        else:
            try:
                y_true = np.asarray(y_holdout, dtype=float).ravel()
                y_pred = _finite_predictions(expr, x_holdout).ravel()
                finite_holdout = bool(y_true.shape == y_pred.shape and np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred)))
                holdout_r_squared = BayesianRefinementEngine.compute_r_squared(y_true, y_pred) if finite_holdout else float("nan")
                evidence["holdout_r_squared"] = float(holdout_r_squared)
                evidence["holdout_finite"] = finite_holdout
                if not finite_holdout:
                    reasons.append("holdout predictions are non-finite or misaligned")
                if min_holdout_r_squared is not None and holdout_r_squared < min_holdout_r_squared:
                    reasons.append("holdout R-squared is below the declared minimum")
                if train_r_squared is not None and max_r2_drop is not None:
                    drop = float(train_r_squared - holdout_r_squared)
                    evidence["r2_drop"] = drop
                    if not np.isfinite(drop) or drop > max_r2_drop:
                        reasons.append("train-to-holdout R-squared degradation exceeds the declared maximum")
                if require_baseline_improvement:
                    if baseline_predictions is None:
                        reasons.append("baseline improvement is required but no baseline was supplied")
                    else:
                        baseline = np.asarray(baseline_predictions, dtype=float).ravel()
                        candidate_mse = float(np.mean((y_pred - y_true) ** 2))
                        baseline_mse = float(np.mean((baseline - y_true) ** 2)) if baseline.shape == y_true.shape and np.all(np.isfinite(baseline)) else float("inf")
                        evidence["candidate_holdout_mse"] = candidate_mse
                        evidence["baseline_holdout_mse"] = baseline_mse
                        if not candidate_mse < baseline_mse:
                            reasons.append("candidate does not improve the required baseline on holdout")
            except (TypeError, ValueError, ArithmeticError, OverflowError) as exc:
                reasons.append(f"holdout evaluation failed: {exc}")
    return AdmissionDecision(AdmissionStatus.REFUSED if reasons else AdmissionStatus.PERSISTED, tuple(reasons), evidence)


__all__ = ["AdmissionDecision", "AdmissionStatus", "admit_candidate", "numeric_conditioning_report"]