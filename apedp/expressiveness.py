# Databricks notebook source
"""Grammar expressiveness audit: detect when no candidate can win honestly.

When the best holdout fit is poor *and* the top candidates cluster at the
same poor error scale *and* residuals carry systematic structure, crowning a
"winner" misleads: the target structure likely lies outside the grammar.
The audit then returns ``STRUCTURE_OUTSIDE_GRAMMAR`` instead of a champion.

This is a documented heuristic, not a proof of unrepresentability. Thresholds
are explicit arguments so every verdict carries its own decision boundary.
``INCONCLUSIVE`` covers the middle ground where poor fit could equally be
noise, omitted variables, or grammar limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ExpressivenessVerdict:
    """Audit outcome over one grammar's holdout predictions."""

    status: str
    best_form: str | None
    best_r_squared: float | None
    top_spread_ratio: float | None
    residual_structure: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "best_form": self.best_form,
            "best_r_squared": self.best_r_squared,
            "top_spread_ratio": self.top_spread_ratio,
            "residual_structure": self.residual_structure,
            "reason": self.reason,
        }


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if total <= 0.0 or not np.all(np.isfinite(y_pred)):
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / total)


def _residual_structure(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 5) -> float:
    """Strength of systematic residual pattern (0 = none, up to ~1).

    Takes the maximum of (a) absolute residual-prediction correlation and
    (b) a binned score: standard deviation of per-bin residual means,
    normalized by residual spread. Component (b) catches oscillatory
    misspecification whose linear correlation is near zero -- the canonical
    too-small-grammar signature.
    """
    residual = y_true - y_pred
    if residual.size < 3 or float(np.std(y_pred)) == 0.0 or float(np.std(residual)) == 0.0:
        return 0.0
    correlation = np.corrcoef(residual, y_pred)[0, 1]
    linear_part = float(abs(correlation)) if np.isfinite(correlation) else 0.0
    order = np.argsort(y_pred)
    binned = np.array_split(residual[order], min(bins, residual.size))
    means = np.array([part.mean() for part in binned if part.size])
    binned_part = float(np.std(means) / np.std(residual)) if means.size >= 2 else 0.0
    return float(max(linear_part, min(binned_part, 1.0)))


def audit_grammar_fit(
    holdout_predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    r_squared_threshold: float = 0.9,
    top_spread_factor: float = 2.0,
    structure_threshold: float = 0.3,
    top_k: int = 3,
) -> ExpressivenessVerdict:
    """Audit ``{form: (y_true, y_pred)}`` holdout predictions.

    * ``SUFFICIENT``: best R-squared clears the threshold -- grammar suffices
      for this data, whatever its other limits.
    * ``STRUCTURE_OUTSIDE_GRAMMAR``: best R-squared misses the threshold,
      the top-``k`` forms cluster within ``top_spread_factor`` in RMSE (no
      candidate is pulling away), and best-form residuals carry structure
      above ``structure_threshold``.
    * ``INCONCLUSIVE``: everything else.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    stats: list[tuple[str, float, float]] = []
    ranked: list[tuple[str, np.ndarray, np.ndarray]] = []
    for form, (truth, pred) in holdout_predictions.items():
        y_true = np.asarray(truth, dtype=float).ravel()
        y_pred = np.asarray(pred, dtype=float).ravel()
        if y_true.shape != y_pred.shape or y_true.size == 0:
            continue
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if np.all(np.isfinite(y_pred)) else float("inf")
        stats.append((form, _r_squared(y_true, y_pred), rmse))
        ranked.append((form, y_true, y_pred))
    if not stats:
        return ExpressivenessVerdict("INCONCLUSIVE", None, None, None, None, "no evaluable predictions")
    stats.sort(key=lambda item: (not np.isfinite(item[2]), item[2]))
    best_form, best_r2, best_rmse = stats[0]
    if np.isfinite(best_r2) and best_r2 >= r_squared_threshold:
        return ExpressivenessVerdict("SUFFICIENT", best_form, best_r2, None, None, "best holdout R-squared clears the threshold")
    top = stats[: min(top_k, len(stats))]
    finite = [rmse for _, _, rmse in top if np.isfinite(rmse) and rmse > 0.0]
    spread = (max(finite) / min(finite)) if finite else None
    _form, y_true, y_pred = ranked[[form for form, _, _ in stats].index(best_form)]
    structure = _residual_structure(y_true, y_pred)
    if (
        (not np.isfinite(best_r2) or best_r2 < r_squared_threshold)
        and spread is not None
        and spread <= top_spread_factor
        and structure >= structure_threshold
    ):
        return ExpressivenessVerdict(
            "STRUCTURE_OUTSIDE_GRAMMAR",
            best_form,
            best_r2 if np.isfinite(best_r2) else None,
            spread,
            structure,
            "poor best fit with clustered top candidates and structured residuals: grammar likely insufficient",
        )
    return ExpressivenessVerdict(
        "INCONCLUSIVE",
        best_form,
        best_r2 if np.isfinite(best_r2) else None,
        spread,
        structure,
        "poor fit without the full outside-grammar signature; could be noise, omitted variables, or grammar limits",
    )


__all__ = ["ExpressivenessVerdict", "audit_grammar_fit"]
