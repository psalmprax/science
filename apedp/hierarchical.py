# Databricks notebook source
"""Hierarchical partial pooling for cross-group transfer.

Instead of a binary transfer verdict (shared family, refit coefficients),
partial pooling shrinks per-group coefficient estimates toward the pooled
mean with empirical-Bayes weights. A new, unseen group then receives a
*prior predictive* distribution -- transfer quantified as an interval with
honest width, rather than refused outright or asserted point-wise.

Scope (documented): linear-in-parameters families (polynomials,
Shomate-class, log-linear forms). Shrinkage is per coefficient via
method-of-moments between-group variance; no MCMC, no priors beyond the
pooled mean. When groups disagree violently, ``tau2`` grows and the prior
predictive widens instead of pretending consensus.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HierarchicalFit:
    """Pooled fit plus per-group shrunk coefficients and prior predictive."""

    group_ids: tuple[Any, ...]
    group_params: dict[Any, list[float]]
    pooled_params: list[float]
    between_variance: list[float]
    shrinkage_weights: dict[Any, list[float]]
    n_per_group: dict[Any, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_ids": [str(group) for group in self.group_ids],
            "group_params": {str(group): list(values) for group, values in self.group_params.items()},
            "pooled_params": list(self.pooled_params),
            "between_variance": list(self.between_variance),
            "shrinkage_weights": {str(group): list(values) for group, values in self.shrinkage_weights.items()},
            "n_per_group": {str(group): count for group, count in self.n_per_group.items()},
        }


def _ols_with_se(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef, residuals, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    dof = max(int(design.shape[0] - rank), 1)
    rss = float(np.sum((target - design @ coef) ** 2))
    sigma2 = rss / dof
    try:
        cov = sigma2 * np.linalg.inv(design.T @ design)
        standard_errors = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        standard_errors = np.full(coef.shape, np.inf)
    return np.asarray(coef, dtype=float), np.asarray(standard_errors, dtype=float)


def partial_pool_linear(
    designs: list[np.ndarray],
    targets: list[np.ndarray],
    group_ids: list[Any] | None = None,
) -> HierarchicalFit:
    """Shrink per-group OLS coefficients toward the pooled mean.

    Weight per coefficient and group: ``w = tau2 / (tau2 + se_k^2)`` with
    ``tau2`` the method-of-moments between-group variance. Small, noisy
    groups shrink hardest; large precise groups stand on their own.
    """
    if len(designs) != len(targets) or len(designs) < 2:
        raise ValueError("partial pooling needs at least two groups with aligned designs/targets")
    ids = list(group_ids) if group_ids is not None else list(range(len(designs)))
    if len(ids) != len(designs):
        raise ValueError("group_ids must align with designs")
    params: dict[Any, np.ndarray] = {}
    errors: dict[Any, np.ndarray] = {}
    counts: dict[Any, int] = {}
    n_params: int | None = None
    for group, design, target in zip(ids, designs, targets):
        design_arr = np.asarray(design, dtype=float)
        target_arr = np.asarray(target, dtype=float).ravel()
        if design_arr.ndim != 2 or design_arr.shape[0] != target_arr.shape[0] or design_arr.shape[0] == 0:
            raise ValueError(f"group {group}: design/target misaligned or empty")
        if n_params is None:
            n_params = design_arr.shape[1]
        elif design_arr.shape[1] != n_params:
            raise ValueError("all groups must share the same parameter basis")
        coef, standard_errors = _ols_with_se(design_arr, target_arr)
        if not np.all(np.isfinite(coef)):
            raise ValueError(f"group {group}: non-finite OLS estimate")
        params[group] = coef
        errors[group] = standard_errors
        counts[group] = int(design_arr.shape[0])
    stacked = np.vstack([params[group] for group in ids])
    pooled = stacked.mean(axis=0)
    mean_within = np.vstack([errors[group] ** 2 for group in ids]).mean(axis=0)
    between = np.maximum(stacked.var(axis=0, ddof=1) - mean_within, 0.0)
    shrunk: dict[Any, list[float]] = {}
    weights: dict[Any, list[float]] = {}
    for group in ids:
        denominator = between + errors[group] ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(denominator > 0.0, between / np.maximum(denominator, np.finfo(float).tiny), 1.0)
        weight = np.clip(weight, 0.0, 1.0)
        shrunk[group] = (weight * params[group] + (1.0 - weight) * pooled).tolist()
        weights[group] = weight.tolist()
    return HierarchicalFit(
        tuple(ids),
        {group: params[group].tolist() for group in ids},
        pooled.tolist(),
        between.tolist(),
        weights,
        counts,
    )


def prior_predictive_interval(
    fit: HierarchicalFit,
    x_new: np.ndarray,
    *,
    z: float = 1.645,
) -> tuple[np.ndarray, np.ndarray]:
    """Prior predictive band for an unseen group: pooled mean +/- spread.

    Spread combines between-group variance propagated through ``x_new`` with
    the average within-group residual scale implied by the shrinkage target.
    This is a transfer-uncertainty band, not a calibrated conformal interval.
    """
    features = np.asarray(x_new, dtype=float)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    pooled = np.asarray(fit.pooled_params, dtype=float)
    between = np.asarray(fit.between_variance, dtype=float)
    if features.shape[1] != pooled.shape[0]:
        raise ValueError("x_new must match the fitted parameter basis")
    center = features @ pooled
    spread = z * np.sqrt(np.maximum(features**2 @ between, 0.0))
    return center - spread, center + spread


__all__ = ["HierarchicalFit", "partial_pool_linear", "prior_predictive_interval"]
