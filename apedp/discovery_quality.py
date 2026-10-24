# Databricks notebook source
"""Quality gates around symbolic discovery results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from apedp.discovery import MultiVariableSRE


@dataclass(frozen=True)
class CandidateQuality:
    expression: str
    train_mse: float
    weighted_train_mse: float
    expanded_output_rms: float
    finite: bool
    domain_valid: bool
    output_bounds: tuple[float, float]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {"expression": self.expression, "train_mse": self.train_mse, "weighted_train_mse": self.weighted_train_mse, "expanded_output_rms": self.expanded_output_rms, "finite": self.finite, "domain_valid": self.domain_valid, "output_bounds": list(self.output_bounds), "seed": self.seed}


def screen_candidate(expression: sp.Expr, x_train: dict[str, np.ndarray], y_train: np.ndarray, *, expansion_factor: float = 1.25, output_bounds: tuple[float, float] | None = None, observation_uncertainties: np.ndarray | None = None) -> CandidateQuality:
    """Measure fit, optional uncertainty-weighted fit, and extrapolation safety."""
    if expansion_factor < 1.0:
        raise ValueError("expansion_factor must be at least 1")
    names = sorted(x_train)
    function = sp.lambdify([sp.Symbol(name) for name in names], expression, modules=["numpy"])
    train_args = [np.asarray(x_train[name], dtype=float) for name in names]
    y = np.asarray(y_train, dtype=float).ravel()
    train_pred = np.broadcast_to(np.asarray(function(*train_args), dtype=float), y.shape)
    finite = bool(np.all(np.isfinite(train_pred)))
    residual = train_pred - y
    if observation_uncertainties is None:
        weighted_train_mse = float(np.mean(residual ** 2)) if finite else float("inf")
    else:
        sigma = np.asarray(observation_uncertainties, dtype=float).ravel()
        if sigma.shape != y.shape or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("observation_uncertainties must be finite, positive, and aligned with y_train")
        weighted_train_mse = float(np.average(residual ** 2, weights=1.0 / sigma ** 2)) if finite else float("inf")
    expanded_args = []
    for name in names:
        values = np.asarray(x_train[name], dtype=float)
        center = float(np.mean(values))
        expanded_args.append(center + (values - center) * expansion_factor)
    expanded_pred = np.asarray(function(*expanded_args), dtype=float)
    finite = bool(finite and np.all(np.isfinite(expanded_pred)))
    minimum = float(np.nanmin(expanded_pred)) if expanded_pred.size else float("nan")
    maximum = float(np.nanmax(expanded_pred)) if expanded_pred.size else float("nan")
    domain_valid = finite if output_bounds is None else bool(finite and minimum >= output_bounds[0] and maximum <= output_bounds[1])
    return CandidateQuality(str(expression), float(np.mean(residual ** 2)) if finite else float("inf"), weighted_train_mse, float(np.sqrt(np.mean(expanded_pred ** 2))) if finite else float("inf"), finite, domain_valid, (minimum, maximum), -1)


def multi_seed_discovery(engine_factory, x_data: dict[str, np.ndarray], y_data: np.ndarray, *, seeds: tuple[int, ...] = (1, 2, 3, 4, 5), generations: int = 30, pop_size: int = 100, output_bounds: tuple[float, float] | None = None, observation_uncertainties: np.ndarray | None = None) -> dict[str, Any]:
    """Run independent seeds and report stability instead of one lucky winner."""
    results = []
    for seed in seeds:
        engine: MultiVariableSRE = engine_factory(seed)
        expression, _, _ = engine.run_discovery(x_data, y_data, generations=generations, pop_size=pop_size, seed=seed)
        if expression is not None:
            quality = screen_candidate(expression, x_data, y_data, output_bounds=output_bounds, observation_uncertainties=observation_uncertainties)
            results.append(CandidateQuality(quality.expression, quality.train_mse, quality.weighted_train_mse, quality.expanded_output_rms, quality.finite, quality.domain_valid, quality.output_bounds, seed))
    valid = [item for item in results if item.finite and item.domain_valid]
    return {"results": [item.to_dict() for item in results], "seed_count": len(seeds), "successful_seed_count": len(results), "valid_seed_count": len(valid), "stability_rate": len(valid) / len(seeds) if seeds else 0.0, "best_train_mse": min((item.train_mse for item in valid), default=float("inf")), "best_weighted_train_mse": min((item.weighted_train_mse for item in valid), default=float("inf")), "best_expanded_output_rms": min((item.expanded_output_rms for item in valid), default=float("inf"))}


__all__ = ["CandidateQuality", "multi_seed_discovery", "screen_candidate"]