"""Generic domain-behavior forensics for picked candidates (P1).

Every measure evaluates *behavior over declared domains*, never syntactic
patterns: a denominator near zero far outside the data range is inert, while
an exponential overflow on-domain is catastrophic. All functions take plain
numpy callables so any engine (or hand-written candidate) can be audited.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DomainReport:
    """Assembled forensic record for one candidate on one domain."""

    finite_fraction: float
    pole_locations: tuple[float, ...]
    derivative_runs: int
    asymptotic_class: str
    coefficient_sensitivity: float | None
    max_magnitude_ratio: float | None
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "finite_fraction": self.finite_fraction,
            "pole_locations": list(self.pole_locations),
            "derivative_runs": self.derivative_runs,
            "asymptotic_class": self.asymptotic_class,
            "coefficient_sensitivity": self.coefficient_sensitivity,
            "max_magnitude_ratio": self.max_magnitude_ratio,
            "notes": list(self.notes),
        }


def _safe_eval(function: Callable[[np.ndarray], np.ndarray], grid: np.ndarray) -> np.ndarray:
    try:
        with np.errstate(all="ignore"):
            values = np.asarray(function(np.asarray(grid, dtype=float)))
    except (TypeError, ValueError, ArithmeticError):
        return np.full(grid.shape, np.nan)
    if np.iscomplexobj(values):
        return np.full(grid.shape, np.nan)
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return np.full(grid.shape, np.nan)
    if values.shape != grid.shape:
        return np.full(grid.shape, np.nan)
    return values


def pole_scan(function: Callable[[np.ndarray], np.ndarray], grid: np.ndarray) -> dict[str, Any]:
    """Finite fraction plus locations where the candidate blows up."""
    values = _safe_eval(function, grid)
    finite = np.isfinite(values)
    poles = [float(grid[index]) for index in np.where(~finite)[0]]
    return {"finite_fraction": float(np.mean(finite)), "pole_locations": poles, "n_grid": int(grid.size)}


def derivative_runs(function: Callable[[np.ndarray], np.ndarray], grid: np.ndarray) -> int:
    """Sign runs of the first derivative: oscillation proxy on the domain."""
    values = _safe_eval(function, grid)
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return 0
    signs = np.sign(np.diff(finite))
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def asymptotic_behavior(
    function: Callable[[np.ndarray], np.ndarray],
    reference: np.ndarray,
    *,
    extension: float = 10.0,
    n_points: int = 200,
    details: bool = False,
) -> str | dict[str, Any]:
    """Classify far-field behavior: bounded/polynomial/exponential/oscillatory/singular/decaying.

    Extends the observed domain geometrically and reads growth, not syntax:
    a divergent truth and a bounded surrogate classify differently even when
    their finite-window errors agree.

    With ``details=True`` returns the class plus a confidence record: the
    runner-up class and the margin separating them. Behavioral classifiers
    are diagnostic evidence, not structural labels -- over finite intervals,
    polynomial, exponential, and mixed forms can mimic one another, so a
    narrow margin means ambiguity, not a verdict.
    """
    lo, hi = float(np.min(reference)), float(np.max(reference))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo or lo <= 0.0:
        return {"class": "unclassified", "confidence": 0.0, "runner_up": None} if details else "unclassified"
    grid = np.geomspace(hi, hi * extension, n_points)
    values = _safe_eval(function, grid)
    if not np.all(np.isfinite(values)):
        result = {"class": "singular", "confidence": 1.0, "runner_up": None}
        return result if details else "singular"
    # Pole hunt between grid points: a sign change of the reciprocal with
    # large adjacent magnitude means a pole was stepped over (geometric grids
    # almost never land exactly on one). Zeros look similar in the
    # reciprocal, so require large function magnitude on at least one side.
    with np.errstate(all="ignore"):
        reciprocal = 1.0 / values
    finite_rec = np.isfinite(reciprocal)
    med = float(np.median(np.abs(values[np.isfinite(values)]))) + 1e-300
    for i in range(len(grid) - 1):
        if (finite_rec[i] and finite_rec[i + 1] and reciprocal[i] * reciprocal[i + 1] < 0.0
                and max(abs(values[i]), abs(values[i + 1])) > 100.0 * med):
                result = {"class": "singular", "confidence": 0.8, "runner_up": None,
                          "note": "pole stepped over between grid points"}
                return result if details else "singular"
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    log_abs = np.log(np.abs(values) / scale + 1e-300)
    log_x = np.log(grid / hi)
    runs = derivative_runs(function, grid)
    if runs > n_points // 4:
        result = {"class": "oscillatory", "confidence": 0.8,
                  "runner_up": None, "derivative_runs": runs}
        return result if details else "oscillatory"
    # Curvature in log-log space separates power laws (zero curvature)
    # from exponentials (strong positive curvature); plain slopes cannot,
    # since x**3 and e^{kx} can share a local slope.
    quad = np.polyfit(log_x, log_abs, 2)
    slope = float(np.polyfit(log_x, log_abs, 1)[0])
    linear_r2 = _fit_r2(log_x, log_abs, 1)
    quad_r2 = _fit_r2(log_x, log_abs, 2)
    if quad[0] > 0.5:
        result = {"class": "exponential", "confidence": _confidence(quad_r2, linear_r2),
                  "runner_up": "polynomial" if slope > 0.5 else "bounded"}
        return result if details else "exponential"
    if slope > 0.5:
        result = {"class": "polynomial", "confidence": _confidence(linear_r2, quad_r2),
                  "runner_up": "exponential"}
        return result if details else "polynomial"
    if slope < -0.5:
        result = {"class": "decaying", "confidence": _confidence(linear_r2, quad_r2),
                  "runner_up": "bounded"}
        return result if details else "decaying"
    result = {"class": "bounded", "confidence": _confidence(linear_r2, 0.0),
              "runner_up": "decaying" if slope < 0 else "polynomial"}
    return result if details else "bounded"


def _fit_r2(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    """R-squared of a polynomial fit (0 on degenerate input)."""
    if x.size < degree + 2 or float(np.std(y)) == 0.0:
        return 0.0
    try:
        coef = np.polyfit(x, y, degree)
        pred = np.polyval(coef, x)
        total = float(np.sum((y - np.mean(y)) ** 2))
        if total <= 0.0:
            return 0.0
        return max(0.0, float(1.0 - np.sum((y - pred) ** 2) / total))
    except (TypeError, ValueError, ArithmeticError):
        return 0.0


def _confidence(winner_r2: float, runner_r2: float) -> float:
    """Confidence from the R-squared margin, capped: close margins mean
    ambiguity, not a verdict."""
    margin = max(0.0, min(1.0, winner_r2 - runner_r2))
    return round(float(0.5 + 0.5 * margin), 3)


def coefficient_sensitivity(
    function_factory: Callable[[dict[str, float]], Callable[[np.ndarray], np.ndarray]],
    parameters: dict[str, float],
    grid: np.ndarray,
    *,
    epsilon: float = 0.01,
) -> float | None:
    """Relative prediction shift under a uniform fractional perturbation."""
    try:
        base = _safe_eval(function_factory(parameters), grid)
        perturbed = _safe_eval(
            function_factory({name: value * (1.0 + epsilon) for name, value in parameters.items()}), grid
        )
    except (TypeError, ValueError, ArithmeticError):
        return None
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(perturbed)):
        return None
    denominator = float(np.mean(np.abs(base))) + 1e-300
    return float(np.mean(np.abs(perturbed - base)) / denominator / epsilon)


def domain_report(
    function: Callable[[np.ndarray], np.ndarray],
    grid: np.ndarray,
    *,
    parameters: dict[str, float] | None = None,
    function_factory: Callable[[dict[str, float]], Callable[[np.ndarray], np.ndarray]] | None = None,
) -> DomainReport:
    """Assemble the forensic record; notes flag the salient pathologies."""
    scan = pole_scan(function, grid)
    runs = derivative_runs(function, grid)
    notes: list[str] = []
    if scan["finite_fraction"] < 1.0:
        notes.append(f"non-finite on {len(scan['pole_locations'])} grid points")
    if runs > max(10, grid.size // 10):
        notes.append("high oscillation on the declared domain")
    sensitivity = None
    if parameters is not None and function_factory is not None:
        sensitivity = coefficient_sensitivity(function_factory, parameters, grid)
        if sensitivity is not None and sensitivity > 10.0:
            notes.append("high coefficient sensitivity")
    values = _safe_eval(function, grid)
    finite = values[np.isfinite(values)]
    magnitude_ratio = (float(np.max(np.abs(finite)) / (np.mean(np.abs(finite)) + 1e-300)) if finite.size else None)
    return DomainReport(
        scan["finite_fraction"],
        tuple(scan["pole_locations"]),
        runs,
        asymptotic_behavior(function, grid),  # type: ignore[arg-type]  # details=False returns str
        sensitivity,
        magnitude_ratio,
        tuple(notes),
    )


__all__ = [
    "DomainReport",
    "asymptotic_behavior",
    "coefficient_sensitivity",
    "derivative_runs",
    "domain_report",
    "pole_scan",
]
