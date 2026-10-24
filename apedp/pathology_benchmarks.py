# Databricks notebook source
"""Pathology-benchmark data generators (review §23).

Synthetic truth functions spanning polynomial, reciprocal, exponential,
smooth-transcendental, Shomate-like, and -- crucially -- genuinely
oscillatory dynamics. Benchmark F exists as a safeguard: any smoothness or
anti-oscillation bias must prove it does not reject oscillation when
oscillation is the truth. Pure data generation; no engine code is touched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PathologyCase:
    """One benchmark truth function with its scientific metadata."""

    case_id: str
    family: str
    oscillatory_truth: bool
    smooth_truth: bool
    x_range: tuple[float, float]
    n_points: int
    noise_std: float
    seed: int

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "oscillatory_truth": self.oscillatory_truth,
            "smooth_truth": self.smooth_truth,
            "x_range": list(self.x_range),
            "n_points": self.n_points,
            "noise_std": self.noise_std,
            "seed": self.seed,
        }


def _grid(case: PathologyCase) -> np.ndarray:
    return np.linspace(case.x_range[0], case.x_range[1], case.n_points)


def _noise(case: PathologyCase, size: int) -> np.ndarray:
    return np.random.default_rng(case.seed).normal(0.0, case.noise_std, size)


def generate(case: PathologyCase) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y_noisy)`` for the case's truth function."""
    x = _grid(case)
    if case.case_id == "A-poly":
        truth = 3.0 * x**3 + 2.0 * x**2 + x + 7.0
    elif case.case_id == "B-reciprocal":
        truth = 3.0 * x + 4.0 / x**2 + 2.0
    elif case.case_id == "C-exponential":
        truth = 5.0 * np.exp(0.2 * x) + 2.0
    elif case.case_id == "D-smooth-transcendental":
        truth = 3.0 * np.sin(x) + 2.0 * x
    elif case.case_id == "E-shomate-like":
        truth = 30.0 + 8.0 * x + 2.0 * x**2 + 0.5 * x**3 + 0.2 / x**2
    elif case.case_id == "F-oscillatory-truth":
        truth = 2.0 * np.sin(3.0 * x) + 1.2 * np.sin(7.0 * x) + 0.1 * x
    else:
        raise KeyError(f"unknown pathology case {case.case_id!r}")
    return x, truth + _noise(case, x.shape[0])


CASES: tuple[PathologyCase, ...] = (
    PathologyCase("A-poly", "polynomial", False, True, (0.5, 3.0), 40, 0.05, 101),
    PathologyCase("B-reciprocal", "reciprocal", False, True, (0.5, 3.0), 40, 0.05, 102),
    PathologyCase("C-exponential", "exponential", False, True, (0.5, 3.0), 40, 0.05, 103),
    PathologyCase("D-smooth-transcendental", "smooth-transcendental", False, True, (0.5, 3.0), 40, 0.05, 104),
    PathologyCase("E-shomate-like", "shomate-like", False, True, (0.5, 3.0), 40, 0.05, 105),
    # Safeguard case: the truth itself oscillates, so a correct pathology
    # layer must NOT flag oscillation here. Noise kept low so the signal is
    # unambiguous; x-range spans multiple periods of both components.
    PathologyCase("F-oscillatory-truth", "oscillatory", True, False, (0.0, 6.0), 200, 0.02, 106),
)


__all__ = ["CASES", "PathologyCase", "generate"]
