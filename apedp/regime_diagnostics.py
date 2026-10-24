# Databricks notebook source
"""Diagnostics for regime coverage and extrapolation claims."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RegimeCoverage:
    status: str
    reason: str
    observed_turns: int
    extrapolation_requested: bool

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "observed_turns": self.observed_turns, "extrapolation_requested": self.extrapolation_requested}


def assess_regime_coverage(x: np.ndarray, y: np.ndarray, *, extrapolation_factor: float = 1.0) -> RegimeCoverage:
    """Detect whether observed data contains a regime turn before extrapolation."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size < 4:
        return RegimeCoverage("INSUFFICIENT", "at least four aligned observations are required", 0, extrapolation_factor > 1.0)
    order = np.argsort(x)
    slopes = np.diff(y[order]) / np.diff(x[order])
    signs = np.sign(slopes[np.abs(slopes) > 1e-12])
    turns = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0
    if extrapolation_factor > 1.0 and turns == 0:
        return RegimeCoverage("UNDERDETERMINED", "no observed slope reversal supports a new regime outside the measured domain", turns, True)
    return RegimeCoverage("COVERED", "observed slope behavior supports the requested regime check", turns, extrapolation_factor > 1.0)


__all__ = ["RegimeCoverage", "assess_regime_coverage"]