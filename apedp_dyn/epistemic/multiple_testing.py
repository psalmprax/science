"""Validated multiple-testing corrections for externally computed p-values.

This module adjusts a declared family of p-values. It does not estimate
p-values, validate study design, or establish that hypotheses were specified
independently of the observed data.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MultipleTestingResult:
    """Adjusted p-values and decisions in the caller's original input order."""

    method: str
    alpha: float
    p_values: tuple[float, ...]
    adjusted_p_values: tuple[float, ...]
    rejected: tuple[bool, ...]
    effective_threshold: float | None


def adjust_p_values(
    p_values: Sequence[float],
    alpha: float,
    method: str = "bh",
) -> MultipleTestingResult:
    """Adjust a nonempty p-value family with BH, BY, or Holm.

    ``alpha`` must be supplied by the caller so this utility cannot silently
    introduce an unregistered significance threshold. BH controls FDR under
    independence or suitable positive dependence; BY is the more conservative
    FDR procedure for arbitrary dependence. Holm controls family-wise error.
    """
    if method not in {"bh", "by", "holm"}:
        raise ValueError(f"unknown multiple-testing method: {method!r}")
    if isinstance(alpha, bool):
        raise ValueError("alpha must be numeric and strictly between 0 and 1")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be numeric and strictly between 0 and 1") from exc
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    if len(p_values) == 0:
        raise ValueError("p_values must contain at least one hypothesis")

    if any(isinstance(value, bool) for value in p_values):
        raise ValueError("p-values must be numeric, not booleans")
    try:
        values = tuple(float(value) for value in p_values)
    except (TypeError, ValueError) as exc:
        raise ValueError("p-values must be numeric") from exc
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must be finite and in [0, 1]")

    ordered = sorted(range(len(values)), key=values.__getitem__)
    sorted_values = [values[index] for index in ordered]
    family_size = len(values)
    adjusted_sorted = [0.0] * family_size

    if method in {"bh", "by"}:
        dependence_factor = (
            sum(1.0 / rank for rank in range(1, family_size + 1))
            if method == "by"
            else 1.0
        )
        running_minimum = 1.0
        for position in range(family_size - 1, -1, -1):
            rank = position + 1
            candidate = (
                sorted_values[position]
                * family_size
                * dependence_factor
                / rank
            )
            running_minimum = min(running_minimum, candidate, 1.0)
            adjusted_sorted[position] = running_minimum
    else:
        running_maximum = 0.0
        for position, value in enumerate(sorted_values):
            candidate = (family_size - position) * value
            running_maximum = max(running_maximum, candidate)
            adjusted_sorted[position] = min(running_maximum, 1.0)

    adjusted = [0.0] * family_size
    rejected = [False] * family_size
    for position, original_index in enumerate(ordered):
        adjusted[original_index] = adjusted_sorted[position]
        rejected[original_index] = adjusted_sorted[position] <= alpha

    rejected_values = [
        value for value, is_rejected in zip(values, rejected, strict=True)
        if is_rejected
    ]
    return MultipleTestingResult(
        method=method,
        alpha=float(alpha),
        p_values=values,
        adjusted_p_values=tuple(adjusted),
        rejected=tuple(rejected),
        effective_threshold=max(rejected_values) if rejected_values else None,
    )
