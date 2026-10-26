"""MOSAIC-2 Structure & Sparsity Audit Engine (Phase P2).

Status: experimental

Before choosing representation approximations (polynomials, delta banking,
prefix sharing, or direct phase synthesis), the structure audit engine
diagnoses whether the underlying data contains exploitable structure:
- Non-zero fraction & Sparsity
- Distinct values & Repeat fraction
- Empirical Shannon entropy
- Walsh-Hadamard transform spectral rank
- Givens tree sign invariant (cos(theta) > 0 <==> bit9 == bit8 in 10-bit words)

A candidate family is only permitted to proceed to synthesis if its
prerequisite structural metrics pass their respective gates.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any
import numpy as np


@dataclasses.dataclass(frozen=True)
class StructureAuditReport:
    """Diagnostic profile of candidate data structure."""
    benchmark: str
    total_elements: int
    non_zero_count: int
    non_zero_fraction: float
    distinct_values: int
    repeat_fraction: float
    distribution_entropy: float
    walsh_rank: int | None = None
    givens_sign_invariant_passed: bool | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def is_sparse(self) -> bool:
        """True if non-zero fraction is below the 40% threshold."""
        return self.non_zero_fraction <= 0.40

    @property
    def is_compressible(self) -> bool:
        """True if repeat fraction >= 35%."""
        return self.repeat_fraction >= 0.35

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compute_distribution_entropy(values: np.ndarray) -> float:
    """Compute empirical Shannon entropy in bits for an integer or categorical array."""
    flat = np.asarray(values).ravel()
    if flat.size == 0:
        return 0.0
    _, counts = np.unique(flat, return_counts=True)
    probs = counts / flat.size
    return float(-np.sum(probs * np.log2(probs)))


def compute_walsh_rank(values: np.ndarray, max_power_of_two: int = 14) -> int:
    """Compute number of non-zero coefficients in Fast Walsh-Hadamard Transform spectrum."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    n = flat.size
    # Truncate or pad to power of two
    p2 = 1 << min(max_power_of_two, int(math.log2(max(1, n))))
    arr = flat[:p2].copy()

    # In-place 1D Fast Walsh-Hadamard Transform
    h = 1
    while h < p2:
        for i in range(0, p2, h * 2):
            for j in range(i, i + h):
                x = arr[j]
                y = arr[j + h]
                arr[j] = x + y
                arr[j + h] = x - y
        h *= 2

    # Count non-zero coefficients outside numerical epsilon
    tol = 1e-6
    return int(np.sum(np.abs(arr) > tol))


def verify_givens_sign_invariant(words_10bit: np.ndarray) -> tuple[bool, float]:
    """Verify that cos(theta) > 0 <==> bit9 == bit8 in 10-bit quantized representation.

    Returns:
        (all_passed: bool, match_fraction: float)
    """
    w10 = np.asarray(words_10bit, dtype=np.int64).ravel() % 1024
    if w10.size == 0:
        return True, 1.0
    b9 = (w10 >> 9) & 1
    b8 = (w10 >> 8) & 1
    matches = (b9 == b8)
    match_frac = float(np.mean(matches))
    all_passed = bool(np.all(matches))
    return all_passed, match_frac


def audit_array_structure(
    values: np.ndarray,
    *,
    benchmark: str = "",
    compute_walsh: bool = False,
    metadata: dict[str, Any] | None = None,
) -> StructureAuditReport:
    """Compute the comprehensive structure audit report for an array."""
    arr = np.asarray(values)
    total = int(arr.size)
    if total == 0:
        return StructureAuditReport(
            benchmark=benchmark,
            total_elements=0,
            non_zero_count=0,
            non_zero_fraction=0.0,
            distinct_values=0,
            repeat_fraction=0.0,
            distribution_entropy=0.0,
            walsh_rank=0 if compute_walsh else None,
            givens_sign_invariant_passed=None,
            metadata=metadata or {},
        )

    flat = arr.ravel()
    non_zero = int(np.count_nonzero(flat))
    non_zero_frac = float(non_zero / total)

    unique_vals = np.unique(flat)
    distinct = int(len(unique_vals))
    repeat_frac = float(1.0 - (distinct / total))
    entropy = compute_distribution_entropy(flat)

    walsh_r = compute_walsh_rank(flat) if compute_walsh else None

    # Check Givens sign invariant if values are within [0, 1024)
    givens_ok = None
    if arr.max() < 1024 and arr.min() >= 0:
        givens_ok, _ = verify_givens_sign_invariant(flat)

    return StructureAuditReport(
        benchmark=benchmark,
        total_elements=total,
        non_zero_count=non_zero,
        non_zero_fraction=non_zero_frac,
        distinct_values=distinct,
        repeat_fraction=repeat_frac,
        distribution_entropy=entropy,
        walsh_rank=walsh_r,
        givens_sign_invariant_passed=givens_ok,
        metadata=metadata or {},
    )
