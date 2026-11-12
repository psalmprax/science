#!/usr/bin/env python3
"""Exact rounded integer Horner model; not a reversible circuit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev as cheb

from audit_normalized_invsqrt import FULL_MAX_X, fit_segments


def round_shift(values: np.ndarray, bits: int) -> np.ndarray:
    """Signed integer division by 2**bits, nearest with ties to even."""
    if not 0 <= bits <= 60:
        raise ValueError("shift must lie in [0, 60]")
    divisor = 1 << bits
    quotient, remainder = np.divmod(values, divisor)
    increment = (2 * remainder > divisor) | (
        (2 * remainder == divisor) & ((quotient & 1) != 0)
    )
    return quotient + increment.astype(np.int64)


def coefficient_table(degree: int, segments: int, bits: int) -> np.ndarray:
    if not 24 <= bits <= 29 or not 1 <= degree <= 9:
        raise ValueError("require 24–29 fractional bits and degree 1–9")
    coefficients = np.array([cheb.cheb2poly(c) for c in fit_segments(degree, segments)])
    # Separate parity rows fold sqrt(2) scaling into the coefficients.
    table = np.rint(
        coefficients[None, :, :] * np.array([1.0, 2**-0.5])[:, None, None] * 2**bits
    ).astype(np.int64)
    # For |t| <= 1, this conservative bound covers every intermediate.
    bound = int(np.max(np.sum(np.abs(table), axis=2))) + degree
    if bound * 2**bits >= 2**63:
        raise ValueError("int64 product bound exceeded")
    return table


def evaluate_integer(
    inputs: np.ndarray, table: np.ndarray, bits: int
) -> tuple[np.ndarray, int]:
    xs = np.asarray(inputs)
    if (
        xs.ndim != 1 or not np.issubdtype(xs.dtype, np.integer)
        or np.any(xs < 1) or np.any(xs > FULL_MAX_X)
    ):
        raise ValueError("inputs must be a vector of benchmark-domain integers")
    if not 24 <= bits <= 29:
        raise ValueError("fractional bits must lie in [24, 29]")
    exponents = np.frexp(xs)[1].astype(np.int64) - 1
    segments = table.shape[1]
    # Benchmark integers are <=22 bits, so these shifts preserve mantissas exactly.
    scaled = np.left_shift(xs.astype(np.int64), bits - exponents)
    indices = np.minimum(((scaled - 2**bits) * segments) >> bits, segments - 1)
    local = 2 * (scaled - 2**bits) * segments - (2 * indices + 1) * 2**bits
    coefficients = table[exponents & 1, indices]
    state = coefficients[:, -1].copy()
    maximum = int(np.max(np.abs(state)))
    for column in range(coefficients.shape[1] - 2, -1, -1):
        state = round_shift(local * state, bits) + coefficients[:, column]
        maximum = max(maximum, int(np.max(np.abs(state))))
    # 48-bit phase word: exponent parity is already in the coefficient table.
    phase = np.left_shift(state, 48 - bits - exponents // 2)
    return phase, maximum


def audit(degree: int, segments: int, bits: int) -> dict:
    table = coefficient_table(degree, segments, bits)
    worst_ratio = 0.0
    worst_x = 1
    violations = 0
    maximum_state = 0
    for start in range(1, FULL_MAX_X + 1, 131072):
        xs = np.arange(start, min(start + 131072, FULL_MAX_X + 1), dtype=np.int64)
        phase, maximum = evaluate_integer(xs, table, bits)
        maximum_state = max(maximum_state, maximum)
        target = 1 / np.sqrt(xs)
        error = phase / 2**48 - target
        error -= np.rint(error)
        ratios = np.abs(error) / (2**-24 * target)
        worst = int(np.argmax(ratios))
        if ratios[worst] > worst_ratio:
            worst_ratio, worst_x = float(ratios[worst]), int(xs[worst])
        violations += int(np.count_nonzero(ratios >= 1))
    width = maximum_state.bit_length() + 1
    return {
        "degree": degree,
        "segments": segments,
        "fractional_bits": bits,
        "tested_inputs": FULL_MAX_X,
        "worst_error_over_tolerance": worst_ratio,
        "worst_x": worst_x,
        "violating_inputs": violations,
        "passes_integer_model_tolerance": violations == 0,
        "observed_signed_accumulator_bits": width,
        "input_plus_observed_accumulator_bits": 22 + width,
        "includes_normalization_multiply_or_cleanup_workspace": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for segments, degree in ((1, 9), (2, 7), (4, 5), (8, 5)):
        for bits in (25, 26, 27, 28):
            result = audit(degree, segments, bits)
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {
        "results": results,
        "limitations": [
            "coefficient fitting and target comparison use float64",
            "integer Horner recurrence is exact with ties-to-even rounding",
            "rounded multiplication is not an injective accumulator update",
            "single classical accumulator cannot be overwritten reversibly without retaining information",
            "observed widths are not proven quantum resource bounds",
            "no rotation synthesis error or generated quantum circuit",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
