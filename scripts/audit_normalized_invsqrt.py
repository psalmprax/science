#!/usr/bin/env python3
"""Numerical full-domain feasibility audit; no quantum resource claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev as cheb

FULL_MAX_X = 3_133_452


def fit_segments(degree: int, segments: int) -> list[np.ndarray]:
    if degree < 1 or segments < 1 or segments & (segments - 1):
        raise ValueError("positive degree and power-of-two segment count required")
    nodes = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
    coefficients = []
    for segment in range(segments):
        midpoint = 1 + (segment + 0.5) / segments
        mantissas = midpoint + nodes / (2 * segments)
        coefficients.append(cheb.chebfit(nodes, 1 / np.sqrt(mantissas), degree))
    return coefficients


def evaluate(
    xs: np.ndarray, coefficients: list[np.ndarray], fractional_bits: int | None = None
) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    if np.any(xs < 1) or np.any(xs > FULL_MAX_X) or np.any(xs != np.floor(xs)):
        raise ValueError("inputs must be benchmark-domain integers")
    exponents = np.frexp(xs)[1] - 1
    mantissas = np.ldexp(xs, -exponents)
    segments = len(coefficients)
    indices = np.minimum(((mantissas - 1) * segments).astype(int), segments - 1)
    local = 2 * (mantissas - 1) * segments - (2 * indices + 1)
    values = np.empty_like(xs)
    for index, coefficient in enumerate(coefficients):
        mask = indices == index
        if fractional_bits is None:
            values[mask] = cheb.chebval(local[mask], coefficient)
        else:
            scale = 2**fractional_bits
            rounded = lambda value: np.rint(value * scale) / scale
            c = rounded(coefficient)
            t = rounded(local[mask])
            next_state = np.zeros_like(t)
            after_next = np.zeros_like(t)
            for term in c[:0:-1]:
                state = rounded(rounded(2 * t * next_state) - after_next + term)
                after_next, next_state = next_state, state
            values[mask] = rounded(rounded(t * next_state) - after_next + c[0])
    scaled = values * np.exp2(-exponents / 2)
    if fractional_bits is not None:
        # This only models phase rounding, not a reversible implementation.
        scaled = np.rint(scaled * 2**48) / 2**48
    return scaled


def audit(
    coefficients: list[np.ndarray],
    block_size: int = 131072,
    fractional_bits: int | None = None,
) -> dict:
    worst_ratio = 0.0
    worst_x = 1
    violations = 0
    minimum_margin = float("inf")
    for start in range(1, FULL_MAX_X + 1, block_size):
        xs = np.arange(start, min(start + block_size, FULL_MAX_X + 1), dtype=float)
        target = 1 / np.sqrt(xs)
        errors = evaluate(xs, coefficients, fractional_bits) - target
        errors -= np.rint(errors)
        tolerance = 2**-24 * target
        ratios = np.abs(errors) / tolerance
        worst = int(np.argmax(ratios))
        if ratios[worst] > worst_ratio:
            worst_ratio = float(ratios[worst])
            worst_x = int(xs[worst])
        violations += int(np.count_nonzero(ratios >= 1))
        minimum_margin = min(minimum_margin, float(np.min(tolerance - np.abs(errors))))
    return {
        "tested_inputs": FULL_MAX_X,
        "worst_error_over_tolerance": worst_ratio,
        "worst_x": worst_x,
        "violating_inputs": violations,
        "minimum_margin_turns": minimum_margin,
        "passes_numerical_tolerance": violations == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for segments in (1, 2, 4, 8):
        for degree in (3, 5, 7, 9, 11):
            coefficients = fit_segments(degree, segments)
            result = {"segments": segments, "degree": degree, **audit(coefficients)}
            result["coefficient_count"] = segments * (degree + 1)
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {
        "method": "piecewise Chebyshev interpolation on normalized mantissa [1,2)",
        "results": results,
        "rounded_arithmetic_results": [],
        "limitations": [
            "floating-point audit, not a formal error proof",
            "no fixed-point arithmetic, coefficient quantization, or synthesis error included",
            "no generated circuit; qubit and T costs are unknown",
            "exponent scaling also requires a reversible implementation",
        ],
    }
    for segments, degree in ((1, 9), (2, 7), (4, 5), (8, 5)):
        for bits in (24, 28, 32):
            result = {
                "segments": segments,
                "degree": degree,
                "fractional_bits": bits,
                **audit(fit_segments(degree, segments), fractional_bits=bits),
            }
            report["rounded_arithmetic_results"].append(result)
            print(json.dumps(result), flush=True)
    report["limitations"].append(
        "rounded model uses two Clenshaw states and unbounded signed range; no overflow model"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
