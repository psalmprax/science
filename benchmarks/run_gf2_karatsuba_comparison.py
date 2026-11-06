"""Compare exact GF(2^12) schoolbook and recursive Karatsuba factors.

The reported XOR count is an unshared linear-circuit estimate, not a reversible
gate or compiled quantum-circuit cost. Runtime measures this Python evaluator.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gf2mul_llm_adjustable import (
    TensorDecomposition,
    build_karatsuba_polynomial_decomposition,
    build_multiplication_tensor,
    build_schoolbook_decomposition,
)


def _row_masks(factors: np.ndarray) -> list[int]:
    return [
        sum(int(bit) << index for index, bit in enumerate(row))
        for row in factors
    ]


def _evaluate(masks: tuple[list[int], list[int], list[int]], a: int, b: int) -> int:
    left, right, outputs = masks
    result = 0
    for u_mask, v_mask, w_mask in zip(left, right, outputs, strict=True):
        if ((a & u_mask).bit_count() & (b & v_mask).bit_count() & 1):
            result ^= w_mask
    return result


def _xor_cost(decomposition: TensorDecomposition) -> dict[str, int]:
    left_masks = _row_masks(decomposition.U)
    right_masks = _row_masks(decomposition.V)
    output_masks = _row_masks(decomposition.W)
    degree = decomposition.U.shape[1]
    input_xors = sum(
        max(0, mask.bit_count() - 1)
        for mask in (*left_masks, *right_masks)
    )
    output_xors = sum(
        max(0, sum((mask >> bit) & 1 for mask in output_masks) - 1)
        for bit in range(degree)
    )
    return {
        "input_linear_xors_unshared": input_xors,
        "output_linear_xors_unshared": output_xors,
        "total_linear_xors_unshared": input_xors + output_xors,
    }


def _reference_multiply(a: int, b: int, degree: int, modulus: int) -> int:
    product = 0
    while b:
        if b & 1:
            product ^= a
        b >>= 1
        a <<= 1
        if a & (1 << degree):
            a ^= modulus
    return product


def _benchmark(
    masks: tuple[list[int], list[int], list[int]],
    pairs: list[tuple[int, int]],
    repeats: int,
) -> dict[str, float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for a, b in pairs:
            _evaluate(masks, a, b)
        samples.append((time.perf_counter() - started) / len(pairs) * 1_000_000)
    return {
        "median_microseconds_per_multiply": round(statistics.median(samples), 3),
        "min_microseconds_per_multiply": round(min(samples), 3),
        "max_microseconds_per_multiply": round(max(samples), 3),
        "samples": repeats,
        "multiplications_per_sample": len(pairs),
    }


def run(samples: int, repeats: int, seed: int) -> dict:
    degree, modulus = 12, 0x1053
    tensor = build_multiplication_tensor(degree, modulus)
    decompositions = {
        "schoolbook": build_schoolbook_decomposition(degree, modulus),
        "recursive_karatsuba": build_karatsuba_polynomial_decomposition(degree, modulus),
    }
    rng = np.random.default_rng(seed)
    pairs = [
        (int(a), int(b))
        for a, b in rng.integers(0, 1 << degree, size=(samples, 2))
    ]
    results = {}
    for name, decomposition in decompositions.items():
        exact, mismatches = decomposition.reconstruct(tensor)
        if not exact:
            raise AssertionError(f"{name} reconstruction failed ({len(mismatches)} mismatches)")
        masks = (
            _row_masks(decomposition.U),
            _row_masks(decomposition.V),
            _row_masks(decomposition.W),
        )
        for a, b in pairs:
            actual = _evaluate(masks, a, b)
            expected = _reference_multiply(a, b, degree, modulus)
            if actual != expected:
                raise AssertionError(f"{name} returned {actual:#x}, expected {expected:#x}")
        results[name] = {
            "bilinear_rank_and_and_count": decomposition.rank,
            "exact_tensor_reconstruction": True,
            **_xor_cost(decomposition),
            "python_evaluator_runtime": _benchmark(masks, pairs, repeats),
        }
    return {
        "degree": degree,
        "modulus_int": modulus,
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "runtime_scope": "packed-bit Python evaluator; not compiled-circuit or hardware runtime",
        "xor_count_scope": "unshared input/output linear-circuit estimate",
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error("--samples and --repeats must be positive")
    report = run(args.samples, args.repeats, args.seed)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
