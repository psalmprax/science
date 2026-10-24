"""Benchmark-I proposer and proposer-side screen for inverse square roots.

Status: experimental

The proposer varies the reference Newton-Raphson pipeline's fixed-point
widths.  It does *not* decide acceptance: :func:`screen_precision` evaluates the
exact integer model and the circuit's static resource counts, and the official
verifier in :mod:`apedp_dyn.quantum_compilation.referee` remains the only
authority for a verification receipt.

The screen deliberately evaluates the full model error before trusting a
resource estimate.  A cheaper circuit that misses the phase tolerance is not a
candidate; it is a rejected screen with its reason retained.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np

from apedp_dyn.quantum_compilation.candidates import sha256_file
from apedp_dyn.quantum_compilation.contracts import VerifierContract

__all__ = [
    "PRECISION_FIELDS",
    "ScreenResult",
    "base_precision",
    "ensure_runtime",
    "full_domain_xs",
    "neighbor_precisions",
    "refinement_precisions",
    "screen_precision",
    "subset_xs",
    "write_circuit",
]

PRECISION_FIELDS = (
    "delta_bits",
    "coef_bits",
    "square_bits",
    "product_bits",
    "result_bits",
)

_FIELD_BOUNDS = {
    "delta_bits": (6, 20),
    "coef_bits": (10, 22),
    "square_bits": (18, 34),
    "product_bits": (18, 34),
    "result_bits": (22, 36),
}


@dataclasses.dataclass(frozen=True)
class ScreenResult:
    """Outcome of the proposer-side exact-model screen."""

    passed: bool
    reason: str | None
    score_estimate: float | None
    metrics: dict[str, Any]


def ensure_runtime(contract: VerifierContract) -> tuple[Any, Any]:
    """Import the shipped verifier/reference modules and ``kickmix``.

    Raises a clear error when called from an interpreter that cannot load the
    extension (the bundled wheel targets CPython 3.13).
    """
    for path in (
        str(contract.kickmix_root),
        str(contract.verifier_root.parent),
        str(contract.verifier_root),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        import kickmix  # noqa: F401 - import is the capability check
    except ImportError as exc:  # pragma: no cover - depends on interpreter
        raise RuntimeError(
            "kickmix could not be imported. Run the proposer with the Python "
            "3.13 interpreter that matches the shipped kickmix extension and "
            "put the competition's kickmix folder on sys.path."
        ) from exc

    import invsqrt.nr_reference_circuit as nr_reference_circuit
    import invsqrt.verify_submission_invsqrt as verify_submission_invsqrt

    expected = contract.verifier_root.resolve()
    actual = Path(verify_submission_invsqrt.__file__).resolve().parent
    if actual != expected:
        raise RuntimeError(
            f"imported invsqrt from {actual}, expected {expected}; refuse to "
            "verify against a different copy of the benchmark."
        )
    return nr_reference_circuit, verify_submission_invsqrt


def base_precision(contract: VerifierContract) -> dict[str, int]:
    """The reference generator's default precision point."""
    reference, _ = ensure_runtime(contract)
    return dict(dataclasses.asdict(reference.Precision()))


def full_domain_xs(contract: VerifierContract) -> np.ndarray:
    """Every ``x`` in the benchmark domain, in ascending order."""
    _, verifier = ensure_runtime(contract)
    n_p = int(contract.constants["N_P"])
    spec = verifier.ProblemSpecInvSqrt.for_n_p(n_p)
    return np.arange(1, spec.max_x + 1, dtype=np.int64)


def subset_xs(max_x: int, *, prefix: int = 8192, stride: int = 37) -> np.ndarray:
    """A deterministic development subset with dense low-x coverage.

    Small ``x`` are where the phase tolerance is largest, while large ``x``
    are where the normalized error tends to be tight.  The prefix plus a fixed
    stride covers both without pretending to be a proof over the full domain.
    """
    prefix_points: np.ndarray = np.arange(1, min(max_x, prefix) + 1, dtype=np.int64)
    stride_points: np.ndarray = np.arange(
        min(max_x, prefix) + 1, max_x + 1, stride, dtype=np.int64
    )
    boundaries: np.ndarray = np.asarray(
        [1, 2, 3, 4, 5, max_x - 1, max_x]
        + [2**k for k in range(1, 32) if 1 <= 2**k <= max_x],
        dtype=np.int64,
    )
    points = np.concatenate([prefix_points, stride_points, boundaries])
    points = points[(points >= 1) & (points <= max_x)]
    return cast(np.ndarray, np.unique(points))


def _phase_margin_units(
    reference: Any,
    verifier: Any,
    n_p: int,
    precision: Any,
    xs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(margin_units, phase_units)`` in the verifier's 2^-64-turn units."""
    y1 = reference.fixed_point_model(n_p, precision, xs)
    output_bits = precision.output_bits(n_p)
    if output_bits > 64:
        raise ValueError("output_bits exceeds the 64-bit phase accumulator")
    phase = y1.astype(np.uint64) << np.uint64(64 - output_bits)
    spec = verifier.ProblemSpecInvSqrt.for_n_p(n_p)
    target, tolerance = verifier.reference_values(spec, xs)
    phase_delta = (phase - target).astype(np.int64)
    if (phase_delta == np.iinfo(np.int64).min).any():
        raise ValueError("phase difference saturates the signed accumulator")
    margin = tolerance.astype(np.int64) - np.abs(phase_delta)
    return margin, phase


def screen_precision(
    contract: VerifierContract,
    parameters: dict[str, int],
    *,
    xs: np.ndarray | None = None,
    safety_ratio: float = 1.0,
) -> ScreenResult:
    """Screen one precision point with the exact integer model.

    ``safety_ratio`` ``< 1`` reserves part of the tolerance for rotation
    synthesis.  The default is the official model's own boundary; callers can
    pass ``0.98`` while searching and then re-check the winner at ``1.0``.
    """
    reference, verifier = ensure_runtime(contract)
    n_p = int(contract.constants["N_P"])
    if xs is None:
        xs = full_domain_xs(contract)
    xs = np.asarray(xs, dtype=np.int64)
    domain_kind = "full-domain" if xs.size >= int(3 * (2**n_p - 2) ** 2) else "subset"

    try:
        precision = reference.Precision(**parameters)
    except TypeError as exc:
        return ScreenResult(False, f"invalid precision fields: {exc}", None, {})

    spec = verifier.ProblemSpecInvSqrt.for_n_p(n_p)
    try:
        relative_error = reference.relative_error(n_p, precision, xs)
        max_relative_error = float(np.max(relative_error))
        tolerance = 2.0 ** -int(spec.precision_bits)
        ratio = max_relative_error / tolerance
        if not math.isfinite(max_relative_error) or not math.isfinite(ratio):
            return ScreenResult(False, "non-finite model error", None, {})
        if ratio >= safety_ratio:
            return ScreenResult(
                False,
                f"model error ratio {ratio:.6f} >= {safety_ratio:.6f}",
                None,
                {
                    "max_relative_error": max_relative_error,
                    "error_tolerance_ratio": ratio,
                    "domain_size": int(xs.size),
                    "domain_kind": domain_kind,
                },
            )
        margin, _ = _phase_margin_units(reference, verifier, n_p, precision, xs)
        min_margin = int(np.min(margin))
        if min_margin <= 0:
            return ScreenResult(
                False,
                "no phase tolerance left for rotation synthesis",
                None,
                {
                    "max_relative_error": max_relative_error,
                    "error_tolerance_ratio": ratio,
                    "min_phase_margin_units": min_margin,
                    "domain_size": int(xs.size),
                    "domain_kind": domain_kind,
                },
            )
        circuit = reference.build_nr_circuit(n_p, prec=precision)
    except (OverflowError, ValueError, IndexError, FloatingPointError) as exc:
        return ScreenResult(False, f"circuit/model construction rejected: {exc}", None, {})

    statistics = circuit.operation_statistics()
    rotation_count = int(
        statistics.num_z_pow(min_clifford_hierarchy_level=4)
    )
    t_base = float(statistics.num_t_like)
    toffoli_count = float(statistics.num_toffoli_like)
    if rotation_count:
        eps = math.pi * float(min_margin) / 2.0**64 / rotation_count
        if not eps > 0:
            return ScreenResult(False, "non-positive rotation synthesis error", None, {})
        t_count = t_base + rotation_count * verifier.rotation_t_count(eps)
    else:
        eps = None
        t_count = t_base
    toffoli_equivalent = toffoli_count + t_count / float(contract.constants["T_PER_TOFFOLI"])
    score = float(circuit.num_qubits) * toffoli_equivalent

    metrics = {
        "max_relative_error": max_relative_error,
        "error_tolerance_ratio": ratio,
        "min_phase_margin_units": min_margin,
        "num_qubits": int(circuit.num_qubits),
        "toffoli_count": toffoli_count,
        "rotation_count": rotation_count,
        "rotation_eps": eps,
        "estimated_t_count": t_count,
        "estimated_toffoli_equivalent": toffoli_equivalent,
        "estimated_score": score,
        "domain_size": int(xs.size),
        "domain_kind": domain_kind,
        "precision": dict(parameters),
    }
    return ScreenResult(True, None, score, metrics)


def write_circuit(
    contract: VerifierContract,
    parameters: dict[str, int],
    output_path: Path,
) -> str:
    """Build and write the canonical ``.kmb`` artifact; return its SHA-256."""
    reference, _ = ensure_runtime(contract)
    n_p = int(contract.constants["N_P"])
    precision = reference.Precision(**parameters)
    circuit = reference.build_nr_circuit(n_p, prec=precision)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    circuit.to_file(output_path, format="kmb")
    return sha256_file(output_path)


def _shift(
    parameters: dict[str, int], field: str, delta: int
) -> dict[str, int] | None:
    low, high = _FIELD_BOUNDS[field]
    value = int(parameters[field]) + delta
    if not low <= value <= high:
        return None
    candidate = dict(parameters)
    candidate[field] = value
    return candidate


def neighbor_precisions(parameters: dict[str, int]) -> Iterator[dict[str, int]]:
    """Yield single-coordinate neighbours in a deterministic order."""
    for field in PRECISION_FIELDS:
        for delta in (-4, -3, -2, -1, 1, 2, 3, 4):
            candidate = _shift(parameters, field, delta)
            if candidate is not None:
                yield candidate


def refinement_precisions(
    parameters: dict[str, int],
    *,
    fields: tuple[str, ...] = ("delta_bits", "product_bits", "result_bits"),
    offsets: tuple[int, ...] = (-2, -1, 1, 2),
) -> Iterator[dict[str, int]]:
    """Yield single- and pairwise-coordinate neighbours for local refinement.

    Precision fields interact (a narrower delta can buy a narrower result and
    still leave phase margin), so a strictly one-coordinate walk can miss
    combinations that are inside the local error budget.
    """
    for order in range(1, len(fields) + 1):
        for field_set in itertools.combinations(fields, order):
            for offset_set in itertools.product(offsets, repeat=order):
                candidate: dict[str, int] | None = dict(parameters)
                for field, offset in zip(field_set, offset_set, strict=True):
                    if candidate is None:
                        break
                    candidate = _shift(candidate, field, offset)
                if candidate is not None:
                    yield candidate
