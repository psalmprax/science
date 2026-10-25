"""QROM-based coordinate-search helpers for Benchmark I (inverse square roots).

Status: experimental

Provides single-coordinate neighborhood generation and exact-model screening
over the piecewise-linear QROM seed precision parameters defined in
:mod:`apedp_dyn.quantum_compilation.invsqrt_qrom_line`.  Delegates all fixed-point
modeling, 64-bit integer phase margin calculation, and circuit construction to
:mod:`apedp_dyn.quantum_compilation.invsqrt_qrom_line`.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from apedp_dyn.quantum_compilation.candidates import sha256_file
from apedp_dyn.quantum_compilation.contracts import VerifierContract
from apedp_dyn.quantum_compilation.invsqrt_qrom_line import (
    QPrecision,
    build_qrom_line_circuit,
    ensure_runtime,
    fixed_point_model,
    reference_phase_values,
)

__all__ = [
    "QROM_PRECISION_FIELDS",
    "QROMScreenResult",
    "qrom_base_precision",
    "qrom_neighbor_precisions",
    "qrom_screen_precision",
    "qrom_write_circuit",
]

QROM_PRECISION_FIELDS = (
    "index_bits",
    "coef_bits",
    "slope_bits",
    "delta_bits",
    "square_bits",
    "product_bits",
    "result_bits",
)

_QROM_FIELD_BOUNDS = {
    "index_bits": (5, 9),
    "coef_bits": (12, 20),
    "slope_bits": (10, 16),
    "delta_bits": (8, 16),
    "square_bits": (24, 32),
    "product_bits": (24, 32),
    "result_bits": (24, 32),
}


@dataclasses.dataclass(frozen=True)
class QROMScreenResult:
    passed: bool
    reason: str | None
    score_estimate: float | None
    metrics: dict[str, Any]


def qrom_base_precision(contract: VerifierContract | None = None) -> dict[str, int]:
    """Return the default QROM piecewise-linear seed precision dictionary."""
    return dataclasses.asdict(QPrecision())


def qrom_screen_precision(
    contract: VerifierContract,
    parameters: dict[str, int],
    *,
    xs: np.ndarray | None = None,
    safety_ratio: float = 1.0,
) -> QROMScreenResult:
    """Screen a QROM precision point with the exact 64-bit integer phase model."""
    n_p = int(contract.constants["N_P"])
    max_x = int(3 * (2**n_p - 2) ** 2)
    if xs is None:
        xs_arr = np.arange(1, max_x + 1, dtype=np.int64)
    else:
        xs_arr = np.asarray(xs, dtype=np.int64)
    domain_kind = "full-domain" if xs_arr.size >= max_x else "subset"

    try:
        prec = QPrecision(**parameters)
    except TypeError as exc:
        return QROMScreenResult(False, f"invalid qrom precision: {exc}", None, {})

    output_bits = prec.output_bits(n_p)
    if output_bits > 64:
        return QROMScreenResult(
            False, "output_bits exceeds the 64-bit phase accumulator", None, {}
        )

    try:
        y1 = fixed_point_model(n_p, prec, xs_arr)
        phase = y1.astype(np.uint64) << np.uint64(64 - output_bits)
        target, tol = reference_phase_values(n_p, xs_arr)
        error = (phase - target).view(np.int64)
        if (error == np.iinfo(np.int64).min).any():
            raise OverflowError("phase difference saturates signed 64-bit accumulator")
        used = np.abs(error).view(np.uint64)
        ratio_arr = np.divide(
            used, tol, out=np.zeros(xs_arr.size, dtype=np.float64), where=tol > 0
        )
        max_ratio = float(np.max(ratio_arr))
        min_margin = int((tol.astype(np.int64) - used.astype(np.int64)).min())
        if not math.isfinite(max_ratio) or max_ratio >= safety_ratio:
            return QROMScreenResult(
                False,
                f"qrom model error ratio {max_ratio:.6f} >= {safety_ratio:.6f}",
                None,
                {
                    "error_tolerance_ratio": max_ratio,
                    "min_phase_margin_units": min_margin,
                    "domain_size": int(xs_arr.size),
                    "domain_kind": domain_kind,
                },
            )
        if min_margin <= 0:
            return QROMScreenResult(
                False,
                "no phase tolerance left for rotation synthesis",
                None,
                {
                    "error_tolerance_ratio": max_ratio,
                    "min_phase_margin_units": min_margin,
                    "domain_size": int(xs_arr.size),
                    "domain_kind": domain_kind,
                },
            )

        _, verifier = ensure_runtime(contract)
        circuit = build_qrom_line_circuit(n_p, prec)
        stats = circuit.operation_statistics()
        rotations = int(stats.num_z_pow(min_clifford_hierarchy_level=4))
        t_base = float(stats.num_t_like)
        toffoli_count = float(stats.num_toffoli_like)
        if rotations > 0:
            eps = math.pi * float(min_margin) / (2.0**64) / rotations
            if not eps > 0:
                return QROMScreenResult(
                    False, "non-positive rotation synthesis error", None, {}
                )
            t_count = t_base + rotations * verifier.rotation_t_count(eps)
        else:
            eps = None
            t_count = t_base
        t_per_toffoli = float(contract.constants.get("T_PER_TOFFOLI", 4.0))
        toffoli_equiv = toffoli_count + t_count / t_per_toffoli
        score = float(circuit.num_qubits) * toffoli_equiv

        metrics = {
            "error_tolerance_ratio": max_ratio,
            "min_phase_margin_units": min_margin,
            "num_qubits": int(circuit.num_qubits),
            "toffoli_count": toffoli_count,
            "t_count": t_base,
            "rotation_count": rotations,
            "rotation_eps": eps,
            "estimated_t_count": t_count,
            "estimated_toffoli_equivalent": toffoli_equiv,
            "estimated_score": score,
            "domain_size": int(xs_arr.size),
            "domain_kind": domain_kind,
            "precision": dict(parameters),
            "structure": "qrom",
        }
        return QROMScreenResult(True, None, score, metrics)
    except (OverflowError, ValueError, IndexError, FloatingPointError, RuntimeError) as exc:
        return QROMScreenResult(False, f"qrom construction failed: {exc}", None, {})


def qrom_write_circuit(
    contract: VerifierContract,
    parameters: dict[str, int],
    output_path: Path,
) -> str:
    """Build and write the QROM ``.kmb`` circuit artifact; return its SHA-256."""
    ensure_runtime(contract)
    n_p = int(contract.constants["N_P"])
    prec = QPrecision(**parameters)
    circuit = build_qrom_line_circuit(n_p, prec)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    circuit.to_file(output_path, format="kmb")
    return sha256_file(output_path)


def _qrom_shift(
    parameters: dict[str, int], field: str, delta: int
) -> dict[str, int] | None:
    low, high = _QROM_FIELD_BOUNDS[field]
    value = int(parameters[field]) + delta
    if not low <= value <= high:
        return None
    candidate = dict(parameters)
    candidate[field] = value
    return candidate


def qrom_neighbor_precisions(parameters: dict[str, int]) -> Iterator[dict[str, int]]:
    """Yield single-coordinate neighbors in deterministic order."""
    for field in QROM_PRECISION_FIELDS:
        for delta in (-2, -1, 1, 2):
            candidate = _qrom_shift(parameters, field, delta)
            if candidate is not None:
                yield candidate
