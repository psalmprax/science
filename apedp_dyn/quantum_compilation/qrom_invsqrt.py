"""QROM-based search for Benchmark I (inverse square roots).

Alternative circuit structure using piecewise-linear approximation with QROM.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from apedp_dyn.quantum_compilation.candidates import sha256_file
from apedp_dyn.quantum_compilation.contracts import VerifierContract

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


def ensure_qrom_runtime(contract: VerifierContract):
    for path in (
        str(contract.kickmix_root),
        str(contract.verifier_root.parent),
        str(contract.verifier_root),
        str(Path(__file__).parent.parent.parent / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        import kickmix  # noqa
    except ImportError as exc:
        raise RuntimeError(f"kickmix import failed: {exc}") from exc
    
    try:
        # Import from alternatives
        from scripts import nr_qrom
        return nr_qrom
    except Exception as exc:
        raise RuntimeError(f"nr_qrom import failed: {exc}") from exc


def qrom_base_precision(contract: VerifierContract) -> dict[str, int]:
    return {
        "index_bits": 7,
        "coef_bits": 16,
        "slope_bits": 12,
        "delta_bits": 11,
        "square_bits": 28,
        "product_bits": 27,
        "result_bits": 28,
    }


def qrom_screen_precision(
    contract: VerifierContract,
    parameters: dict[str, int],
    *,
    xs: np.ndarray | None = None,
    safety_ratio: float = 1.0,
) -> QROMScreenResult:
    nr_qrom = ensure_qrom_runtime(contract)
    n_p = int(contract.constants["N_P"])
    
    if xs is None:
        from apedp_dyn.quantum_compilation.invsqrt import full_domain_xs
        xs = full_domain_xs(contract)
    xs = np.asarray(xs, dtype=np.int64)
    
    try:
        prec = nr_qrom.QPrecision(**parameters)
    except Exception as exc:
        return QROMScreenResult(False, f"invalid qrom precision: {exc}", None, {})
    
    try:
        approx = nr_qrom.fixed_point_model(n_p, prec, xs)
        output_bits = prec.output_bits(n_p)
        approx_float = approx / (2.0 ** output_bits)
        exact = 1.0 / np.sqrt(xs.astype(np.float64))
        rel_err = np.abs(approx_float - exact) / exact
        max_rel_err = float(np.max(rel_err))
        tolerance = 2.0 ** -24  # approx tolerance for n_p=10
        ratio = max_rel_err / tolerance
        if ratio >= safety_ratio:
            return QROMScreenResult(False, f"qrom model error ratio {ratio:.6f}", None, 
                                    {"max_relative_error": max_rel_err, "error_tolerance_ratio": ratio})
        
        # Build circuit for cost estimation
        circuit = nr_qrom.build_circuit(n_p, prec)
        stats = circuit.operation_statistics()
        rotations = int(stats.num_z_pow(min_clifford_hierarchy_level=4))
        # rough margin estimate
        margin = 1.0  # placeholder
        if rotations:
            eps = 1e-12  # rough
            from invsqrt import verify_submission_invsqrt
            t_count = 1 + rotations * verify_submission_invsqrt.rotation_t_count(eps)
        else:
            t_count = 1
        toffoli_equiv = float(stats.num_toffoli_like) + t_count / 4.0
        score = float(circuit.num_qubits) * toffoli_equiv
        
        metrics = {
            "max_relative_error": max_rel_err,
            "error_tolerance_ratio": ratio,
            "num_qubits": int(circuit.num_qubits),
            "toffoli_count": float(stats.num_toffoli_like),
            "rotation_count": rotations,
            "estimated_score": score,
            "precision": dict(parameters),
            "structure": "qrom",
        }
        return QROMScreenResult(True, None, score, metrics)
    except Exception as exc:
        return QROMScreenResult(False, f"qrom construction failed: {exc}", None, {})


def qrom_write_circuit(
    contract: VerifierContract,
    parameters: dict[str, int],
    output_path: Path,
) -> str:
    nr_qrom = ensure_qrom_runtime(contract)
    n_p = int(contract.constants["N_P"])
    prec = nr_qrom.QPrecision(**parameters)
    circuit = nr_qrom.build_circuit(n_p, prec)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    circuit.to_file(str(output_path), format="kmb")
    return sha256_file(output_path)


def _qrom_shift(parameters: dict[str, int], field: str, delta: int) -> dict[str, int] | None:
    low, high = _QROM_FIELD_BOUNDS[field]
    value = int(parameters[field]) + delta
    if not low <= value <= high:
        return None
    candidate = dict(parameters)
    candidate[field] = value
    return candidate


def qrom_neighbor_precisions(parameters: dict[str, int]) -> Iterator[dict[str, int]]:
    for field in QROM_PRECISION_FIELDS:
        for delta in (-2, -1, 1, 2):
            candidate = _qrom_shift(parameters, field, delta)
            if candidate is not None:
                yield candidate
