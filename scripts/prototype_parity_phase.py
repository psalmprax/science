#!/usr/bin/env python3
"""Full-domain feasibility audit of sparse Walsh parity-phase approximations."""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

FULL_MAX_X = 3_133_452
ADDRESS_BITS = 22


def walsh_transform(values: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh-Hadamard transform in binary-mask order."""
    result = np.asarray(values, dtype=np.float64).copy()
    if result.ndim != 1 or result.size == 0 or result.size & (result.size - 1):
        raise ValueError("values must be a nonempty power-of-two vector")
    width = 1
    while width < result.size:
        blocks = result.reshape(-1, 2 * width)
        left = blocks[:, :width].copy()
        right = blocks[:, width:].copy()
        blocks[:, :width] = left + right
        blocks[:, width:] = left - right
        width *= 2
    return result


def benchmark_phases() -> np.ndarray:
    """Smooth phase lift; integer phase at x=1 is equivalent to zero modulo one.

    The continuation outside the benchmark domain is a design choice affecting
    sparsity, not an additional correctness requirement.
    """
    values = np.arange(1 << ADDRESS_BITS, dtype=np.float64)
    values[0] = 1
    return 1 / np.sqrt(values)


def sparse_terms(coefficients: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= count <= coefficients.size:
        raise ValueError("term count must be between one and the vector size")
    indices = np.argpartition(np.abs(coefficients), -count)[-count:]
    indices = indices[np.argsort(-np.abs(coefficients[indices]))]
    return indices, coefficients[indices]


def audit_terms(
    indices: np.ndarray, weights: np.ndarray, reference: np.ndarray, max_x: int
) -> dict:
    if not 1 <= max_x < reference.size:
        raise ValueError("max_x must index the supplied reference vector")
    spectrum = np.zeros_like(reference)
    spectrum[indices] = weights
    approximation = walsh_transform(spectrum)
    errors = approximation[1:max_x + 1] - reference[1:max_x + 1]
    errors -= np.rint(errors)
    ratios = np.abs(errors) / (2**-24 * reference[1:max_x + 1])
    worst = int(np.argmax(ratios))
    return {
        "terms": int(indices.size),
        "tested_inputs": max_x,
        "worst_x": worst + 1,
        "worst_error_over_tolerance": float(ratios[worst]),
        "violating_inputs": int(np.count_nonzero(ratios >= 1)),
        "passes_model_tolerance": bool(np.all(ratios < 1)),
        "includes_rotation_synthesis_error": False,
    }


def build_parity_circuit(indices: np.ndarray, weights: np.ndarray, address_bits: int):
    """Implement the selected model exactly as binary64 rational rotations."""
    import kickmix as km

    if not 1 <= address_bits <= ADDRESS_BITS:
        raise ValueError("address_bits must be in [1, 22]")
    if len(indices) != len(weights) or any(int(mask) >= 1 << address_bits for mask in indices):
        raise ValueError("parity masks must match weights and fit the input register")
    builder = km.CircuitBuilder()
    builder.set_max_qubits(50)
    x = builder.create_quantum_register(address_bits, name="x")
    global_phase = sum((Fraction(float(weight)) for weight in weights), Fraction())
    if global_phase:
        ancilla = builder.alloc_qubits(1)
        builder.x(ancilla)
        builder.z_pow(ancilla, 2 * global_phase)
        builder.x(ancilla)
        builder.free(ancilla)
    for mask, weight in zip(indices, weights, strict=True):
        bits = [bit for bit in range(address_bits) if (int(mask) >> bit) & 1]
        if not bits or weight == 0:
            continue
        pivot = x[bits[0]]
        for bit in bits[1:]:
            builder.cx(x[bit], pivot)
        builder.z_pow(pivot, -4 * Fraction(float(weight)))
        for bit in reversed(bits[1:]):
            builder.cx(x[bit], pivot)
    return builder.finish_circuit()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", type=int, nargs="+", default=[64, 256, 1024, 4096])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--circuit-terms", type=int, default=256)
    args = parser.parse_args()
    if any(not 1 <= count <= 1 << ADDRESS_BITS for count in args.terms + [args.circuit_terms]):
        parser.error("term counts must lie in [1, 2**22]")
    reference = benchmark_phases()
    coefficients = walsh_transform(reference) / reference.size
    results = []
    for count in sorted(set(args.terms + [args.circuit_terms])):
        indices, weights = sparse_terms(coefficients, count)
        result = audit_terms(indices, weights, reference, FULL_MAX_X)
        results.append(result)
        print(json.dumps(result), flush=True)
    indices, weights = sparse_terms(coefficients, args.circuit_terms)
    circuit = build_parity_circuit(indices, weights, ADDRESS_BITS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    circuit_path = args.output_dir / "candidate.kmb"
    circuit.to_file(str(circuit_path), format="kmb")
    import kickmix as km
    stats = circuit.operation_statistics()
    report = {
        "scope": "experimental-full-domain-model-audit",
        "method": "largest-magnitude Walsh coefficients with smooth phase continuation",
        "audits": results,
        "circuit_terms": args.circuit_terms,
        "num_qubits": circuit.num_qubits,
        "static_toffoli": int(stats.num_toffoli_like),
        "t_like": int(stats.num_t_like),
        "rotations": int(stats.num_z_pow(min_clifford_hierarchy_level=4)),
        "num_ops": len(circuit),
        "circuit_sha256": hashlib.sha256(circuit_path.read_bytes()).hexdigest(),
        "kickmix_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
        "officially_verified": False,
        "numerical_audit_is_not_a_formal_error_bound": True,
    }
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(args.output_dir / "terms.npz", indices=indices, weights=weights)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
