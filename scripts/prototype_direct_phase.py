#!/usr/bin/env python3
"""Experimental shared-predicate phase oracle, independent of FAROS."""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import kickmix as km

from prototype_streamed_invsqrt import phase_words


def build_direct(words: list[int], precision_bits: int):
    """Apply fixed-point phases using a shared prefix tree of input predicates.

    Missing table entries receive zero phase, not periodic prefix aliasing.
    All 22 input bits are included in each predicate.
    """
    if not 2 <= len(words) <= 1 << 22:
        raise ValueError("phase table must contain between 2 and 2**22 entries")
    if not 32 <= precision_bits <= 48:
        raise ValueError("precision_bits must be in [32, 48]")
    if any(not 0 <= word < 1 << precision_bits for word in words):
        raise ValueError("phase words must fit the declared precision")
    builder = km.CircuitBuilder()
    builder.set_max_qubits(50)
    x = builder.create_quantum_register(22, name="x")
    scratch = builder.alloc_qubits(21)

    def visit(depth, start, stop, control):
        if start >= len(words) or not any(words[start:min(stop, len(words))]):
            return
        if depth == 22:
            builder.z_pow(control, Fraction(2 * words[start], 1 << precision_bits))
            return
        qubit = x[21 - depth]
        middle = (start + stop) // 2
        for value, lower, upper in ((0, start, middle), (1, middle, stop)):
            if lower >= len(words) or not any(words[lower:min(upper, len(words))]):
                continue
            if value == 0:
                builder.x(qubit)
            if control is None:
                visit(depth + 1, lower, upper, qubit)
            else:
                predicate = scratch[depth - 1]
                builder.ccx(control, qubit, predicate)
                visit(depth + 1, lower, upper, predicate)
                builder.ccx(control, qubit, predicate)
            if value == 0:
                builder.x(qubit)

    visit(0, 0, 1 << 22, None)
    builder.free(scratch)
    return builder.finish_circuit()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-x", type=int, default=4096)
    parser.add_argument("--precision-bits", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_x <= 65535:
        parser.error("max-x must be in [1, 65535] to bound experimental circuit growth")
    if not 32 <= args.precision_bits <= 48:
        parser.error("precision-bits must be in [32, 48]")
    circuit = build_direct(phase_words(args.max_x, args.precision_bits), args.precision_bits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    circuit.to_file(str(args.output), format="kmb")
    stats = circuit.operation_statistics()
    report = {
        "scope": "experimental-prefix-only",
        "max_x": args.max_x,
        "precision_bits": args.precision_bits,
        "num_qubits": circuit.num_qubits,
        "static_toffoli": int(stats.num_toffoli_like),
        "t_like": int(stats.num_t_like),
        "rotations": int(stats.num_z_pow(min_clifford_hierarchy_level=4)),
        "num_ops": len(circuit),
        "outside_prefix_behavior": "zero phase; not a full-domain solution",
        "circuit_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "kickmix_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
        "full_domain_verified": False,
    }
    args.output.with_suffix(".resources.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
