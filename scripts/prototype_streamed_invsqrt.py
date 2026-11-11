#!/usr/bin/env python3
"""Experimental bit-sliced phase lookup; prefix evidence, not a benchmark submission."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import kickmix as km


def phase_words(max_x: int, precision_bits: int) -> list[int]:
    """Round phases to fixed point; x=0 is outside the verifier's domain."""
    scale = 1 << precision_bits
    return [0] + [round(scale / math.sqrt(x)) % scale for x in range(1, max_x + 1)]


def build_streamed(words: list[int], precision_bits: int, chunk_bits: int):
    builder = km.CircuitBuilder()
    builder.set_max_qubits(50)
    x = builder.create_quantum_register(22, name="x")
    address_bits = (len(words) - 1).bit_length()
    padded = words + [0] * ((1 << address_bits) - len(words))
    address = x[:address_bits]
    scratch = builder.alloc_qubits(max(0, address_bits - 1))

    def toggle_lookup(table, target):
        def visit(depth, index, control):
            if depth == address_bits:
                for bit, qubit in enumerate(target):
                    if (table[index] >> bit) & 1:
                        builder.cx(control, qubit)
                return
            qubit = address[depth]
            for value in (0, 1):
                if value == 0:
                    builder.x(qubit)
                if control is None:
                    visit(depth + 1, index | (value << depth), qubit)
                else:
                    temporary = scratch[depth - 1]
                    builder.ccx(control, qubit, temporary)
                    visit(depth + 1, index | (value << depth), temporary)
                    builder.ccx(control, qubit, temporary)
                if value == 0:
                    builder.x(qubit)
        visit(0, 0, None)
    for offset in range(0, precision_bits, chunk_bits):
        width = min(chunk_bits, precision_bits - offset)
        target = builder.alloc_qubits(width)
        table = [(word >> offset) & ((1 << width) - 1) for word in padded]
        toggle_lookup(table, target)
        for bit, qubit in enumerate(target):
            builder.z_pow(qubit, Fraction(2) ** (offset + bit + 1 - precision_bits))
        toggle_lookup(table, target)
        builder.free(target)
    builder.free(scratch)
    return builder.finish_circuit()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-x", type=int, default=4095)
    parser.add_argument("--precision-bits", type=int, default=40)
    parser.add_argument("--chunk-bits", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_x <= 65535:
        parser.error("max-x must be in [1, 65535]; this prototype bounds lookup growth")
    if not 32 <= args.precision_bits <= 48:
        parser.error("precision-bits must be in [32, 48]")
    max_chunk = 29 - args.max_x.bit_length()
    if not 1 <= args.chunk_bits <= max_chunk:
        parser.error(f"chunk-bits must be in [1, {max_chunk}] to respect the 50-qubit cap")
    words = phase_words(args.max_x, args.precision_bits)
    scale = 1 << args.precision_bits
    worst_ratio = max(
        abs(words[x] / scale - (1 / math.sqrt(x)) % 1) / (2**-24 / math.sqrt(x))
        for x in range(1, args.max_x + 1)
    )
    circuit = build_streamed(words, args.precision_bits, args.chunk_bits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    circuit.to_file(str(args.output), format="kmb")
    stats = circuit.operation_statistics()
    report = {
        "scope": "experimental-prefix-only",
        "max_x": args.max_x,
        "precision_bits": args.precision_bits,
        "chunk_bits": args.chunk_bits,
        "num_qubits": circuit.num_qubits,
        "static_toffoli": int(stats.num_toffoli_like),
        "t_like": int(stats.num_t_like),
        "rotations": int(stats.num_z_pow(min_clifford_hierarchy_level=4)),
        "num_ops": len(circuit),
        "model_worst_error_over_tolerance": worst_ratio,
        "circuit_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "kickmix_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
        "full_domain_verified": False,
        "outside_prefix_behavior": "phase lookup repeats modulo the address-table size",
    }
    args.output.with_suffix(".resources.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
