#!/usr/bin/env python3
"""Plan-only screening for DFTHCSOS Arm 2: Reed-Muller angle buffering.

This tool never claims to compile or verify a quantum circuit. Buffer width, phase-fixup
cost, and candidate resources must come from the exact implementation and official verifier.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

CANDIDATES = (
    {"name": "rm_block2_j4", "block_bits": 1, "angle_batch_size": 4},
    {"name": "rm_block4_j6", "block_bits": 2, "angle_batch_size": 6},
)


def make_plan(*, address_count: int = 3976, word_bits: int = 12,
              selection_qubits: int = 12, ancilla_qubits: int = 11,
              qubit_cap: int = 128, shared_total_qubits_constant: int = 634,
              buffered_delta_bits_per_entry: int | None = None,
              phase_fixup_qubits: int | None = None) -> list[dict]:
    values = (address_count, word_bits, selection_qubits, ancilla_qubits,
              qubit_cap, shared_total_qubits_constant)
    if any(v <= 0 for v in values):
        raise ValueError("all dimensions must be positive")
    if buffered_delta_bits_per_entry is not None and buffered_delta_bits_per_entry <= 0:
        raise ValueError("buffered_delta_bits_per_entry must be positive")
    if phase_fixup_qubits is not None and phase_fixup_qubits < 0:
        raise ValueError("phase_fixup_qubits cannot be negative")

    rows = []
    for c in CANDIDATES:
        block_size = 1 << c["block_bits"]
        buffered_words = block_size - 1
        base_angle_qubits = selection_qubits + c["angle_batch_size"] * word_bits + ancilla_qubits
        width_known = (buffered_delta_bits_per_entry is not None and phase_fixup_qubits is not None)
        buffer_width = buffered_words * buffered_delta_bits_per_entry if width_known else None
        phase_width = phase_fixup_qubits if width_known else None
        peak = base_angle_qubits + buffer_width + phase_width if width_known else None
        rows.append({
            "candidate": c["name"],
            "arm": "arm2_reed_muller_angle_buffering_only",
            "address_count": address_count,
            "word_bits": word_bits,
            "block_bits": c["block_bits"],
            "block_size": block_size,
            "angle_batch_size_J": c["angle_batch_size"],
            "buffered_words_per_block": buffered_words,
            "selection_qubits": selection_qubits,
            "ancilla_qubits_baseline": ancilla_qubits,
            "base_angle_qubits_before_buffer": base_angle_qubits,
            "buffered_delta_bits_per_entry": buffered_delta_bits_per_entry,
            "phase_fixup_qubits": phase_fixup_qubits,
            "peak_angle_qubits_estimate": peak,
            "qubit_cap": qubit_cap,
            "feasibility_by_width_estimate": (peak <= qubit_cap) if peak is not None else None,
            "total_qubits_formula_note": f"analytical estimate only: {shared_total_qubits_constant} + {word_bits}*J",
            "total_qubits_estimate": shared_total_qubits_constant + word_bits*c["angle_batch_size"],
            "phase_fixup_toffolis": None,
            "angle_toffolis": None,
            "lambda_eff": None,
            "official_cost": None,
            "status": "PLAN_ONLY_NEEDS_MEASURED_WIDTHS" if not width_known else (
                "PLAN_ONLY_WIDTH_FEASIBLE" if peak <= qubit_cap else "REJECTED_WIDTH_ESTIMATE"
            ),
            "must_measure": [
                "exact encoded Reed-Muller delta width and all temporary qubits",
                "phase-fixup circuit width and Toffoli-like cost",
                "all ANGLE circuit peak qubits and Toffoli-like counts",
                "full chemistry verifier lambda_eff, Q_total, T_total, and score",
                "angle lookup equality, input restoration, clean scratch, and measurement phase semantics",
            ],
            "warning": "Planning estimate only; do not use estimates for promotion or claim official verification.",
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--address-count", type=int, default=3976)
    p.add_argument("--word-bits", type=int, default=12)
    p.add_argument("--selection-qubits", type=int, default=12)
    p.add_argument("--ancilla-qubits", type=int, default=11)
    p.add_argument("--qubit-cap", type=int, default=128)
    p.add_argument("--shared-total-qubits-constant", type=int, default=634)
    p.add_argument("--buffered-delta-bits-per-entry", type=int, default=None,
                   help="measured encoded delta width; omitting leaves feasibility unknown")
    p.add_argument("--phase-fixup-qubits", type=int, default=None,
                   help="measured incremental phase-fixup qubits; omitting leaves feasibility unknown")
    p.add_argument("--output", type=Path, default=Path("artifacts/quantum_compilation/dfthcsos/arm2/plan.json"))
    a = p.parse_args()
    rows = make_plan(address_count=a.address_count, word_bits=a.word_bits,
                     selection_qubits=a.selection_qubits, ancilla_qubits=a.ancilla_qubits,
                     qubit_cap=a.qubit_cap,
                     shared_total_qubits_constant=a.shared_total_qubits_constant,
                     buffered_delta_bits_per_entry=a.buffered_delta_bits_per_entry,
                     phase_fixup_qubits=a.phase_fixup_qubits)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"output": str(a.output), "candidate_layouts": len(rows),
                      "compiled": False, "official_verification_required": True}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
