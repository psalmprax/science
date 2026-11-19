#!/usr/bin/env python3
"""Plan-only screening for DFTHCSOS Arm 3: SelectSwap and Reed-Muller together.

Arm 3 is the interaction arm. It asks whether the two isolated levers add up, so it
refuses to produce a layout until Arm 1 and Arm 2 each hold a PASS official receipt on
their own (and each strictly beats the shared baseline). Planning figures are never
costs and are never promoted; the exact widths and the combined cost must come from
the implementation and the official verifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARM1 = "arm1_selectswap_enabled"
ARM2 = "arm2_reed_muller_angle_buffering_only"
ARM3 = "arm3_combined_selectswap_and_reed_muller"

CANDIDATES = (
    {"name": "combined_rm_block2_j4", "block_bits": 1, "angle_batch_size": 4},
    {"name": "combined_rm_block4_j6", "block_bits": 2, "angle_batch_size": 6},
)


def status_and_score(obj: dict[str, Any], label: str) -> tuple[str, float]:
    """Read ``(status, score)`` from a raw receipt or an Arm 2 receipt wrapper."""
    receipt = obj.get("official_receipt", obj)
    if not isinstance(receipt, dict):
        raise ValueError(f"{label}: expected an official receipt object")
    status = receipt.get("status")
    score = receipt.get("score")
    if not isinstance(status, str) or not status:
        raise ValueError(f"{label}: official receipt status is required")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not score > 0:
        raise ValueError(f"{label}: official receipt score must be a positive number")
    return status, float(score)


def make_plan(
    *,
    baseline: dict[str, Any] | None = None,
    arm1_isolated: dict[str, Any] | None = None,
    arm2_isolated: dict[str, Any] | None = None,
    address_count: int = 3976,
    word_bits: int = 12,
    selection_qubits: int = 12,
    ancilla_qubits: int = 11,
    qubit_cap: int = 128,
    shared_total_qubits_constant: int = 634,
    buffered_delta_bits_per_entry: int | None = None,
    phase_fixup_qubits: int | None = None,
) -> list[dict[str, Any]]:
    if baseline is None or arm1_isolated is None or arm2_isolated is None:
        raise ValueError(
            "Arm 3 is the interaction arm: supply the baseline and both isolated official "
            "receipts (--baseline/--arm1-receipt/--arm2-receipt). Without them the "
            "combination has no premise to test."
        )
    dims = (address_count, word_bits, selection_qubits, ancilla_qubits,
            qubit_cap, shared_total_qubits_constant)
    if any(v <= 0 for v in dims):
        raise ValueError("all dimensions must be positive")
    if buffered_delta_bits_per_entry is not None and buffered_delta_bits_per_entry <= 0:
        raise ValueError("buffered_delta_bits_per_entry must be positive")
    if phase_fixup_qubits is not None and phase_fixup_qubits < 0:
        raise ValueError("phase_fixup_qubits cannot be negative")

    base_status, base_score = status_and_score(baseline, "baseline")
    if base_status.upper() != "PASS":
        raise ValueError("baseline official receipt must be PASS")
    scores: dict[str, float] = {}
    for label, receipt in (("arm1_isolated", arm1_isolated), ("arm2_isolated", arm2_isolated)):
        status, score = status_and_score(receipt, label)
        if status.upper() != "PASS":
            raise ValueError(f"{label}: the isolated arm must be PASS before combining, got {status!r}")
        scores[label] = score

    arm1_gain = base_score / scores["arm1_isolated"]
    arm2_gain = base_score / scores["arm2_isolated"]
    if not (arm1_gain > 1.0 and arm2_gain > 1.0):
        raise ValueError(
            "each isolated arm must strictly beat the baseline before Arm 3 is plannable: "
            f"arm1_gain={arm1_gain:.6g}, arm2_gain={arm2_gain:.6g}"
        )
    expected_gain = arm1_gain * arm2_gain

    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        block_size = 1 << candidate["block_bits"]
        buffered_words = block_size - 1
        base_angle_qubits = (
            selection_qubits + candidate["angle_batch_size"] * word_bits + ancilla_qubits
        )
        width_known = (
            buffered_delta_bits_per_entry is not None and phase_fixup_qubits is not None
        )
        buffer_width = buffered_words * buffered_delta_bits_per_entry if width_known else None
        phase_width = phase_fixup_qubits if width_known else None
        peak = base_angle_qubits + buffer_width + phase_width if width_known else None
        rows.append({
            "candidate": candidate["name"],
            "arm": ARM3,
            "levers": {ARM1: True, ARM2: True},
            "address_count": address_count,
            "word_bits": word_bits,
            "block_bits": candidate["block_bits"],
            "block_size": block_size,
            "angle_batch_size_J": candidate["angle_batch_size"],
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
            "total_qubits_estimate": shared_total_qubits_constant + word_bits * candidate["angle_batch_size"],
            "baseline_score": base_score,
            "arm1_isolated_score": scores["arm1_isolated"],
            "arm2_isolated_score": scores["arm2_isolated"],
            "arm1_isolated_gain": arm1_gain,
            "arm2_isolated_gain": arm2_gain,
            "expected_multiplicative_gain": expected_gain,
            "expected_combined_score_estimate": base_score / expected_gain,
            "official_cost": None,
            "status": "PLAN_ONLY_NEEDS_COMBINED_RECEIPT",
            "additivity_is_hypothesis": True,
            "must_measure": [
                "exact combined width: SelectSwap block encoding plus buffered RMS deltas",
                "phase-fixup circuit width and Toffoli-like cost under both levers at once",
                "all ANGLE circuit peak qubits and Toffoli-like counts",
                "full chemistry verifier lambda_eff, Q_total, T_total, and score for the combination",
                "lookup equality, input restoration, clean scratch, and measurement phase semantics",
                "the isolated Arm 1 and Arm 2 receipts' own constraints, re-checked on the combined build",
            ],
            "warning": (
                "Planning estimate only. The isolated gains compose multiplicatively only if the "
                "levers are independent, which is the hypothesis Arm 3 tests -- do not use the "
                "expected score for promotion or claim official verification."
            ),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--arm1-receipt", type=Path, required=True)
    parser.add_argument("--arm2-receipt", type=Path, required=True)
    parser.add_argument("--address-count", type=int, default=3976)
    parser.add_argument("--word-bits", type=int, default=12)
    parser.add_argument("--selection-qubits", type=int, default=12)
    parser.add_argument("--ancilla-qubits", type=int, default=11)
    parser.add_argument("--qubit-cap", type=int, default=128)
    parser.add_argument("--shared-total-qubits-constant", type=int, default=634)
    parser.add_argument("--buffered-delta-bits-per-entry", type=int, default=None,
                        help="measured encoded delta width; omitting leaves feasibility unknown")
    parser.add_argument("--phase-fixup-qubits", type=int, default=None,
                        help="measured incremental phase-fixup qubits; omitting leaves feasibility unknown")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/quantum_compilation/dfthcsos/arm3/plan.json"))
    args = parser.parse_args()
    rows = make_plan(
        baseline=json.loads(args.baseline.read_text()),
        arm1_isolated=json.loads(args.arm1_receipt.read_text()),
        arm2_isolated=json.loads(args.arm2_receipt.read_text()),
        address_count=args.address_count,
        word_bits=args.word_bits,
        selection_qubits=args.selection_qubits,
        ancilla_qubits=args.ancilla_qubits,
        qubit_cap=args.qubit_cap,
        shared_total_qubits_constant=args.shared_total_qubits_constant,
        buffered_delta_bits_per_entry=args.buffered_delta_bits_per_entry,
        phase_fixup_qubits=args.phase_fixup_qubits,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "candidate_layouts": len(rows),
                      "compiled": False, "official_verification_required": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
