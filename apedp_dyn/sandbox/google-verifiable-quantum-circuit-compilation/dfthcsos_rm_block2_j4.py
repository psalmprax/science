#!/usr/bin/env python3
"""Build and profile the *table-level* rm_block2_j4 transform.

This module intentionally does not emit .kmb circuits.  The official ANGLE word
semantics must be supplied by the user's local reference implementation through
an adapter.  Once supplied, this module pairs adjacent address rows and produces
one high-address lookup row containing [base_words, xor_delta_words].

Expected table convention:
  batch_xor_words[k, address, word] is the exact 12-bit output word of official
  angle_k.kmb's classical lookup at `address`. For batch k>0 this is the XOR
  difference between angle batch k and k-1, exactly as required by the verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Callable, Any

import numpy as np

ADDRESS_COUNT = 3976
NUM_ANGLES = 75
WORD_BITS = 12
J = 4
ANGLE_BATCH_COUNT = math.ceil(NUM_ANGLES / J)  # 19, including final padded batch
SELECTION_QUBITS = 12
ANGLE_ANCILLA_QUBITS = 11
ANGLE_BUFFER_BITS = J * WORD_BITS  # 48-bit XOR difference for one adjacent address
ANGLE_BASE_WIDTH = SELECTION_QUBITS + J * WORD_BITS + ANGLE_ANCILLA_QUBITS  # 71
ANGLE_PEAK_BEFORE_PHASE_FIXUP = ANGLE_BASE_WIDTH + ANGLE_BUFFER_BITS  # 119
QUBIT_CAP = 128

# Values pinned from the user's current baseline receipt/context; these are not
# recomputed from an official verifier in this package.
EXPECTED_SUBMISSION_SHA256 = "276d464ff14da828b6b33e8487d99975494cbdf0c3cb2bd37dd3c3ddc4002695"
BASELINE_LAMBDA_EFF = 49.30136925057438
BASELINE_Q_TOTAL = 730
BASELINE_T_TOTAL = 93237
BASELINE_T_ANGLES_FORWARD_REVERSE = 79480
BASELINE_NON_ANGLE_T = BASELINE_T_TOTAL - BASELINE_T_ANGLES_FORWARD_REVERSE
BASELINE_COST = 3355599588.3155365
BASELINE_INPUT_ROWS_PER_QROM = ADDRESS_COUNT
BASELINE_ANGLE_CIRCUITS = 10

# Historical design-note range only. The actual phase-fixup circuit must be built
# and counted before these can be replaced with measured resource counts.
PHASE_FIXUP_TOFFOLI_LOW = 64
PHASE_FIXUP_TOFFOLI_HIGH = 95


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    c = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(c.dtype).encode())
    h.update(json.dumps(list(c.shape), separators=(",", ":")).encode())
    h.update(c.tobytes(order="C"))
    return h.hexdigest()


def import_callable(spec: str) -> Callable[..., Any]:
    """Resolve a user adapter as ``package.module:function``."""
    if ":" not in spec:
        raise ValueError("--adapter must be specified as package.module:function")
    module_name, function_name = spec.split(":", 1)
    if not module_name or not function_name:
        raise ValueError("--adapter must be specified as package.module:function")
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name, None)
    if not callable(fn):
        raise TypeError(f"adapter {spec!r} is not a callable")
    return fn


def validate_batch_xor_words(words: np.ndarray) -> np.ndarray:
    """Validate the exact official lookup table shape and 12-bit word domain."""
    arr = np.asarray(words)
    expected = (ANGLE_BATCH_COUNT, ADDRESS_COUNT, J)
    if arr.shape != expected:
        raise ValueError(f"batch_xor_words shape must be {expected}, got {arr.shape}")
    if arr.dtype.kind not in "iu":
        raise TypeError(f"batch_xor_words must contain integers, got dtype {arr.dtype}")
    if np.any(arr < 0) or np.any(arr >= (1 << WORD_BITS)):
        raise ValueError("each ANGLE output word must be in [0, 4095]")
    # uint16 makes XOR behavior explicit and safe across signed input arrays.
    return arr.astype(np.uint16, copy=False)


def pair_adjacent_addresses(batch_xor_words: np.ndarray) -> np.ndarray:
    """Pack every even/odd address pair as [base(4 words), delta(4 words)].

    For high address h and low bit b:
      base[h]  = table[2*h]
      delta[h] = table[2*h] XOR table[2*h+1]
      answer   = base[h] XOR (b * delta[h])

    The packed table has 1,988 rows, 8 x 12-bit outputs (96 bits), per ANGLE
    batch. This is the exact classical truth-table transformation; the quantum
    implementation of the low-bit-controlled XOR and measured cleanup is separate.
    """
    words = validate_batch_xor_words(batch_xor_words)
    even = words[:, 0::2, :]
    odd = words[:, 1::2, :]
    if even.shape != odd.shape or even.shape[1] * 2 != ADDRESS_COUNT:
        raise ValueError("address count must consist of complete even/odd pairs")
    delta = np.bitwise_xor(even, odd)
    packed = np.concatenate((even, delta), axis=2)
    assert packed.shape == (ANGLE_BATCH_COUNT, ADDRESS_COUNT // 2, 2 * J)
    return packed.astype(np.uint16, copy=False)


def reconstruct_addresses(packed: np.ndarray) -> np.ndarray:
    """Invert pair_adjacent_addresses and reconstruct every original lookup row."""
    arr = np.asarray(packed)
    expected = (ANGLE_BATCH_COUNT, ADDRESS_COUNT // 2, 2 * J)
    if arr.shape != expected:
        raise ValueError(f"packed table shape must be {expected}, got {arr.shape}")
    if arr.dtype.kind not in "iu" or np.any(arr < 0) or np.any(arr >= (1 << WORD_BITS)):
        raise ValueError("packed table values must be integers in [0, 4095]")
    base = arr[:, :, :J].astype(np.uint16, copy=False)
    delta = arr[:, :, J:].astype(np.uint16, copy=False)
    out = np.empty((ANGLE_BATCH_COUNT, ADDRESS_COUNT, J), dtype=np.uint16)
    out[:, 0::2, :] = base
    out[:, 1::2, :] = np.bitwise_xor(base, delta)
    return out


def _popcount12(values: np.ndarray) -> np.ndarray:
    # Vectorized table-size is small enough that a short Python-free 12-bit loop is fine.
    v = values.astype(np.uint16, copy=True)
    count = np.zeros(v.shape, dtype=np.uint8)
    for _ in range(WORD_BITS):
        count += (v & 1).astype(np.uint8)
        v >>= 1
    return count


def profile_actual_table(words: np.ndarray, packed: np.ndarray) -> dict[str, Any]:
    words = validate_batch_xor_words(words)
    recon = reconstruct_addresses(packed)
    equivalent = bool(np.array_equal(words, recon))
    even = words[:, 0::2, :]
    odd = words[:, 1::2, :]
    deltas = np.bitwise_xor(even, odd)
    per_batch = []
    for k in range(ANGLE_BATCH_COUNT):
        d = deltas[k]
        per_batch.append({
            "batch_index": k,
            "base_unique_rows": int(np.unique(even[k], axis=0).shape[0]),
            "delta_unique_rows": int(np.unique(d, axis=0).shape[0]),
            "zero_delta_word_fraction": float(np.mean(d == 0)),
            "zero_delta_row_fraction": float(np.mean(np.all(d == 0, axis=1))),
            "delta_mean_hamming_weight_per_48_bit_row": float(np.sum(_popcount12(d), axis=1).mean()),
            "delta_max_hamming_weight_per_48_bit_row": int(np.sum(_popcount12(d), axis=1).max()),
        })
    return {
        "status": "TABLE_TRANSFORMED_CLASSICAL_EQUIVALENCE_PASS" if equivalent else "TABLE_TRANSFORM_FAILED",
        "compiled_kmb": False,
        "official_verifier_run": False,
        "lookup_equivalence_all_addresses": equivalent,
        "angle_batches_J4": ANGLE_BATCH_COUNT,
        "addresses_per_batch": ADDRESS_COUNT,
        "words_per_address": J,
        "word_bits": WORD_BITS,
        "pair_rows_per_batch": ADDRESS_COUNT // 2,
        "packed_words_per_pair": 2 * J,
        "packed_payload_bits_per_pair": 2 * J * WORD_BITS,
        "raw_table_bytes": int(words.nbytes),
        "packed_table_bytes": int(packed.nbytes),
        "raw_table_sha256": sha256_array(words),
        "packed_table_sha256": sha256_array(packed),
        "delta_word_zero_fraction": float(np.mean(deltas == 0)),
        "delta_row_zero_fraction": float(np.mean(np.all(deltas == 0, axis=2))),
        "per_batch": per_batch,
        "caveat": "Classical truth-table equivalence only; this does not validate HMR phase fixup or emit a KMB circuit.",
    }


def design_projection() -> dict[str, Any]:
    """Return transparent analytical projection, never report it as measured."""
    high_rows = ADDRESS_COUNT // 2
    lookup_t_per_circuit = high_rows - 1  # standard QROM structural estimate only
    raw_qrom_t = ANGLE_BATCH_COUNT * lookup_t_per_circuit
    fixup_low = ANGLE_BATCH_COUNT * PHASE_FIXUP_TOFFOLI_LOW
    fixup_high = ANGLE_BATCH_COUNT * PHASE_FIXUP_TOFFOLI_HIGH
    angle_low = raw_qrom_t + fixup_low
    angle_high = raw_qrom_t + fixup_high
    total_low = BASELINE_NON_ANGLE_T + angle_low
    total_high = BASELINE_NON_ANGLE_T + angle_high
    q_total = 634 + WORD_BITS * J
    score_low = BASELINE_LAMBDA_EFF * q_total * total_low
    score_high = BASELINE_LAMBDA_EFF * q_total * total_high
    return {
        "candidate": "rm_block2_j4",
        "arm": "arm2_reed_muller_angle_buffering_only",
        "status": "ANALYTICAL_PROJECTION_ONLY",
        "compiled_kmb": False,
        "official_verifier_run": False,
        "input_submission_sha256_expected": EXPECTED_SUBMISSION_SHA256,
        "angle_batch_size_J": J,
        "angle_batch_count": ANGLE_BATCH_COUNT,
        "reason_for_19_circuits": "ceil(75 logical angles / 4 per batch) = 19; final batch is padded according to the official table semantics",
        "address_count": ADDRESS_COUNT,
        "address_block_size": 2,
        "high_address_rows": high_rows,
        "selection_qubits": SELECTION_QUBITS,
        "word_register_qubits": J * WORD_BITS,
        "baseline_angle_ancilla_qubits": ANGLE_ANCILLA_QUBITS,
        "angle_base_width_before_buffer": ANGLE_BASE_WIDTH,
        "buffer_payload_bits_for_four_word_delta": ANGLE_BUFFER_BITS,
        "peak_angle_qubits_before_phase_fixup": ANGLE_PEAK_BEFORE_PHASE_FIXUP,
        "qubit_cap": QUBIT_CAP,
        "extra_phase_fixup_qubits_allowed": QUBIT_CAP - ANGLE_PEAK_BEFORE_PHASE_FIXUP,
        "projected_Q_total_formula": "634 + 12*J",
        "projected_Q_total": q_total,
        "baseline_Q_total": BASELINE_Q_TOTAL,
        "projected_Q_saving": BASELINE_Q_TOTAL - q_total,
        "structural_QROM_Toffolis_per_candidate_batch": lookup_t_per_circuit,
        "projected_candidate_QROM_Toffolis_all_19_batches": raw_qrom_t,
        "phase_fixup_Toffolis_per_batch_design_note_range": [PHASE_FIXUP_TOFFOLI_LOW, PHASE_FIXUP_TOFFOLI_HIGH],
        "projected_phase_fixup_Toffolis_all_19_batches": [fixup_low, fixup_high],
        "baseline_T_ANGLES_forward_reverse": BASELINE_T_ANGLES_FORWARD_REVERSE,
        "projected_T_ANGLES": [angle_low, angle_high],
        "baseline_non_angle_T_assumed_unchanged": BASELINE_NON_ANGLE_T,
        "projected_T_total": [total_low, total_high],
        "baseline_lambda_eff_assumed_unchanged": BASELINE_LAMBDA_EFF,
        "projected_chemistry_cost": [score_low, score_high],
        "projected_gain_vs_frozen_baseline": [BASELINE_COST / score_high, BASELINE_COST / score_low],
        "assumptions": [
            "one high-address QROM loads 48-bit base plus 48-bit XOR delta for each pair",
            "structural QROM Toffoli cost is approximated as high_address_rows - 1 per batch",
            "phase-fixup costs 64-95 Toffolis per batch based on prior design notes, not this run",
            "HMR cleanup adds no Toffolis as planned but remains unimplemented and unverified",
            "lambda_eff and all non-ANGLE costs remain unchanged",
            "Q_total = 634 + 12*J is the existing analytical score-model formula",
        ],
        "promotion_gate": {
            "input_npz_sha256_must_equal": EXPECTED_SUBMISSION_SHA256,
            "all_19_angle_circuits_within_128_qubits": True,
            "exact_peak_qubits_and_Toffoli_counts_measured": False,
            "lookup_semantics_pass": False,
            "phase_semantics_pass": False,
            "official_verifier_status": "NOT_RUN",
            "strict_official_cost_improvement": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/quantum_compilation/dfthcsos/arm2/rm_block2_j4"))
    p.add_argument("--submission-npz", type=Path, help="frozen submission.npz used by the official reference")
    p.add_argument("--expected-input-sha256", default=EXPECTED_SUBMISSION_SHA256,
                   help="frozen input SHA-256; override only for a new, explicitly documented baseline")
    p.add_argument("--tables-npz", type=Path,
                   help="NPZ containing batch_xor_words with shape (19,3976,4), exported by the official reference table adapter")
    p.add_argument("--adapter", help="optional local table adapter module:function; returns batch_xor_words ndarray")
    a = p.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=True)

    projection = design_projection()
    (a.output_dir / "design_projection.json").write_text(json.dumps(projection, indent=2) + "\n")
    print(json.dumps({"design_projection": str(a.output_dir / "design_projection.json"),
                      "projection_status": projection["status"],
                      "table_transform_run": False}, indent=2))

    if not a.tables_npz and not a.adapter:
        return 0
    if a.submission_npz is None:
        raise SystemExit("--submission-npz is required with --tables-npz or --adapter")
    if not a.submission_npz.is_file():
        raise SystemExit(f"submission input not found: {a.submission_npz}")
    actual_sha = sha256_file(a.submission_npz)
    if actual_sha.lower() != a.expected_input_sha256.lower():
        raise SystemExit(f"frozen submission.npz hash mismatch: expected {a.expected_input_sha256}, got {actual_sha}")

    if a.tables_npz:
        if not a.tables_npz.is_file():
            raise SystemExit(f"table NPZ not found: {a.tables_npz}")
        with np.load(a.tables_npz, allow_pickle=False) as z:
            if "batch_xor_words" not in z:
                raise SystemExit("tables NPZ must contain array 'batch_xor_words'")
            words = validate_batch_xor_words(z["batch_xor_words"])
    else:
        fn = import_callable(a.adapter)
        words = validate_batch_xor_words(fn(a.submission_npz))

    packed = pair_adjacent_addresses(words)
    recon = reconstruct_addresses(packed)
    if not np.array_equal(words, recon):
        raise SystemExit("pair-table exhaustive lookup-equivalence check FAILED")
    np.savez_compressed(a.output_dir / "rm_block2_j4_pair_tables.npz", pair_tables=packed)
    report = profile_actual_table(words, packed)
    report["submission_npz_sha256"] = actual_sha
    report["source_adapter"] = a.adapter or None
    report["table_source_path"] = str(a.tables_npz) if a.tables_npz else None
    report["phase_semantics_pass"] = None
    report["phase_semantics_note"] = "Not testable in this table-only harness; requires actual Kickmix HMR circuit and verifier trajectory"
    (a.output_dir / "table_profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"table_profile": str(a.output_dir / "table_profile.json"),
                      "pair_tables": str(a.output_dir / "rm_block2_j4_pair_tables.npz"),
                      "lookup_equivalence_all_addresses": report["lookup_equivalence_all_addresses"],
                      "compiled_kmb": False,
                      "official_verifier_run": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
