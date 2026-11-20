#!/usr/bin/env python3
"""Does the champion pay for zero-angle rotations?

Status: audit-only

`src/invsqrt_calibration.json` records 48 phase deltas. One original rotation
(index 18) is already dropped, and three more carry an angle of exactly zero
(indices 15, 16, 17). A `z_pow` with a zero angle is the identity, so if those
three survive into the compiled circuit the score pays 7.16 Toffoli-equivalents
each for doing nothing.

This script rebuilds the champion through the shipped path, applies the
calibration twice (the shipped drop set, then the shipped set plus 15-17), and
verifies both with the official verifier at the same shot budget. It prints the
rotation count, the Toffoli-equivalent and the score for each, so the answer is
measured rather than argued.

Usage:
    .venv-kickmix/bin/python scripts/try_drop_zero_rotations.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(
    0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix")
)
sys.path.insert(
    0,
    str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/invsqrt"),
)

import kickmix as km
import signed_digit_invsqrt_v3 as core
import portable_exact_domain as domain_compiler
import portable_calibration as cal
import upstream_functions as uf
from verify_submission_invsqrt import ProblemSpecInvSqrt, verify

import test_shared_pddd_circuit as builder

ZERO_ANGLE_INDICES = (15, 16, 17)


def build_raw():
    """The champion's raw circuit, built exactly as the shipped script builds it."""
    cfg, tables, _ = core.load_or_build_tables()
    cv_curve = np.array([tables[(i // 128, i % 128)][0] for i in range(256)], dtype=np.int64)
    gray = cv_curve ^ (cv_curve >> 1)
    choice = json.loads((REPO_ROOT / "src/shared_davio_results.json").read_text())["candidates"][0]
    plan = builder.build_shared_plan(
        {"curve4": (gray >> 4) & 1, "curve5": (gray >> 5) & 1, "curve6": (gray >> 6) & 1},
        order=tuple(choice["order"]),
        polarity=int(choice["polarity"]),
    )
    c6_choice = json.loads((REPO_ROOT / "src/boolean_choices90.json").read_text())["coarse6"]
    raw, meta = builder.build_candidate(
        cfg, tables, coarse_choice=c6_choice, shared_curve_plan=plan
    )
    opt, _ = domain_compiler.optimize_circuit(raw, linear=True)
    return opt, meta


def calibrate(opt, dropped: set[int]):
    fixed = json.loads((REPO_ROOT / "src/invsqrt_calibration.json").read_text())
    text = cal._serialize(opt, "kmx").decode()
    lines, rots = cal.parse_kmx(text)
    values = fixed["deltasPhaseUnits"]
    rewritten = uf.rewrite(
        lines,
        rots,
        {j: int(v) for j, v in enumerate(values) if j not in dropped},
        dropped,
    )
    return km.Circuit(rewritten)


def verify_at(cand, shots_per_x: int = 1) -> dict:
    spec = ProblemSpecInvSqrt.for_n_p(10, shots_per_x=shots_per_x)
    spec = ProblemSpecInvSqrt(
        width=spec.width,
        max_x=25600,
        beta=spec.beta,
        precision_bits=spec.precision_bits,
        shots_per_x=shots_per_x,
        system_qubits=spec.system_qubits,
    )
    return verify(cand, spec)


def summarise(label: str, cand, receipt: dict) -> None:
    stats = cand.operation_statistics()
    print(
        f"{label:<22s} Q={receipt['num_qubits']:3d} "
        f"rotations={receipt['rotation_count']:5.1f} "
        f"T_mean={receipt['toffoli_count']:9.4f} "
        f"T_eq={receipt['toffoli_equivalent']:9.4f} "
        f"score={receipt['score']:12.4f} "
        f"status={receipt['status']} "
        f"worst/tol={receipt['worst_error_over_tolerance']:.10f} "
        f"static={stats.num_toffoli_like}"
    )


def main() -> int:
    opt, meta = build_raw()
    print(f"raw circuit: Q={meta['qubits']}, static Toffolis={meta['static_toffoli']}")

    shipped_drop = {18}
    baseline = calibrate(opt, shipped_drop)
    candidate = calibrate(opt, shipped_drop | set(ZERO_ANGLE_INDICES))

    for label, cand in (("shipped drop {18}", baseline), ("drop {15,16,17,18}", candidate)):
        receipt = verify_at(cand)
        summarise(label, cand, receipt)

    base_receipt = verify_at(baseline)
    cand_receipt = verify_at(candidate)
    delta = cand_receipt["score"] - base_receipt["score"]
    print(
        f"\nscore change: {delta:+.4f} units "
        f"({100 * delta / base_receipt['score']:+.3f}%)"
    )
    print(
        "rotations freed:",
        base_receipt["rotation_count"] - cand_receipt["rotation_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
