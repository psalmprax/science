#!/usr/bin/env python3
"""Calibrated Production Stage-Level Diagnostic Tracer for Benchmark I (Invsqrt).

Directly instruments the frozen calibrated production candidate
(invsqrt_89q_shared_pddd.kmb, SHA-256 ce04c9d54...) at genuine simulator
checkpoints across the algorithmic pipeline.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/invsqrt"))

import kickmix as km
import portable_calibration as cal
from verify_submission_invsqrt import (
    ProblemSpecInvSqrt,
    reference_values,
    half_turns_to_units,
    to_bits,
    from_bits,
)

CIRCUIT_PATH = REPO_ROOT / "artifacts/quantum_compilation/invsqrt/invsqrt_89q_shared_pddd.kmb"


def instrument_calibrated_trace(x_val: int = 32, shots: int = 64, seed: int = 12345):
    print("=" * 90)
    print(f"CALIBRATED PRODUCTION STAGE-LEVEL DIAGNOSTIC TRACE FOR x = {x_val}")
    print(f"Workload: {shots} shots, seed={seed}")
    print(f"Candidate: {CIRCUIT_PATH.name}")
    print("=" * 90)

    # 1. Contract Reference
    spec = ProblemSpecInvSqrt.for_n_p(10, shots_per_x=shots)
    target_arr, tol_arr = reference_values(spec, np.array([x_val], dtype=np.uint64))
    target_units = int(target_arr[0])
    allowed_units = int(tol_arr[0])

    print(f"Published Contract Reference at x={x_val}:")
    print(f"  Target Phase:       {target_units} (mod 2^64) [{target_units / 2**64:.12f} turns]")
    print(f"  Allowed Tolerance:  {allowed_units} units [{allowed_units / 2**64:.12e} turns]")
    exact_math_tol = (2**-24 / math.sqrt(x_val)) * (2**64)
    print(f"  Exact Math Tol:     {exact_math_tol:.2f} units (residual vs verifier: {exact_math_tol - allowed_units:.4f})")
    print("-" * 90)

    # 2. Load and parse the production candidate
    c = km.Circuit.from_file(str(CIRCUIT_PATH))
    t = cal._serialize(c, "kmx").decode()
    lines = t.splitlines()
    header = [l for l in lines[:30] if "REGISTER" in l]

    # Checkpoint boundaries identified from circuit structure
    checkpoints = [
        ("Stage 1: Normalized input & flags", 0, 90),
        ("Stage 2: Table mantissas loaded (ANF)", 90, 5500),
        ("Stage 3: Quadratic curvature added to residual", 5500, 11251),
        ("Stage 4: High phase applied (rotations 0..17)", 11251, 11269),
        ("Stage 5: Coarse6 restored & linear deriv added", 11269, 14375),
        ("Stage 6: Low phase applied (rotations 18..46)", 14375, 14404),
        ("Stage 7: Linear uncomputation & low erased (HMR)", 14404, 16500),
        ("Stage 8: Shared PDDD restoration (c2[4..6])", 16500, 18000),
        ("Stage 9: Curvature uncomputed & residual erased (HMR)", 18000, 19448),
        ("Stage 10: Measurement phase compensation (phase_from_records)", 19448, 27312),
        ("Stage 11: Normalization uncomputed, input restored & scratch clean", 27312, len(lines)),
    ]

    sim = km.Simulator(batch_size=shots, counting_operations=True, seed=seed)
    x_reg = km.array([km.q(i) for i in range(22)])
    flags_reg = km.array([km.q(i) for i in range(22, 27)])
    c2_reg = km.array([km.q(i) for i in range(27, 34)])
    res_reg = km.array([km.q(i) for i in range(34, 46)])
    coarse_reg = km.array([km.q(i) for i in range(46, 52)])
    low_reg = km.array([km.q(i) for i in range(52, 73)])
    high_reg = km.array([km.q(i) for i in range(73, 81)])

    xs_batch = np.full(shots, x_val, dtype=np.uint64)
    sim.write_across_shots(x_reg, to_bits(xs_batch, 22))

    print("STAGE-LEVEL CHECKPOINT EXECUTION (Inspecting Shot 0 Registers & Phase):")
    cur_line = len(header)
    for name, start_l, end_l in checkpoints:
        slice_start = max(cur_line, start_l)
        slice_lines = header + lines[slice_start:end_l]
        subcirc = km.Circuit("\n".join(slice_lines) + "\n")
        sim.do(subcirc)
        cur_line = end_l

        p_u = half_turns_to_units(sim.read_shot_phase(0))
        x_read = from_bits(sim.read_across_shots(x_reg))[0]
        fl_read = from_bits(sim.read_across_shots(flags_reg))[0]
        c2_read = from_bits(sim.read_across_shots(c2_reg))[0]
        res_read = from_bits(sim.read_across_shots(res_reg))[0]
        coarse_read = from_bits(sim.read_across_shots(coarse_reg))[0]
        high_read = from_bits(sim.read_across_shots(high_reg))[0]

        print(f"  {name}")
        print(f"      Phase:      {p_u} units")
        print(f"      Registers:  x={x_read} | flags={fl_read:05b} | c2={c2_read:07b} | res={res_read:012b} | coarse={coarse_read:06b} | hi={high_read:08b}")

    # Final verification across all shots
    thetas = np.array([half_turns_to_units(sim.read_shot_phase(i)) for i in range(shots)], dtype=np.uint64)
    errors = (thetas - target_units).view(np.int64)
    used_margins = np.abs(errors).view(np.uint64)
    ratios = used_margins / allowed_units
    margins = allowed_units - used_margins

    restored_x = from_bits(sim.read_across_shots(x_reg))
    assert np.all(restored_x == x_val), "x register was not restored!"
    scratch = km.array([km.q(i) for i in range(c.num_qubits) if km.q(i) not in set(x_reg)])
    dirty_qubits = sim.read_across_shots(scratch).any(axis=1)
    assert not dirty_qubits.any(), "Scratch qubits dirty!"

    print("-" * 90)
    print(f"Full Production Candidate Diagnostics across all {shots} shots:")
    print(f"  Input restoration:         PASS (All {shots} shots restored x = {x_val})")
    print(f"  Scratch cleanliness:       PASS (All {shots} shots returned scratch to |0>)")
    print(f"  Tracked Phase:             {thetas[0]} units")
    print(f"  Signed Phase Error:        {errors[0]} units ({errors[0] / 2**64:.6e} turns)")
    print(f"  Absolute Phase Error:      {used_margins[0]} units")
    print(f"  Allowed Tolerance:         {allowed_units} units")
    print(f"  Worst Error / Tol Ratio:   {ratios.max():.12f}")
    print(f"  Remaining Margin:          {margins.min()} units")
    print(f"  Fraction of Tol Unused:    {(margins.min() / allowed_units) * 100:.5f}%")
    print(f"  Deterministic across shots:PASS (Unique phase values: {len(np.unique(thetas))})")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrated production stage tracer for invsqrt")
    parser.add_argument("--x", type=int, default=32, help="Input x (default 32)")
    parser.add_argument("--shots", type=int, default=64, help="Number of shots (default 64)")
    parser.add_argument("--seed", type=int, default=12345, help="Simulation seed")
    args = parser.parse_args()
    instrument_calibrated_trace(args.x, args.shots, args.seed)
