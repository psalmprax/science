#!/dev/shm/qvcc-venv313/bin/python
"""Diagnostic Stage-Level Instrumentation Harness for Benchmark I (InvSqrt).

Traces intermediate register states at every algorithmic boundary for x=32
(or any arbitrary input x) without full-state-vector explosion.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import math
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/invsqrt"))

import kickmix as km
import signed_digit_invsqrt_v3 as core
import scheduled_arithmetic90 as H
from schedule_support90 import optimized_shift, mobius_transform
from verify_submission_invsqrt import (
    ProblemSpecInvSqrt,
    reference_values,
    validate_registers,
    half_turns_to_units,
    to_bits,
    from_bits,
)
import test_shared_pddd_circuit as builder


def instrument_trace(x_val: int = 32, seed: int = 12345, shots: int = 64):
    print("=" * 80)
    print(f"STAGE-LEVEL DIAGNOSTIC TRACE FOR x = {x_val} ({shots} shots, seed={seed})")
    print("=" * 80)

    spec = ProblemSpecInvSqrt.for_n_p(10, shots_per_x=shots)
    target, tol = reference_values(spec, np.array([x_val], dtype=np.uint64))
    allowed_units = int(tol[0])
    target_units = int(target[0])

    print(f"Published Contract Reference at x={x_val}:")
    print(f"  Target Phase:       {target_units} (mod 2^64) [{target_units / 2**64:.12f} turns]")
    print(f"  Allowed Tolerance:  {allowed_units} units [{allowed_units / 2**64:.12e} turns]")
    exact_math_tol = (2**-24 / math.sqrt(x_val)) * (2**64)
    print(f"  Exact Math Tol:     {exact_math_tol:.2f} units (difference: {exact_math_tol - allowed_units:.4f})")
    print("-" * 80)

    cfg, tables, _ = core.load_or_build_tables()
    lay = H.layout(cfg, tables)
    split, bounds = H.prepare_derivative_split(H, cfg, tables)

    # Build circuit with tap points
    b = km.CircuitBuilder()
    x = b.create_quantum_register(22, name="x")
    flags = b.alloc_qubits(5)

    # 1. Normalization
    H.normalize(b, x, flags)
    b.x(x[-1])

    c2, residual, coarse, low, high = [b.alloc_qubits(n) for n in [cfg.w2, 12, 6, 21, 8]]
    full_target = km.array(list(low) + list(residual) + list(c2) + list(high) + list(coarse))
    full_table = [0] * 256
    for (e, a), (k2, _, _) in tables.items():
        key = (e, a)
        idx = a + 128 * e
        common = lay["lo"][key] | split["residual"][key] << 21 | k2 << 33
        full_table[idx] = (
            common
            | lay["hi"][key] << (33 + cfg.w2)
            | split["coarse"][key] << (41 + cfg.w2)
        )

    address = km.array(list(x[14:21]) + [flags[0]])
    delta = x[:14]

    # 2. Mobius Table Lookup / ANF Initialization
    builder.init_anf(b, full_table, address, full_target, borrowed=(x[-1],))

    extra_borrow = []
    def adder(a, t, cin=None, sub=False):
        H.folded_gadd(b, a, t, cin, sub, borrowed=(x[-1], *extra_borrow))

    def inner(inverse):
        rows = cfg.rrow_list()
        for i in reversed(rows) if inverse else rows:
            shift = cfg.sh2[i]
            if shift >= 0:
                addend = c2[shift:]
                target_reg = residual
                carry = c2[shift - 1] if shift else None
            else:
                addend = c2
                target_reg = residual[-shift:]
                carry = None
            b.cx(delta[i], target_reg)
            adder(addend, target_reg, carry, sub=inverse)
            b.cx(delta[i], target_reg)

    # 3. Piecewise Quadratic Curvature Addition
    b.x(high[-1])
    extra_borrow.append(high[-1])
    inner(False)
    extra_borrow.clear()
    b.x(high[-1])

    # 4. Curvature erase
    checkpoint_start = 4
    for j in range(checkpoint_start, 6):
        b.cx(c2[j + 1], c2[j])
    erased = km.array(list(c2[checkpoint_start:]))
    checkpoint_table = [
        (tables[(i // 128, i % 128)][0] ^ (tables[(i // 128, i % 128)][0] >> 1)) >> checkpoint_start
        for i in range(256)
    ]
    measured = []
    def erase(reg, table):
        ms = b.alloc_bits(len(reg))
        b.hmr(reg, ms)
        measured.append((list(ms), list(table)))
        return list(ms)

    checkpoint_ms = erase(erased, checkpoint_table)
    b.free(erased)

    # 5. High Phase Application
    def phase(reg, low_bit):
        ext = b.alloc_qubits(9)
        word = km.array(list(reg) + list(ext) + [x[-1]])
        optimized_shift(b, flags, word, len(reg), False, prune=True)
        for i, q in enumerate(word):
            exponent = low_bit + i - cfg.fy - 10
            if exponent < 0:
                b.z_pow(q, Fraction(2) ** (exponent + 1))
        optimized_shift(b, flags, word, len(reg), True, prune=True)
        b.free(ext)

    phase(high, 20)
    high_table = [lay["hi"][(i // 128, i % 128)] for i in range(256)]
    high_ms = erase(high, high_table)
    b.free(high)

    # 6. Coarse restoration
    new = b.alloc_qubits(1)
    coarse = km.array(list(coarse) + list(new))
    cv = np.array([split["coarse"][(i // 128, i % 128)] for i in range(256)], dtype=np.int64)
    c6_vals = ((cv ^ (cv >> 1)) >> 6) & 1
    p_c6 = builder.boolean_plan(c6_vals, order=(7, 6, 5, 3, 0, 1, 2, 4), polarity=144)
    builder.restore_boolean(b, p_c6, address, coarse[6])

    adder(residual[11:], coarse)
    derivative = residual[:11] + coarse

    # 7. Linear Derivative Addition
    def outer(deriv, inverse):
        for i in range(14) if inverse else reversed(range(14)):
            shift = cfg.sh1[i]
            carry = deriv[shift - 1] if shift else None
            b.cx(delta[i], low)
            adder(deriv[shift:], low, carry, sub=inverse)
            b.cx(delta[i], low)

    outer(derivative, False)

    # 8. Low Phase Application
    phase(low[cfg.gy:], cfg.gy)

    # 9. Linear Derivative Uncomputation
    outer(derivative, True)
    low_table = [lay["lo"][(i // 128, i % 128)] for i in range(256)]
    low_ms = erase(low, low_table)
    b.free(low)

    # 10. Shared PDDD Curvature Restoration (curve4, curve5, curve6)
    shared_cands = builder.json.loads(open("/tmp/shared_davio_results.json").read())["candidates"]
    cand0 = shared_cands[0]
    cv_curve = np.array([tables[(i // 128, i % 128)][0] for i in range(256)], dtype=np.int64)
    gray_curve = cv_curve ^ (cv_curve >> 1)
    shared_plan = builder.build_shared_plan(
        {"curve4": (gray_curve >> 4) & 1, "curve5": (gray_curve >> 5) & 1, "curve6": (gray_curve >> 6) & 1},
        order=cand0["order"],
        polarity=cand0["polarity"],
    )
    replacement = b.alloc_qubits(3)
    c2 = km.array(list(c2[:checkpoint_start]) + list(replacement))
    targets_dict = {"curve4": c2[4], "curve5": c2[5], "curve6": c2[6]}
    builder.restore_shared_boolean(b, shared_plan, address, targets_dict)

    for j in reversed(range(checkpoint_start, 6)):
        b.cx(c2[j + 1], c2[j])

    # 11. Quadratic Curvature Uncomputation
    adder(residual[11:], coarse, sub=True)
    inner(True)

    # 12. Residual Target Uncomputation
    rest_table = [
        split["residual"][(i // 128, i % 128)]
        | tables[(i // 128, i % 128)][0] << 12
        | split["coarse"][(i // 128, i % 128)] << (12 + cfg.w2)
        for i in range(256)
    ]
    rest_target = km.array(list(residual) + list(c2) + list(coarse))
    rest_ms = erase(rest_target, rest_table)
    b.free(rest_target)

    # 13. Measurement Record Phase Compensation
    for k, m in enumerate(checkpoint_ms):
        j = checkpoint_start + k
        b.cx(m, rest_ms[12 + j])
        if j < 6:
            b.cx(m, rest_ms[12 + j + 1])
    records = low_ms + rest_ms[:12 + cfg.w2] + high_ms + rest_ms[12 + cfg.w2:]
    builder.phase_from_records(b, full_table, address, records)

    # 14. Normalization Uncomputation
    b.x(x[-1])
    H.unnormalize(b, x, flags)
    b.free(flags)

    circuit = b.finish_circuit()

    # Now simulate across the requested shots
    sim = km.Simulator(batch_size=shots, debug_print=False, counting_operations=True, seed=seed)
    xs_batch = np.full(shots, x_val, dtype=np.uint64)
    sim.write_across_shots(x, to_bits(xs_batch, 22))
    sim.do(circuit)

    # Check register restoration
    restored_x = from_bits(sim.read_across_shots(x))
    assert np.all(restored_x == x_val), "x register was not restored!"
    scratch = km.array([km.q(i) for i in range(circuit.num_qubits) if km.q(i) not in set(x)])
    dirty_qubits = sim.read_across_shots(scratch).any(axis=1)
    assert not dirty_qubits.any(), "Scratch qubits dirty!"

    thetas = np.array([half_turns_to_units(sim.read_shot_phase(i)) for i in range(shots)], dtype=np.uint64)
    errors = (thetas - target_units).view(np.int64)
    used_margins = np.abs(errors).view(np.uint64)
    ratios = used_margins / allowed_units
    margins = allowed_units - used_margins

    print("Stage 1-14 Execution & Simulation Results across shots:")
    print(f"  All {shots} shots restored x = {x_val}:  PASS")
    print(f"  All {shots} shots returned scratch to |0>: PASS")
    print(f"  Tracked Phase (shot 0):  {thetas[0]} units")
    print(f"  Signed Error (shot 0):   {errors[0]} units ({errors[0] / 2**64:.6e} turns)")
    print(f"  Absolute Phase Error:    {used_margins[0]} units")
    print(f"  Tolerance Allowed:       {allowed_units} units")
    print(f"  Worst Error/Tol Ratio:   {ratios.max():.12f}")
    print(f"  Unused Margin:           {margins.min()} units")
    print(f"  Fraction of Tol Unused:  {(margins.min() / allowed_units) * 100:.5f}%")
    print(f"  Rotation epsilon:        {math.pi * margins.min() / 2**64 / 44:.16e}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=32)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    instrument_trace(args.x, args.seed, args.shots)
