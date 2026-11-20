#!/usr/bin/env python3
"""Prototype: a bit-serial square root that phases each digit as it appears.

Status: experimental

The 89-qubit champion materialises the whole answer word before phasing it: its
peak is set by an `adder` allocating carries on top of the 21-bit `low` and
8-bit `high` registers, 81 live qubits in total. The 62-qubit public circuit
never holds a word -- it produces one digit at a time, phases it at its own bit
weight, and reuses the qubit. This script builds that schedule in miniature so
the claim can be checked rather than believed.

It computes `r = floor(sqrt(w))` for `w < 2^(2k)` one digit at a time:

  * digit `i` (MSB first) is a Boolean function of the top `2(k-i)` bits of `w`
    -- measured, not assumed, and true for every `k` tried;
  * the digit is restored into a single reusable qubit with the shared-PDDD
    machinery, phased by `2^(i-k)` turns, and restored again, which returns the
    qubit to `|0>` for the next digit;
  * so the workspace is one digit qubit plus the plan's transient nodes, and no
    part of the answer is ever held as a word.

`build_materialised` is the same arithmetic in the champion's shape: all `k`
digits live in a register, phased together, then uncomputed. Comparing the two
is the point of the script -- it isolates the schedule from the arithmetic.

Every circuit is checked against the exact phase with the competition's own
simulator, so a pass is a statement about the circuit and not about this model.

Usage:
    APEDP_Q_KICKMIX_PYTHON=... .venv-kickmix/bin/python scripts/prototype_bitserial_sqrt.py
"""
from __future__ import annotations

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

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

import numpy as np  # noqa: E402

import kickmix as km  # noqa: E402
from test_shared_pddd_circuit import boolean_plan, restore_boolean  # noqa: E402
from verify_submission_invsqrt import half_turns_to_units  # noqa: E402

BATCH = 64


def digit_table(k: int, i: int) -> tuple[np.ndarray, int]:
    """`table[hi] = bit i of floor(sqrt(w))` for the top `m = 2(k-i)` bits of `w`.

    The digit is a function of that prefix alone (checked by `digit_prefix_dependence`),
    so the table is well defined: every `w` with the same prefix gets the same digit.
    """
    m = 2 * (k - i)
    n = 2 * k
    hi = np.arange(1 << m, dtype=np.int64)
    w = hi << (n - m)
    # Round the prefix down to its first representative; the digit is constant
    # across the block, which is what makes the prefix address sufficient.
    table = np.array([(math.isqrt(int(v)) >> i) & 1 for v in w], dtype=np.int64)
    return table, m


def digit_prefix_dependence(k: int) -> list[int]:
    """The measured prefix length each digit actually needs."""
    n = 2 * k
    w = np.arange(1 << n, dtype=np.int64)
    r = np.array([math.isqrt(int(v)) for v in w], dtype=np.int64)
    out = []
    for i in reversed(range(k)):
        bit = (r >> i) & 1
        needed = n
        for m in range(0, n + 1):
            prefix = (w >> (n - m)) if m else np.zeros_like(w)
            seen: dict[int, int] = {}
            ok = True
            for p, d in zip(prefix.tolist(), bit.tolist()):
                if seen.setdefault(p, d) != d:
                    ok = False
                    break
            if ok:
                needed = m
                break
        out.append(needed)
    return out


def build_bitserial(k: int) -> tuple[km.Circuit, dict]:
    """One digit qubit, reused: compute -> phase -> uncompute, MSB first."""
    n = 2 * k
    b = km.CircuitBuilder()
    x = b.create_quantum_register(n, name="x")
    q = b.alloc_qubits(1)[0]
    for i in reversed(range(k)):
        table, m = digit_table(k, i)
        plan = boolean_plan(list(table), order=tuple(range(m)), polarity=0)
        addr = list(x[n - m :])
        restore_boolean(b, plan, addr, q)
        b.z_pow(q, Fraction(2) ** (i - k + 1))  # exponent in half turns
        restore_boolean(b, plan, addr, q)  # self-inverse: q returns to |0>
    circuit = b.finish_circuit()
    return circuit, {"qubits": circuit.num_qubits,
                     "toffoli": int(circuit.operation_statistics().num_toffoli_like),
                     "rotations": int(circuit.operation_statistics().num_z_pow(min_clifford_hierarchy_level=4))}


def build_materialised(k: int) -> tuple[km.Circuit, dict]:
    """The champion's shape: hold every digit in a register, phase, then erase."""
    n = 2 * k
    b = km.CircuitBuilder()
    x = b.create_quantum_register(n, name="x")
    y = b.alloc_qubits(k)
    for i in reversed(range(k)):
        table, m = digit_table(k, i)
        plan = boolean_plan(list(table), order=tuple(range(m)), polarity=0)
        restore_boolean(b, plan, list(x[n - m :]), y[i])
    for i in range(k):
        b.z_pow(y[i], Fraction(2) ** (i - k + 1))
    for i in reversed(range(k)):
        table, m = digit_table(k, i)
        plan = boolean_plan(list(table), order=tuple(range(m)), polarity=0)
        restore_boolean(b, plan, list(x[n - m :]), y[i])
    circuit = b.finish_circuit()
    return circuit, {"qubits": circuit.num_qubits,
                     "toffoli": int(circuit.operation_statistics().num_toffoli_like),
                     "rotations": int(circuit.operation_statistics().num_z_pow(min_clifford_hierarchy_level=4))}


def verify(circuit: km.Circuit, k: int) -> dict:
    """Compare every shot's tracked phase with `r / 2^k` turns, exactly."""
    n = 2 * k
    x_reg = km.array(circuit.register_data["x"])
    scratch = km.array([km.q(j) for j in range(circuit.num_qubits) if km.q(j) not in set(x_reg)])
    total = 1 << n
    sim = km.Simulator(batch_size=BATCH, debug_print=False,
                       counting_operations=True, seed=20261103)
    worst = 0.0
    for start in range(0, total, BATCH):
        xs = np.arange(start, start + BATCH, dtype=np.uint64)
        bits = ((xs[None, :] >> np.arange(n, dtype=np.uint64)[:, None]) & 1) == 1
        sim.clear_for_shot()
        sim.write_across_shots(x_reg, bits)
        sim.do(circuit)
        if not np.array_equal(sim.read_across_shots(x_reg), bits):
            raise AssertionError(f"x register not restored at w={start}")
        if len(scratch) and sim.read_across_shots(scratch).any():
            raise AssertionError(f"scratch not clean at w={start}")
        r = np.array([math.isqrt(int(v)) for v in xs], dtype=np.int64)
        # The simulator tracks phase in half turns, so the expected value is
        # 2 * (r / 2^k) -- the digit weights sum to r / 2^k turns.
        expected = [half_turns_to_units(Fraction(2 * int(v), 1 << k)) for v in r]
        for j in range(BATCH):
            got = half_turns_to_units(sim.read_shot_phase(j))
            diff = (int(got) - int(expected[j])) % (1 << 64)
            if diff > (1 << 63):
                diff -= 1 << 64
            worst = max(worst, abs(diff) / 2**64)
    return {"inputs_checked": total, "worst_phase_error_turns": worst}


def main() -> int:
    print(f"{'k':>2} {'w bits':>7} {'digits ask for':>16} "
          f"{'Q ser':>6} {'Q word':>7} {'T ser':>7} {'T word':>7}  {'phase check':>18}")
    rows = []
    # k <= 6 is checked against the exact phase (feasible shot budget); k >= 7 measures
    # the width and gate cost only, which is what decides whether the schedule scales.
    # `digit_prefix_dependence` enumerates every `w < 2^(2k)`, so it stops at k = 6: at
    # k = 11 that is 4.2M inputs times 11 digits of prefix search.
    for k in (3, 4, 5, 6, 7, 8, 9, 10, 11):
        need = digit_prefix_dependence(k) if k <= 6 else "not enumerated"
        ser, s_meta = build_bitserial(k)
        mat, m_meta = build_materialised(k)
        check = verify(ser, k) if k <= 5 else None
        control = verify(mat, k) if k <= 5 else None
        rows.append({"k": k, "prefix_bits": need,
                     "bitserial": s_meta, "materialised": m_meta,
                     "verification": check, "control": control})
        verdict = ("exact" if check and check["worst_phase_error_turns"] == 0.0
                   else f"{check['worst_phase_error_turns']:.3e}" if check else "not simulated")
        print(f"{k:2d} {2*k:7d} {str(need):>16} "
              f"{s_meta['qubits']:6d} {m_meta['qubits']:7d} "
              f"{s_meta['toffoli']:7d} {m_meta['toffoli']:7d}  {verdict:>18}")
    print()
    for row in rows:
        if row["verification"]:
            print(f"k={row['k']}: streaming  {row['verification']['inputs_checked']:5d} inputs, "
                  f"worst |phase error| = {row['verification']['worst_phase_error_turns']:.3e} turns")
            print(f"      materialised {row['control']['inputs_checked']:5d} inputs, "
                  f"worst |phase error| = {row['control']['worst_phase_error_turns']:.3e} turns")

    if "--json" in sys.argv:
        # Consumed by tests/test_dyn_bitserial_prototype.py, which pins the widths
        # and the exact-phase result so a schedule change cannot move them quietly.
        print(json.dumps(rows, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
