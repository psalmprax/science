#!/usr/bin/env python3
"""Empirical Test of the Viable Optimization Path: Cross-Register Shared PDDD vs Wire-Lifetime Constraints.

This script measures, rather than asserts:
1. Exact Boolean node count of the joint 4-output {coarse6, curve4, curve5, curve6} PDDD
   against the incumbent's disjoint {11 (coarse6) + 28 (curve4..6)} node count.
2. Two-phase DAG decomposition (Phase 1: coarse6 before outer; Phase 2: curves after outer).
3. Peak logical qubit allocation, measured by building both schedules to the same point,
   under monolithic vs two-phase evaluation.
4. Score impact, computed from the measured node counts, the measured peak widths, and the
   pinned 89Q receipt baseline (Q, T_mean, toffoli_equivalent, official score).

Every printed number in Step 4 is derived from a measured value or from the receipt; there
are no typed-in results. The peak widths are structural counts of the built schedule (the
circuit is not run through the official verifier here).
"""
from __future__ import annotations

import json
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
from schedule_support90 import optimized_shift
from fractions import Fraction
import test_shared_pddd_circuit as builder

RECEIPT_PATH = REPO_ROOT / "artifacts/quantum_compilation/invsqrt/invsqrt_89q_shared_pddd_full_receipt.json"
# src/ holds the tracked copy (artifacts/ is gitignored, so a clone only has this one).
SHARED_PDDD_PATH = next(
    (
        p
        for p in (
            REPO_ROOT / "src/shared_davio_results.json",
            REPO_ROOT / "artifacts/quantum_compilation/invsqrt/shared_davio_results.json",
        )
        if p.exists()
    ),
    REPO_ROOT / "src/shared_davio_results.json",
)
COARSE_CHOICES_PATH = REPO_ROOT / "src/boolean_choices90.json"


def run_test():
    print("=" * 80)
    print("EMPIRICAL TEST OF THE VIABLE OPTIMIZATION PATH")
    print("=" * 80)

    cfg, tables, _ = core.load_or_build_tables()
    lay = H.layout(cfg, tables)
    split, _bounds = H.prepare_derivative_split(H, cfg, tables)

    cv_curve = np.array([tables[(i // 128, i % 128)][0] for i in range(256)], dtype=np.int64)
    gray_curve = cv_curve ^ (cv_curve >> 1)
    c4 = (gray_curve >> 4) & 1
    c5 = (gray_curve >> 5) & 1
    c6 = (gray_curve >> 6) & 1

    cv = np.array([split["coarse"][(i // 128, i % 128)] for i in range(256)], dtype=np.int64)
    c6_coarse = ((cv ^ (cv >> 1)) >> 6) & 1

    # Per-output parameters used by the incumbent build (11-node coarse6 and 28-node
    # curve4..6 plan). Read from the files the champion build reads, never hardcoded.
    coarse_choice = json.loads(COARSE_CHOICES_PATH.read_text())["coarse6"]
    c6_order = tuple(coarse_choice["order"])
    c6_pol = int(coarse_choice["polarity"])
    curve_choice = json.loads(SHARED_PDDD_PATH.read_text())["candidates"][0]
    curve_order = tuple(curve_choice["order"])
    curve_pol = int(curve_choice["polarity"])

    print("Step 1: Evaluating Boolean Structure of Joint 4-Output PDDD...")
    order_4 = (7, 4, 3, 2, 1, 0, 6, 5)
    polarity_4 = 32
    four_plan = builder.build_shared_plan(
        {"coarse6": c6_coarse, "curve4": c4, "curve5": c5, "curve6": c6},
        order=order_4,
        polarity=polarity_4,
    )
    n_joint = len(four_plan["nodes"])

    plan_c6_alone = builder.boolean_plan(c6_coarse, order=c6_order, polarity=c6_pol)
    plan_curves_alone = builder.build_shared_plan(
        {"curve4": c4, "curve5": c5, "curve6": c6}, order=curve_order, polarity=curve_pol
    )
    n_c6_alone = len(plan_c6_alone["nodes"])
    n_curves_alone = len(plan_curves_alone["nodes"])
    n_incumbent = n_c6_alone + n_curves_alone

    print(f"  Joint 4-output plan (order={order_4}, polarity={polarity_4}): {n_joint} nodes")
    print(f"  Incumbent disjoint plan: {n_c6_alone} (coarse6) + {n_curves_alone} (curve4..6) = {n_incumbent} nodes")
    print(f"  Measured node difference: {n_incumbent} - {n_joint} = {n_incumbent - n_joint} nodes")
    print(f"  Toffoli difference at 2 per node: 2 * {n_incumbent - n_joint} = {2 * (n_incumbent - n_joint)}")

    print("\nStep 2: Testing Phase 1 and Phase 2 Sub-DAGs Without Live Ancilla Retention...")
    plan_p1 = builder.build_shared_plan({"coarse6": c6_coarse}, order=order_4, polarity=polarity_4)
    plan_p2 = builder.build_shared_plan(
        {"curve4": c4, "curve5": c5, "curve6": c6}, order=order_4, polarity=polarity_4
    )
    n_p1 = len(plan_p1["nodes"])
    n_p2 = len(plan_p2["nodes"])
    print(f"  Phase 1 (coarse6 under joint order): {n_p1} nodes (single-output optimum is {n_c6_alone})")
    print(f"  Phase 2 (curves under joint order):  {n_p2} nodes (single-output optimum is {n_curves_alone})")
    print(f"  Sum of nodes without live sharing:   {n_p1 + n_p2} nodes")
    print(
        f"  Measured penalty for splitting the joint order: "
        f"({n_p1} + {n_p2}) - {n_incumbent} = {(n_p1 + n_p2) - n_incumbent} nodes "
        f"= {2 * ((n_p1 + n_p2) - n_incumbent)} Toffolis"
    )

    def build_partial(monolithic: bool) -> int:
        """Builds up to and including the low phase and returns the peak logical width.

        `monolithic` restores the joint 4-output plan (coarse6 + curve4..6) in one call
        before `outer`, so all of its node ancillas are live at the same time; the
        two-phase schedule restores coarse6 before `outer` and the curves afterwards,
        which is what the incumbent build does.
        """
        b = km.CircuitBuilder()
        x = b.create_quantum_register(22, name="x")
        flags = b.alloc_qubits(5)
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
                    target = residual
                    carry = c2[shift - 1] if shift else None
                else:
                    addend = c2
                    target = residual[-shift:]
                    carry = None
                b.cx(delta[i], target)
                adder(addend, target, carry, sub=inverse)
                b.cx(delta[i], target)

        b.x(high[-1])
        extra_borrow.append(high[-1])
        inner(False)
        extra_borrow.clear()
        b.x(high[-1])

        checkpoint_start = 4
        for j in range(checkpoint_start, 6):
            b.cx(c2[j + 1], c2[j])
        erased = km.array(list(c2[checkpoint_start:]))
        ms = b.alloc_bits(len(erased))
        b.hmr(erased, ms)
        b.free(erased)

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
        ms_h = b.alloc_bits(len(high))
        b.hmr(high, ms_h)
        b.free(high)

        new = b.alloc_qubits(1)
        coarse = km.array(list(coarse) + list(new))
        targets_dict = {"coarse6": coarse[6], "curve4": None, "curve5": None, "curve6": None}
        if monolithic:
            # One restore of all four outputs before `outer`: every node ancilla of the
            # joint DAG is live at the same point, including the three curve bits.
            replacement = b.alloc_qubits(3)
            c2 = km.array(list(c2[:checkpoint_start]) + list(replacement))
            targets_dict.update(
                {"curve4": c2[4], "curve5": c2[5], "curve6": c2[6]}
            )
            builder.restore_shared_boolean(b, four_plan, address, targets_dict)
        else:
            builder.restore_boolean(b, plan_c6_alone, address, coarse[6])

        adder(residual[11:], coarse)
        derivative = residual[:11] + coarse

        def outer(deriv, inverse):
            for i in range(14) if inverse else reversed(range(14)):
                shift = cfg.sh1[i]
                carry = deriv[shift - 1] if shift else None
                b.cx(delta[i], low)
                adder(deriv[shift:], low, carry, sub=inverse)
                b.cx(delta[i], low)

        outer(derivative, False)
        phase(low[cfg.gy:], cfg.gy)
        outer(derivative, True)

        low_ms = b.alloc_bits(len(low))
        b.hmr(low, low_ms)
        b.free(low)

        if monolithic:
            # init_and / del_and are involutions on the computational basis, so a second
            # pass uncomputes the restored bits at the same stage the two-phase schedule
            # uncomputes its curve restore.
            builder.restore_shared_boolean(b, four_plan, address, targets_dict)
        else:
            replacement = b.alloc_qubits(3)
            c2 = km.array(list(c2[:checkpoint_start]) + list(replacement))
            targets_dict.update(
                {"curve4": c2[4], "curve5": c2[5], "curve6": c2[6]}
            )
            builder.restore_shared_boolean(
                b, plan_curves_alone, address, {k: v for k, v in targets_dict.items() if k != "coarse6"}
            )

        return b.finish_circuit().num_qubits

    print("\nStep 3: Measuring Peak Logical Width of Each Schedule...")
    q_two_phase = build_partial(monolithic=False)
    q_monolithic = build_partial(monolithic=True)
    print(f"  Two-phase schedule (incumbent):   peak width = {q_two_phase} qubits")
    print(f"  Monolithic joint restore schedule: peak width = {q_monolithic} qubits")
    print(f"  Measured width difference:        {q_monolithic - q_two_phase:+d} qubits "
          f"({n_joint} joint nodes vs {n_c6_alone} coarse6 nodes live at the same point)")

    print("\nStep 4: Computing Score Impact from the Measured Counts...")
    receipt = json.loads(RECEIPT_PATH.read_text())
    q0 = int(receipt["num_qubits"])
    t0 = float(receipt["toffoli_count"])
    e0 = float(receipt["toffoli_equivalent"])
    s0 = float(receipt["score"])
    other = e0 - t0  # rotations / measurement / T-gate equivalent overhead
    if abs(q0 * e0 - s0) > 1e-3:
        raise SystemExit(
            f"receipt is inconsistent: {q0} * {e0} = {q0 * e0} but score = {s0}"
        )
    if q_two_phase != q0:
        raise SystemExit(
            f"rebuilt two-phase width {q_two_phase} does not reproduce the receipt width {q0}"
        )
    print(f"  Receipt baseline: Q = {q0}, T_mean = {t0:.6f}, toffoli_equivalent = {e0:.6f}")
    print(f"                    other (non-Toffoli) term = {other:.6f}, score = {s0:.5f}")
    print(f"  Score formula: Q * (T_mean + dT + {other:.6f}); 2 Toffolis saved/added per node.")

    def score(q: int, dt: float) -> float:
        return q * (t0 + dt + other)

    # Each case is (label, peak width, Toffoli delta) with both terms measured or derived
    # from measured node counts above.
    baseline_case = ("Incumbent (two-phase, 11 + 28 nodes)", q_two_phase, 0)
    dt_monolithic = -2 * (n_incumbent - n_joint)
    dt_two_phase = 2 * ((n_p1 + n_p2) - n_incumbent)
    cases = [
        baseline_case,
        ("Monolithic joint restore before outer", q_monolithic, dt_monolithic),
        ("Two-phase, joint order, no live sharing", q_two_phase, dt_two_phase),
    ]
    # Retaining k shared nodes across outer costs one wire each and saves 2 Toffolis each.
    for k in range(1, (n_incumbent - n_joint) + 1):
        cases.append((f"Two-phase + {k} shared node(s) kept live", q_two_phase + k, -2 * k))

    print(f"\n  {'Schedule':<44s} | {'Q':>4s} | {'dT':>7s} | {'T_mean':>11s} | {'Score':>14s} | {'dScore':>10s}")
    print("  " + "-" * 106)
    best_label, best_score = None, None
    for label, q, dt in cases:
        s = score(q, dt)
        marker = ""
        if best_score is None or s < best_score:
            best_label, best_score, marker = label, s, ""
        print(f"  {label:<44s} | {q:4d} | {dt:+7.1f} | {t0 + dt:11.3f} | {s:14.3f} | {s - s0:+10.3f}{marker}")

    print("\n" + "=" * 80)
    print("COMPUTED VERDICT")
    print("=" * 80)
    print(f"  Lowest computed score: {best_score:.3f} ({best_score - s0:+.3f} vs the {s0:.3f} receipt baseline)")
    print(f"  Best modelled schedule: {best_label}")
    if best_score >= s0 - 1e-9:
        print("  The receipt baseline is not improved by any schedule modelled here.")
    else:
        print(f"  This schedule is {s0 - best_score:.3f} better than the receipt baseline in the Q x C model.")
        print("  That is a model prediction only: it has not been compiled, verified, or scored by the")
        print("  official verifier, so it is not a result until it produces a passing receipt.")
    print()
    print("  Scope: these cases cover three rearrangements of the same 39/36-node restoration DAG.")
    print("  They are not a statement about the whole EEA schedule space, and no global optimality")
    print("  claim is made from them.")
    print("=" * 80)


if __name__ == "__main__":
    run_test()
