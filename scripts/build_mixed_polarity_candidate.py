#!/usr/bin/env python3
"""Builds and verifies the mixed-polarity two-phase candidate for the 89Q invsqrt circuit.

`scripts/search_mixed_polarity_pddd.py` finds a curve-restoration plan with 26 shared
nodes instead of 28, by letting each AND node choose its control literal (x_v or NOT
x_v) instead of forcing the positive literal. This script turns that plan into a
circuit, verifies it with the official verifier, and compares it against a rebuild of
the incumbent under identical settings and two shot budgets.

Why the extra budget: the score uses the *mean* executed Toffoli count, which depends
on mid-circuit measurement outcomes, so a small difference between two circuits can be
sampling noise. The comparison at 8 shots per x (204,800 verified shots) is there to
separate a real per-shot saving from jitter in the 25,600-shot estimate.

How per-node polarity is realised: `init_and(control, parity, target)` computes
control AND parity, so a negative-polarity node is built by flipping the control's
address lane around the init_and/del_and pair. X gates are not Toffolis, so the score
does not pay for them; the lane is restored before any later read of it.

The incumbent build (`scripts/test_shared_pddd_circuit.py`) is untouched: this script
swaps the restore helper on its module object for the duration of each build, so the
shipped champion's code path and hash stay reproducible.
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
import portable_exact_domain as domain_compiler
import portable_calibration as cal
import upstream_functions as uf
from verify_submission_invsqrt import verify, ProblemSpecInvSqrt

import test_shared_pddd_circuit as B
from search_mixed_polarity_pddd import plan_mixed, permuted_table

# Found by scripts/search_mixed_polarity_pddd.py (section 4): 26 nodes vs 28.
CURVE_ORDER = (4, 7, 2, 1, 0, 3, 6, 5)
CURVE_POLARITY = 99
CURVE_PATTERNS = {"curve4": 0, "curve5": 96, "curve6": 40}

ORIGINAL_RESTORE = B.restore_shared_boolean


def restore_shared_polarity(b, plan, address, targets_dict):
    """`B.restore_shared_boolean` with a per-node control literal."""
    address = list(address)
    n = len(address)
    order = plan["order"]
    pol = plan["polarity"]
    flipped = [address[j] for j in range(n) if (pol >> j) & 1]
    if flipped:
        b.x(km.array(flipped))
    wires = [address[j] for j in order]
    nodes = list(b.alloc_qubits(len(plan["nodes"])))
    parity = b.alloc_qubits(1)[0]
    wires += nodes

    def form(f):
        if f[0]:
            b.x(parity)
        for j in sorted(f[1]):
            b.cx(wires[j], parity)

    def node_op(index: int, *, adjoint: bool):
        variable, negative, right = plan["nodes"][index]
        lane = wires[variable]
        form(right)
        if negative:
            b.x(lane)
        if adjoint:
            b.del_and(lane, parity, nodes[index])
        else:
            b.init_and(lane, parity, nodes[index])
        if negative:
            b.x(lane)
        form(right)

    for i in range(len(plan["nodes"])):
        node_op(i, adjoint=False)

    for name, target in targets_dict.items():
        constant, terms = plan["results"][name]
        if constant:
            b.x(target)
        for term in sorted(terms):
            b.cx(wires[term], target)

    for i in reversed(range(len(plan["nodes"]))):
        node_op(i, adjoint=True)

    b.free(km.array(nodes))
    b.free(parity)
    if flipped:
        b.x(km.array(flipped))


def build(mixed_plan, label: str):
    """Builds, compiles and calibrates one circuit. `mixed_plan=None` is the incumbent."""
    cfg, tables, _ = core.load_or_build_tables()
    c6_choice = json.loads((REPO_ROOT / "src/boolean_choices90.json").read_text())["coarse6"]

    B.restore_shared_boolean = restore_shared_polarity if mixed_plan is not None else ORIGINAL_RESTORE
    if mixed_plan is None:
        # Rebuild the incumbent exactly as scripts/test_shared_pddd_circuit.py does.
        choice = json.loads((REPO_ROOT / "src/shared_davio_results.json").read_text())["candidates"][0]
        c4, c5, c6 = _curve_functions(tables)
        raw, meta = B.build_candidate(
            cfg,
            tables,
            coarse_choice=c6_choice,
            shared_curve_plan=B.build_shared_plan(
                {"curve4": c4, "curve5": c5, "curve6": c6},
                order=tuple(choice["order"]),
                polarity=int(choice["polarity"]),
            ),
        )
    else:
        raw, meta = B.build_candidate(
            cfg, tables, coarse_choice=c6_choice, shared_curve_plan=mixed_plan
        )
    B.restore_shared_boolean = ORIGINAL_RESTORE

    opt, comp = domain_compiler.optimize_circuit(raw, linear=True)
    st = opt.operation_statistics()

    fixed_deltas = json.loads((REPO_ROOT / "src/invsqrt_calibration.json").read_text())
    text = cal._serialize(opt, "kmx").decode()
    lines, rots = cal.parse_kmx(text)
    values = fixed_deltas.get("deltasPhaseUnits", [])
    dropped = set(fixed_deltas.get("droppedOriginalRotationIndices", [18]))
    rewritten = uf.rewrite(
        lines, rots, {j: int(v) for j, v in enumerate(values) if j not in dropped}, dropped
    )
    cand = km.Circuit(rewritten)
    st_final = cand.operation_statistics()
    print(
        f"[{label}] raw Toffolis={meta['static_toffoli']} -> compiled={st.num_toffoli_like} "
        f"({len(comp.replacements)} replacements) -> calibrated={st_final.num_toffoli_like}, "
        f"Q={cand.num_qubits}, rotations={st_final.num_z_pow(min_clifford_hierarchy_level=4)}"
    )
    return cand


def _curve_functions(tables):
    cv_curve = np.array([tables[(i // 128, i % 128)][0] for i in range(256)], dtype=np.int64)
    gray = cv_curve ^ (cv_curve >> 1)
    return [(gray >> 4) & 1, (gray >> 5) & 1, (gray >> 6) & 1]


def verify_at(cand, shots_per_x: int):
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


def main() -> int:
    cfg, tables, _ = core.load_or_build_tables()
    functions = {
        "curve4": _curve_functions(tables)[0],
        "curve5": _curve_functions(tables)[1],
        "curve6": _curve_functions(tables)[2],
    }
    permuted = {k: permuted_table(v, CURVE_ORDER, CURVE_POLARITY) for k, v in functions.items()}
    plan = plan_mixed(permuted, CURVE_ORDER, CURVE_PATTERNS, polarity=CURVE_POLARITY)
    print(
        f"curve restore plan: {len(plan['nodes'])} nodes (incumbent 28), "
        f"order={CURVE_ORDER} polarity={CURVE_POLARITY}, "
        f"negative-polarity nodes={sum(1 for _, p, _ in plan['nodes'] if p)}"
    )
    if len(plan["nodes"]) != 26:
        raise SystemExit(f"expected 26 nodes, plan has {len(plan['nodes'])}")

    incumbent = build(None, "incumbent")
    candidate = build(plan, "mixed-polarity")

    results = {}
    for shots in (1, 8):
        print(f"\nverifying both at {25600 * shots:,} shots ({shots} per x)...")
        a = verify_at(incumbent, shots)
        b = verify_at(candidate, shots)
        results[shots] = (a, b)
        print(
            f"  incumbent: {a['status']}  score={a['score']:.5f}  "
            f"T_mean={a['toffoli_count']:.6f}  static={a['max_toffoli_count']}  "
            f"err/tol={a['worst_error_over_tolerance']:.13f}"
        )
        print(
            f"  candidate: {b['status']}  score={b['score']:.5f}  "
            f"T_mean={b['toffoli_count']:.6f}  static={b['max_toffoli_count']}  "
            f"err/tol={b['worst_error_over_tolerance']:.13f}"
        )
        print(f"  candidate - incumbent: {b['score'] - a['score']:+.5f} score units, "
              f"T_mean {b['toffoli_count'] - a['toffoli_count']:+.6f}")

    a1, b1 = results[1]
    a8, b8 = results[8]
    print("\n" + "=" * 74)
    print("READING OF THE RESULT")
    print("=" * 74)
    if a1["max_toffoli_count"] == b1["max_toffoli_count"]:
        print(f"  Static Toffoli count is IDENTICAL ({a1['max_toffoli_count']} both): the exact")
        print("  domain compiler's linear pass removes two more Toffolis from the 28-node")
        print("  plan than from the 26-node one, so the node saving does not reach the")
        print("  static gate count.")
    print(f"  T_mean delta at 25,600 shots:     {b1['toffoli_count'] - a1['toffoli_count']:+.6f}")
    print(f"  T_mean delta at 204,800 shots:    {b8['toffoli_count'] - a8['toffoli_count']:+.6f}")
    print(f"  score delta at 204,800 shots:     {b8['score'] - a8['score']:+.5f}")
    print("  A saving that shrinks toward zero as the sample grows is sampling noise in")
    print("  the mean, not a per-shot gate saving.")
    print("=" * 74)

    out = Path("/tmp/invsqrt_89q_mixed_polarity.kmb")
    candidate.to_file(str(out), format="kmb")
    print(f"written: {out}")
    return 0 if all(r["status"] == "PASS" for pair in results.values() for r in pair) else 1


if __name__ == "__main__":
    raise SystemExit(main())
