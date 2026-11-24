"""Emit the packed decoder's uncompute pass and total the three stages.

A reversible decoder has to give its syndrome qubits back. The reference pays for that with
an adjoint EEA pass (``dqi_gluzdov_v6.build_syndrome_decoding_mbuc`` runs the forward EEA,
the Forney phase, then the adjoint EEA). The packed update is a product of
``ixor_gf2_mul`` operations, each of which is its own inverse and none of which touches its
own operands, so its adjoint is the same gate multiset in reverse order -- the same
Toffolis. This script does not assume that: it emits the reversed stream and counts it.

Both directions are built and measured at three fields, then combined with the measured
Chien/Forney sweep (``scripts/emit_chien_forney_stage.py``) into a four-term total:

    key equation (forward) + coefficient sweep + Forney block + uncompute

    python3 scripts/emit_uncompute_pass.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
#: Point this at a rebuilt tree (``KICKMIX_PATH=/dev/shm/kmbuild``) to measure against an
#: extension other than the pinned one.
KICKMIX_PATH = os.environ.get("KICKMIX_PATH") or str(BUNDLE / "kickmix")
if KICKMIX_PATH not in sys.path:
    sys.path.insert(0, KICKMIX_PATH)

import kickmix as km  # noqa: E402

ARCHIVE_TOFFOLIS = 396919.33
FORNEY_PER_POSITION = {4: 18.0, 8: 108.0, 12: 315.0}  # measured by emit_chien_forney_stage
POSITIONS = 4095


def build_iteration(degree: int, t: int, *, adjoint: bool) -> tuple[km.Circuit, dict]:
    """One packed BM iteration, or its adjoint (the same gates in reverse order)."""
    n_cells = 2 * t + 3
    field = km.GF2Field(degree)
    builder = km.CircuitBuilder()

    array = builder.create_quantum_register(n_cells * degree, "bm_array")
    cells = [array[i * degree:(i + 1) * degree] for i in range(n_cells)]
    sigma = cells[:t + 1]
    b_poly = cells[t + 1:2 * t + 2]
    disc = cells[2 * t + 2]
    syndromes = [builder.create_quantum_register(degree, f"S{j}") for j in range(2 * t)]

    def discrepancy() -> None:
        for j in range(t + 1):
            builder.ixor_gf2_mul(sigma[j], syndromes[t - j], target=disc, field=field)

    def update() -> None:
        for j in range(1, t + 1):
            builder.ixor_gf2_mul(disc, b_poly[j - 1], target=sigma[j], field=field)

    if adjoint:
        # Reverse order, and each `ixor_gf2_mul` is its own inverse.
        for j in range(t, 0, -1):
            builder.ixor_gf2_mul(disc, b_poly[j - 1], target=sigma[j], field=field)
        for j in range(t, -1, -1):
            builder.ixor_gf2_mul(sigma[j], syndromes[t - j], target=disc, field=field)
    else:
        discrepancy()
        update()

    circuit = builder.finish_circuit()
    stats = circuit.operation_statistics()
    return circuit, {
        "degree": degree,
        "t": t,
        "direction": "adjoint" if adjoint else "forward",
        "toffolis": int(stats.num_toffoli_like),
        "gf_multiplies": 2 * t + 1,
        "qubits": int(circuit.num_qubits),
        "ops": {k: int(v) for k, v in stats.items() if int(v)},
    }


def build_measured_uncompute(degree: int, t: int) -> tuple[km.Circuit, dict]:
    """Clear the array with a per-qubit measurement instead of replaying the adjoint.

    ``mr`` is used when the extension has it, because it measures in Z, clears the qubit and
    leaves the tracked phase alone (``docs/MR_INSTRUCTION_SPEC.md``). The pinned build has no
    ``mr``, so there the pass falls back to ``reset`` and the phase-safety caveat is recorded
    rather than assumed away. Every one of ``reset``, ``hmr`` and ``mr`` emits zero ANDs, so
    the Toffoli count is a property of the clear being measured, not of the instruction used.
    """
    n_cells = 2 * t + 3
    builder = km.CircuitBuilder()
    array = builder.create_quantum_register(n_cells * degree, "bm_array")
    has_mr = hasattr(builder, "mr")

    if has_mr:
        # Only the scalar binding is exposed in this snapshot; one ``mr`` per qubit is the
        # same instruction stream the broadcast form would emit.
        outputs = builder.alloc_bits(n_cells * degree)
        for qubit, bit in zip(list(array), list(outputs)):
            builder.mr(qubit, bit)
    else:
        for q in list(array):
            builder.reset(q)

    circuit = builder.finish_circuit()
    stats = circuit.operation_statistics()
    return circuit, {
        "degree": degree,
        "t": t,
        "cells": n_cells,
        "cleared_qubits": n_cells * degree,
        "toffolis": int(stats.num_toffoli_like),
        "ops": {k: int(v) for k, v in stats.items() if int(v)},
        "instruction": ("mr per qubit (phase-preserving Z-basis clear)" if has_mr
                        else "reset per qubit (this build has no mr and no h)"),
        "phase_caveat": ("none: mr measures in Z and leaves the tracked phase alone" if has_mr
                         else "reset clears but does not preserve the tracked phase; a "
                              "phase-safe clear needs `mr` (see scripts/kickmix_mr_opcode.patch "
                              "and docs/MR_INSTRUCTION_SPEC.md), which emits zero ANDs as "
                              "well, so the Toffoli count is unchanged"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--t", type=int, default=35)
    parser.add_argument("--out", type=pathlib.Path, help="write the adjoint iteration as .kmb")
    parser.add_argument("--json", type=pathlib.Path, help="write the receipt here")
    args = parser.parse_args()

    rows = []
    adjoint_circuit = None
    for degree, t in ((4, 5), (8, 10), (args.degree, args.t)):
        forward, fwd = build_iteration(degree, t, adjoint=False)
        reverse, rev = build_iteration(degree, t, adjoint=True)
        iterations = 2 * t
        row = {
            "degree": degree,
            "t": t,
            "iterations": iterations,
            "forward_toffolis_per_iteration": fwd["toffolis"],
            "adjoint_toffolis_per_iteration": rev["toffolis"],
            "adjoint_equals_forward": fwd["toffolis"] == rev["toffolis"],
            "key_equation_toffolis": fwd["toffolis"] * iterations,
            "uncompute_toffolis": rev["toffolis"] * iterations,
            "forney_per_position": FORNEY_PER_POSITION[degree],
            "forney_sweep_toffolis": FORNEY_PER_POSITION[degree] * POSITIONS,
            "coefficient_sweep_toffolis": 0,
        }
        row["four_stage_total"] = (row["key_equation_toffolis"] + row["uncompute_toffolis"]
                                   + row["forney_sweep_toffolis"]
                                   + row["coefficient_sweep_toffolis"])
        rows.append(row)
        if degree == args.degree and t == args.t:
            adjoint_circuit = reverse
        print(f"GF(2^{degree}) t={t}: forward {fwd['toffolis']:,} T/iteration, adjoint "
              f"{rev['toffolis']:,} T/iteration (equal: {row['adjoint_equals_forward']}), "
              f"4-stage total {row['four_stage_total']:,}")

    mr_circuit, mr_row = build_measured_uncompute(args.degree, args.t)
    print(f"measured-clear uncompute: {mr_row['toffolis']} T over "
          f"{mr_row['cleared_qubits']} qubits, ops {sorted(mr_row['ops'].items())[:4]}")

    target = rows[-1]
    receipt = {
        "script": "scripts/emit_uncompute_pass.py",
        "module": "apedp_dyn/quantum_compilation/inplace_bm.py",
        "class": "measured gate stream for one iteration and for its adjoint; the loops are "
                 "projections",
        "why_the_adjoint_costs_the_same": (
            "every gate in the loop is an ixor_gf2_mul, which is its own inverse and never "
            "touches its own operands, so the adjoint is the same multiset in reverse order; "
            "the two counts are compared, not assumed"
        ),
        "rows": rows,
        "measure_based_uncompute": mr_row,
        "accounting": {
            "archive_toffolis": ARCHIVE_TOFFOLIS,
            "stages": {
                "key_equation": target["key_equation_toffolis"],
                "coefficient_sweep": target["coefficient_sweep_toffolis"],
                "forney_block": target["forney_sweep_toffolis"],
                "uncompute": target["uncompute_toffolis"],
                "uncompute_measured": mr_row["toffolis"],
            },
            "four_stage_total": target["four_stage_total"],
            "ratio_vs_archive": target["four_stage_total"] / ARCHIVE_TOFFOLIS,
            "excess_over_archive": target["four_stage_total"] - ARCHIVE_TOFFOLIS,
            "archive_share_of_total": ARCHIVE_TOFFOLIS / target["four_stage_total"],
            "conclusion": (
                f"the complete packed decoder totals {target['four_stage_total']:,} Toffolis, "
                f"{target['four_stage_total'] / ARCHIVE_TOFFOLIS:.2f}x the archive's "
                f"{ARCHIVE_TOFFOLIS:,.0f} -- the archive is "
                f"{100 * ARCHIVE_TOFFOLIS / target['four_stage_total']:.1f}% of the measured "
                "construction. The uncompute pass is not cheaper than the forward one "
                f"({target['uncompute_toffolis']:,} against "
                f"{target['key_equation_toffolis']:,}), and the Forney sweep is the dominant "
                f"term at {100 * target['forney_sweep_toffolis'] / target['four_stage_total']:.1f}%"
            ),
        },
        "mr_availability": (
            "the shipped extension exposes reset, hmr and del_and -- and no mr, and no "
            "Hadamard -- so the clear is measured with reset; `kickmix_mr_opcode.patch` adds "
            "a phase-preserving mr to the source. reset, hmr and mr all emit zero ANDs, so "
            "the count below is the count for all three, and only phase safety differs"
        ),
    }
    a = receipt["accounting"]
    print(f"stages: {a['stages']}")
    print(f"four-stage total {a['four_stage_total']:,} T = {a['ratio_vs_archive']:.2f}x the "
          f"archive ({a['excess_over_archive']:,.0f} T over); Forney is "
          f"{100 * a['stages']['forney_block'] / a['four_stage_total']:.1f}% of it")

    if args.out and adjoint_circuit is not None:
        adjoint_circuit.to_file(args.out, format="kmb")
        print(f"wrote {args.out}")
    if args.json:
        args.json.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
