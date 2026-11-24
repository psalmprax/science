"""Emit one packed-BM iteration as a real gate stream and count its Toffolis.

The verified packed decoder (``apedp_dyn/quantum_compilation/inplace_bm.py``) is an
algorithm, not a circuit. This script emits the circuit for **one iteration of the
worst case** -- the discrepancy dot product over ``t + 1`` terms plus the update over
``t + 1`` terms -- with the competition's own builder, so the Toffoli cost per
iteration is measured rather than modelled, and then projects the unrolled loop.

What the number means: ``2t`` iterations at this per-iteration cost is the decoder's
key-equation stage. The Chien search, the Forney phase kickback and the uncompute pass
are **not** in it, so the projection is a lower bound for a complete decoder, not a
submission score.

    python3 scripts/emit_bm_iteration.py --degree 12 --t 35 --out /tmp/bm_iter12.kmb
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
if str(BUNDLE / "kickmix") not in sys.path:
    sys.path.insert(0, str(BUNDLE / "kickmix"))

import kickmix as km  # noqa: E402

ARCHIVE_TOFFOLIS = 396919.33


def build_iteration(degree: int, t: int) -> tuple[km.Circuit, dict]:
    """One iteration: ``t + 1`` discrepancy terms into the array's own cell, then
    ``t + 1`` multiply-accumulates for the update."""
    n_cells = 2 * t + 3
    field = km.GF2Field(degree)
    builder = km.CircuitBuilder()

    array = builder.create_quantum_register(n_cells * degree, "bm_array")
    cells = [array[i * degree:(i + 1) * degree] for i in range(n_cells)]
    sigma = cells[:t + 1]
    b_poly = cells[t + 1:2 * t + 2]
    disc = cells[2 * t + 2]
    syndromes = [builder.create_quantum_register(degree, f"S{j}") for j in range(2 * t)]

    # Discrepancy: d ^= sigma_j * S_{i-j}, accumulated in the array's own cell.
    for j in range(t + 1):
        builder.ixor_gf2_mul(sigma[j], syndromes[t - j], target=disc, field=field)

    # Update: sigma_j ^= d * B_{j-shift}, with shift = 1 and the shift folded into
    # the index, which is classical in an unrolled loop.
    for j in range(t + 1):
        if j == 0:
            continue
        builder.ixor_gf2_mul(disc, b_poly[j - 1], target=sigma[j], field=field)

    circuit = builder.finish_circuit()
    stats = circuit.operation_statistics()
    return circuit, {
        "degree": degree,
        "t": t,
        "cells": n_cells,
        "array_qubits": n_cells * degree,
        "syndrome_qubits": 2 * t * degree,
        "gf_multiplies": 2 * t + 1,
        "toffolis": int(stats.num_toffoli_like),
        "qubits": int(circuit.num_qubits),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--t", type=int, default=35)
    parser.add_argument("--out", type=pathlib.Path, help="write the circuit as .kmb here")
    parser.add_argument("--json", type=pathlib.Path, help="write the receipt here")
    args = parser.parse_args()

    rows = []
    for degree, t in ((4, 5), (8, 10), (args.degree, args.t)):
        circuit, row = build_iteration(degree, t)
        iterations = 2 * t
        row["iterations"] = iterations
        row["key_equation_toffolis"] = row["toffolis"] * iterations
        row["toffolis_per_gf_multiply"] = row["toffolis"] / row["gf_multiplies"]
        if degree == args.degree and t == args.t:
            row["archive_toffolis"] = ARCHIVE_TOFFOLIS
            row["key_equation_vs_archive"] = row["key_equation_toffolis"] / ARCHIVE_TOFFOLIS
            if args.out:
                circuit.to_file(args.out, format="kmb")
                row["written"] = str(args.out)
        rows.append(row)
        print(f"GF(2^{degree}) t={t}: {row['toffolis']:,} T/iteration "
              f"({row['toffolis_per_gf_multiply']:.1f} per GF multiply), "
              f"x{iterations} = {row['key_equation_toffolis']:,} T key-equation stage")

    target = rows[-1]
    receipt = {
        "script": "scripts/emit_bm_iteration.py",
        "module": "apedp_dyn/quantum_compilation/inplace_bm.py",
        "class": "measured gate stream for one iteration; the loop is a projection",
        "what_is_counted": "discrepancy dot product + update accumulate, i.e. the "
                           "key-equation stage only (no Chien, Forney or uncompute)",
        "rows": rows,
        "against_the_archive": {
            "archive_toffolis": ARCHIVE_TOFFOLIS,
            "key_equation_toffolis": target["key_equation_toffolis"],
            "ratio": target["key_equation_vs_archive"],
            "conclusion": (
                "the key-equation stage alone costs "
                f"{target['key_equation_vs_archive']:.2f}x the archive's complete "
                "decoder, so the packed algorithm is not cheaper at GF(2^12) without "
                "the uncompute savings the archive's 396,919 Toffolis must contain"
            ),
        },
    }
    if args.json:
        args.json.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
