"""Emit and count the Chien/Forney sweep of the packed BM decoder.

The key-equation stage is measured at 4,473 Toffolis per GF(2^12) iteration
(``scripts/emit_bm_iteration.py``), 313,110 for the 70-iteration loop. That leaves
396,919.33 - 313,110 = 83,809 Toffolis for everything the archive must still do: the
Chien search, the Forney phase, and the uncompute pass. This script measures one
*position* of that sweep as a gate stream and reconciles the remainder against it.

The per-position block is the reference's, not an invention of mine
(``scripts/run_mr_gated_arms.py``, ``build_forney_mbuc``):

* step every locator/omega coefficient by a **classical constant** (``gf2_imul`` by
  ``alpha^k``), which is a linear map -- CX only, no ANDs by construction, and measured;
* read ``sigma(alpha^j) == 0`` with a Z-basis read (no Toffoli);
* when it is zero, invert ``sigma'`` with ``init_gf2_inverse_with_scaffold``, AND that
  against omega, apply ``gf2_phase_by_product`` and uncompute all three.

So a position's Toffoli bill is the gated Forney block plus whatever the constant
multiplies actually cost. Both are measured here:

    python3 scripts/emit_chien_forney_stage.py --degree 12 --t 35 --positions 8
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
KEY_EQUATION_TOFFOLIS = 313110  # measured by scripts/emit_bm_iteration.py
POSITIONS = 4095                # block_length = 2^degree - 1


def build_forney_block(degree: int, t: int, positions: int) -> tuple[km.Circuit, dict]:
    """``positions`` copies of the reference's gated Forney block."""
    field = km.GF2Field(degree)
    builder = km.CircuitBuilder()

    sigma_prime = builder.create_quantum_register(degree, "sigma_prime")
    omega = builder.create_quantum_register(degree, "omega")
    flag = builder.alloc_bits(1)[0]
    # The phase product's mask is a degree-wide register, as in the reference's
    # `error_mbuc[j]`: one bit per field element, all of which clear on uncompute.
    error_mask = builder.alloc_bits(degree)
    inv = builder.alloc_qubits(degree)
    located = builder.alloc_qubits(degree)

    for j in range(positions):
        builder.x(flag)
        _, chain = builder.init_gf2_inverse_with_scaffold(sigma_prime, target=inv, field=field)
        builder.init_and(flag, omega, located)
        builder.gf2_phase_by_product(located, inv, mask=error_mask, field=field)
        builder.del_and(flag, omega, located)
        builder.del_gf2_inverse_with_scaffold(
            sigma_prime, target=inv, scaffold=chain, field=field)
        builder.free(chain)
        builder.x(flag)

    circuit = builder.finish_circuit()
    stats = circuit.operation_statistics()
    return circuit, {
        "degree": degree,
        "t": t,
        "positions": positions,
        "toffolis": int(stats.num_toffoli_like),
        "toffolis_per_position": int(stats.num_toffoli_like) / max(1, positions),
        "qubits": int(circuit.num_qubits),
        "ops": {k: int(v) for k, v in stats.items() if int(v)},
    }


def build_constant_sweep(degree: int, t: int, positions: int) -> tuple[km.Circuit, dict]:
    """Only the sweep's coefficient steps: multiply by a classical ``alpha^k``."""
    field = km.GF2Field(degree)
    builder = km.CircuitBuilder()
    cells = [builder.create_quantum_register(degree, f"c{k}") for k in range(2 * t + 1)]
    steps = [field.pow(field.primitive_element, k) for k in range(1, 2 * t + 2)]

    for _ in range(positions):
        for cell, step in zip(cells, steps):
            builder.gf2_imul(step, target=cell, field=field)

    circuit = builder.finish_circuit()
    stats = circuit.operation_statistics()
    return circuit, {
        "degree": degree,
        "t": t,
        "positions": positions,
        "steps_per_position": len(cells),
        "toffolis": int(stats.num_toffoli_like),
        "toffolis_per_position": int(stats.num_toffoli_like) / max(1, positions),
        "qubits": int(circuit.num_qubits),
        "ops": {k: int(v) for k, v in stats.items() if int(v)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--t", type=int, default=35)
    parser.add_argument("--positions", type=int, default=8,
                        help="block length to emit; the full sweep is projected")
    parser.add_argument("--out", type=pathlib.Path, help="write the emitted block as .kmb")
    parser.add_argument("--json", type=pathlib.Path, help="write the receipt here")
    args = parser.parse_args()

    rows = []
    block_circuit = None
    for degree, t in ((4, 5), (8, 10), (args.degree, args.t)):
        circuit, row = build_forney_block(degree, t, args.positions)
        row["stage"] = "forney block (inversion + AND + phase product)"
        row["projected_to_4095_positions"] = row["toffolis_per_position"] * POSITIONS
        rows.append(row)
        print(f"GF(2^{degree}) t={t}: {row['toffolis']:,} T over {row['positions']} positions "
              f"({row['toffolis_per_position']:.1f}/position), projected "
              f"{row['projected_to_4095_positions']:,.0f} T over {POSITIONS:,}")
        if degree == args.degree and t == args.t:
            block_circuit = circuit

    sweep_rows = []
    for positions in (1, args.positions):
        _, row = build_constant_sweep(args.degree, args.t, positions)
        row["stage"] = "constant-multiply coefficient sweep"
        sweep_rows.append(row)
        print(f"sweep, {positions} positions: {row['toffolis']:,} T "
              f"({row['toffolis_per_position']:.1f}/position) over "
              f"{row['steps_per_position']} steps, ops {sorted(row['ops'].items())[:5]}")

    target = rows[-1]
    per_position = target["toffolis_per_position"]
    full_sweep = per_position * POSITIONS
    remainder = ARCHIVE_TOFFOLIS - KEY_EQUATION_TOFFOLIS
    receipt = {
        "script": "scripts/emit_chien_forney_stage.py",
        "module": "apedp_dyn/quantum_compilation/inplace_bm.py",
        "reference_block": "scripts/run_mr_gated_arms.py build_forney_mbuc",
        "class": "measured gate stream for a block of positions; the full 4,095-position "
                 "sweep is a projection",
        "positions": POSITIONS,
        "forney_block": rows,
        "constant_multiply_sweep": sweep_rows,
        "accounting": {
            "archive_toffolis": ARCHIVE_TOFFOLIS,
            "key_equation_toffolis": KEY_EQUATION_TOFFOLIS,
            "remainder_after_key_equation": remainder,
            "measured_forney_full_sweep": full_sweep,
            "complete_decoder_lower_bound": KEY_EQUATION_TOFFOLIS + full_sweep,
            "ratio_vs_archive": (KEY_EQUATION_TOFFOLIS + full_sweep) / ARCHIVE_TOFFOLIS,
            "positions_the_remainder_affords": remainder / per_position,
            "fraction_of_search_the_remainder_affords": remainder / per_position / POSITIONS,
            "conclusion": (
                f"a position costs {per_position:,.0f} Toffolis, so the {POSITIONS:,}-position "
                f"sweep costs {full_sweep:,.0f} -- "
                f"{(KEY_EQUATION_TOFFOLIS + full_sweep) / ARCHIVE_TOFFOLIS:.2f}x the archive's "
                f"whole decoder. The {remainder:,.0f} Toffolis left after the key-equation "
                f"stage afford {remainder / per_position:.0f} positions, "
                f"{100 * remainder / per_position / POSITIONS:.1f}% of the search, so the "
                "archive cannot contain a per-position Chien/Forney sweep at all: its "
                "remaining budget buys 1/30th of one. Whatever locates the errors there, "
                "it is not a 4,095-position loop"
            ),
        },
    }
    a = receipt["accounting"]
    print(f"remainder after key equation: {a['remainder_after_key_equation']:,.0f} T | "
          f"measured sweep over all positions: {a['measured_forney_full_sweep']:,.0f} T "
          f"({a['ratio_vs_archive']:.2f}x the archive's whole decoder) | "
          f"remainder affords {a['positions_the_remainder_affords']:.1f} positions "
          f"({100 * a['fraction_of_search_the_remainder_affords']:.1f}% of the search)")

    if args.out and block_circuit is not None:
        block_circuit.to_file(args.out, format="kmb")
        print(f"wrote {args.out}")
    if args.json:
        args.json.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
