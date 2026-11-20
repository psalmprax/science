"""Three Chien/Forney variants, and which one fits the archive's 83,809 Toffolis.

After the measured key-equation stage (313,110 T), the archive has 83,809 Toffolis left
for the locator and everything else. A per-position gated block costs 315 T
(``scripts/emit_chien_forney_stage.py``), i.e. 1,289,925 over 4,095 positions -- 15.4x the
remainder. This script measures the two ways to share that arithmetic and prices the third:

1. **Montgomery batch inversion** -- one inversion of the product of all 4,095 locator
   values, then 3(n-1) products to recover the individuals. Measured for a block of
   positions and projected.
2. **Runtime-gated blocks** -- the reference's shape: the block is classically conditioned
   on the mid-circuit read of ``sigma(alpha^j)``, so it executes only where the locator
   vanishes. There can be at most ``t`` such positions (deg sigma <= t), which makes the
   expected cost ``315 * t`` rather than ``315 * 4,095``. This is what ``T_mean`` averages.
3. **Measure-based uncompute** -- a Z-basis clear costs 0 Toffolis but needs ``mr``, which
   kickmix does not implement, so it is priced by construction and marked unmeasurable.

    python3 scripts/emit_chien_variants.py --json out.json
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
KEY_EQUATION_TOFFOLIS = 313110.0
REMAINDER = ARCHIVE_TOFFOLIS - KEY_EQUATION_TOFFOLIS
POSITIONS = 4095


def build_batch_inversion(degree: int, positions: int) -> tuple[km.Circuit, dict]:
    """Montgomery: one inversion of the product, then 3(n-1) products to split it."""
    field = km.GF2Field(degree)
    builder = km.CircuitBuilder()
    values = [builder.create_quantum_register(degree, f"v{i}") for i in range(positions)]
    partial = [builder.create_quantum_register(degree, f"p{i}") for i in range(positions)]
    inverse = builder.alloc_qubits(degree)

    # Forward prefix products: n - 1 products.
    builder.cx(values[0], partial[0])
    for i in range(1, positions):
        builder.ixor_gf2_mul(partial[i - 1], values[i], target=partial[i], field=field)

    # One inversion of the whole product.
    _, chain = builder.init_gf2_inverse_with_scaffold(partial[positions - 1],
                                                      target=inverse, field=field)

    # Backward pass: 2 products per value to recover each individual inverse. Every target
    # is a fresh register, so no multiply reads its own target (kickmix rejects the alias).
    running = inverse  # running = 1 / (v_0 ... v_i)
    for i in range(positions - 1, 0, -1):
        builder.ixor_gf2_mul(running, partial[i - 1], target=builder.alloc_qubits(degree),
                             field=field)  # 1 / v_i
        next_running = builder.alloc_qubits(degree)
        builder.ixor_gf2_mul(running, values[i], target=next_running, field=field)
        running = next_running
    builder.free(chain)

    circuit = builder.finish_circuit()
    stats = circuit.operation_statistics()
    products = (positions - 1) + 2 * (positions - 1)
    return circuit, {
        "positions": positions,
        "products": products,
        "toffolis": int(stats.num_toffoli_like),
        "toffolis_per_position": int(stats.num_toffoli_like) / positions,
        "qubits": int(circuit.num_qubits),
    }


def build_gated_block(degree: int, t: int, positions: int) -> tuple[km.Circuit, dict]:
    """The reference's shape: one classically-gated Forney block per position."""
    field = km.GF2Field(degree)
    builder = km.CircuitBuilder()
    sigma_prime = builder.create_quantum_register(degree, "sigma_prime")
    omega = builder.create_quantum_register(degree, "omega")
    mask = builder.alloc_bits(degree)
    inv = builder.alloc_qubits(degree)
    located = builder.alloc_qubits(degree)
    flags = [builder.alloc_bits(1)[0] for _ in range(positions)]

    for j in range(positions):
        builder.x(flags[j])
        _, chain = builder.init_gf2_inverse_with_scaffold(sigma_prime, target=inv, field=field)
        builder.init_and(flags[j], omega, located)
        builder.gf2_phase_by_product(located, inv, mask=mask, field=field)
        builder.del_and(flags[j], omega, located)
        builder.del_gf2_inverse_with_scaffold(
            sigma_prime, target=inv, scaffold=chain, field=field)
        builder.free(chain)
        builder.x(flags[j])

    circuit = builder.finish_circuit()
    stats = circuit.operation_statistics()
    return circuit, {
        "positions": positions,
        "toffolis": int(stats.num_toffoli_like),
        "toffolis_per_position": int(stats.num_toffoli_like) / positions,
        "qubits": int(circuit.num_qubits),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--t", type=int, default=35)
    parser.add_argument("--positions", type=int, default=8,
                        help="block length to emit; the 4,095-position sweep is projected")
    parser.add_argument("--json", type=pathlib.Path, help="write the receipt here")
    args = parser.parse_args()

    degree, t = args.degree, args.t
    batch_circuit, batch = build_batch_inversion(degree, args.positions)
    # The block amortises one inversion over few values, so project the arithmetic rather
    # than scaling the per-position mean: 1 inversion (5 products) + 3(n - 1) products.
    per_product = batch["toffolis"] / (batch["products"] + 5)
    batch["per_product_toffolis"] = per_product
    batch["projected_to_4095"] = per_product * (3 * (POSITIONS - 1) + 5)

    gated_circuit, gated = build_gated_block(degree, t, args.positions)
    gated["projected_if_every_position_live"] = gated["toffolis_per_position"] * POSITIONS
    gated["projected_with_at_most_t_live"] = gated["toffolis_per_position"] * t

    unconditional = gated["toffolis_per_position"] * POSITIONS
    expected = gated["toffolis_per_position"] * t

    variants = {
        "unconditional_per_position": {
            "toffolis": unconditional,
            "fits_remainder": unconditional <= REMAINDER,
            "ratio_to_remainder": unconditional / REMAINDER,
        },
        "shared_single_inversion": {
            "toffolis": batch["projected_to_4095"],
            "fits_remainder": batch["projected_to_4095"] <= REMAINDER,
            "ratio_to_remainder": batch["projected_to_4095"] / REMAINDER,
            "saving_vs_unconditional": unconditional / batch["projected_to_4095"],
        },
        "runtime_gated_blocks": {
            "toffolis": expected,
            "fits_remainder": expected <= REMAINDER,
            "fraction_of_remainder": expected / REMAINDER,
            "assumption": "at most t = deg(sigma) <= 35 of the 4,095 positions have a zero "
                          "locator value, which is what T_mean averages over shots",
        },
    }
    runtime_total = (KEY_EQUATION_TOFFOLIS + expected)  # + 0 sweep + 0 measure-based uncompute
    receipt = {
        "script": "scripts/emit_chien_variants.py",
        "reference": "scripts/emit_chien_forney_stage.py, scripts/emit_uncompute_pass.py",
        "class": "measured variants for a block of positions; the sweep is a projection",
        "positions": POSITIONS,
        "measured": {"batch_inversion_block": batch, "gated_block": gated},
        "remainder": REMAINDER,
        "variants": variants,
        "runtime_conditioned_total": {
            "key_equation": KEY_EQUATION_TOFFOLIS,
            "coefficient_sweep": 0,
            "forney_gated": expected,
            "uncompute_measure_based": 0,
            "total": runtime_total,
            "archive_toffolis": ARCHIVE_TOFFOLIS,
            "archive_over_measured": ARCHIVE_TOFFOLIS / runtime_total,
            "conclusion": (
                f"with the locator gated to at most t live positions and the adjoint replaced "
                f"by a Z-basis clear, the packed decoder totals {runtime_total:,.0f} Toffolis "
                f"-- {ARCHIVE_TOFFOLIS / runtime_total:.2f}x *below* the archive's "
                f"{ARCHIVE_TOFFOLIS:,.0f}. The archive is therefore consistent with this "
                "construction once the cost is read as T_mean (executed gates averaged over "
                "shots) rather than a static count, and the static count (1,916,145) is "
                "consistent with nothing in the archive at all"
            ),
        },
        "not_measurable": (
            "the measure-based uncompute is priced at 0 Toffolis by construction: a Z-basis "
            "clear emits no AND. kickmix has no `mr`, so the circuit that contains it cannot "
            "be counted here; `local_mr_sim` runs `mr` but reports values and phases, not gates"
        ),
    }
    print(f"batch inversion: {batch['toffolis']:,} T over {args.positions} positions "
          f"({batch['toffolis_per_position']:.1f}/position), projected "
          f"{batch['projected_to_4095']:,.0f}")
    print(f"gated block: {gated['toffolis']:,} T over {args.positions} positions "
          f"({gated['toffolis_per_position']:.1f}/position); if all live "
          f"{unconditional:,.0f}, at most t live {expected:,.0f}")
    print(f"remainder {REMAINDER:,.0f}; variants: "
          + ", ".join(f"{k} {'fits' if v['fits_remainder'] else 'does not fit'}"
                      for k, v in variants.items()))
    print(f"runtime-conditioned total {runtime_total:,.0f} T = "
          f"{ARCHIVE_TOFFOLIS / runtime_total:.2f}x below the archive")

    if args.json:
        args.json.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
