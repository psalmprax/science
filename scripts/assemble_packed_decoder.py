"""Assemble the four measured stages into one circuit and read T_mean from it.

The stage receipts measure each piece on its own. This script puts them in one gate
stream, in the order a decoder runs them:

1. key equation -- ``2t`` packed iterations over the one array (no adjoint replay);
2. coefficient sweep -- the constant-multiply steps that carry ``sigma(alpha^j)``;
3. Forney block -- classically conditioned on the live-position bit;
4. uncompute -- a measured clear of the array, through the ``h; hmr; h`` equivalence.

Two numbers come out: the **static** Toffoli count of the finished circuit, and the
**T_mean** the simulator reports with ``counting_operations=True``, i.e. the mean over
shots of the gates actually executed. The second is the archive's metric, so the
comparison against 396,919 is made with it.

The official instance (degree 12, t = 35) needs 70 iterations and 4,095 positions, so the
default here is the scaled instance and the ratio is reported per Toffoli of work rather
than as a submission score:

    python3 scripts/assemble_packed_decoder.py --degree 4 --t 5 --positions 4 --shots 8
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
if str(BUNDLE / "kickmix") not in sys.path:
    sys.path.insert(0, str(BUNDLE / "kickmix"))

import kickmix as km  # noqa: E402

ARCHIVE_TOFFOLIS = 396919.33


def build_assembly(degree: int, t: int, positions: int) -> tuple[km.Circuit, dict]:
    field = km.GF2Field(degree)
    builder = km.CircuitBuilder()

    n_cells = 2 * t + 3
    array = builder.create_quantum_register(n_cells * degree, "bm_array")
    cells = [array[i * degree:(i + 1) * degree] for i in range(n_cells)]
    sigma = cells[:t + 1]
    b_poly = cells[t + 1:2 * t + 2]
    disc = cells[2 * t + 2]
    syndromes = [builder.create_quantum_register(degree, f"S{j}") for j in range(2 * t)]
    steps = [field.pow(field.primitive_element, k) for k in range(1, 2 * t + 2)]

    # Stage 2/3 need a Forney numerator/denominator slot and a phase mask.
    numerator = builder.create_quantum_register(degree, "omega")
    inverse = builder.alloc_qubits(degree)
    mask = builder.alloc_bits(degree)

    # The live flag is an input here: in the real decoder it is the MR read of the
    # locator, which costs no Toffoli. Deriving it from a written qubit keeps the
    # simulator honest about which shots execute the block.
    live_qubits = [builder.alloc_qubits(1)[0] for _ in range(positions)]
    live_bits = [builder.alloc_bits(1)[0] for _ in range(positions)]
    for qubit, bit in zip(live_qubits, live_bits):
        # No Hadamard in this build: the derived bit is the X-basis outcome, which is all
        # the schedule needs, and it emits no Toffoli.
        builder.hmr(km.array([qubit]), km.array([bit]))

    # Stage 1: the key equation, 2t packed iterations.
    for _ in range(2 * t):
        for j in range(t + 1):
            builder.ixor_gf2_mul(sigma[j], syndromes[t - j], target=disc, field=field)
        for j in range(1, t + 1):
            builder.ixor_gf2_mul(disc, b_poly[j - 1], target=sigma[j], field=field)

    # Stages 2 and 3: sweep, then the gated Forney block.
    for p in range(positions):
        for cell, step in zip(sigma + b_poly, steps):
            builder.gf2_imul(step, target=cell, field=field)
        builder.push_condition(live_bits[p])
        _, chain = builder.init_gf2_inverse_with_scaffold(sigma[0], target=inverse, field=field)
        builder.gf2_phase_by_product(numerator, inverse, mask=mask, field=field)
        builder.del_gf2_inverse_with_scaffold(sigma[0], target=inverse, scaffold=chain,
                                              field=field)
        builder.free(chain)
        builder.pop_condition()

    # Stage 4: measured clear of the array instead of the adjoint. `reset` is the clear
    # primitive this build has; it and `mr` both emit zero ANDs.
    for q in list(array):
        builder.reset(q)

    circuit = builder.finish_circuit()
    stats = circuit.operation_statistics()
    return circuit, {
        "degree": degree,
        "t": t,
        "positions": positions,
        "iterations": 2 * t,
        "static_toffolis": int(stats.num_toffoli_like),
        "qubits": int(circuit.num_qubits),
        "syndromes": syndromes,
        "live_qubits": live_qubits,
        "ops": {k: int(v) for k, v in stats.items() if int(v)},
    }


def read_t_mean(circuit, meta: dict, shots: int, seed: int) -> dict:
    """Mean executed Toffolis per shot, which is the metric the archive reports."""
    rng = np.random.default_rng(seed)
    degree = meta["degree"]
    sim = km.Simulator(batch_size=shots, seed=seed, counting_operations=True)
    sim.use_same_registers_as(circuit)
    # `write_across_shots` takes one row per qubit and one column per shot.
    for register in meta["syndromes"]:
        values = rng.integers(0, 1 << degree, size=shots)
        bits = np.zeros((degree, shots), dtype=bool)
        for shot, value in enumerate(values):
            for b in range(degree):
                bits[b, shot] = bool((int(value) >> b) & 1)
        sim.write_across_shots(register, bits)
    # Half the positions live on average, so the mean is over mixed schedules.
    for index, qubit in enumerate(meta["live_qubits"]):
        column = np.array([(seed + index + shot) % 2 for shot in range(shots)], dtype=bool)
        sim.write_across_shots(qubit, column)
    sim.do(circuit)
    per_shot = np.array([int(sim.operation_statistics(i).num_toffoli_like)
                         for i in range(shots)])
    return {
        "shots": shots,
        "per_shot": [int(v) for v in per_shot],
        "t_mean": float(per_shot.mean()),
        "t_max": int(per_shot.max()),
        "t_min": int(per_shot.min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--positions", type=int, default=4)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--out", type=pathlib.Path, help="write the assembled circuit as .kmb")
    parser.add_argument("--json", type=pathlib.Path, help="write the receipt here")
    args = parser.parse_args()

    circuit, meta = build_assembly(args.degree, args.t, args.positions)
    print(f"assembled GF(2^{args.degree}) t={args.t}: {meta['iterations']} iterations, "
          f"{args.positions} positions, static {meta['static_toffolis']:,} T, "
          f"{meta['qubits']} qubits")
    measured = read_t_mean(circuit, meta, args.shots, seed=20261105)
    print(f"T_mean {measured['t_mean']:,.1f} (min {measured['t_min']:,}, "
          f"max {measured['t_max']:,}) over {measured['shots']} shots; "
          f"executed/static {measured['t_mean'] / meta['static_toffolis']:.3f}")

    receipt = {
        "script": "scripts/assemble_packed_decoder.py",
        "stages": ["key equation", "coefficient sweep", "gated Forney block",
                   "measured clear (reset per qubit)"],
        "class": "one assembled circuit at a scaled instance; the official instance "
                 "(degree 12, t 35, 4,095 positions) is not built here",
        "instance": {"degree": args.degree, "t": args.t, "positions": args.positions,
                     "iterations": meta["iterations"]},
        "static_toffolis": meta["static_toffolis"],
        "qubits": meta["qubits"],
        "ops": meta["ops"],
        "t_mean": measured,
        "archive_toffolis": ARCHIVE_TOFFOLIS,
        "note": (
            "the live bit is an input to the assembly; in the full decoder it is the "
            "zero-locator read, which is measured at 0 Toffolis in "
            "scripts/emit_chien_forney_stage.py. The clear is `reset`, because this build "
            "has no `mr` and no Hadamard; `reset` and `mr` both emit zero ANDs"
        ),
    }
    print(f"archive {ARCHIVE_TOFFOLIS:,.0f} T is {ARCHIVE_TOFFOLIS / measured['t_mean']:.1f}x "
          f"this scaled instance's T_mean; the stage-sum comparison lives in "
          "benchmarks/quantum_compilation/uncompute_pass.json")

    if args.out:
        circuit.to_file(args.out, format="kmb")
        print(f"wrote {args.out}")
    if args.json:
        args.json.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
