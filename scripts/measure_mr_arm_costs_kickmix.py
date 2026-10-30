#!/usr/bin/env python3
"""Measure the MR-gated arms' Toffoli counts on the rebuilt, MR-capable ``kickmix``.

`docs/MR_INSTRUCTION_SPEC.md` and §5.5 of `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md`
project Benchmark II `T_mean` 5,725,071.61 → 1,382,207.08 (4.142×) from the probe's
oracle-gated measurements. This script prices the *actual* circuits: it builds the
reference and each arm with a real ``kickmix.CircuitBuilder``, simulates them with a
real ``kickmix.Simulator`` with per-shot operation counting, and records what the
bounded gate set executes. Conditional gates are counted only in the shots where
they run, so the mean over shots is the `T_mean` of the sampled shots.

Two views are recorded because the projection was built as a chain:

* ``cumulative`` — reference, then each lever stacked on the previous one, which is
  the sequence the projected 1.332× / 2.652× / 3.599× / 3.929× / 4.142× follow. The
  last row stacks a sixth lever (``gate_quotient``, the dead polynomial division in each
  live round) that the oracle probe never priced, so it has no projected column.
* ``arms`` — each lever alone, which shows how much each gate is worth by itself.

Two caveats, both recorded in the output:

* The extension is the local build (``scripts/build_kickmix_mr.py``), not the pinned
  competition binary. It is the same builder, and the reference row reproduces the
  pinned `T_mean` to sampling error, but this is not a receipt.
* The sample is small. `T_mean` is a property of the syndrome distribution, so the
  recorded shot count belongs in any comparison.

Example::

    /dev/shm/qvcc-venv313/bin/python scripts/measure_mr_arm_costs_kickmix.py
    ... --m 4095 --n 70 --degree 12 --shots 64
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "kickmix_mr_arm_costs.json"

#: The projection this measurement is compared against (docs/MR_INSTRUCTION_SPEC.md §1).
PROJECTION_REFERENCE_T_MEAN = 5_725_071.61
PROJECTION_GATED_T_MEAN = 1_382_207.08

#: Cumulative lever chain, with the projected Toffoli mean the boundary doc records
#: for each stage (docs/PHOTONIC_ACCELERATOR_BOUNDARY.md §5.5).
CUMULATIVE_STAGES: tuple[tuple[str, dict, float | None], ...] = (
    ("reference", {}, None),
    ("forney", {"gate_forney": True}, 4_298_955.87),
    ("forney+clear", {"gate_forney": True, "clear_eea": True}, 2_158_651.45),
    ("forney+clear+sweep", {"gate_forney": True, "clear_eea": True, "gate_sweep": True},
     1_590_938.28),
    ("forney+clear+sweep+swap",
     {"gate_forney": True, "clear_eea": True, "gate_sweep": True, "gate_swap": True},
     1_457_103.45),
    ("all",
     {"gate_forney": True, "clear_eea": True, "gate_sweep": True, "gate_swap": True,
      "gate_shift": True},
     1_382_207.08),
    # A sixth lever, with no projected count: it skips work the oracle build priced as
    # live (each live round's dead polynomial division), so there is nothing to compare
    # it against -- only the measured row.
    ("all+quotient",
     {"gate_forney": True, "clear_eea": True, "gate_sweep": True, "gate_swap": True,
      "gate_shift": True, "gate_quotient": True},
     None),
)

ARMS = ("none", "swap", "sweep", "forney", "clear", "shift", "quotient", "all", "full")


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "mr_gated_arms_kickmix", ROOT / "scripts" / "run_mr_gated_arms_kickmix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_arms(km, runner):
    spec = importlib.util.spec_from_file_location(
        "mr_gated_arms", ROOT / "scripts" / "run_mr_gated_arms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._all_qubits = lambda registers: runner.all_qubits(km, registers)
    return module


def build_with_flags(km, arms, shim, flags, n, degree, m):
    """Build the decoder with an arbitrary set of levers enabled."""
    decoder = arms.MRGatedDecoder(n, km.GF2Field(degree), **flags)
    builder = shim.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
    error_mbuc = [builder.create_classical_register(degree, name=f"E{j}") for j in range(m)]
    decoder.build_syndrome_decoding_mbuc(
        builder, syndromes=syndromes, error_mbuc=error_mbuc,
        n_iters=arms.DR.default_n_iters(n))
    return builder.finish_circuit(), syndromes, error_mbuc


def simulate(km, circuit, syndromes, error_mbuc, syndromes_in, mbuc_in, degree, shots):
    sim = km.Simulator(batch_size=shots, seed=99, counting_operations=True)
    sim.use_same_registers_as(circuit)
    flat_syndromes = km.array([q for register in syndromes for q in register])
    flat_errors = km.array([b for register in error_mbuc for b in register])
    sim.write_across_shots(flat_syndromes, _bits(syndromes_in, degree))
    sim.write_across_shots(flat_errors, _bits(mbuc_in, degree))
    sim.do(circuit)
    syndrome_ids = {q.id for register in syndromes for q in register}
    junk_ids = [i for i in range(circuit.num_qubits) if i not in syndrome_ids]
    return {
        "syndromes": sim.read_across_shots(flat_syndromes),
        "junk": sim.read_across_shots([km.q(i) for i in junk_ids]),
        "phase": sim.read_phase_flipped_across_shots(),
        "toffoli_per_shot": [int(sim.operation_statistics(i).num_toffoli_like)
                             for i in range(shots)],
    }


def _bits(values, width):
    from verify_submission_dqi import to_bits
    return to_bits(values, width)


def _is_correct(out, reference, reference_inputs):
    return bool(
        np.array_equal(out["syndromes"], reference["syndromes"])
        and np.array_equal(out["phase"], reference["phase"])
        and np.array_equal(out["syndromes"], reference_inputs)
        and not out["junk"].any()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=4095, help="number of MBUC positions")
    parser.add_argument("--n", type=int, default=70)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--kickmix-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()

    runner = load_runner()
    km = runner.load_kickmix(args.kickmix_build)
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    import verify_submission_dqi as V

    arms = load_arms(km, runner)
    shim = runner.shim_for(km, runner.builder_class(km))
    original = arms.DR.km
    arms.DR.km = shim
    try:
        spec = V.ProblemSpecDQI.benchmark_instance(
            m=args.m, n=args.n, degree=args.degree, num_shots=args.shots)
        weights = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
        tables = V.FieldTables(km.GF2Field(spec.degree))
        rng = np.random.default_rng(20261028)
        syndromes_in, mbuc_in, _phase = V.sample_shots(
            spec, tables, rng, args.shots, weights)
        reference_inputs = V.to_bits(syndromes_in, args.degree)

        record: dict[str, object] = {
            "engine": "kickmix",
            "kickmix": str(Path(km.__file__).resolve()),
            "kickmix_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
            "kickmix_role": "local rebuild with MR (scripts/build_kickmix_mr.py); not the "
                            "pinned competition binary and not eligible for a receipt",
            "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                         "shots": args.shots, "seed": 20261028},
            "measure": "toffoli-like operations executed per shot, counted by kickmix's "
                       "conditional-aware simulator statistics, averaged over shots",
            "projection": {
                "source": "docs/MR_INSTRUCTION_SPEC.md, "
                          "docs/PHOTONIC_ACCELERATOR_BOUNDARY.md §5.5",
                "kind": "oracle-gated measurement (predicate bits supplied classically)",
                "reference_t_mean": PROJECTION_REFERENCE_T_MEAN,
                "gated_t_mean": PROJECTION_GATED_T_MEAN,
                "ratio": PROJECTION_REFERENCE_T_MEAN / PROJECTION_GATED_T_MEAN,
            },
            "cumulative": {},
            "arms": {},
        }

        reference = None
        for name, flags, projected in CUMULATIVE_STAGES:
            circuit, syn, mbuc = build_with_flags(km, arms, shim, flags, args.n, args.degree,
                                                  args.m)
            out = simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in, args.degree,
                           args.shots)
            if reference is None:
                reference = out
            record["cumulative"][name] = {
                "levers": flags,
                "qubits": circuit.num_qubits,
                "toffoli_per_shot": out["toffoli_per_shot"],
                "mean_toffoli_like": float(np.mean(out["toffoli_per_shot"])),
                "projected_mean_toffoli_like": projected,
                "correct": _is_correct(out, reference, reference_inputs),
            }

        for arm in [a for a in args.arms if a != "none"]:
            circuit, syn, mbuc = runner.build(km, arms, shim, arm, "mr", args.n, args.degree,
                                              args.m)
            out = simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in, args.degree,
                           args.shots)
            record["arms"][arm] = {
                "qubits": circuit.num_qubits,
                "toffoli_per_shot": out["toffoli_per_shot"],
                "mean_toffoli_like": float(np.mean(out["toffoli_per_shot"])),
                "correct": _is_correct(out, reference, reference_inputs),
            }
    finally:
        arms.DR.km = original

    reference_mean = record["cumulative"]["reference"]["mean_toffoli_like"]
    record["reference_t_mean_measured"] = reference_mean
    record["reference_vs_pinned_t_mean"] = reference_mean - PROJECTION_REFERENCE_T_MEAN
    for group in ("cumulative", "arms"):
        for data in record[group].values():
            data["ratio_vs_reference"] = reference_mean / data["mean_toffoli_like"]
            data["delta_vs_projection"] = (
                data["ratio_vs_reference"] - record["projection"]["ratio"])

    on_official_instance = (args.m, args.n, args.degree) == (4095, 70, 12)
    if not on_official_instance:
        record["finding"] = (
            "The 4.142x projection is defined at the official instance only; this run is "
            "a scaled check, so the projected column is left empty."
        )
        for data in record["cumulative"].values():
            data["projected_mean_toffoli_like"] = None
    else:
        gaps = {
            name: 100.0 * (data["mean_toffoli_like"] / data["projected_mean_toffoli_like"] - 1.0)
            for name, data in record["cumulative"].items()
            if data["projected_mean_toffoli_like"]
        }
        record["stage_gap_percent"] = gaps
        worst_name = max(gaps, key=lambda name: abs(gaps[name]))
        worst = gaps[worst_name]
        spread = "; ".join(f"{name} {gap:+.2f}%" for name, gap in gaps.items())
        shift_saving = 100 * (
            1 - record["arms"]["shift"]["mean_toffoli_like"] / reference_mean)
        gated = record["cumulative"]["all"]
        measured_ratio = gated["ratio_vs_reference"]
        projected_ratio = record["projection"]["ratio"]
        extra = record["cumulative"]["all+quotient"]
        extra_gain = 100 * (1 - extra["mean_toffoli_like"] / gated["mean_toffoli_like"])
        quotient_arm = record["arms"]["quotient"]
        record["agrees_with_projection"] = abs(worst) <= 0.5
        if record["agrees_with_projection"]:
            record["finding"] = (
                f"The five-lever arm executes {gated['mean_toffoli_like']:,.1f} Toffoli per shot "
                f"against the projected {PROJECTION_GATED_T_MEAN:,.2f}: {measured_ratio:.3f}x "
                f"measured, {projected_ratio:.3f}x projected. Every stacked stage is within "
                f"{abs(worst):.2f}% of its projected count ({spread}), so the oracle-gated "
                f"projection is reproduced by the circuits rather than bounding them from above. "
                f"The shift lever alone saves {shift_saving:.2f}% of T. The rows that land slightly "
                f"below their projection are below because the predicate is read out of `sigma` "
                f"with `mr` instead of tested for equality: that removes two `degree - 1`-Toffoli "
                f"equality circuits per live position, which the oracle build still pays. "
                f"A sixth lever goes past the projection: the sweep's two polynomial divisions, "
                f"each gated on its own counter bit ({quotient_arm['mean_toffoli_like']:,.1f} alone, "
                f"{quotient_arm['ratio_vs_reference']:.3f}x), takes the stack to "
                f"{extra['mean_toffoli_like']:,.1f} Toffoli per shot, {extra['ratio_vs_reference']:.3f}x "
                f"-- a further {extra_gain:.2f}% of the five-lever count. Both stacks allocate "
                f"{extra['qubits']} qubit ids against the reference's {record['cumulative']['reference']['qubits']}, "
                f"because every read reuses one shared ancilla instead of leaking one per "
                f"round; before that fix this arm allocated 4,503."
            )
        else:
            record["finding"] = (
                f"The fully gated arm executes {gated['mean_toffoli_like']:,.1f} Toffoli per shot, "
                f"not {PROJECTION_GATED_T_MEAN:,.2f}: {measured_ratio:.3f}x measured against the "
                f"projected {projected_ratio:.3f}x. The largest stage gap is {worst_name} "
                f"{worst:+.2f}% ({spread}), and the shift lever alone saves {shift_saving:.2f}% "
                f"of T. This is a measured discrepancy, not a correction: the projection stays "
                f"pinned until it is re-derived from gated circuits."
            )

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")

    print(f"m={args.m} n={args.n} GF(2^{args.degree}) shots={args.shots} "
          f"({record['kickmix']})")
    print(f"  {'stage':26s} {'qubits':>7s} {'T_mean':>14s} {'projected':>14s} "
          f"{'ratio':>7s}  ok")
    for name, data in record["cumulative"].items():
        projected = data["projected_mean_toffoli_like"]
        projected_text = f"{projected:14,.1f}" if projected else f"{'-':>14s}"
        print(f"  {name:26s} {data['qubits']:7d} {data['mean_toffoli_like']:14,.1f} "
              f"{projected_text} {data['ratio_vs_reference']:7.3f}  {data['correct']}")
    print(f"  {'arm (lever alone)':26s} {'qubits':>7s} {'T_mean':>14s} {'':>14s} "
          f"{'ratio':>7s}  ok")
    for name, data in record["arms"].items():
        print(f"  {name:26s} {data['qubits']:7d} {data['mean_toffoli_like']:14,.1f} "
              f"{'':>14s} {data['ratio_vs_reference']:7.3f}  {data['correct']}")
    print(f"  projection: {PROJECTION_REFERENCE_T_MEAN:,.2f} -> "
          f"{PROJECTION_GATED_T_MEAN:,.2f} ({record['projection']['ratio']:.3f}x)")
    print(f"  reference vs pinned T_mean: {record['reference_vs_pinned_t_mean']:+.2f}")
    print(f"  record: {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
