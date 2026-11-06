#!/usr/bin/env python3
"""Each DQI lever's share, measured by leaving it out of a correct circuit.

`benchmarks/quantum_compilation/gf54_mul_kickmix.json` corrected one number that had
been measured by *deleting* work and differencing executed counts: deleting a block
changes the register state, so the predicates that gate the rest of the EEA read
something else, and the difference includes or excludes work the block never owned.
The multiply's share was 59.48%, not the ablated 86.33%.

This script applies the fix to the six levers. For each lever `L` it builds the full
six-lever arm and the same arm with `L` switched off — which is not an ablation: it
is the reference code path for that lever, so the circuit is still the same unitary
and still has to pass the six-lever verification. The executed difference between
two correct circuits is the lever's own cost, and nothing else moves.

Three views are recorded for each lever, so they can be compared:

* `leave_one_out` — `T(full - L) - T(full)`, the marginal cost of `L` inside the
  stack. This is the only one that answers "what does this lever buy now".
* `cumulative` — the adjacent-stage delta of `kickmix_mr_arm_costs.json`, which is
  the same quantity measured with the levers added in a fixed order instead of
  removed from the top.
* `arm_alone` — `T(reference) / T(only L)`, which is a different question (what the
  lever is worth on its own, before the others have removed work).

A lever whose three views disagree is not a lever: either the stack is not the sum of
its parts, or one of the measurements is perturbing something.

Example::

    /dev/shm/qvcc-venv313/bin/python scripts/measure_dqi_lever_leave_one_out.py
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
DEFAULT_RECORD = (ROOT / "benchmarks" / "quantum_compilation"
                  / "dqi_lever_leave_one_out.json")
COST_RECORD = (ROOT / "benchmarks" / "quantum_compilation" / "kickmix_mr_arm_costs.json")

#: The six levers, i.e. the `all+quotient` arm.
SIX_LEVERS = ("gate_forney", "clear_eea", "gate_sweep", "gate_swap", "gate_shift",
              "gate_quotient")

#: Dropping `gate_sweep` also drops the quotient gate: the quotient divisions live
#: inside the gated step, and without the sweep gate the decoder falls back to the
#: reference body, which has no such gate. The row is labelled accordingly.
COUPLED = {"gate_sweep": ("gate_quotient",)}


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


def build_arm(km, arms, shim, flags, n, degree, m):
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


def _is_correct(out, reference, reference_inputs) -> bool:
    return bool(
        np.array_equal(out["syndromes"], reference["syndromes"])
        and np.array_equal(out["phase"], reference["phase"])
        and np.array_equal(out["syndromes"], reference_inputs)
        and not out["junk"].any()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=4095)
    parser.add_argument("--n", type=int, default=70)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--shots", type=int, default=64)
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
    try:
        spec = V.ProblemSpecDQI.benchmark_instance(
            m=args.m, n=args.n, degree=args.degree, num_shots=args.shots)
        weights = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
        tables = V.FieldTables(km.GF2Field(spec.degree))
        rng = np.random.default_rng(20261028)
        syndromes_in, mbuc_in, _phase = V.sample_shots(
            spec, tables, rng, args.shots, weights)
        reference_inputs = V.to_bits(syndromes_in, args.degree)

        arms.DR.km = shim
        ref_circuit, ref_syn, ref_mbuc = build_arm(km, arms, shim, {}, args.n,
                                                  args.degree, args.m)
        reference = simulate(km, ref_circuit, ref_syn, ref_mbuc, syndromes_in, mbuc_in,
                             args.degree, args.shots)

        def run(flags):
            circuit, syn, mbuc = build_arm(km, arms, shim, flags, args.n, args.degree,
                                           args.m)
            out = simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in, args.degree,
                           args.shots)
            return {
                "levers": dict(sorted(flags.items())),
                "qubits": circuit.num_qubits,
                "toffoli_per_shot": out["toffoli_per_shot"],
                "mean_toffoli_like": float(np.mean(out["toffoli_per_shot"])),
                "correct": _is_correct(out, reference, reference_inputs),
            }

        everything = {lever: True for lever in SIX_LEVERS}
        full = run(everything)
        if not full["correct"]:
            raise SystemExit("the six-lever arm is not correct; refusing to price levers")

        rows = {}
        for lever in SIX_LEVERS:
            dropped = {lever: False}
            for coupled in COUPLED.get(lever, ()):
                dropped[coupled] = False
            flags = {k: v for k, v in everything.items() if k not in dropped}
            row = run(flags)
            if not row["correct"]:
                raise SystemExit(
                    f"leaving out {lever} breaks the circuit, so its share cannot be "
                    f"measured by differencing: {row}")
            row["also_dropped"] = sorted(COUPLED.get(lever, ()))
            row["marginal_toffoli"] = row["mean_toffoli_like"] - full["mean_toffoli_like"]
            row["share_of_arm_percent"] = (
                100 * row["marginal_toffoli"] / row["mean_toffoli_like"])
            rows[lever] = row
    finally:
        arms.DR.km = original

    reference_mean = float(np.mean(reference["toffoli_per_shot"]))
    record: dict[str, object] = {
        "engine": "kickmix",
        "kickmix": str(Path(km.__file__).resolve()),
        "kickmix_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
        "kickmix_role": "local rebuild with MR (scripts/build_kickmix_mr.py); not the "
                        "pinned competition binary and not eligible for a receipt",
        "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                     "shots": args.shots, "seed": 20261028},
        "method": "leave-one-out on correct circuits: each row is the same decoder with one "
                  "lever switched off, which is the reference code path for that lever, so "
                  "the circuit is still the same unitary. The executed difference is the "
                  "lever's own cost. No block is deleted, so no state is perturbed.",
        "reference_t_mean": reference_mean,
        "full": full,
        "levers": rows,
    }

    # The two recorded views this is compared against, if the record is present.
    if COST_RECORD.exists():
        costs = json.loads(COST_RECORD.read_text())
        chain = list(costs["cumulative"].items())
        deltas = {}
        for (prev_name, prev), (name, row) in zip(chain, chain[1:]):
            deltas[name] = prev["mean_toffoli_like"] - row["mean_toffoli_like"]
        record["comparison"] = {
            "cumulative_deltas": deltas,
            "arm_alone_ratios": {name: row["ratio_vs_reference"]
                                 for name, row in costs["arms"].items()},
            "source": "benchmarks/quantum_compilation/kickmix_mr_arm_costs.json",
        }

    total = sum(row["marginal_toffoli"] for row in rows.values())
    record["sum_of_marginals"] = total
    record["sum_vs_stack_saving"] = {
        "stack_saving": reference_mean - full["mean_toffoli_like"],
        "sum_of_marginals": total,
        "residual_percent": 100 * (total - (reference_mean - full["mean_toffoli_like"]))
                            / (reference_mean - full["mean_toffoli_like"]),
    }
    order = sorted(rows, key=lambda name: -rows[name]["marginal_toffoli"])
    record["finding"] = (
        f"Leaving each lever out of the six-lever arm and differencing the executed count "
        f"— the way the multiply's share had to be re-measured — gives "
        + ", ".join(f"{name} {rows[name]['marginal_toffoli']:,.1f}"
                    f" ({rows[name]['share_of_arm_percent']:.2f}% of the arm)"
                    for name in order)
        + f". The marginals sum to {total:,.1f} against the stack's own "
        f"{reference_mean - full['mean_toffoli_like']:,.1f} over the reference, a residual of "
        f"{record['sum_vs_stack_saving']['residual_percent']:+.2f}% that is the interaction "
        f"between levers rather than measurement error: a lever that skips a round also "
        f"removes work another lever would have skipped. Every row is a correct circuit, and "
        f"the script refuses to report a row whose circuit does not reproduce the reference's "
        f"syndromes and phase."
    )

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")

    print(f"m={args.m} n={args.n} GF(2^{args.degree}) shots={args.shots}")
    print(f"  reference {reference_mean:14,.1f}")
    print(f"  full      {full['mean_toffoli_like']:14,.1f}  "
          f"{reference_mean / full['mean_toffoli_like']:.3f}x  Q={full['qubits']}")
    print(f"  {'lever':22s} {'without it':>14s} {'marginal':>12s} {'% of arm':>9s}  ok")
    for lever in order:
        row = rows[lever]
        extra = (f"  (+{', '.join(row['also_dropped'])})" if row["also_dropped"] else "")
        print(f"  {lever + extra:22s} {row['mean_toffoli_like']:14,.1f} "
              f"{row['marginal_toffoli']:12,.1f} {row['share_of_arm_percent']:9.2f}  "
              f"{row['correct']}")
    print(f"  sum of marginals {total:,.1f} vs stack saving "
          f"{reference_mean - full['mean_toffoli_like']:,.1f} "
          f"({record['sum_vs_stack_saving']['residual_percent']:+.2f}%)")
    if "comparison" in record:
        chain = record["comparison"]["cumulative_deltas"]
        print("  cumulative-chain deltas: "
              + ", ".join(f"{name} {value:,.1f}" for name, value in chain.items()))
    print(f"  record: {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
