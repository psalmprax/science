#!/usr/bin/env python3
"""Oracle-gate each region of the conditional EEA body, one region at a time.

The six levers were each measured the same way: gate a region on the predicate
that makes it dead, run the arm, and difference the executed counts. That leaves
three regions of the conditional EEA body ungated and unmeasured -- `_update_flag`,
`_update_metadata`, and the per-round control block (`eea_step` outside any
nested mark). `docs/EEA_AND_PREDICATE_PLAN.md` §6 calls for their ledger.

This script supplies the missing predicates and measures what each may remove.

**The predicate for `_update_flag` and `_update_metadata`.** Both regions are
gated round by round on the four one-hot counter bits `c0..c3`, which
`_counter_controls` produces from `is_hc_zero`. When the halting counter is not
zero those bits are all zero, so every term in both regions is controlled off and
the region's net effect is the identity (in `_update_flag` the one unconditional
`iadd`/`isub` pair cancels). Skipping them there is therefore exact, not an
approximation -- and the arm already computes `is_hc_zero`, so reading it into a
classical bit (`cx` into the shared readout ancilla, then `MR`) costs no
Toffoli. This is a real gate, not an oracle; the oracle framing in the plan was
about predicates that had to be supplied as truth tables.

**The regions that stay oracle-only.** `_update_metadata`'s per-round updates are
all `+1`/`-1` under exactly one counter bit, so within a live round it is never
identity; `eea_step`'s counter decode and halting arithmetic are unconditional.
For those the ledger reports the executed count and states that no cheaper
predicate is known, rather than inventing a removable figure.

Output: `benchmarks/quantum_compilation/dqi_eea_oracle_ledger.json`, which carries
the per-region executed counts of the six-lever arm (from
`scripts/measure_dqi_region_ledger.py`), the removable amount each measured lever
delivered (same record), and the removable amount the halt gate delivers for the
two regions it reaches.

Example::

    /dev/shm/qvcc-venv313/bin/python scripts/measure_dqi_region_ledger.py
    /dev/shm/qvcc-venv313/bin/python scripts/measure_eea_oracle_ledger.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"
DEFAULT_REGION_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "dqi_region_ledger.json"
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "dqi_eea_oracle_ledger.json"


def load_ledger_module():
    spec = importlib.util.spec_from_file_location(
        "region_ledger", ROOT / "scripts" / "measure_dqi_region_ledger.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HaltGatedDecoder:
    """Mixin: skip `_update_flag` and `_update_metadata` in rounds with no live sub-step.

    `_synchronized_eea` is the reference loop with two steps wrapped in a
    `push_condition` on a classical read of `is_hc_zero`. Everything else is
    copied verbatim so the only difference from the six-lever arm is the gate.
    """

    def _synchronized_eea(self, builder, reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag,
                          counter, halting_counter, *, n_iters, is_adjoint):
        eea_step = builder.mark("eea_step")
        for it in (range(n_iters - 1, -1, -1) if is_adjoint else range(n_iters)):
            with eea_step:
                if is_adjoint:
                    self._update_counter(builder, counter, flag, is_adjoint=True)

                is_c3 = builder.alloc_qubits(1)[0]
                builder.init_and(counter[0], counter[1], is_c3)
                if is_adjoint:
                    self._update_halting_counter(builder, n_A, n_B, n_b, halting_counter,
                                                 flag, is_c3, is_adjoint=True)

                is_hc_zero = builder.alloc_qubits(1)[0]
                builder.flip_if_equal(halting_counter, 0, target=is_hc_zero)
                ctrl = self._counter_controls(builder, counter, is_c3, is_hc_zero)

                # The gate: `is_hc_zero` read through the shared ancilla. `mr` clears
                # the ancilla, so the read costs no Toffoli and leaves the qubit alone.
                live = self._read_bit(builder, is_hc_zero)

                steps = [
                    lambda: self._gated(builder, live[0], lambda: self._update_flag(
                        builder, reg_b, n_A, n_B, n_a, n_b, n_q, flag, ctrl)),
                    lambda: self._multiply_and_add_or_sub(builder, reg_a, reg_b, n_A, n_a,
                                                          n_q, ctrl, it,
                                                          is_adjoint=is_adjoint),
                    lambda: self._shift_b_array_right(builder, reg_b, n_B, n_b, flag, ctrl,
                                                      it, is_adjoint=is_adjoint),
                    lambda: self._gated(builder, live[0], lambda: self._update_metadata(
                        builder, n_B, n_b, n_q, flag, ctrl, is_adjoint=is_adjoint)),
                    lambda: self._swap_a_and_b(builder, reg_a, reg_b, n_A, n_B, n_a, n_b,
                                               flag, ctrl),
                ]
                for step in (reversed(steps) if is_adjoint else steps):
                    step()

                self._del_counter_controls(builder, counter, is_c3, is_hc_zero, ctrl)
                builder.flip_if_equal(halting_counter, 0, target=is_hc_zero)
                builder.free(is_hc_zero)

                if not is_adjoint:
                    self._update_halting_counter(builder, n_A, n_B, n_b, halting_counter,
                                                 flag, is_c3, is_adjoint=False)
                builder.del_and(counter[0], counter[1], is_c3)
                builder.free(is_c3)
                if not is_adjoint:
                    self._update_counter(builder, counter, flag, is_adjoint=False)

    @staticmethod
    def _gated(builder, bit, body) -> None:
        builder.push_condition(bit)
        body()
        builder.pop_condition()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=4095)
    parser.add_argument("--n", type=int, default=70)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--kickmix-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--region-record", type=Path, default=DEFAULT_REGION_RECORD)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()

    if not args.region_record.exists():
        raise SystemExit(f"run scripts/measure_dqi_region_ledger.py first: "
                         f"{args.region_record} is missing")
    regions = json.loads(args.region_record.read_text())

    ledger = load_ledger_module()
    runner = ledger.load_runner()
    km = runner.load_kickmix(args.kickmix_build)
    arms = ledger.load_arms(km, runner)
    base = runner.builder_class(km)
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    import verify_submission_dqi as V

    shim = runner.shim_for(km, ledger.native_builder(base, "and54"))
    levers = dict(gate_forney=True, clear_eea=True, gate_sweep=True, gate_swap=True,
                  gate_shift=True, gate_quotient=True)
    n_iters = arms.DR.default_n_iters(args.n)

    spec = V.ProblemSpecDQI.benchmark_instance(
        m=args.m, n=args.n, degree=args.degree, num_shots=args.shots)
    weights = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    tables = V.FieldTables(km.GF2Field(spec.degree))
    rng = np.random.default_rng(20261028)
    syndromes_in, mbuc_in, _ = V.sample_shots(spec, tables, rng, args.shots, weights)
    reference_inputs = V.to_bits(syndromes_in, args.degree)

    def build(decoder_type):
        builder = shim.CircuitBuilder()
        syndromes = [builder.create_quantum_register(args.degree, name=f"S{i}")
                     for i in range(args.n)]
        error_mbuc = [builder.create_classical_register(args.degree, name=f"E{j}")
                      for j in range(args.m)]
        decoder_type(args.n, km.GF2Field(args.degree), **levers).build_syndrome_decoding_mbuc(
            builder, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=n_iters)
        return builder.finish_circuit(), syndromes, error_mbuc

    original = arms.DR.km
    arms.DR.km = shim
    try:
        ref_circuit, ref_syn, ref_mbuc = ledger.build_reference(km, arms, shim, args.n,
                                                               args.degree, args.m)
        reference = ledger.simulate(km, ref_circuit, ref_syn, ref_mbuc, syndromes_in,
                                    mbuc_in, args.degree, args.shots)
        halt_type = type("HaltGatedDecoder", (HaltGatedDecoder, arms.MRGatedDecoder), {})
        measured = {}
        for label, decoder_type in (("six_levers", arms.MRGatedDecoder),
                                    ("halt_gated", halt_type)):
            circuit, syn, mbuc = build(decoder_type)
            out = ledger.simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in,
                                  args.degree, args.shots)
            measured[label] = {
                "qubits": circuit.num_qubits,
                "mean_toffoli_like": out["mean_toffoli_like"],
                "ledger": ledger.region_ledger(circuit, out, args.shots),
                "syndromes_restored": bool(np.array_equal(out["syndromes"],
                                                          reference_inputs)),
                "junk_clean": bool(not out["junk"].any()),
                "phase_matches_reference": bool(np.array_equal(out["phase"],
                                                               reference["phase"])),
            }
    finally:
        arms.DR.km = original

    control = measured["six_levers"]
    gated = measured["halt_gated"]
    control_rows = {row["region"]: row for row in control["ledger"]["rows"]}
    gated_rows = {row["region"]: row for row in gated["ledger"]["rows"]}

    # Removable per region, from the two instruments: the measured levers recorded by
    # the region ledger, and the halt gate measured here.
    lever_removed: dict[str, dict[str, float]] = {}
    for name, row in (regions.get("per_lever") or {}).items():
        for region, delta in row["regions_that_moved"].items():
            lever_removed.setdefault(region, {})[name] = delta
    halt_removed = {
        region: control_rows[region]["executed_toffoli_per_shot"]
        - gated_rows.get(region, {}).get("executed_toffoli_per_shot", 0.0)
        for region in control_rows
    }

    table = []
    for region, row in sorted(control_rows.items(),
                              key=lambda kv: -kv[1]["executed_toffoli_per_shot"]):
        levers_for_region = lever_removed.get(region, {})
        largest = (max(levers_for_region.items(), key=lambda kv: kv[1])
                   if levers_for_region else None)
        halt = max(halt_removed.get(region, 0.0), 0.0)
        table.append({
            "region": region,
            "executed_toffoli_per_shot": row["executed_toffoli_per_shot"],
            "static_toffoli": row["static_toffoli"],
            # Per lever, not summed: `without_clear` removes the whole reverse EEA, so its
            # per-region figures already contain what `without_sweep` removes, and adding
            # them would double count.
            "removable_by_lever": levers_for_region,
            "largest_lever": largest[0] if largest else None,
            "largest_lever_removable": largest[1] if largest else 0.0,
            "removable_by_halt_gate": halt,
        })

    correct = (gated["syndromes_restored"] and gated["junk_clean"]
               and gated["phase_matches_reference"])
    record = {
        "engine": "kickmix",
        "kickmix": str(Path(km.__file__).resolve()),
        "instance": regions["instance"],
        "policy": "and54",
        "levers": levers,
        "iteration_control": {"fixed_iterations": n_iters,
                              "early_halting": False},
        "method": "each region is gated on the predicate that makes it dead; the "
                  "executed counts before and after are the removable amount. The "
                  "six-lever figure comes from per-lever arms, the halt-gate figure "
                  "from the arm measured here.",
        "control": {
            "qubits": control["qubits"],
            "mean_toffoli_like": control["mean_toffoli_like"],
            "correct": bool(control["syndromes_restored"] and control["junk_clean"]),
        },
        "halt_gated": {
            "qubits": gated["qubits"],
            "mean_toffoli_like": gated["mean_toffoli_like"],
            "delta_per_shot": gated["mean_toffoli_like"] - control["mean_toffoli_like"],
            "correct": bool(correct),
            "regions": gated["ledger"]["rows"],
        },
        "region_table": table,
        "totals": {
            "executed_per_shot": control["mean_toffoli_like"],
            "halt_gate_removable_per_shot": sum(row["removable_by_halt_gate"]
                                                for row in table),
            "halt_gate_delta_per_shot": (gated["mean_toffoli_like"]
                                         - control["mean_toffoli_like"]),
            "note": "lever removals are per lever and are not summed across levers or with "
                    "the halt gate: they are measured against different arms",
        },
        "no_known_predicate": [
            region for region, row in control_rows.items()
            if not lever_removed.get(region) and halt_removed.get(region, 0.0) <= 1e-6
        ],
        "checks": {
            "control_matches_region_ledger": bool(
                abs(control["mean_toffoli_like"]
                    - regions["six_levers"]["mean_toffoli_like"]) < 0.5),
            "halt_gate_is_correct": bool(correct),
            "halt_gate_delta_matches_region_deltas": bool(
                abs((gated["mean_toffoli_like"] - control["mean_toffoli_like"])
                    + sum(min(value, 0.0) for value in halt_removed.values())) < 0.51),
            "fixed_iterations_is_222": n_iters == 222,
        },
    }

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")

    print(f"control six levers: Q={control['qubits']} T={control['mean_toffoli_like']:.1f}/shot")
    print(f"halt-gated arm:     Q={gated['qubits']} T={gated['mean_toffoli_like']:.1f}/shot "
          f"({record['halt_gated']['delta_per_shot']:+.1f}), correct={correct}")
    print(f"{'region':28s} {'executed':>12s} {'largest lever':>28s} {'halt':>8s}")
    for row in table:
        lever = (f"{row['largest_lever']} {row['largest_lever_removable']:.1f}"
                 if row["largest_lever"] else "-")
        print(f"{row['region']:28s} {row['executed_toffoli_per_shot']:12.1f} "
              f"{lever:>28s} {row['removable_by_halt_gate']:8.1f}")
    print(f"halt gate removes {record['totals']['halt_gate_removable_per_shot']:.1f}/shot "
          f"({record['halt_gated']['delta_per_shot']:+.1f} on the arm)")
    print(f"no known predicate for: {', '.join(record['no_known_predicate']) or 'none'}")
    print(f"record: {args.record}")
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
