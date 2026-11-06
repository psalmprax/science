#!/usr/bin/env python3
"""Stack the two measured control-logic gates into one arm and measure the result.

`docs/EEA_AND_PREDICATE_PLAN.md` §2.4 reports two independently measured gates:

* predicate reuse in `_update_flag` -- 1,776 per shot (8 Toffolis x 222 rounds),
  `scripts/measure_flag_predicate_reuse.py`;
* halting-round gating of `_update_flag` and `_update_metadata` -- 1,224 per shot,
  `scripts/measure_eea_oracle_ledger.py`.

They were measured on separate arms and have never run in one circuit. This script
runs them together, and it does not assume the savings add.

**The prediction, written before the combined arm is built.** The reuse saves 8
Toffolis in each round in which `_update_flag` executes. The halt gate removes
`_update_flag` entirely from rounds whose halting counter is non-zero, so in those
rounds there is nothing left for the reuse to save. The two savings are therefore
*not* additive:

    combined = 8 x (rounds - dead_rounds) + halt_gate_delta

This script measures the control arm and the halt-gated arm first, derives
`dead_rounds` from the difference between their `_update_flag` counts and the
region's per-round cost, and prices the reuse from its measured unit costs. Only
then does it build the combined arm. Both quantities therefore come from the same
run and the same instance as the arm being predicted, and the committed
single-gate records are used only as cross-checks.

At the pinned instance the arithmetic is `8 x 216 + 1,224 = 2,952`, against a
naive sum of the published figures of 3,000: the two differ by exactly the reuse
forgone in the six dead rounds (48 Toffolis). The record states both models and
the verdict names the one the measurement matched.

Example::

    /dev/shm/qvcc-venv313/bin/python scripts/build_kickmix_mr.py
    /dev/shm/qvcc-venv313/bin/python scripts/measure_dqi_combined_arm.py
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "dqi_combined_arm.json"
PATCH = ROOT / "scripts" / "kickmix_mr_opcode.patch"

#: The reconciled six-lever arm at the pinned instance, 64 shots.
PINNED_INSTANCE = (4095, 70, 12)
LEDGER_TOTAL = 1_221_466.8
LEDGER_QUBITS = 1_913
#: The published single-lever figures, used only to cross-check the pinned instance.
PUBLISHED_PREDICATE_REUSE = 1_776.0
PUBLISHED_HALT_GATE = 1_224.0
PUBLISHED_DEAD_ROUNDS = 6
#: `_update_flag` static cost per round at the pinned instance: 32,856 / 222.
UPDATE_FLAG_PER_ROUND_PINNED = 32_856 / 222


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=PINNED_INSTANCE[0])
    parser.add_argument("--n", type=int, default=PINNED_INSTANCE[1])
    parser.add_argument("--degree", type=int, default=PINNED_INSTANCE[2])
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--kickmix-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--halt-record", type=Path,
                        default=ROOT / "benchmarks" / "quantum_compilation"
                        / "dqi_eea_oracle_ledger.json")
    args = parser.parse_args()

    pinned = (args.m, args.n, args.degree) == PINNED_INSTANCE

    ledger = load_module("region_ledger", ROOT / "scripts" / "measure_dqi_region_ledger.py")
    reuse = load_module("flag_predicate_reuse",
                        ROOT / "scripts" / "measure_flag_predicate_reuse.py")
    halt = load_module("eea_oracle_ledger",
                       ROOT / "scripts" / "measure_eea_oracle_ledger.py")

    runner = ledger.load_runner()
    km = runner.load_kickmix(args.kickmix_build)
    arms = ledger.load_arms(km, runner)
    base = runner.builder_class(km)
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    import verify_submission_dqi as V

    shim = runner.shim_for(km, ledger.native_builder(base, "and54"))
    n_iters = arms.DR.default_n_iters(args.n)
    logm = (args.n + 3).bit_length() + 1

    # ---- unit prices, before any arm is built --------------------------------
    units = reuse.unit_costs(km, base, logm)
    per_iteration = (2 * units["equality_flip_controlled"]
                     - (units["equality_flip_uncontrolled"] + 2 * units["ccx"]))

    spec = V.ProblemSpecDQI.benchmark_instance(
        m=args.m, n=args.n, degree=args.degree, num_shots=args.shots)
    weights = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    tables = V.FieldTables(km.GF2Field(spec.degree))
    rng = np.random.default_rng(20261028)
    syndromes_in, mbuc_in, _ = V.sample_shots(spec, tables, rng, args.shots, weights)
    reference_inputs = V.to_bits(syndromes_in, args.degree)

    levers = dict(gate_forney=True, clear_eea=True, gate_sweep=True, gate_swap=True,
                  gate_shift=True, gate_quotient=True)

    def build(decoder_type):
        builder = shim.CircuitBuilder()
        syndromes = [builder.create_quantum_register(args.degree, name=f"S{i}")
                     for i in range(args.n)]
        error_mbuc = [builder.create_classical_register(args.degree, name=f"E{j}")
                      for j in range(args.m)]
        decoder_type(args.n, km.GF2Field(args.degree), **levers).build_syndrome_decoding_mbuc(
            builder, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=n_iters)
        return builder.finish_circuit(), syndromes, error_mbuc

    combined_type = type("CombinedControlDecoder",
                         (reuse.ReusedFlagPredicateDecoder,
                          halt.HaltGatedDecoder,
                          arms.MRGatedDecoder), {})
    halt_type = type("HaltGatedDecoder", (halt.HaltGatedDecoder, arms.MRGatedDecoder), {})

    def measure(decoder_type, reference) -> dict:
        circuit, syn, mbuc = build(decoder_type)
        out = ledger.simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in,
                              args.degree, args.shots)
        row = {
            "qubits": circuit.num_qubits,
            "mean": out["mean_toffoli_like"],
            "regions": ledger.region_ledger(circuit, out, args.shots),
            "syndromes": bool(np.array_equal(out["syndromes"], reference_inputs)),
            "junk": bool(not out["junk"].any()),
            "phase": bool(np.array_equal(out["phase"], reference["phase"])),
            "eea_steps": sum(1 for _index, text in circuit.info_tags()
                             if text == "@enter:eea_step"),
        }
        del circuit, out
        gc.collect()
        return row

    def region_row(arm, name) -> float:
        for entry in arm["regions"]["rows"]:
            if entry["region"] == name:
                return entry["executed_toffoli_per_shot"]
        return 0.0

    original = arms.DR.km
    arms.DR.km = shim
    try:
        ref_circuit, ref_syn, ref_mbuc = ledger.build_reference(km, arms, shim, args.n,
                                                               args.degree, args.m)
        reference = ledger.simulate(km, ref_circuit, ref_syn, ref_mbuc, syndromes_in,
                                    mbuc_in, args.degree, args.shots)
        del ref_circuit
        gc.collect()

        control = measure(arms.MRGatedDecoder, reference)
        halted_arm = measure(halt_type, reference)

        # ---- prediction, before the combined arm is built --------------------
        flag_cost_per_round = region_row(control, "update_flag") / n_iters
        dead_rounds_measured = ((region_row(control, "update_flag")
                                - region_row(halted_arm, "update_flag"))
                                / flag_cost_per_round)
        halt_delta = control["mean"] - halted_arm["mean"]
        naive_additive = per_iteration * n_iters + halt_delta
        predicted = per_iteration * (n_iters - dead_rounds_measured) + halt_delta
        prediction = {
            "written_before_combined_arm": True,
            "hypothesis": "the predicate-reuse arm and the halting-round-gate arm compose: "
                          "one circuit, both gates, correct, no qubit or iteration change",
            "unit_costs": units,
            "register_width": logm,
            "iterations": n_iters,
            "toffoli_per_iteration_saved_by_reuse": per_iteration,
            "measured_before_prediction": {
                "control_total_per_shot": control["mean"],
                "halt_gate_delta_per_shot": halt_delta,
                "update_flag_per_round": flag_cost_per_round,
                "dead_rounds": dead_rounds_measured,
            },
            "live_rounds": n_iters - dead_rounds_measured,
            "naive_additive_prediction_per_shot": naive_additive,
            "interaction_prediction_per_shot": predicted,
            "interaction_penalty_per_shot": naive_additive - predicted,
            "why": "the halt gate removes _update_flag from dead rounds, so the reuse "
                   "cannot save 8 Toffolis there; only the naive sum assumes it does",
            "published_single_gate_figures": {
                "predicate_reuse_per_shot": PUBLISHED_PREDICATE_REUSE,
                "halt_gate_per_shot": PUBLISHED_HALT_GATE,
                "dead_rounds": PUBLISHED_DEAD_ROUNDS,
                "naive_additive_per_shot": PUBLISHED_PREDICATE_REUSE + PUBLISHED_HALT_GATE,
                "source": str(args.halt_record),
                "comparable": pinned,
            },
        }
        print(f"prediction (before the combined arm): reuse {per_iteration} x "
              f"{n_iters - dead_rounds_measured:.0f} live rounds + halt gate "
              f"{halt_delta:.1f} = {predicted:+.1f}/shot; naive additive "
              f"{naive_additive:+.1f}")

        combined = measure(combined_type, reference)
    finally:
        arms.DR.km = original

    combined_delta = control["mean"] - combined["mean"]
    distance_interaction = abs(combined_delta - predicted)
    distance_additive = abs(combined_delta - naive_additive)
    model = ("interaction" if distance_interaction < 0.5
             else "additive" if distance_additive < 0.5 else "neither")
    correct = combined["syndromes"] and combined["junk"] and combined["phase"]

    def q2t(arm) -> float:
        return float(arm["qubits"]) ** 2 * float(arm["mean"])

    checks = [
        {
            "check": "the gates compose: combined arm is correct and clean",
            "requirement": "syndromes restored, phase matched, no junk",
            "measured": {"syndromes_restored": combined["syndromes"],
                         "phase_matches_reference": combined["phase"],
                         "junk_clean": combined["junk"]},
            "passed": correct,
        },
        {
            "check": "combined saving matches the interaction prediction",
            "requirement": f"{predicted:.1f} +- 0.5 per shot",
            "measured": combined_delta,
            "passed": distance_interaction < 0.5,
        },
        {
            "check": "the naive additive model is the one that fails",
            "requirement": (f"observed within 0.5 of the interaction model and more than "
                            f"0.5 from the sum of the two single-gate figures"),
            "measured": {"interaction_distance": distance_interaction,
                         "additive_distance": distance_additive},
            "passed": distance_interaction < 0.5 and distance_additive > 0.5,
        },
        {
            "check": "peak qubits unchanged by either gate",
            "requirement": "Q identical in all three arms"
                           + (f" and equal to {LEDGER_QUBITS}" if pinned else ""),
            "measured": {"control": control["qubits"], "halt_gated": halted_arm["qubits"],
                         "combined": combined["qubits"]},
            "passed": (control["qubits"] == halted_arm["qubits"]
                       == combined["qubits"])
                      and (not pinned or combined["qubits"] == LEDGER_QUBITS),
        },
        {
            "check": "the iteration control is unchanged",
            "requirement": f"{n_iters} forward EEA rounds in all three arms",
            "measured": {"control": control["eea_steps"],
                         "halt_gated": halted_arm["eea_steps"],
                         "combined": combined["eea_steps"]},
            "passed": all(arm["eea_steps"] == n_iters
                          for arm in (control, halted_arm, combined)),
        },
        {
            "check": "every arm's region rows partition its measured total",
            "requirement": "reconciles",
            "measured": {name: arm["regions"]["reconciles_with_measured_total"]
                         for name, arm in (("control", control),
                                           ("halt_gated", halted_arm),
                                           ("combined", combined))},
            "passed": all(arm["regions"]["reconciles_with_measured_total"]
                          for arm in (control, halted_arm, combined)),
        },
        {
            "check": "pinned instance: control reproduces the reconciled six-lever total",
            "requirement": f"{LEDGER_TOTAL:.1f} per shot" if pinned else "not the pinned instance",
            "measured": control["mean"] if pinned else None,
            "passed": (abs(control["mean"] - LEDGER_TOTAL) < 0.5) if pinned else True,
        },
        {
            "check": "pinned instance: each gate reproduces its published single-arm figure",
            "requirement": (f"halt {PUBLISHED_HALT_GATE:.0f}, reuse "
                            f"{PUBLISHED_PREDICATE_REUSE:.0f}, {PUBLISHED_DEAD_ROUNDS} dead "
                            f"rounds") if pinned else "not the pinned instance",
            "measured": {"halt_gate_delta": halt_delta,
                         "dead_rounds": dead_rounds_measured,
                         "reuse_per_live_round": per_iteration} if pinned else None,
            "passed": (abs(halt_delta - PUBLISHED_HALT_GATE) < 0.5
                       and abs(dead_rounds_measured - PUBLISHED_DEAD_ROUNDS) < 1e-9
                       and abs(per_iteration
                               - PUBLISHED_PREDICATE_REUSE / n_iters) < 1e-9
                       ) if pinned else True,
        },
        {
            "check": "reproducibility: frozen hashes and record path",
            "requirement": "extension, patch, script and record pinned",
            "measured": {
                "kickmix_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
                "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "record": str(args.record),
                "git_rev": subprocess.run(
                    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=False,
                    capture_output=True, text=True).stdout.strip(),
            },
            "passed": True,
        },
    ]
    all_passed = all(check["passed"] for check in checks)

    record = {
        "experiment": "stack predicate reuse and the halting-round gate in one arm",
        "engine": "kickmix",
        "kickmix": str(Path(km.__file__).resolve()),
        "kickmix_role": "local rebuild with MR and per-instruction execution counts "
                        "(scripts/build_kickmix_mr.py); not the pinned competition binary",
        "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                     "shots": args.shots, "seed": 20261028, "sim_seed": 99,
                     "pinned": pinned},
        "policy": "and54",
        "levers": levers,
        "prediction": prediction,
        "arms": {
            name: {
                "qubits": arm["qubits"],
                "mean_toffoli_like": arm["mean"],
                "update_flag_toffoli_per_shot": region_row(arm, "update_flag"),
                "update_metadata_toffoli_per_shot": region_row(arm, "update_metadata"),
                "q2t": q2t(arm),
                "regions": arm["regions"]["rows"],
                "correct": bool(arm["syndromes"] and arm["junk"] and arm["phase"]),
            }
            for name, arm in (("control", control), ("halt_gated", halted_arm),
                              ("combined", combined))
        },
        "observation": {
            "halt_gate_delta_per_shot": halt_delta,
            "combined_delta_per_shot": combined_delta,
            "combined_delta_percent_of_arm": 100 * combined_delta / control["mean"],
            "predicted_delta_per_shot": predicted,
            "naive_additive_per_shot": naive_additive,
            "interaction_penalty_per_shot": naive_additive - combined_delta,
            "dead_rounds": dead_rounds_measured,
            "model_matched": model,
            "update_flag_after_both_gates": {
                "control": region_row(control, "update_flag"),
                "halt_gated": region_row(halted_arm, "update_flag"),
                "combined": region_row(combined, "update_flag"),
            },
            "update_metadata_after_both_gates": {
                "control": region_row(control, "update_metadata"),
                "halt_gated": region_row(halted_arm, "update_metadata"),
                "combined": region_row(combined, "update_metadata"),
            },
        },
        # Promotion is decided by the checks; the two models are hypotheses about
        # how the gates compose and are reported as matched or falsified.
        "promotion_checks": checks,
        "all_checks_passed": bool(all_passed),
        "verdict": "PROMOTE" if all_passed else "REJECT: a promotion check failed",
        "scope": {
            "is_official_result": False,
            "reason": "MR-gated arms cannot be compiled by the pinned competition binary; "
                      "this is a measured local figure, not a receipt",
        },
    }

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")

    print(f"control:    Q={control['qubits']} T={control['mean']:.1f}/shot")
    print(f"halt gate:  Q={halted_arm['qubits']} T={halted_arm['mean']:.1f}/shot "
          f"({halt_delta:+.1f})")
    print(f"combined:   Q={combined['qubits']} T={combined['mean']:.1f}/shot "
          f"({combined_delta:+.1f}), correct={correct}")
    print(f"  predicted {predicted:+.1f} (interaction) vs {naive_additive:+.1f} (additive); "
          f"model matched: {model}")
    for check in checks:
        print(f"  {'PASS' if check['passed'] else 'FAIL'}  {check['check']}")
    print(f"VERDICT: {record['verdict']}")
    print(f"record: {args.record}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
