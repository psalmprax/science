#!/usr/bin/env python3
"""The first controlled experiment from the non-multiply ledger: share `_update_flag`.

`docs/NONMULTIPLY_LEDGER.md` §5 names the predicate block as the first experiment:
`_update_flag` (32,856 static Toffolis), `_update_halting_counter` (10,656) and
`_update_counter` (222). It asks for a hypothesis and a prediction written *before*
the edit, a control that is the same arm with the same 222 iterations, and eight
promotion gates that all have to pass.

This script runs that experiment on the predicate-reuse hypothesis:

* **Hypothesis.** `_update_flag` tests `(n_A == n_B)` twice, once under `c0` and
  once under `c1`. Compute the comparison once into the region's scratch qubit,
  drive `flag` from it with two `CCX`, and clear the scratch with `MR` (a
  Z-basis measurement fused with a reset: deterministic on a basis-state value,
  no phase change, no Toffolis) instead of uncomputing it.
* **Why `MR` rather than uncompute.** Computed into an ancilla, the shared
  comparison must be cleaned before the next sub-step uses the scratch. A second
  equality flip would cost exactly what it saved; a measurement costs nothing.
  This is the same trade `gen_gf54_mul` makes for its windows.
* **Prediction, measured before the arms are built.** The unit costs of one
  controlled and one uncontrolled equality flip on an 8-bit register are measured
  first, and the prediction is `2 * controlled - (uncontrolled + 2 CCX)` per
  iteration times 222 iterations. It is printed and stored before either arm is
  simulated; if the measured delta is outside the predicted value the observation
  fails the prediction even if the gates pass.

The eight gates are the ones the ledger lists: the non-product total reconciles,
the multiply total is reported separately, the components sum to the total, `Q`
is measured including ancillas, `Q²T` is recomputed from the same run,
correctness and cleanup hold, the run is still exactly 222 iterations, and the
hashes and record path are frozen.

Example::

    /dev/shm/qvcc-venv313/bin/python scripts/measure_flag_predicate_reuse.py
"""
from __future__ import annotations

import argparse
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
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "dqi_flag_predicate_reuse.json"
PATCH = ROOT / "scripts" / "kickmix_mr_opcode.patch"

#: The ledger's reconciled control values at the pinned instance, 64 shots.
LEDGER_TOTAL = 1_221_466.8
LEDGER_PRODUCT = 693_660.0 + 8_382.7 + 46_656.0
LEDGER_RESIDUAL = 472_768.2
LEDGER_UPDATE_FLAG_STATIC = 32_856


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def product_tags(tag_names) -> set[str]:
    """Generator-level marks that contain a field product (and the controlled add)."""
    return {name for name in set(tag_names)
            if "mul" in name and name not in LEDGER_REGIONS}


LEDGER_REGIONS = {
    "initial_shift", "shift_step", "forward_eea", "eea_step", "update_flag",
    "update_metadata", "update_halting_counter", "update_counter",
    "multiply_and_add_or_sub", "sweep_positions", "quotient_div",
    "shift_b_array_right", "swap_a_and_b", "chien_forney_mbuc", "pre_align_reg_b",
    "forney_step", "unalign_reg_b", "measured_clear",
}


class ReusedFlagPredicateDecoder:
    """Mixin form: the six-lever decoder with `_update_flag` sharing one comparison."""

    def _update_flag(self, builder, reg_b, n_A, n_B, n_a, n_b, n_q, flag, ctrl):
        with builder.mark("update_flag"):
            c0, c1, c2, c3 = ctrl
            m = len(reg_b)
            aux = builder.alloc_qubits(1)[0]

            # Sub-step 0, unchanged: the less-than round trip is its own term.
            builder.flip_if_less_than(n_A, n_B, target=aux, control=c0)
            builder.flip_if_equal(n_q, 0, target=flag, control=aux)
            builder.flip_if_less_than(n_A, n_B, target=aux, control=c0)

            # The shared comparison: `c0` and `c1` both test (n_A == n_B), so compute
            # it once, use it twice, and clear the scratch by measurement.
            builder.flip_if_equal(n_A, n_B, target=aux)
            builder.ccx(c0, aux, flag)
            builder.ccx(c1, aux, flag)
            pair_bit = builder.alloc_bits(1)
            builder.mr(aux, pair_bit[0])
            builder.free(pair_bit)

            # Sub-step 1, minus its own (n_A == n_B) flip.
            builder.cx(c1, flag)
            builder.flip_if_equal(n_B, 1, target=aux, control=c1)
            builder.cx(c1, aux)
            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)
            builder.cx(c1, aux)
            builder.flip_if_equal(n_B, 1, target=aux, control=c1)

            # Sub-step 2's terms involve (n_a == n_b + 1) between the iadd and the isub,
            # so they are not the same predicate as sub-step 3's (n_a == n_b) and
            # cannot share a comparison.
            builder.cx(c2, flag)
            builder.flip_if_equal(n_b, 0, target=aux, control=c2)
            builder.cx(c2, aux)
            builder.flip_if_equal(reg_b[0], 0, target=flag, control=aux)
            builder.cx(c2, aux)
            builder.flip_if_equal(n_b, 0, target=aux, control=c2)
            builder.iadd(1, target=n_b)
            builder.flip_if_equal(n_a, n_b, target=flag, control=c2)
            builder.isub(1, target=n_b)

            # Sub-step 3.
            builder.flip_if_equal(n_a, n_b, target=flag, control=c3)
            builder.flip_if_equal(n_q, 1, target=flag, control=c3)

            builder.free(aux)


def unit_costs(km, base: type, width: int) -> dict:
    """Toffolis per equality flip, with and without an outer control; plus CCX and MR."""

    def count(build) -> int:
        builder = base()
        build(builder)
        circuit = builder.finish_circuit()
        return int(circuit.operation_statistics().num_toffoli_like)

    def without_control(b) -> None:
        reg = b.alloc_qubits(width)
        target = b.alloc_qubits(1)[0]
        b.flip_if_equal(reg, 0, target=target)

    def with_control(b) -> None:
        reg = b.alloc_qubits(width)
        target = b.alloc_qubits(1)[0]
        control = b.alloc_qubits(1)[0]
        b.flip_if_equal(reg, 0, target=target, control=control)

    def ccx(b) -> None:
        q = b.alloc_qubits(3)
        b.ccx(q[0], q[1], q[2])

    def mr(b) -> None:
        q = b.alloc_qubits(1)[0]
        bit = b.alloc_bits(1)[0]
        b.mr(q, bit)

    return {
        "width": width,
        "equality_flip_uncontrolled": count(without_control),
        "equality_flip_controlled": count(with_control),
        "ccx": count(ccx),
        "mr": count(mr),
    }


def build_reference(km, arms, shim, n: int, degree: int, m: int):
    builder = shim.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
    error_mbuc = [builder.create_classical_register(degree, name=f"E{j}") for j in range(m)]
    arms.DR.ProosAndZalkaDecoder(n, km.GF2Field(degree)).build_syndrome_decoding_mbuc(
        builder, syndromes=syndromes, error_mbuc=error_mbuc,
        n_iters=arms.DR.default_n_iters(n))
    return builder.finish_circuit(), syndromes, error_mbuc


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

    ledger = load_module("region_ledger", ROOT / "scripts" / "measure_dqi_region_ledger.py")
    runner = ledger.load_runner()
    km = runner.load_kickmix(args.kickmix_build)
    arms = ledger.load_arms(km, runner)
    base = runner.builder_class(km)
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    import verify_submission_dqi as V

    shim = runner.shim_for(km, ledger.native_builder(base, "and54"))
    n_iters = arms.DR.default_n_iters(args.n)
    # `_update_flag` compares `n_A`/`n_B`, which are `logm` wide, not `degree` wide:
    # pricing the equality flip at the field degree over-predicts by one Toffoli per
    # register bit (12 and 13 instead of 8 and 9 at m = 73).
    logm = (args.n + 3).bit_length() + 1

    # ---- prediction, before any arm is built ---------------------------------
    units = unit_costs(km, base, logm)
    per_iteration = (2 * units["equality_flip_controlled"]
                     - (units["equality_flip_uncontrolled"] + 2 * units["ccx"]))
    predicted_delta = float(per_iteration * n_iters)
    prediction = {
        "written_before_measurement": True,
        "register_width": logm,
        "hypothesis": "compute (n_A == n_B) once for the c0 and c1 sub-steps, drive "
                      "flag with two CCX, clear the scratch with MR; leaves function, "
                      "Q and the 222-iteration schedule unchanged",
        "unit_costs": units,
        "toffoli_per_iteration_saved": per_iteration,
        "iterations": n_iters,
        "predicted_delta_toffoli_per_shot": predicted_delta,
    }
    print(f"prediction (before measuring): {per_iteration} Toffolis per iteration "
          f"x {n_iters} = {predicted_delta:+.1f} per shot")

    # ---- the two arms --------------------------------------------------------
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

    original = arms.DR.km
    arms.DR.km = shim
    try:
        ref_circuit, ref_syn, ref_mbuc = build_reference(km, arms, shim, args.n,
                                                         args.degree, args.m)
        reference = ledger.simulate(km, ref_circuit, ref_syn, ref_mbuc, syndromes_in,
                                    mbuc_in, args.degree, args.shots)

        variant_type = type("ReusedFlagPredicateDecoder",
                            (ReusedFlagPredicateDecoder, arms.MRGatedDecoder), {})
        measured = {}
        for label, decoder_type in (("control", arms.MRGatedDecoder),
                                    ("reused", variant_type)):
            circuit, syn, mbuc = build(decoder_type)
            out = ledger.simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in,
                                  args.degree, args.shots)
            measured[label] = {"circuit": circuit, "out": out,
                               "qubits": circuit.num_qubits,
                               "mean": out["mean_toffoli_like"]}
    finally:
        arms.DR.km = original

    def split(circuit, out) -> dict:
        spans, tag_names = ledger.region_spans(circuit)
        counts = out["counts"]
        mask = out["toffoli_mask"]
        toffoli = counts * mask
        product = 0.0
        per_region_product: dict[str, float] = {}
        per_region_total: dict[str, float] = {}
        for start, stop, region, in_product in spans:
            executed = int(toffoli[start:stop].sum())
            if not executed:
                continue
            per_region_total[region] = per_region_total.get(region, 0.0) + executed / args.shots
            if in_product:
                product += executed / args.shots
                per_region_product[region] = (per_region_product.get(region, 0.0)
                                              + executed / args.shots)
        total = float(out["mean_toffoli_like"])
        return {
            "total": total,
            "multiply": product,
            "non_product": total - product,
            "regions": ledger.region_ledger(circuit, out, args.shots),
            "product_tags": sorted(product_tags(tag_names)),
            "non_product_by_region": {
                region: value - per_region_product.get(region, 0.0)
                for region, value in per_region_total.items()},
        }

    control = measured["control"]
    reused = measured["reused"]
    control_split = split(control["circuit"], control["out"])
    reused_split = split(reused["circuit"], reused["out"])

    def region_row(split_row, name) -> float:
        for row in split_row["regions"]["rows"]:
            if row["region"] == name:
                return row["executed_toffoli_per_shot"]
        return 0.0

    control_flag = region_row(control_split, "update_flag")
    reused_flag = region_row(reused_split, "update_flag")
    delta_flag = control_flag - reused_flag
    delta_total = control["mean"] - reused["mean"]

    eea_steps = sum(1 for name in control_split["regions"]["tag_names"] if name == "eea_step")
    reused_eea_steps = sum(1 for name in reused_split["regions"]["tag_names"]
                           if name == "eea_step")

    def q2t(row) -> float:
        return float(row["qubits"]) ** 2 * float(row["mean"])

    correct = (bool(np.array_equal(reused["out"]["syndromes"], reference_inputs))
               and bool(not reused["out"]["junk"].any())
               and bool(np.array_equal(reused["out"]["phase"], reference["phase"])))

    # Components from the per-instruction instrument (product term plus each
    # region's non-product part) against the total from the type counters: two
    # independent instruments, so this is a reconciliation and not an identity.
    component_total = control_split["multiply"] + sum(
        control_split["non_product_by_region"].values())
    gates = [
        {
            "gate": "mean non-product T-count reconciles with the instrumented total",
            "requirement": f"{LEDGER_RESIDUAL:.1f} per shot",
            "measured": control_split["non_product"],
            "passed": abs(control_split["non_product"] - LEDGER_RESIDUAL) < 0.5,
        },
        {
            "gate": "mean multiply T-count reported separately",
            "requirement": f"{LEDGER_PRODUCT:.1f} per shot",
            "measured": control_split["multiply"],
            "passed": abs(control_split["multiply"] - LEDGER_PRODUCT) < 0.5,
        },
        {
            "gate": "total T-count: components sum to it",
            "requirement": f"{control['mean']:.1f} +- 0.1",
            "measured": component_total,
            "passed": abs(component_total - control["mean"]) < 0.1,
        },
        {
            "gate": "peak qubits Q measured, including added ancillas",
            "requirement": "measured",
            "measured": {"control": control["qubits"], "reused": reused["qubits"]},
            "passed": True,
        },
        {
            "gate": "Q^2 T recomputed from the same run",
            "requirement": "recomputed",
            "measured": {"control": q2t(control), "reused": q2t(reused),
                         "percent_change": 100 * (q2t(reused) / q2t(control) - 1)},
            "passed": True,
        },
        {
            "gate": "correctness and cleanup",
            "requirement": "syndromes restored, phase matched, no junk",
            "measured": {
                "syndromes_restored": bool(np.array_equal(reused["out"]["syndromes"],
                                                          reference_inputs)),
                "junk_clean": bool(not reused["out"]["junk"].any()),
                "phase_matches_reference": bool(np.array_equal(
                    reused["out"]["phase"], reference["phase"])),
            },
            "passed": correct,
        },
        {
            "gate": "iteration behaviour fixed at 222",
            "requirement": "222 forward EEA rounds, unchanged",
            "measured": {"control": eea_steps, "reused": reused_eea_steps,
                         "expected": n_iters},
            "passed": eea_steps == reused_eea_steps == n_iters,
        },
        {
            "gate": "reproducibility: frozen hashes and record path",
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

    all_gates = all(gate["passed"] for gate in gates)
    # `delta_total` is control minus variant, i.e. Toffolis saved, and so is the
    # prediction: both are positive when the variant is cheaper.
    within_prediction = abs(delta_total - predicted_delta) < 0.5
    if not all_gates:
        verdict = "REJECT: a promotion gate failed"
    elif within_prediction:
        verdict = "PROMOTE"
    else:
        verdict = "PROMOTE WITH A FALSIFIED COST MODEL"
    record = {
        "experiment": "share the (n_A == n_B) comparison in _update_flag",
        "engine": "kickmix",
        "kickmix": str(Path(km.__file__).resolve()),
        "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                     "shots": args.shots, "seed": 20261028, "sim_seed": 99},
        "policy": "and54",
        "levers": levers,
        "prediction": prediction,
        "control": {
            "qubits": control["qubits"],
            "mean_toffoli_like": control["mean"],
            "multiply_toffoli_per_shot": control_split["multiply"],
            "non_product_toffoli_per_shot": control_split["non_product"],
            "update_flag_toffoli_per_shot": control_flag,
            "q2t": q2t(control),
            "regions": control_split["regions"]["rows"],
        },
        "variant": {
            "qubits": reused["qubits"],
            "mean_toffoli_like": reused["mean"],
            "multiply_toffoli_per_shot": reused_split["multiply"],
            "non_product_toffoli_per_shot": reused_split["non_product"],
            "update_flag_toffoli_per_shot": reused_flag,
            "q2t": q2t(reused),
            "regions": reused_split["regions"]["rows"],
        },
        "observation": {
            "delta_update_flag_per_shot": delta_flag,
            "delta_total_per_shot": delta_total,
            "predicted_delta_per_shot": predicted_delta,
            "within_prediction": bool(within_prediction),
            "delta_percent_of_arm": 100 * delta_total / control["mean"],
            "delta_percent_of_region": (100 * delta_flag / control_flag
                                        if control_flag else None),
        },
        "promotion_gates": gates,
        "all_gates_passed": bool(all_gates),
        "prediction_matched": bool(within_prediction),
        # Promotion is decided by the eight gates; the prediction is a separate
        # hypothesis about the cost model and is reported as matched or falsified.
        "verdict": verdict,
        "reconciliation": {
            "ledger_total": LEDGER_TOTAL,
            "ledger_product": LEDGER_PRODUCT,
            "ledger_residual": LEDGER_RESIDUAL,
            "ledger_update_flag_static": LEDGER_UPDATE_FLAG_STATIC,
            "control_matches_ledger_total": abs(control["mean"] - LEDGER_TOTAL) < 0.5,
        },
    }

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")

    print(f"control six-lever arm: Q={control['qubits']} T={control['mean']:.1f}/shot "
          f"(multiply {control_split['multiply']:.1f}, non-product "
          f"{control_split['non_product']:.1f})")
    print(f"variant:               Q={reused['qubits']} T={reused['mean']:.1f}/shot "
          f"(update_flag {control_flag:.1f} -> {reused_flag:.1f})")
    print(f"observed delta: {delta_total:+.1f}/shot "
          f"({100 * delta_total / control['mean']:+.3f}% of the arm), "
          f"predicted {predicted_delta:+.1f}")
    for gate in gates:
        print(f"  {'PASS' if gate['passed'] else 'FAIL'}  {gate['gate']}")
    print(f"VERDICT: {record['verdict']}")
    print(f"record: {args.record}")
    return 0 if record["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
