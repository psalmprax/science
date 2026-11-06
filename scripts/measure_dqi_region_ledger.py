#!/usr/bin/env python3
"""Split the six-lever arm's executed Toffoli stream region by region.

`benchmarks/quantum_compilation/dqi_nonmultiply_ledger.json` reconciles the arm's
1,221,466.8 executed Toffolis per shot into three product terms and a 472,768.2
non-product residual, and stops there: 80% of the residual is "unattributed",
because `operation_statistics()` counts by operation *type*. Two CCX
instructions of the same type are one row in that table, so a per-region split
cannot be read off it.

The local `MR` build now carries per-instruction execution counts
(`sim.counting_op_executions` and `sim.operation_execution_counts()`), and the
circuit carries its own region boundaries (`circuit.info_tags()`, the
`@enter:`/`@exit:` pairs `builder.mark(...)` emits). A region's executed
Toffolis are therefore the sum of the counts of the Toffoli-like instructions
whose innermost mark is that region -- an exact partition of the measured total,
not an estimate.

Two things are measured, and they answer different questions:

* **executed** -- what each region costs in the six-lever arm, per shot. This
  splits the residual completely, so the ledger's 80% becomes a table.
* **removable** -- what each region costs when its lever is *not* applied,
  minus what it costs with the lever. The difference is the measured saving the
  lever actually delivered in that region, which is the same oracle-style
  ceiling `scripts/probe_dqi_predicate_gating.py` reports, read off the real
  arm instead of a probe.

Nothing here is a score. The counts come from the local MR rebuild
(`scripts/build_kickmix_mr.py`); the pinned competition binary is untouched.

Example::

    /dev/shm/qvcc-venv313/bin/python scripts/build_kickmix_mr.py
    /dev/shm/qvcc-venv313/bin/python scripts/measure_dqi_region_ledger.py
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "dqi_region_ledger.json"

#: The six levers, in the order the ledger reports them.
LEVERS = ("forney", "clear", "sweep", "swap", "shift", "quotient")

#: Names `builder.mark(...)` uses for the regions this ledger reports. Anything
#: else at the innermost level is generator internals (a product's window, an
#: in-place add) or a wrapping region, and is attributed to its nearest named
#: ancestor.
REGIONS = (
    "initial_shift", "shift_step",
    "forward_eea", "eea_step", "update_flag", "update_metadata",
    "update_halting_counter", "update_counter",
    "multiply_and_add_or_sub", "sweep_positions", "quotient_div",
    "shift_b_array_right", "swap_a_and_b",
    "chien_forney_mbuc", "pre_align_reg_b", "forney_step", "unalign_reg_b",
    "measured_clear",
)

#: `arms.MRGatedDecoder` flags for one lever each.
LEVER_FLAGS = {
    "forney": {"gate_forney": True},
    "clear": {"clear_eea": True},
    "sweep": {"gate_sweep": True},
    "swap": {"gate_swap": True},
    "shift": {"gate_shift": True},
    "quotient": {"gate_sweep": True, "gate_quotient": True},
}

#: Flags to drop from the six-lever arm to instantiate "everything but this lever".
#: The quotient gate refines the sweep gate (its divisions sit inside the gated step),
#: so removing the sweep has to remove both.
WITHOUT = {
    "forney": ("gate_forney",),
    "clear": ("clear_eea",),
    "sweep": ("gate_sweep", "gate_quotient"),
    "swap": ("gate_swap",),
    "shift": ("gate_shift",),
    "quotient": ("gate_quotient",),
}


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


def native_builder(base: type, product: str) -> type:
    """The builder with `gf_product` set at construction time."""

    class WithProduct(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gf_product = product

    WithProduct.__name__ = f"{base.__name__}[{product}]"
    return WithProduct


def region_spans(circuit) -> tuple[list[tuple[int, int, str, bool]], list[str]]:
    """Split the instruction stream into spans of constant region and product state.

    A span is `(start, stop, innermost_named_region, inside_a_product)`. Walking the
    `INFO_TAG` stream only, rather than every instruction, keeps this O(tags)
    instead of O(10^7) tuples for a pinned-instance circuit.

    The second return value is the ordered list of tag names the circuit contains,
    so a caller can see what was actually emitted rather than what was expected.
    """
    tags = circuit.info_tags()
    if not tags:
        return [(0, len(circuit), "unmarked", False)], []

    spans: list[tuple[int, int, str, bool]] = []
    stack: list[str] = []
    names: list[str] = []
    named = "unmarked"
    in_product = False
    start = 0

    def state() -> tuple[str, bool]:
        current = "unmarked"
        for name in reversed(stack):
            if name in REGIONS:
                current = name
                break
        depth = any("mul" in name and name not in REGIONS for name in stack)
        return current, depth

    for index, text in tags:
        if text.startswith("@enter:"):
            stack.append(text[len("@enter:"):])
            names.append(text[len("@enter:"):])
        elif text.startswith("@exit:"):
            closing = text[len("@exit:"):]
            while stack and stack.pop() != closing:
                pass
        else:
            names.append(text)
        new_named, new_product = state()
        if (new_named, new_product) != (named, in_product):
            if index > start:
                spans.append((start, index, named, in_product))
            named, in_product = new_named, new_product
            start = index + 1
    if start < len(circuit):
        spans.append((start, len(circuit), named, in_product))
    return spans, names


def region_of(path: tuple[str, ...]) -> str:
    """The innermost mark that names one of the ledger's regions."""
    for name in reversed(path):
        if name in REGIONS:
            return name
    return "unmarked"


def build_arm(km, arms, shim, flags: dict, n: int, degree: int, m: int):
    decoder = arms.MRGatedDecoder(n, km.GF2Field(degree), **flags)
    builder = shim.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
    error_mbuc = [builder.create_classical_register(degree, name=f"E{j}") for j in range(m)]
    decoder.build_syndrome_decoding_mbuc(
        builder, syndromes=syndromes, error_mbuc=error_mbuc,
        n_iters=arms.DR.default_n_iters(n))
    return builder.finish_circuit(), syndromes, error_mbuc


def build_reference(km, arms, shim, n: int, degree: int, m: int):
    builder = shim.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
    error_mbuc = [builder.create_classical_register(degree, name=f"E{j}") for j in range(m)]
    arms.DR.ProosAndZalkaDecoder(n, km.GF2Field(degree)).build_syndrome_decoding_mbuc(
        builder, syndromes=syndromes, error_mbuc=error_mbuc,
        n_iters=arms.DR.default_n_iters(n))
    return builder.finish_circuit(), syndromes, error_mbuc


def simulate(km, circuit, syndromes, error_mbuc, syndromes_in, mbuc_in, degree, shots):
    """Run one arm and record both the totals and the per-instruction counts."""
    sim = km.Simulator(batch_size=shots, seed=99, counting_operations=True)
    sim.counting_op_executions = True
    sim.use_same_registers_as(circuit)
    flat_syndromes = km.array([q for register in syndromes for q in register])
    flat_errors = km.array([b for register in error_mbuc for b in register])
    from verify_submission_dqi import to_bits
    sim.write_across_shots(flat_syndromes, to_bits(syndromes_in, degree))
    sim.write_across_shots(flat_errors, to_bits(mbuc_in, degree))
    sim.do(circuit)

    counts = np.asarray(sim.operation_execution_counts(), dtype=np.int64)
    if counts.size != len(circuit):
        raise SystemExit(f"per-instruction counts are {counts.size} long, "
                         f"circuit is {len(circuit)}")
    toffoli_mask = np.asarray(circuit.op_toffoli_like, dtype=bool)
    per_shot_total = np.array(
        [int(sim.operation_statistics(i).num_toffoli_like) for i in range(shots)])
    if int(counts[toffoli_mask].sum()) != int(per_shot_total.sum()):
        raise SystemExit("per-instruction counts do not reconcile with "
                         f"num_toffoli_like: {int(counts[toffoli_mask].sum())} vs "
                         f"{int(per_shot_total.sum())}")

    syndrome_ids = {q.id for register in syndromes for q in register}
    junk_ids = [i for i in range(circuit.num_qubits) if i not in syndrome_ids]
    return {
        "counts": counts,
        "toffoli_mask": toffoli_mask,
        "toffoli_per_shot": per_shot_total,
        "mean_toffoli_like": float(per_shot_total.mean()),
        "syndromes": sim.read_across_shots(flat_syndromes),
        "junk": sim.read_across_shots([km.q(i) for i in junk_ids]),
        "phase": sim.read_phase_flipped_across_shots(),
    }


def region_ledger(circuit, out: dict, shots: int) -> dict:
    """Sum the executed Toffolis of each named region, and check the partition."""
    spans, tag_names = region_spans(circuit)
    counts = out["counts"]
    mask = out["toffoli_mask"]
    toffoli = counts * mask

    executed: dict[str, int] = {}
    static: dict[str, int] = {}
    instructions: dict[str, int] = {}
    product_executed: dict[str, int] = {}
    for start, stop, name, in_product in spans:
        instructions[name] = instructions.get(name, 0) + (stop - start)
        static[name] = static.get(name, 0) + int(mask[start:stop].sum())
        executed[name] = executed.get(name, 0) + int(toffoli[start:stop].sum())
        if in_product:
            product_executed[name] = (product_executed.get(name, 0)
                                      + int(toffoli[start:stop].sum()))

    total_executed = sum(executed.values())
    measured = int(toffoli.sum())
    if total_executed != measured:
        raise SystemExit(f"region sums {total_executed} != measured {measured}")
    static_total = int(mask.sum())
    if sum(static.values()) != static_total:
        raise SystemExit(f"static region sums {sum(static.values())} != {static_total}")

    rows = []
    for name in sorted(executed, key=lambda k: -executed[k]):
        rows.append({
            "region": name,
            "instructions": instructions[name],
            "executed_toffoli_per_shot": executed[name] / shots,
            "static_toffoli": static[name],
            "executed_fraction_of_static": (executed[name] / shots / static[name]
                                            if static[name] else None),
            "product_toffoli_per_shot": product_executed.get(name, 0) / shots,
        })
    return {
        "rows": rows,
        "tag_names": tag_names,
        "executed_total_per_shot": total_executed / shots,
        "product_total_per_shot": sum(product_executed.values()) / shots,
        "static_total": static_total,
        "reconciles_with_measured_total": total_executed == measured,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=4095)
    parser.add_argument("--n", type=int, default=70)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--kickmix-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--skip-levers", action="store_true",
                        help="only the six-lever arm; no per-lever reinstantiation")
    args = parser.parse_args()

    runner = load_runner()
    km = runner.load_kickmix(args.kickmix_build)
    if not hasattr(km.Simulator, "operation_execution_counts"):
        raise SystemExit(f"{km.__file__} has no per-instruction counts; rebuild with "
                         f"scripts/build_kickmix_mr.py")
    base = runner.builder_class(km)
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    import verify_submission_dqi as V

    arms = load_arms(km, runner)
    shim = runner.shim_for(km, native_builder(base, "and54"))

    spec = V.ProblemSpecDQI.benchmark_instance(
        m=args.m, n=args.n, degree=args.degree, num_shots=args.shots)
    weights = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    tables = V.FieldTables(km.GF2Field(spec.degree))
    rng = np.random.default_rng(20261028)
    syndromes_in, mbuc_in, _ = V.sample_shots(spec, tables, rng, args.shots, weights)
    reference_inputs = V.to_bits(syndromes_in, args.degree)

    original = arms.DR.km
    arms.DR.km = shim
    try:
        ref_circuit, ref_syn, ref_mbuc = build_reference(km, arms, shim, args.n,
                                                         args.degree, args.m)
        reference = simulate(km, ref_circuit, ref_syn, ref_mbuc, syndromes_in, mbuc_in,
                             args.degree, args.shots)

        six = LEVER_FLAGS["forney"] | LEVER_FLAGS["clear"] | LEVER_FLAGS["sweep"] | \
            LEVER_FLAGS["swap"] | LEVER_FLAGS["shift"] | LEVER_FLAGS["quotient"]
        arms_out = {}
        for label, flags in [("six_levers", six)] + (
                [] if args.skip_levers else [
                    (f"without_{name}",
                     {k: v for k, v in six.items() if k not in WITHOUT[name]})
                    for name in LEVERS]):
            circuit, syn, mbuc = build_arm(km, arms, shim, flags, args.n, args.degree, args.m)
            out = simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in,
                           args.degree, args.shots)
            arms_out[label] = {
                "qubits": circuit.num_qubits,
                "mean_toffoli_like": out["mean_toffoli_like"],
                "ledger": region_ledger(circuit, out, args.shots),
                "syndromes_restored": bool(np.array_equal(out["syndromes"],
                                                         reference_inputs)),
                "junk_clean": bool(not out["junk"].any()),
                "phase_matches_reference": bool(np.array_equal(out["phase"],
                                                               reference["phase"])),
            }
            # A pinned-instance circuit is hundreds of megabytes; holding seven of
            # them is what makes the run unnecessary rather than cheap.
            del circuit, out
            gc.collect()
    finally:
        arms.DR.km = original

    six_row = arms_out["six_levers"]
    if not six_row["syndromes_restored"] or not six_row["junk_clean"]:
        raise SystemExit("the six-lever arm is not correct; refusing to write a ledger")

    record = {
        "engine": "kickmix",
        "kickmix": str(Path(km.__file__).resolve()),
        "kickmix_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
        "kickmix_role": "local rebuild with MR and per-instruction execution counts "
                        "(scripts/build_kickmix_mr.py); not the pinned competition binary",
        "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                     "shots": args.shots, "seed": 20261028},
        "policy": "and54",
        "levers": six,
        "method": "each instruction is attributed to the innermost builder.mark(...) "
                  "region it sits inside; a region's executed count is the sum of the "
                  "per-instruction counts of the Toffoli-like instructions in it, so the "
                  "rows partition the measured total exactly",
        "six_levers": {
            "qubits": six_row["qubits"],
            "mean_toffoli_like": six_row["mean_toffoli_like"],
            "regions": six_row["ledger"]["rows"],
            "executed_total_per_shot": six_row["ledger"]["executed_total_per_shot"],
            "static_total": six_row["ledger"]["static_total"],
            "reconciles_with_measured_total": six_row["ledger"][
                "reconciles_with_measured_total"],
            "correct": bool(six_row["syndromes_restored"] and six_row["junk_clean"]),
        },
    }

    if not args.skip_levers:
        # One lever per arm, so a region's removable cost is its cost in the arm that
        # leaves it ungated minus its cost in the arm that gates it.
        per_lever = {}
        for name in LEVERS:
            arm = arms_out[f"without_{name}"]
            before = {row["region"]: row["executed_toffoli_per_shot"]
                      for row in arm["ledger"]["rows"]}
            after = {row["region"]: row["executed_toffoli_per_shot"]
                     for row in six_row["ledger"]["rows"]}
            removed = {region: before.get(region, 0.0) - after.get(region, 0.0)
                       for region in sorted(set(before) | set(after))}
            applied = {region: delta for region, delta in removed.items()
                       if abs(delta) > 1e-6}
            per_lever[name] = {
                "qubits": arm["qubits"],
                "mean_toffoli_like": arm["mean_toffoli_like"],
                "arm_delta_per_shot": arm["mean_toffoli_like"] - six_row[
                    "mean_toffoli_like"],
                "regions_that_moved": applied,
                "largest_region": (max(applied, key=applied.get) if applied else None),
                "correct": bool(arm["syndromes_restored"] and arm["junk_clean"]
                                and arm["phase_matches_reference"]),
            }
        record["per_lever"] = per_lever
        record["per_lever_check"] = {
            "every_arm_correct": all(row["correct"] for row in per_lever.values()),
            "measured_deltas_match_region_deltas": all(
                abs(sum(row["regions_that_moved"].values()) - row["arm_delta_per_shot"]) < 0.51
                for row in per_lever.values()),
        }

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")

    print(f"six-lever arm: Q={six_row['qubits']} executed T="
          f"{six_row['mean_toffoli_like']:.1f}/shot  static={record['six_levers']['static_total']}")
    print(f"  partition reconciles: {record['six_levers']['reconciles_with_measured_total']}")
    print(f"{'region':28s} {'executed/shot':>14s} {'static':>10s} {'executed':>9s}")
    for row in record["six_levers"]["regions"]:
        frac = row["executed_fraction_of_static"]
        print(f"{row['region']:28s} {row['executed_toffoli_per_shot']:14.1f} "
              f"{row['static_toffoli']:10d} {'' if frac is None else f'{frac:8.1%}'}")
    if "per_lever" in record:
        print()
        for name, row in record["per_lever"].items():
            print(f"  without_{name:8s} Q={row['qubits']} T={row['mean_toffoli_like']:12.1f} "
                  f"delta={row['arm_delta_per_shot']:+11.1f} correct={row['correct']} "
                  f"moved={row['largest_region']}")
        print(f"  every arm correct: {record['per_lever_check']['every_arm_correct']}; "
              f"deltas match regions: "
              f"{record['per_lever_check']['measured_deltas_match_region_deltas']}")
    print(f"\nrecord: {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
