#!/usr/bin/env python3
"""Price the two things a "better circuit" would have to come from: the field multiply
and the static gate stream.

The gate-level levers are exhausted (`docs/MR_INSTRUCTION_SPEC.md` §6a–§6b), so the next
question is whether the arithmetic itself can be cheaper. This script measures, in one
place:

* **The multiply.** `ixor_gf2_mul` against the Karatsuba AND-count model of
  `gen_gf_mul.h`, at six field degrees. If the model matches the measurement at every
  degree, the built-in multiply is the Karatsuba recursion to the gate, and the only
  cheaper multiply left is a different bilinear algorithm.
* **The split search.** Every split `k` the recursion allows at m=12, against the
  measured 63, so "no split beats the built-in" is arithmetic rather than assertion.
* **The static stream.** Static (emitted, not executed) Toffoli-like count for the
  reference and for the five- and six-lever arms. Gating adds conditions instead of
  removing gates, so this is the metric on which the levers do *not* show up — unless the
  lever deletes emission, as `clear_eea` does.
* **The peephole headroom.** Every builder call is recorded in emission order (at the
  default, small instance) and the pass removes immediately-cancelling pairs. This is
  what a local rewrite can find without commuting gates past each other, and it is
  measured rather than assumed to be zero.
* **The multiply's share** (`--official` only). The sweep's multiply is ablated to a
  no-op and the executed count differenced. The ablated circuit is wrong; only its cost
  is used, as the multiplier's share of the bill, and the gain of a hypothetical cheaper
  multiply is derived from that share.

Everything here is a measurement on the local MR rebuild, not a score. Run it under the
interpreter `kickmix` was built for (CPython 3.13)::

    /dev/shm/qvcc-venv313/bin/python scripts/measure_gf_mul_and_static_headroom.py
    /dev/shm/qvcc-venv313/bin/python scripts/measure_gf_mul_and_static_headroom.py \\
        --official --record benchmarks/quantum_compilation/gf_mul_and_static_headroom.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "gf_mul_and_static_headroom.json"

#: The pinned reference `T_mean` these ratios are quoted against.
REFERENCE_T_MEAN = 5_725_071.61

#: Toffoli-like ops a peephole can annihilate as an adjacent pair.
SELF_INVERSE = ("ccx", "cswap", "multi_controlled_x")
TOFFOLI_LIKE = ("ccx", "multi_controlled_x", "cswap", "init_and", "del_and")

LEVER_FLAGS = {
    "reference": {},
    "five levers": {"gate_forney": True, "clear_eea": True, "gate_sweep": True,
                    "gate_swap": True, "gate_shift": True},
    "six levers": {"gate_forney": True, "clear_eea": True, "gate_sweep": True,
                   "gate_swap": True, "gate_shift": True, "gate_quotient": True},
}


def karatsuba(n: int) -> int:
    """ANDs in the in-place Karatsuba product of two n-bit factors (`gen_gf_mul.h`)."""
    if n <= 1:
        return 1
    k = (n + 1) // 2
    return karatsuba(k) + karatsuba(n - k) + karatsuba(k)


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


def _ids(value, km):
    if hasattr(value, "id"):
        return int(value.id)
    if isinstance(value, km.array):
        return tuple(_ids(q, km) for q in value)
    if isinstance(value, (list, tuple)):
        return tuple(_ids(item, km) for item in value)
    return type(value).__name__


class Recording:
    """Mixin: records every builder call, in emission order, on `events`.

    Built as a class factory because the base builder class comes from the extension.
    Allocation, marks and condition handling are not gates and are skipped.
    """

    @staticmethod
    def make(base, km):
        skip = {"mark", "finish_circuit", "alloc_qubits", "alloc_bits", "free", "free_bits",
                "create_quantum_register", "create_classical_register", "use_same_registers_as",
                "push_condition", "pop_condition", "num_free_qubits", "append_circuit"}

        class RecordingBuilder(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.events = []

            def __getattribute__(self, name):
                attr = object.__getattribute__(self, name)
                if name.startswith("_") or name in skip or not callable(attr):
                    return attr
                events = object.__getattribute__(self, "events")

                def recorded(*args, **kwargs):
                    events.append((name, tuple(_ids(a, km) for a in args),
                                   _ids(kwargs.get("control"), km)))
                    return object.__getattribute__(self, name)(*args, **kwargs)

                return recorded

        return RecordingBuilder


def measure_multiply(km, degrees):
    """Static Toffoli-like per `ixor_gf2_mul`, against the Karatsuba model."""
    rows = {}
    for degree in degrees:
        builder = km.CircuitBuilder()
        target = builder.alloc_qubits(degree)
        lhs = builder.alloc_qubits(degree)
        rhs = builder.alloc_qubits(degree)
        builder.ixor_gf2_mul(lhs, rhs, target=target, field=km.GF2Field(degree))
        measured = int(builder.finish_circuit().operation_statistics().num_toffoli_like)
        rows[degree] = {"measured": measured, "karatsuba": karatsuba(degree)}
    return rows


def split_table(n: int) -> dict[int, int]:
    """ANDs for every split the recursion allows, plus schoolbook."""
    table = {k: karatsuba(k) + karatsuba(n - k) + karatsuba(k)
             for k in range((n + 1) // 2, n)}
    table["schoolbook"] = n * n
    return table


def peephole(events) -> Counter:
    """Adjacent pairs that cancel: a repeat of a self-inverse op, or an AND pair.

    An `init_and` immediately followed by its `del_and` with the same operands emits two
    Toffolis that annihilate. A repeated `ccx`/`cswap` cancels the same way. Anything
    else would need a commutation the pass does not attempt.
    """
    removable: Counter = Counter()
    index = 0
    while index + 1 < len(events):
        (name, sig, ctl), (nxt, nsig, nctl) = events[index], events[index + 1]
        same = sig == nsig and ctl == nctl
        if name == nxt and same and name in SELF_INVERSE:
            removable[name] += 1
            index += 2
            continue
        if {name, nxt} == {"init_and", "del_and"} and same:
            removable["and-pair"] += 1
            index += 2
            continue
        index += 1
    return removable


def build(km, runner, arms, flags, m, n, degree, recording=False):
    """Build one arm; returns (circuit, recorded events or [])."""
    base = runner.builder_class(km)
    builder_cls = Recording.make(base, km) if recording else base
    shim = runner.shim_for(km, builder_cls)
    original = arms.DR.km
    arms.DR.km = shim
    try:
        decoder = arms.MRGatedDecoder(n, km.GF2Field(degree), **flags)
        builder = shim.CircuitBuilder()
        syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
        error_mbuc = [builder.create_classical_register(degree, name=f"E{j}") for j in range(m)]
        decoder.build_syndrome_decoding_mbuc(
            builder, syndromes=syndromes, error_mbuc=error_mbuc,
            n_iters=arms.DR.default_n_iters(n))
        circuit = builder.finish_circuit()
        return circuit, getattr(builder, "events", [])
    finally:
        arms.DR.km = original


def static_toffoli(circuit) -> tuple[int, dict]:
    stats = {name: int(count) for name, count in dict(circuit.operation_statistics()).items()
             if count}
    like = sum(stats.get(name, 0) for name in ("CCX", "CSWAP", "AND"))
    return like, stats


def load_mul():
    spec = importlib.util.spec_from_file_location("gf54_mul",
                                                  ROOT / "scripts" / "gf54_mul.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Toffolis per controlled multiply: the ANDs, plus the controlled add into the target.
BUILTIN_ANDS = 63
GF54_ANDS = 54
CONTROLLED_ADD = 12


def multiply_share(km, runner, arms, m, n, degree, shots):
    """Executed cost of the six-lever arm, decomposed into multiply and everything else.

    The share comes from two arms that differ *only* in what the multiply costs: the
    built-in 63-AND one and the 54-AND one from `scripts/gf54_mul.py`. Deleting the
    multiply instead -- the first version of this measurement -- is not valid, because
    the deleted call leaves the registers wrong and every predicate gating the rest of
    the EEA then reads garbage, so the differenced circuit executes a different amount
    of *other* work too. Both arms here are correct circuits and run on the same shots,
    so their CCX difference is exactly `calls * (75 - 66)`.
    """
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    import verify_submission_dqi as V

    mul = load_mul()
    spec = V.ProblemSpecDQI.benchmark_instance(m=m, n=n, degree=degree, num_shots=shots)
    weights = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    tables = V.FieldTables(km.GF2Field(degree))
    rng = np.random.default_rng(20261028)
    syndromes_in, mbuc_in, _phase = V.sample_shots(spec, tables, rng, shots, weights)

    base = runner.builder_class(km)

    def run(builder_cls) -> dict:
        shim = runner.shim_for(km, builder_cls)
        original = arms.DR.km
        arms.DR.km = shim
        try:
            decoder = arms.MRGatedDecoder(n, km.GF2Field(degree), **LEVER_FLAGS["six levers"])
            builder = shim.CircuitBuilder()
            syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
            error_mbuc = [builder.create_classical_register(degree, name=f"E{j}")
                          for j in range(m)]
            decoder.build_syndrome_decoding_mbuc(
                builder, syndromes=syndromes, error_mbuc=error_mbuc,
                n_iters=arms.DR.default_n_iters(n))
            circuit = builder.finish_circuit()
        finally:
            arms.DR.km = original
        sim = km.Simulator(batch_size=shots, seed=99, counting_operations=True)
        sim.use_same_registers_as(circuit)
        sim.write_across_shots(km.array([q for r in syndromes for q in r]),
                               V.to_bits(syndromes_in, degree))
        sim.write_across_shots(km.array([b for r in error_mbuc for b in r]),
                               V.to_bits(mbuc_in, degree))
        sim.do(circuit)
        rows = [dict(sim.operation_statistics(i)) for i in range(shots)]
        return {
            "toffoli": float(np.mean([int(sim.operation_statistics(i).num_toffoli_like)
                                      for i in range(shots)])),
            "ccx": float(np.mean([row.get("CCX", 0) for row in rows])),
        }

    cheap_builder = type("Gf54Builder", (mul.Gf54MulMixin, base), {})
    full = run(base)
    cheap = run(cheap_builder)
    difference = full["ccx"] - cheap["ccx"]
    per_call = (BUILTIN_ANDS + CONTROLLED_ADD) - (GF54_ANDS + CONTROLLED_ADD)
    if difference % per_call:
        raise SystemExit(f"the arms' CCX counts differ by {difference}, not a multiple of "
                         f"{per_call}")
    calls = int(round(difference / per_call))
    multiply = calls * (BUILTIN_ANDS + CONTROLLED_ADD)
    return {
        "six_levers_t_mean": full["toffoli"],
        "six_levers_54_and_t_mean": cheap["toffoli"],
        "executed_multiply_calls": calls,
        "multiply_t_mean": multiply,
        "everything_else_t_mean": full["toffoli"] - multiply,
        "multiply_share_percent": 100 * multiply / full["toffoli"],
        "ratio_vs_reference": REFERENCE_T_MEAN / full["toffoli"],
        "ccx_delta_per_call": per_call,
        "measured_in": "benchmarks/quantum_compilation/gf54_mul_kickmix.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--official", action="store_true",
                        help="official instance (4095/70/GF(2^12)); adds the multiply share")
    parser.add_argument("--m", type=int, default=15)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--shots", type=int, default=8,
                        help="only used by the multiply-share ablation (--official)")
    parser.add_argument("--kickmix-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--record", type=Path, default=None)
    args = parser.parse_args()

    if args.official:
        args.m, args.n, args.degree = 4095, 70, 12
        args.shots = 64

    runner = load_runner()
    km = runner.load_kickmix(args.kickmix_build)
    arms = load_arms(km, runner)

    record: dict[str, object] = {
        "engine": "kickmix",
        "kickmix": str(Path(km.__file__).resolve()),
        "kickmix_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
        "kickmix_role": "local rebuild with MR (scripts/build_kickmix_mr.py); not the "
                        "pinned competition binary and not eligible for a receipt",
        "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                     "shots": args.shots, "official": args.official},
        "multiply": {
            "measure": "static Toffoli-like ops in one uncontrolled ixor_gf2_mul",
            "degrees": measure_multiply(km, (2, 3, 4, 6, 8, 12)),
        },
        "splits_at_degree_12": split_table(12),
        "static": {},
        "peephole": {},
    }
    record["multiply"]["karatsuba_is_exact"] = all(
        row["measured"] == row["karatsuba"]
        for row in record["multiply"]["degrees"].values())

    for label, flags in LEVER_FLAGS.items():
        circuit, events = build(km, runner, arms, flags, args.m, args.n, args.degree,
                                recording=True)
        like, stats = static_toffoli(circuit)
        found = peephole(events)
        record["static"][label] = {"toffoli_like": like, "qubits": circuit.num_qubits,
                                   "ops": stats}
        record["peephole"][label] = {"events": len(events),
                                     "removable": sum(found.values()),
                                     "by_kind": dict(found)}

    static = record["static"]
    reference_static = static["reference"]["toffoli_like"]
    for label, row in static.items():
        row["static_ratio_vs_reference"] = reference_static / row["toffoli_like"]
    for label, row in record["peephole"].items():
        row["removable_percent"] = (100 * row["removable"]
                                    / static[label]["toffoli_like"]
                                    if static[label]["toffoli_like"] else 0.0)

    if args.official:
        executed = multiply_share(km, runner, arms, args.m, args.n, args.degree, args.shots)
        calls = executed["executed_multiply_calls"]
        rest = executed["everything_else_t_mean"]
        # A multiply costs its ANDs plus the controlled add, so `t(ands) = rest + calls *
        # (ands + 12)`. The add is in every row, including the free one.
        for ands in (63, 54, 45, 0):
            executed[f"t_mean_at_{ands}_ands"] = rest + calls * (ands + CONTROLLED_ADD)
        executed["ratio_with_free_multiply"] = REFERENCE_T_MEAN / (
            rest + calls * CONTROLLED_ADD)
        record["multiply"]["executed"] = executed
        record["multiply"]["derived_if_cheaper"] = {
            f"{ands}_ands": {
                "t_mean": rest + calls * (ands + CONTROLLED_ADD),
                "ratio_vs_reference": REFERENCE_T_MEAN / (rest + calls * (ands + CONTROLLED_ADD)),
            }
            for ands in (56, 50, 45, 40)
        }

    record["finding"] = _finding(record)

    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n")

    degrees = record["multiply"]["degrees"]
    print("multiply (static Toffoli per ixor_gf2_mul)")
    for degree, row in degrees.items():
        print(f"  GF(2^{degree:<2}) Karatsuba {row['karatsuba']:4d}  measured {row['measured']:4d}")
    print(f"  exact match at every degree: {record['multiply']['karatsuba_is_exact']}")
    print("splits at m=12: " + ", ".join(f"{k}:{v}" for k, v in
                                         record["splits_at_degree_12"].items()))
    print(f"{'arm':12s} {'static T':>12s} {'ratio':>7s} {'removable':>10s} {'%':>6s}")
    for label, row in static.items():
        pf = record["peephole"][label]
        print(f"{label:12s} {row['toffoli_like']:12,d} {row['static_ratio_vs_reference']:7.3f} "
              f"{pf['removable']:10,d} {pf['removable_percent']:6.3f}")
    if args.official:
        executed = record["multiply"]["executed"]
        print(f"executed, six levers: {executed['six_levers_t_mean']:,.1f} T/shot "
              f"({executed['ratio_vs_reference']:.3f}x)")
        print(f"  multiply: {executed['multiply_t_mean']:,.1f} "
              f"({executed['multiply_share_percent']:.2f}%, "
              f"{executed['executed_multiply_calls']:,} calls); free: "
              f"{executed['ratio_with_free_multiply']:.3f}x")
        for name, row in record["multiply"]["derived_if_cheaper"].items():
            print(f"  {name:8s} -> {row['t_mean']:13,.1f} T/shot  {row['ratio_vs_reference']:.3f}x")
    if args.record is not None:
        print(f"record: {args.record}")
    return 0


def _finding(record: dict) -> str:
    degrees = record["multiply"]["degrees"]
    exact = record["multiply"]["karatsuba_is_exact"]
    best = min(v for k, v in record["splits_at_degree_12"].items() if k != "schoolbook")
    static = record["static"]
    six = static["six levers"]["toffoli_like"]
    five = static["five levers"]["toffoli_like"]
    reference = static["reference"]["toffoli_like"]
    removable = sum(row["removable"] for row in record["peephole"].values())
    text = (
        f"`ixor_gf2_mul` is the Karatsuba recursion to the gate: "
        f"{degrees[12]['measured']} static Toffolis at GF(2^12), matching K(12) = "
        f"{degrees[12]['karatsuba']}"
        + ("" if exact else " -- NOT matching at every degree") +
        f", and matching at every degree measured ("
        + ", ".join(f"GF(2^{d})={r['measured']}" for d, r in degrees.items()) + "). "
        f"No split of the recursion beats it at m=12: the best is {best}, schoolbook is 144. "
        f"Gating is invisible in the emitted stream -- six levers emit {six:,} static "
        f"Toffolis against the reference's {reference:,} ({reference / six:.3f}x), and the "
        f"quotient gate changes that by exactly 0, because it adds conditions rather than "
        f"removing gates; the whole static saving is `clear_eea`, which deletes the reverse "
        f"EEA ({reference:,} vs five levers {five:,}). A local peephole removes "
        f"{removable:,} Toffolis across both arms, so there is no adjacent-cancellation "
        f"headroom. "
    )
    executed = record["multiply"].get("executed")
    if executed:
        text += (
            f"The sweep's multiply is {executed['multiply_share_percent']:.2f}% of the "
            f"six-lever arm's executed Toffolis ({executed['multiply_t_mean']:,.1f} of "
            f"{executed['six_levers_t_mean']:,.1f} per shot, "
            f"{executed['executed_multiply_calls']:,} calls), which makes it the largest "
            f"remaining lever but not the only one: a hypothetical 45-AND multiply would "
            f"give {record['multiply']['derived_if_cheaper']['45_ands']['ratio_vs_reference']:.3f}x, "
            f"and a multiply that cost nothing but its {CONTROLLED_ADD}-Toffoli controlled "
            f"add would give {executed['ratio_with_free_multiply']:.3f}x, still above "
            f"Benchmark II's 445,229. No such cheaper coherent multiply was in hand when this "
            f"was recorded; a free one requires measuring an operand, which "
            f"`docs/CIRCUIT_OPTIMISATION_RULES.md` rules inadmissible."
        )
    return text


if __name__ == "__main__":
    raise SystemExit(main())
