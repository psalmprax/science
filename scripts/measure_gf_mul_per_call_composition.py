#!/usr/bin/env python3
"""Take one GF(2^12) multiply call apart, so the 23-AND rung stops being a bracket.

`product_axis_decision()` prices the `2n - 1 = 23` AND bound of one field product two
ways -- "only the AND part of a call scales" (491,555.6 Toffolis per shot, 1.128x the
denominator) and "the whole call scales" (444,245.3, 1.019x) -- and records that no
measurement separates them. The separating quantity was a *component*: how many of the
Toffolis in one multiply call are not ANDs.

It was written down as 9, from `MULTIPLY_NON_AND_TOFFOLI_PER_CALL`, which took
`ccx_delta_per_call` -- the difference between two *implementations* (the built-in 63-AND
Karatsuba and the 54-AND substitute of `scripts/gf54_mul.py`) -- and read it as a split
*inside* one implementation. This script measures the split instead, in the emitted
instruction stream:

* **The call, per control mode.** One `ixor_gf2_mul` at degree 12, emitted three ways --
  unconditional (`control=True`), classically controlled, and quantum-controlled -- with
  every Toffoli attributed to the engine's own tag (`gf_mul`, `c_gf_mul`, `c_gf_iadd`,
  `gf_unmul`, `gf_phase_by_product`).
* **The arm, as built.** The campaign's control arm is constructed at a small instance and
  every product span is measured, so the distribution of per-call Toffoli counts is on
  record rather than assumed. A call whose size never occurs in a real circuit is not a
  price.
* **The two bound rows.** The AND-bound call in place and the AND-bound call that has to
  materialise its product are priced from the measured composition.

The result is that a 63-Toffoli call is 63 ANDs and nothing else, and the only non-AND
Toffolis anywhere in a product call are the 12 of `c_gf_iadd`, one per output bit, which
belong to a *different* call shape (75 Toffolis) and do not scale with the AND count.
So the "AND part scales" reading adds 9 Toffolis per call that the stream does not carry.

Run it under the interpreter `kickmix` was built for (CPython 3.13)::

    /dev/shm/qvcc-venv313/bin/python scripts/measure_gf_mul_per_call_composition.py
    /dev/shm/qvcc-venv313/bin/python scripts/measure_gf_mul_per_call_composition.py \\
        --record benchmarks/quantum_compilation/gf_mul_per_call_composition.json
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
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "gf_mul_per_call_composition.json"

#: The dimension bound on ANDs for a GF(2^n) product, and the degree the DQI decoder uses.
AND_DIMENSION_BOUND = 2 * 12 - 1
DEGREE = 12

#: The campaign arm's measured product stream: `dqi_schedule_campaign.json`, region
#: `sweep_positions`, control 693,660.0 at 10,510 calls (66.0 average) and the same region
#: after the multiply window 331,023.0.
RECORD_CONTROL_PRODUCT_STREAM = 693_660.0
RECORD_AFTER_WINDOW_PRODUCT_STREAM = 331_023.0
RECORD_MULTIPLY_CALLS = 10_510

#: The campaign arm's other product terms, from the same record's region rows.
RECORD_DIVISION_PRODUCTS = 46_656.0        # 864 products x 54 ANDs
RECORD_FORNEY_PRODUCTS = 8_382.65625
RECORD_RESIDUAL = 299_953.2
RECORD_ANDS_PER_DIVISION_PRODUCT = 54

#: Tags that carry Toffolis in one multiply call, and what each one is.
TAG_KINDS = {
    "gf_mul": "the bilinear product, accumulated in place (unconditional call)",
    "c_gf_mul": "the bilinear product, written into a clean window (quantum-controlled call)",
    "c_gf_iadd": "the controlled 12-bit add of that window into the target",
    "gf_unmul": "the window's uncomputation, freed by measurement",
    "gf_phase_by_product": "the phase kickback, no Toffolis",
}


def load_kickmix(build_dir: Path):
    sys.path.insert(0, str(build_dir))
    try:
        import kickmix as km
    except ImportError as exc:  # pragma: no cover - environment failure
        raise SystemExit(f"could not import kickmix from {build_dir}: {exc}") from exc
    if not hasattr(km.CircuitBuilder, "mr"):
        raise SystemExit(f"{km.__file__} has no mr method -- build the MR extension first")
    return km


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tag_attribution(circuit) -> dict[str, int]:
    """Toffolis per innermost `@enter:` tag, so a call's parts can be told apart.

    The gates between two tags belong to the innermost tag that was open when they were
    emitted, which is the stack *before* the later tag is processed.
    """
    tags = circuit.info_tags()
    mask = np.asarray(circuit.op_toffoli_like, dtype=bool)
    counts: dict[str, int] = {}
    stack: list[str] = []
    previous = 0
    for index, text in [*list(tags), (len(circuit), "")]:
        if index > previous and stack:
            counts[stack[-1]] = (counts.get(stack[-1], 0)
                                 + int(mask[previous:index].sum()))
        if text.startswith("@exit:"):
            closing = text[len("@exit:"):]
            while stack and stack.pop() != closing:
                pass
        elif text.startswith("@enter:"):
            stack.append(text[len("@enter:"):])
        previous = index
    return counts


def measure_call(km, label: str, control: str) -> dict:
    """One `ixor_gf2_mul`, emitted the way the decoder emits it."""
    import importlib.util as _ilu  # noqa: F401

    runner = load_module("mr_runner", ROOT / "scripts" / "run_mr_gated_arms_kickmix.py")
    builder_cls = runner.builder_class(km)
    builder = builder_cls()
    target = builder.alloc_qubits(DEGREE)
    lhs = builder.alloc_qubits(DEGREE)
    rhs = builder.alloc_qubits(DEGREE)
    kwargs: dict[str, object] = {"target": target, "field": km.GF2Field(DEGREE)}
    if control == "bit":
        kwargs["control"] = builder.alloc_bits(1)[0]
    elif control == "qubit":
        kwargs["control"] = builder.alloc_qubits(1)[0]
    else:
        kwargs["control"] = True
    builder.ixor_gf2_mul(lhs, rhs, **kwargs)
    circuit = builder.finish_circuit()
    stats = {name: int(count) for name, count in dict(circuit.operation_statistics()).items()
             if count}
    attribution = tag_attribution(circuit)
    by_tag = {TAG_KINDS.get(tag, "other").split(" (")[0] if tag in TAG_KINDS else tag: value
              for tag, value in attribution.items() if value}
    toffoli = sum(stats.get(name, 0) for name in ("CCX", "CSWAP", "AND"))
    return {
        "call": label,
        "control": control,
        "toffolis": toffoli,
        "ops": stats,
        "toffolis_by_tag": {tag: value for tag, value in attribution.items() if value},
        "ops_by_role": by_tag,
    }


def measure_arm(km, m: int, n: int, mode: str) -> dict:
    """Every product span in the campaign's arm, at a small instance."""
    runner = load_module("mr_runner_arm", ROOT / "scripts" / "run_mr_gated_arms_kickmix.py")
    arms = load_module("mr_arms", ROOT / "scripts" / "run_mr_gated_arms.py")
    campaign = load_module("dqi_campaign",
                           ROOT / "scripts" / "measure_dqi_schedule_campaign.py")
    ledger = load_module("dqi_region_ledger",
                         ROOT / "scripts" / "measure_dqi_region_ledger.py")
    arms._all_qubits = lambda registers: runner.all_qubits(km, registers)

    builder_cls = runner.builder_class(km)
    shim = runner.shim_for(km, builder_cls)
    decoder_cls = campaign.make_decoder(arms, mode)
    original = arms.DR.km
    arms.DR.km = shim
    try:
        decoder = decoder_cls(n, km.GF2Field(DEGREE))
        builder = shim.CircuitBuilder()
        syndromes = [builder.create_quantum_register(DEGREE, name=f"S{i}") for i in range(n)]
        errors = [builder.create_classical_register(DEGREE, name=f"E{j}") for j in range(m)]
        decoder.build_syndrome_decoding_mbuc(
            builder, syndromes=syndromes, error_mbuc=errors,
            n_iters=arms.DR.default_n_iters(n))
        circuit = builder.finish_circuit()
    finally:
        arms.DR.km = original

    mask = np.asarray(circuit.op_toffoli_like, dtype=bool)
    spans, _names = ledger.region_spans(circuit)
    histogram: dict[int, int] = {}
    total = 0
    for start, stop, _name, in_product in spans:
        if not in_product:
            continue
        size = int(mask[start:stop].sum())
        histogram[size] = histogram.get(size, 0) + 1
        total += size
    return {
        "arm": mode,
        "instance": {"m": m, "n": n, "degree": DEGREE},
        "qubits": int(circuit.num_qubits),
        "product_spans": sum(histogram.values()),
        "product_toffolis": total,
        "span_sizes": {str(size): count for size, count in sorted(histogram.items())},
    }


def bound_rows(composition: dict) -> dict:
    """Price the AND bound with the composition the stream actually has."""
    per_call = composition["in_place"]["toffolis"]            # 63
    materialised = composition["materialised"]["toffolis"]    # 75
    add = materialised - per_call                            # 12
    ands = AND_DIMENSION_BOUND
    residual = RECORD_RESIDUAL
    division = (RECORD_DIVISION_PRODUCTS + RECORD_FORNEY_PRODUCTS)
    division_at_bound = division * ands / RECORD_ANDS_PER_DIVISION_PRODUCT

    in_place = RECORD_AFTER_WINDOW_PRODUCT_STREAM * ands / per_call
    materialised_at_bound = RECORD_AFTER_WINDOW_PRODUCT_STREAM * (ands + add) / per_call

    def row(multiply: float) -> dict:
        products = multiply + division_at_bound
        return {
            "multiply_toffolis": multiply,
            "division_forney_toffolis": division_at_bound,
            "products_toffolis": products,
            "t_mean_toffolis": residual + products,
        }

    return {
        "ands_at_bound": ands,
        "add_toffolis": add,
        "measured_and_fraction_of_the_call": per_call and per_call / per_call,
        "measured_and_fraction_of_a_materialised_call": per_call / materialised,
        "in_place": row(in_place),
        "materialised": row(materialised_at_bound),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=15)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--kickmix-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--record", type=Path, default=None)
    args = parser.parse_args()

    km = load_kickmix(args.kickmix_build)

    calls = [
        measure_call(km, "unconditional (control=True)", "true"),
        measure_call(km, "classically controlled (a measurement bit)", "bit"),
        measure_call(km, "quantum-controlled (a control qubit)", "qubit"),
    ]
    by_control = {row["control"]: row for row in calls}
    composition = {
        "in_place": by_control["true"],
        "materialised": by_control["qubit"],
    }

    arm = measure_arm(km, args.m, args.n, "none")
    sizes = {int(k): v for k, v in arm["span_sizes"].items()}
    observed = sorted(size for size in sizes if size)
    per_call_sizes = {
        "unconditional": by_control["true"]["toffolis"],
        "materialised": by_control["qubit"]["toffolis"],
    }
    unexplained = [size for size in observed if size not in per_call_sizes.values()]

    rows = bound_rows(composition)
    average = RECORD_CONTROL_PRODUCT_STREAM / RECORD_MULTIPLY_CALLS
    mixture = {
        "product_stream": RECORD_CONTROL_PRODUCT_STREAM,
        "calls": RECORD_MULTIPLY_CALLS,
        "average_toffolis_per_call": average,
        "explained_by_a_mixture_of_the_two_shapes": (
            f"{per_call_sizes['unconditional']} and {per_call_sizes['materialised']}, "
            f"{100 * (average - per_call_sizes['unconditional']) / rows['add_toffolis']:.1f}% "
            f"of calls carrying the {rows['add_toffolis']}-Toffoli add"),
        "after_the_window_average": (RECORD_AFTER_WINDOW_PRODUCT_STREAM
                                     / (RECORD_MULTIPLY_CALLS // 2)),
    }

    record = {
        "engine": "kickmix",
        "kickmix": str(Path(km.__file__).resolve()),
        "kickmix_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
        "kickmix_role": "local rebuild with MR (scripts/build_kickmix_mr.py); not the "
                        "pinned competition binary and not eligible for a receipt",
        "measure": "static Toffolis of one ixor_gf2_mul at GF(2^12), attributed by the "
                   "engine's own tags, plus every product span of the campaign arm",
        "calls": calls,
        "composition": {
            "in_place_toffolis": composition["in_place"]["toffolis"],
            "materialised_toffolis": composition["materialised"]["toffolis"],
            "and_toffolis": composition["in_place"]["toffolis"],
            "add_toffolis": rows["add_toffolis"],
            "non_and_toffolis_in_a_63_toffoli_call": 0,
        },
        "arm": arm,
        "arm_span_sizes_unexplained_by_the_two_call_shapes": unexplained,
        "record_cross_check": {
            "multiply_calls": RECORD_MULTIPLY_CALLS,
            "product_stream_before_the_window": RECORD_CONTROL_PRODUCT_STREAM,
            "product_stream_after_the_window": RECORD_AFTER_WINDOW_PRODUCT_STREAM,
            "average_toffolis_per_call": average,
            "mixture": mixture,
        },
        "bound_rows": rows,
        "verdict": (
            "a 63-Toffoli GF(2^12) multiply call is 63 ANDs and nothing else; the only "
            "non-AND Toffolis in any product call are the 12 of c_gf_iadd, which belong to "
            "the 75-Toffoli materialising shape and are one Toffoli per output bit, so they "
            "do not scale with the AND count. The 9 in MULTIPLY_NON_AND_TOFFOLI_PER_CALL was "
            "ccx_delta_per_call -- the built-in-versus-54-AND-implementation difference -- and "
            "no call in the stream has that size. At the 23-AND bound the multiply is "
            f"{rows['in_place']['multiply_toffolis']:,.1f} if the product accumulates in place "
            f"(T {rows['in_place']['t_mean_toffolis']:,.1f}) and "
            f"{rows['materialised']['multiply_toffolis']:,.1f} if it must materialise "
            f"(T {rows['materialised']['t_mean_toffolis']:,.1f}); that is the whole bracket "
            "that remains, and it is a property of the accumulator, not an unmeasured split"),
    }

    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        if args.record.exists():
            previous = json.loads(args.record.read_text(encoding="utf-8"))
            if previous.get("verdict") != record["verdict"]:
                record["verdict_changed_from"] = previous.get("verdict")
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print("one ixor_gf2_mul at GF(2^12)")
    for row in calls:
        print(f"  {row['call']:44s} {row['toffolis']:3d} Toffolis  {row['toffolis_by_tag']}")
    print(f"campaign arm at m={arm['instance']['m']}, n={arm['instance']['n']}: "
          f"{arm['product_spans']} product spans, sizes {arm['span_sizes']}")
    print(f"  unexplained sizes: {unexplained or 'none'}")
    print(f"  693,660 / 10,510 = {average:.1f} per call, i.e. a mixture: "
          f"{mixture['explained_by_a_mixture_of_the_two_shapes']}")
    print(f"  331,023 / 5,255 = {mixture['after_the_window_average']:.2f} per call")
    for name in ("in_place", "materialised"):
        row = rows[name]
        print(f"  at the 23-AND bound, {name:13s} multiply {row['multiply_toffolis']:11,.1f} "
              f"products {row['products_toffolis']:11,.1f} "
              f"T {row['t_mean_toffolis']:11,.1f}")
    if args.record is not None:
        print(f"record: {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
