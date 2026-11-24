#!/usr/bin/env python3
"""The division bill's "864 products x 54 ANDs" is an assumption. Take one call apart.

`dqi_division_granularity.json` settled the *granularity* of the division bill -- 378 Toffolis
is one call, and 46,656 is the whole decoder's bill for one shot -- but it reconstructed the
per-shot figure as `864 products x 54 ANDs`, and neither 864 nor 54 was measured. The 378 was.
This script measures the rest the way `measure_gf_mul_per_call_composition.py` measured the
multiply: build one `init_gf2_div_with_scaffold` (and the inverse) at GF(2^12), and attribute
every Toffoli to the engine's own tags, one span at a time.

What one division call turns out to be: **6 `gf_mul` calls of 63 Toffolis each, and nothing
else** (378 = 6 x 63). So the per-shot bill of 46,656 is **740.6 products of 63 ANDs**, not 864
of 54. The bill is unchanged -- it is the measured total either way -- but the *product count*
is 1.167x smaller, and that count is what a lower AND bound multiplies.

The correction is not cosmetic, because the 23-AND rung scales the division products:

* assumed: 864 products x 23 ANDs = **19,872** Toffolis per shot at the bound;
* measured: 740.6 products x 23 ANDs = **17,033.1**.

So `product_axis_decision()`'s in-place row moves from `T` 444,245.3 (0.998x the target) to
440,896.4 (**0.990x**), and the materialised row from 507,297.3 (1.139x) to 505,545.1 (1.136x).
The campaign decision does not change -- the target still needs the 23-AND bound *and* the
verifier peak -- but the margin it clears by at the peak is 4,332.6 Toffolis rather than 984,
and the "864 products of 54 ANDs" figure stops being quoted as a measurement everywhere it came
from: the bill is measured, the decomposition of it was not.

Run it under the interpreter `kickmix` was built for (CPython 3.13)::

    /dev/shm/qvcc-venv313/bin/python scripts/measure_division_product_composition.py
    /dev/shm/qvcc-venv313/bin/python scripts/measure_division_product_composition.py \\
        --record benchmarks/quantum_compilation/dqi_division_composition.json
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
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "dqi_division_composition.json"
GRANULARITY = ROOT / "benchmarks" / "quantum_compilation" / "dqi_division_granularity.json"
PRIMITIVE_COSTS = ROOT / "benchmarks" / "quantum_compilation" / "dqi_primitive_costs.json"

DEGREE = 12
AND_DIMENSION_BOUND = 2 * DEGREE - 1

#: The figures the bill was decomposed with, which this script measures instead.
ASSUMED_PER_SHOT_PRODUCTS = 864
ASSUMED_ANDS_PER_PRODUCT = 54


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
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def emit_call(km, kind: str):
    """One scaffold call, wrapped in a mark so its spans can be attributed."""
    runner = load_module("mr_runner_division", ROOT / "scripts" / "run_mr_gated_arms_kickmix.py")
    builder = runner.builder_class(km)()
    field = km.GF2Field(DEGREE)
    lhs = builder.alloc_qubits(DEGREE)
    rhs = builder.alloc_qubits(DEGREE)
    with builder.mark("division_call"):
        if kind == "div":
            builder.init_gf2_div_with_scaffold(lhs, rhs, field=field)
        else:
            builder.init_gf2_inverse_with_scaffold(lhs, target=builder.alloc_qubits(DEGREE),
                                                   field=field)
    return builder.finish_circuit()


def span_sizes(circuit, tag: str) -> list[int]:
    """The Toffoli count of every `tag` span, one entry per call."""
    mask = np.asarray(circuit.op_toffoli_like, dtype=bool)
    tags = list(circuit.info_tags())
    starts = [index for index, text in tags if text == f"@enter:{tag}"]
    ends = [index for index, text in tags if text == f"@exit:{tag}"]
    return [int(mask[start:stop].sum()) for start, stop in zip(starts, ends, strict=False)]


def measure_call(km, kind: str) -> dict:
    circuit = emit_call(km, kind)
    mask = np.asarray(circuit.op_toffoli_like, dtype=bool)
    stats = {name: int(count) for name, count in dict(circuit.operation_statistics()).items()
             if count}
    multiplies = span_sizes(circuit, "gf_mul")
    controlled = span_sizes(circuit, "c_gf_mul")
    unmul = span_sizes(circuit, "gf_unmul")
    total = int(mask.sum())
    return {
        "call": "init_gf2_div_with_scaffold" if kind == "div" else "init_gf2_inverse_with_scaffold",
        "toffolis": total,
        "ccx": stats.get("CCX", 0),
        "ops": stats,
        "qubits": int(circuit.num_qubits),
        "gf_mul_calls": len(multiplies),
        "gf_mul_sizes": multiplies,
        "c_gf_mul_calls": len(controlled),
        "c_gf_mul_sizes": controlled,
        "gf_unmul_calls": len(unmul),
        "multiply_toffolis": sum(multiplies) + sum(controlled) + sum(unmul),
        "non_multiply_toffolis": total - sum(multiplies) - sum(controlled) - sum(unmul),
    }


def price(calls: dict) -> dict:
    """Re-decompose the per-shot bill with the measured product count, then re-price the bound."""
    model = load_module("dqi_v6_division", ROOT / "apedp_dyn" / "quantum_compilation"
                        / "dqi_gluzdov_v6.py")
    granularity = json.loads(GRANULARITY.read_text(encoding="utf-8"))["granularity"]
    costs = json.loads(PRIMITIVE_COSTS.read_text(encoding="utf-8"))["primitives_per_call"]
    division = calls["init_gf2_div_with_scaffold"]

    bill = float(granularity["per_shot_bill_toffolis"])
    per_call = int(division["toffolis"])
    sizes = division["gf_mul_sizes"]
    ands_per_product = sorted(set(sizes))
    if len(ands_per_product) != 1:
        raise SystemExit(f"one division call emits multiply spans of several sizes: {sizes}")
    ands_per_product = ands_per_product[0]
    products_per_call = division["gf_mul_calls"]
    calls_per_shot = bill / per_call
    products_per_shot = calls_per_shot * products_per_call

    # The model's own constants, not its decision: the decision reads this record, so asking it
    # here would be circular. The "as recorded" rows are therefore recomputed from the constants
    # with the assumed decomposition, and the model's rows are checked against them in the tests.
    residual = model.CAMPAIGN_RESIDUAL_T_MEAN
    products = model.CAMPAIGN_PRODUCT_T_MEAN
    # The measured division-and-Forney term: 55,038.7 = 46,656 division + 8,382.7 Forney.
    inversion_division = model.CAMPAIGN_INVERSION_DIVISION_T_MEAN
    multiply_term = products - inversion_division
    ands = AND_DIMENSION_BOUND

    assumed_division = inversion_division * ands / ASSUMED_ANDS_PER_PRODUCT
    measured_division = inversion_division / float(ands_per_product) * ands

    def row(division_toffolis: float, multiply_shape: str) -> dict:
        if multiply_shape == "in_place":
            multiply = multiply_term * ands / 63.0
        else:
            multiply = multiply_term * (ands + 12) / 63.0
        total_products = multiply + division_toffolis
        t_mean = residual + total_products
        return {
            "multiply_toffolis": multiply,
            "division_toffolis": division_toffolis,
            "products_toffolis": total_products,
            "t_mean_toffolis": t_mean,
            "vs_target": t_mean / model.T_MEAN_TARGET_V6,
        }

    return {
        "per_shot_bill_toffolis": bill,
        "division_share_of_the_term": bill / inversion_division,
        "per_call_toffolis": per_call,
        "primitive_cost_record_toffolis": costs["divide_scaffold_pair"],
        "calls_per_shot": calls_per_shot,
        "products_per_call": products_per_call,
        "ands_per_product": ands_per_product,
        "products_per_shot": products_per_shot,
        "bill_rebuilds_from_the_measurement": products_per_shot * ands_per_product,
        "assumption_it_replaces": {
            "products_per_shot": ASSUMED_PER_SHOT_PRODUCTS,
            "ands_per_product": ASSUMED_ANDS_PER_PRODUCT,
            "bill": ASSUMED_PER_SHOT_PRODUCTS * ASSUMED_ANDS_PER_PRODUCT,
            "product_count_ratio": products_per_shot / ASSUMED_PER_SHOT_PRODUCTS,
        },
        "at_the_and_bound": {
            "ands": ands,
            "division_and_forney_toffolis": inversion_division,
            "products_in_that_term_assumed": inversion_division / ASSUMED_ANDS_PER_PRODUCT,
            "products_in_that_term_measured": inversion_division / float(ands_per_product),
            "division_toffolis_assumed": assumed_division,
            "division_toffolis_measured": measured_division,
            "difference_toffolis": assumed_division - measured_division,
            "in_place_corrected": row(measured_division, "in_place"),
            "in_place_as_recorded": row(assumed_division, "in_place"),
            "materialised_corrected": row(measured_division, "materialised"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kickmix-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--record", type=Path, default=None)
    args = parser.parse_args()

    km = load_kickmix(args.kickmix_build)
    calls = {
        "init_gf2_div_with_scaffold": measure_call(km, "div"),
        "init_gf2_inverse_with_scaffold": measure_call(km, "inv"),
    }
    pricing = price(calls)

    record = {
        "engine": "kickmix",
        "kickmix": str(Path(km.__file__).resolve()),
        "kickmix_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
        "kickmix_role": "local rebuild with MR (scripts/build_kickmix_mr.py); not the "
                        "pinned competition binary and not eligible for a receipt",
        "degree": DEGREE,
        "measure": "every Toffoli of one GF(2^12) scaffold division and inverse call, attributed "
                   "to the engine's own tags, span by span; then the per-shot bill re-decomposed "
                   "from the measured product count and re-priced at the 2n-1 AND bound",
        "calls": calls,
        "pricing": pricing,
        "verdict": (
            f"one division call is {pricing['products_per_call']} gf_mul calls of "
            f"{pricing['ands_per_product']} Toffolis each and nothing else, so the measured "
            f"{pricing['per_shot_bill_toffolis']:,.0f} per-shot bill is "
            f"{pricing['products_per_shot']:,.1f} products, not "
            f"{ASSUMED_PER_SHOT_PRODUCTS} of {ASSUMED_ANDS_PER_PRODUCT}. The bill is unchanged -- "
            "it is the measured total either way -- but at the 23-AND bound the division term is "
            f"{pricing['at_the_and_bound']['division_toffolis_measured']:,.1f} Toffolis rather "
            f"than {pricing['at_the_and_bound']['division_toffolis_assumed']:,.1f}, which moves "
            f"the in-place row from T "
            f"{pricing['at_the_and_bound']['in_place_as_recorded']['t_mean_toffolis']:,.1f} to "
            f"{pricing['at_the_and_bound']['in_place_corrected']['t_mean_toffolis']:,.1f} "
            f"({pricing['at_the_and_bound']['in_place_corrected']['vs_target']:.4f}x the target)"
        ),
    }

    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    for name, call in calls.items():
        print(f"{name}: {call['toffolis']} Toffolis, {call['gf_mul_calls']} gf_mul calls of "
              f"{sorted(set(call['gf_mul_sizes']))}, "
              f"{call['non_multiply_toffolis']} non-multiply, q={call['qubits']}")
    print(f"per shot: {pricing['per_shot_bill_toffolis']:,.0f} Toffolis = "
          f"{pricing['calls_per_shot']:.2f} calls x {pricing['products_per_call']} products = "
          f"{pricing['products_per_shot']:,.1f} products of {pricing['ands_per_product']} ANDs "
          f"(rebuilds to {pricing['bill_rebuilds_from_the_measurement']:,.1f})")
    assumption = pricing["assumption_it_replaces"]
    print(f"  replaces {assumption['products_per_shot']} products of "
          f"{assumption['ands_per_product']} ANDs: the product count is "
          f"{assumption['product_count_ratio']:.3f}x of it")
    bound = pricing["at_the_and_bound"]
    print(f"at the {bound['ands']}-AND bound the division term is "
          f"{bound['division_toffolis_measured']:,.1f} not "
          f"{bound['division_toffolis_assumed']:,.1f} "
          f"({bound['difference_toffolis']:,.1f} apart)")
    print(f"  in place  : T {bound['in_place_corrected']['t_mean_toffolis']:,.1f} "
          f"({bound['in_place_corrected']['vs_target']:.4f}x the target), against the recorded "
          f"{bound['in_place_as_recorded']['t_mean_toffolis']:,.1f} "
          f"({bound['in_place_as_recorded']['vs_target']:.4f}x)")
    print(f"  materialised: T {bound['materialised_corrected']['t_mean_toffolis']:,.1f} "
          f"({bound['materialised_corrected']['vs_target']:.4f}x the target)")
    if args.record is not None:
        print(f"record: {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
