#!/usr/bin/env python3
"""Reconcile the DQI arm's Toffoli budget into exclusive accounting categories.

The review asked for a ledger, not another ratio: every executed Toffoli-like gate
assigned to exactly one category, components summing to the measured total, measured
contributions distinguished from estimates, and the part that current instrumentation
cannot attribute named as such rather than guessed.

Two ledgers are produced, because they are different metrics and must not be mixed:

* **Executed** (the scored quantity, `num_toffoli_like` per shot). The arm's
  1,221,466.8 comes from the native-opcode record: the multiply proper, the products
  inside the inversion, the products inside the division, and the residual. The residual
  is then split as far as measurement allows -- the six unconditional regions are exact
  (they run on every shot, so their static count is their executed count), and the rest
  is one named, un-attributed bucket with its size and the reason it is not split.
* **Static** (emitted gates, state-independent). The arm's 3,486,597 and the reference's
  5,725,461 come from the headroom record; deleting a leaf region from a static count is
  exact, so the leaf inventory and the remainder reconcile without double counting.

Every component carries `measured: true|false` and the method that produced it. Nothing
here is re-derived by hand from a prose number: the script reads the records, checks the
sums, and refuses to write a ledger that does not reconcile.

Example::

    /dev/shm/qvcc-venv313/bin/python scripts/attribute_dqi_nonmultiply_tcount.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT / "benchmarks" / "quantum_compilation"
DEFAULT_RECORD = RECORDS / "dqi_nonmultiply_ledger.json"

TOL = 1e-6


def load(name: str) -> dict:
    return json.loads((RECORDS / name).read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()

    native = load("gf54_native_opcode.json")
    directions = load("designer_directions.json")
    headroom = load("gf_mul_static_headroom.json") if (
        RECORDS / "gf_mul_static_headroom.json").exists() else load(
        "gf_mul_and_static_headroom.json")
    primitives = load("dqi_primitive_costs.json")

    arm = native["arm"]
    decomposition = native["isolated_blocks"]["arm_decomposition"]
    isolated = native["isolated_blocks"]["arm"]
    total = arm["and54"]["mean_toffoli_like"]
    controlled = decomposition["controlled_multiplies"]
    uncontrolled = decomposition["uncontrolled_products"]
    multiply = controlled * 66
    inversion = isolated["forney_inversion"]["mean_executed_products"] * 54
    division = isolated["division_and_sweep"]["mean_executed_products"] * 54
    residual = decomposition["residual_non_product_toffoli"]

    def share(value: float) -> float:
        return 100 * value / total

    components = [
        {"id": "ixor_gf2_mul", "category": "field arithmetic",
         "what": "10,510 executed variable multiplies, each 54 ANDs + 12 controlled add",
         "toffoli_per_shot": multiply, "share_percent": share(multiply),
         "measured": True,
         "method": "CCX difference between two correct arms that differ in the product "
                   "policy only; the call count is that difference over 9"},
        {"id": "inversion_products", "category": "field arithmetic",
         "what": "the products inside init/del_gf2_inverse_with_scaffold, 54 ANDs each, "
                 "no controlled add",
         "toffoli_per_shot": inversion, "share_percent": share(inversion),
         "measured": True,
         "method": "the same reference/arm built with gf_product scoped to the inversion "
                   "entry points only, then differenced against the unscoped arm"},
        {"id": "division_products", "category": "field arithmetic",
         "what": "the products inside init_gf2_div_with_scaffold, 54 ANDs each, no "
                 "controlled add",
         "toffoli_per_shot": division, "share_percent": share(division),
         "measured": True,
         "method": "the same, scoped to the division entry points only"},
        {"id": "non_product_residual", "category": "decoder logic",
         "what": "EEA state updates and their control, predicates, counter and halting "
                 "decode, shifts, metadata, and the multiply-and-add conditionals",
         "toffoli_per_shot": residual, "share_percent": share(residual),
         "measured": True,
         "method": "the measured arm total minus the three product terms; the terms sum "
                   "to the total by construction, and the record checks it"},
    ]
    component_sum = sum(row["toffoli_per_shot"] for row in components)
    if abs(component_sum - total) > 1e-3:
        raise SystemExit(f"components sum to {component_sum}, not the measured {total}")

    # The residual, split as far as the instruments allow. The six unconditional regions
    # are exact; everything else is one bucket, explicitly not attributed.
    ceiling = directions["a_predicate_ceiling"]
    regions = {name: row["ceiling_static_toffoli"]
               for name, row in ceiling["regions"].items()}
    unconditional = sum(regions.values())
    if unconditional != ceiling["total_ceiling_static_toffoli"]:
        raise SystemExit("the region ledger does not sum to its recorded total")
    unattributed = residual - unconditional

    # Static inventory: deletion of a leaf method is exact, so the ledger cannot
    # double count. The remainder is everything else the decoder emits.
    static_arm = headroom["static"]["six levers"]["toffoli_like"]
    static_reference = headroom["static"]["reference"]["toffoli_like"]
    static_remainder = static_arm - unconditional
    if static_arm != ceiling["baseline_static_toffoli"]:
        raise SystemExit("the static totals disagree between records")

    record = {
        "instance": native["instance"],
        "engine": native["engine"],
        "build": {"extension": native["kickmix"],
                  "sha256": native["kickmix_sha256"],
                  "role": native["kickmix_role"]},
        "metric": {
            "executed": "mean Toffoli-like operations per shot (CCX, CSWAP, AND), the "
                        "quantity verify_submission_dqi.py averages over its shots",
            "static": "every Toffoli-like gate emitted, conditions ignored",
            "rule": "one category per gate: the executed categories are defined by "
                    "measurement (policy-scoped differences and the residual), the "
                    "static categories by deleting a leaf region; no gate is counted twice",
        },
        "executed_ledger": {
            "total": total,
            "components": components,
            "component_sum": component_sum,
            "reconciles": bool(abs(component_sum - total) < 1e-3),
            "residual_split": {
                "unconditional_regions": regions,
                "unconditional_total": unconditional,
                "unconditional_share_of_residual_percent": 100 * unconditional / residual,
                "unattributed_executed": unattributed,
                "unattributed_share_of_residual_percent": 100 * unattributed / residual,
                "unattributed_note": (
                    "the conditional EEA body and the gated overrides; the simulator "
                    "reports per-shot counts by op type, not per operation, so this part "
                    "can be bounded (<= its static count) and gated (an oracle lever "
                    "measures what can be removed) but not attributed region by region "
                    "with the current instrumentation"),
            },
        },
        "static_ledger": {
            "arm_total": static_arm,
            "reference_total": static_reference,
            "unconditional_regions": unconditional,
            "remainder": static_remainder,
            "remainder_share_percent": 100 * static_remainder / static_arm,
            "reconciles": bool(unconditional + static_remainder == static_arm),
            "method": "each region deleted from a fresh build and the static count "
                      "differenced; static counts are state-independent, so the "
                      "difference is that region's own gates",
            "not_deletable": [
                "generator internals (gen_gf_*, gen_iadd, multi-control decompositions)",
                "_counter_controls and the arm's gated overrides, which return values "
                "their callers consume and so cannot be replaced by a no-op",
            ],
        },
        "unit_prices": {
            "toffoli_per_eea_iteration": primitives["slopes"]["per_iteration"],
            "toffoli_per_chien_position": primitives["slopes"]["per_position"],
            "toffoli_per_constant_multiply": primitives["primitives_per_call"][
                "multiply_by_constant"],
            "note": "measured on the reference decoder, so they price the arm's EEA body "
                    "rather than split it; the arm gates the body and executes fewer "
                    "iterations' worth of work",
        },
        "measurement_gaps": [
            {"gap": "executed Toffolis per region inside the conditional EEA body",
             "why": "the simulator exposes per-shot counts by operation type only",
             "how_to_close": "per-operation execution counts in the simulator, or an "
                             "oracle-gated variant per region (which measures the "
                             "removable part, not the total)"},
            {"gap": "the arm's EEA iteration count and its cost per iteration",
             "why": "only the reference slope is measured",
             "how_to_close": "count executed Toffolis around one iteration with the "
                             "existing marks on the arm"},
            {"gap": "whether the gated Forney body or another stage sets the peak width",
             "why": "Q is measured for the whole circuit only",
             "how_to_close": "per-frame liveness, already implemented in "
                             "scripts/measure_dqi_scratch_pool.py"},
        ],
        "first_experiment": {
            "target": "_update_flag and the counter/halting predicates",
            "why": "the largest measured region in the residual (32,856 of 94,470 "
                   "unconditional Toffolis) and the one the review's 'predicate reuse' "
                   "hypothesis would change",
            "control": "the same arm at the fixed 222 iterations, unchanged",
            "prediction_required_before_the_edit": True,
            "promotion_gates": [
                "exact output equivalence: syndromes restored, measured phase, no junk",
                "restored workspace and no unintended phase changes",
                "unchanged fixed-iteration behaviour (no early halting)",
                "mean non-product T-count and static T-count both reported, Q reported, "
                "Q^2*T recomputed from the same run",
                "the complete regression suite, and a frozen source + record",
            ],
        },
    }

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")

    print(f"executed ledger, {native['instance']}")
    print(f"  total {total:,.1f} Toffoli-like per shot")
    for row in components:
        print(f"    {row['id']:22s} {row['toffoli_per_shot']:12,.1f}  "
              f"{row['share_percent']:5.1f}%  {row['category']}")
    print(f"  reconciles: {record['executed_ledger']['reconciles']}")
    split = record["executed_ledger"]["residual_split"]
    print(f"  residual {residual:,.1f}: unconditional {unconditional:,} "
          f"({split['unconditional_share_of_residual_percent']:.1f}%), "
          f"unattributed {unattributed:,.1f} "
          f"({split['unattributed_share_of_residual_percent']:.1f}%)")
    print(f"static ledger: arm {static_arm:,} = {unconditional:,} + {static_remainder:,} "
          f"(reconciles: {record['static_ledger']['reconciles']})")
    print(f"record: {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
