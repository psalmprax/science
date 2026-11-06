#!/usr/bin/env python3
"""Read one scaffold division call off the circuit the builder actually emits.

Status: audit-only

`docs/BENCHMARK2_TARGET_REACHABILITY.md` §7.6 carried two prices for "one division": 378 from
`benchmarks/quantum_compilation/dqi_primitive_costs.json` and 46,656 from §2's 864 products x 54
ANDs. They are 124x apart, and the single-array trade depends on which one an extra call costs.

This script settles it by counting the gates instead of the prose: it builds one
`init_gf2_div_with_scaffold` call at GF(2^12), counts its CCX operations, and does the same for
`init_gf2_inverse_with_scaffold`. If a call really carries 378 Toffolis, then 378 is the per-call
price and 46,656 must be something else -- it is the whole decoder's division bill for one shot
(864 products / 54 per call x 378 = about 123 calls).

Writes `benchmarks/quantum_compilation/dqi_division_granularity.json`. `--check` re-measures and
fails if the committed record or the primitive-cost record disagree, so a builder change that
moves either number is caught rather than read past.

Usage:
    APEDP_Q_KICKMIX_PYTHON=... PYTHONPATH=...kickmix build/qvcc-venv313/bin/python \\
        scripts/verify_division_granularity.py [--check]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACT = REPO_ROOT / "benchmarks/quantum_compilation/dqi_division_granularity.json"
PRIMITIVE_COSTS = REPO_ROOT / "benchmarks/quantum_compilation/dqi_primitive_costs.json"
KICKMIX_DIR = (
    REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"
)

#: The decoder's division budget for one shot, from §2's 864 products, and the ANDs per product.
PER_SHOT_PRODUCTS = 864
ANDS_PER_PRODUCT = 54

KICKMIX_HINT = (
    "kickmix is not importable. Run this with the kickmix interpreter and PYTHONPATH, e.g.\n"
    f"  PYTHONPATH={KICKMIX_DIR} APEDP_Q_KICKMIX_PYTHON=<python-with-kickmix> \\\n"
    "    build/qvcc-venv313/bin/python scripts/verify_division_granularity.py"
)


def measure(kind: str) -> dict[str, int]:
    """Build one scaffold call and count the operations it emits."""
    import kickmix as km

    builder = km.CircuitBuilder()
    field = km.GF2Field(12)
    lhs = builder.create_quantum_register(12, name="lhs")
    rhs = builder.create_quantum_register(12, name="rhs")
    if kind == "div":
        builder.init_gf2_div_with_scaffold(lhs, rhs, field=field)
    else:
        target = builder.create_quantum_register(12, name="target")
        builder.init_gf2_inverse_with_scaffold(lhs, target=target, field=field)
    circuit = builder.finish_circuit()
    stats = {str(key): int(value) for key, value in dict(circuit.operation_statistics()).items()}
    return {
        "ccx": stats.get("CCX", 0),
        "cx": stats.get("CX", 0),
        "swap": stats.get("SWAP", 0),
        "r": stats.get("R", 0),
        "peak_qubits": int(circuit.num_qubits),
    }


def build_artifact() -> dict:
    div = measure("div")
    inverse = measure("inv")
    costs = json.loads(PRIMITIVE_COSTS.read_text(encoding="utf-8"))["primitives_per_call"]
    per_call = div["ccx"]
    per_shot_bill = PER_SHOT_PRODUCTS * ANDS_PER_PRODUCT
    return {
        "what": (
            "CCX emitted by one GF(2^12) scaffold call, read off the built circuit rather than "
            "off a table"
        ),
        "script": "scripts/verify_division_granularity.py",
        "builder": "kickmix.CircuitBuilder",
        "field_degree": 12,
        "calls": {
            "init_gf2_div_with_scaffold": div,
            "init_gf2_inverse_with_scaffold": inverse,
        },
        "record_cross_check": {
            "record": PRIMITIVE_COSTS.name,
            "path": "primitives_per_call",
            "divide_scaffold_pair": costs["divide_scaffold_pair"],
            "inverse_scaffold_pair": costs["inverse_scaffold_pair"],
            "agrees": (
                per_call == costs["divide_scaffold_pair"]
                and inverse["ccx"] == costs["inverse_scaffold_pair"]
            ),
        },
        "granularity": {
            "per_call_toffolis": per_call,
            "per_shot_products": PER_SHOT_PRODUCTS,
            "ands_per_product": ANDS_PER_PRODUCT,
            "per_shot_bill_toffolis": per_shot_bill,
            "implied_calls_per_shot": round(per_shot_bill / per_call, 2),
        },
        "verdict": (
            f"{per_call} Toffolis is one division call, measured; {per_shot_bill:,} is the whole "
            f"decoder's division bill for one shot, about {per_shot_bill / per_call:.1f} calls. "
            "Multiplying the per-shot bill by the EEA round count overstates the single-array "
            "trade by that same factor"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare against the committed record")
    args = parser.parse_args()

    try:
        artifact = build_artifact()
    except ImportError as exc:
        print(f"{KICKMIX_HINT}\n({exc})", file=sys.stderr)
        return 2

    if args.check:
        if not ARTIFACT.is_file():
            print(f"missing {ARTIFACT}", file=sys.stderr)
            return 1
        committed = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        for key in ("calls", "granularity"):
            if committed.get(key) != artifact[key]:
                print(f"{key} disagrees with the committed artifact:\n"
                      f"  committed: {committed.get(key)}\n  measured:  {artifact[key]}",
                      file=sys.stderr)
                return 1
        print(f"granularity record agrees: {artifact['granularity']['per_call_toffolis']} Toffolis "
              f"per call, {artifact['granularity']['per_shot_bill_toffolis']:,} per shot")
        return 0

    ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    print(f"\nwrote {ARTIFACT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
