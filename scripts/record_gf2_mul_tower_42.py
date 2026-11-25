#!/usr/bin/env python3
"""Record the 42-product multiplication in GF(2^12) and its 42-Toffoli circuit, with every check.

Status: audit-only

A known bound made explicit, not a discovery: the literature's best bound on the number of
products is 42 (``apedp_dyn/quantum/tower_reference.py`` names the source), and this writes the
decomposition, the circuit and what was checked into one record, which ``--check`` rebuilds.

    python scripts/record_gf2_mul_tower_42.py                 # write the record
    python scripts/record_gf2_mul_tower_42.py --check         # rebuild and compare

The record's projection onto the decoder benchmark is arithmetic on the repository's own measured
figures (``benchmarks/quantum_compilation/gf_mul_and_static_headroom.json``), not a measurement:
the circuit has not been put into the benchmark's decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
from typing import Any

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from apedp_dyn.quantum import tower_reference as tower  # noqa: E402
from apedp_dyn.quantum.field_tensor_search import (  # noqa: E402
    DEGREE,
    MODULUS_INT,
    MODULUS_POLY,
    build_54_reference,
    build_karatsuba_reference,
    build_multiplication_tensor,
    gf_mul,
    solve_gf2_bilinear_weights,
)
from apedp_dyn.quantum.gf2_rank_search import solve_weights  # noqa: E402

RECORD = REPO_ROOT / "benchmarks" / "gf2_multiplication" / "gf2_mul_tower_42.json"
HEADROOM = REPO_ROOT / "benchmarks" / "quantum_compilation" / "gf_mul_and_static_headroom.json"
TAG = "DEV-GF2-12-TOWER-42-2026-11"
PAIRS, STATES, SAMPLE_SEED = 4096, 20000, 20261125
PUBLISHED = {
    "toffolis": 51, "cnots": 1506, "parity_controlled_toffolis": 51,
    "source": "Khattar, Shutty, Gidney, Zalcman, Yosri, Maslov, Babbush, Jordan, 'Verifiable Quantum Advantage via "
              "Optimized DQI Circuits', arXiv:2510.10967 (October 2025), Table 7, row m = 12",
    "how_built": "the synthesis of Yosri, Gavinsky, Maslov, arXiv:2511.20618: a modified Karatsuba multiplication with "
                 "local optimisations by templates; the paper does not call the count optimal",
    "also_m_10_and_11": "39 and 47 Toffolis (46 parity-controlled) in the same table; the survey's bounds on the number "
                        "of products there are 33 and 39, which this repository has not built",
    "in_this_repository": "no copy: field_tensor_search.load_published_51_reference returns zeros",
}


def _rows(matrix: np.ndarray) -> list[int]:
    return [sum(int(bit) << k for k, bit in enumerate(row)) for row in matrix]


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.uint8).tobytes()).hexdigest()


def _gates_text(gates: list[tower.Gate]) -> list[str]:
    return [f"C {control} {target}" if kind == tower.CNOT else f"T {control} {second} {target}"
            for kind, control, second, target in gates]


def _least_weight(U: np.ndarray) -> int:
    """The fewest products that are non-zero for a non-zero x: at least 12 in any correct multiplication."""
    forms = _rows(U)
    return min(sum(bin(form & x).count("1") & 1 for form in forms) for x in range(1, 1 << DEGREE))


def _through_the_builder(decomposition: Any, tensor: np.ndarray) -> dict[str, Any]:
    account = tower.replay(tower.build_netlist(decomposition))
    right = bool(np.array_equal(account.pop("tensor"), tensor)) and account["registers_restored"]
    return {"toffolis": account["toffolis"], "cnots": account["cnots"], "qubits": account["qubits"], "right": right}


def projection(ands: int) -> dict[str, Any]:
    """The repository's measured model of the decoder's cost, read at ``ands`` products a multiplication."""
    executed = json.loads(HEADROOM.read_text())["multiply"]["executed"]
    calls, at_63, at_54 = executed["executed_multiply_calls"], executed["t_mean_at_63_ands"], executed["t_mean_at_54_ands"]
    per_and = (at_63 - at_54) / (63 - 54) / calls                 # measured: one Toffoli a product a call
    mean = at_63 - calls * per_and * (63 - ands)
    reference = at_63 * executed["ratio_vs_reference"]
    return {"what": "arithmetic on the repository's measured figures; NOT a measurement of this circuit in the decoder",
            "from": str(HEADROOM.relative_to(REPO_ROOT)), "arm": "six levers", "multiplications_a_shot": calls,
            "toffolis_a_product_a_call": per_and, "mean_toffolis_at_63": at_63, "mean_toffolis_at_54_measured": at_54,
            f"mean_toffolis_at_{ands}": mean, "change_from_63_percent": 100.0 * (mean - at_63) / at_63,
            "ratio_to_the_reference_at_63": executed["ratio_vs_reference"], f"ratio_to_the_reference_at_{ands}": reference / mean,
            "not_covered": ["the controlled call and its 12-Toffoli add into the target", "the window the decoder's "
                            "multiplication writes", "the CNOTs, which the benchmark does not price",
                            "any receipt: the local simulator is not the pinned build"]}


def build() -> dict[str, Any]:
    tensor = build_multiplication_tensor()
    decomposition = tower.build_tower_42_reference()
    U, W = decomposition.U, decomposition.W
    solved, other_W = solve_weights(tensor, U, U)
    dropped = [solve_gf2_bilinear_weights(np.delete(U, r, axis=0), np.delete(U, r, axis=0), tensor)[0] for r in range(len(U))]
    rng = random.Random(SAMPLE_SEED)
    wrong_pairs = 0
    forms, weights = _rows(U), _rows(W)
    for _ in range(PAIRS):
        a, b = rng.randrange(1 << DEGREE), rng.randrange(1 << DEGREE)
        product = 0
        for form, weight in zip(forms, weights, strict=True):
            if bin(form & a).count("1") & bin(form & b).count("1") & 1:
                product ^= weight
        wrong_pairs += product != gf_mul(a, b, MODULUS_INT)
    gates, account = tower.build_tower_42_circuit()
    wrong_states = 0
    for _ in range(STATES):
        x, y, z = (rng.randrange(1 << DEGREE) for _ in range(3))
        wrong_states += tower.run(gates, x, y, z) != (x, y, z ^ gf_mul(x, y, MODULUS_INT))
    text = _gates_text(gates)
    return {
        "tag": TAG, "development": True,
        "what": "multiplication in GF(2^12) with 42 products and a 42-Toffoli circuit on the three registers alone: a "
                "known bound made explicit and checked exactly; not a discovery and not a proof that 42 is the least",
        "field": {"modulus_poly": list(MODULUS_POLY), "modulus_int": MODULUS_INT, **{
            key: decomposition.metadata[key] for key in ("subfield_generator", "construction")}},
        "source_of_the_bound": tower.SOURCE,
        "decomposition": {"rank": decomposition.rank, "symmetric": bool(np.array_equal(U, decomposition.V)),
                          "U_rows_as_integers_bit_k_is_coefficient_k": forms, "W_rows_as_integers": weights,
                          "U_sha256": _digest(U), "W_sha256": _digest(W), "tensor_sha256": _digest(tensor)},
        "checks": {
            "tensor_entries_that_differ_of_1728": len(tower.verify_tensor_decomposition(tensor, decomposition)[1]),
            "weights_from_the_repository_s_other_solver_agree": bool(solved and np.array_equal(other_W, W)),
            "products_that_can_be_dropped_alone": int(sum(dropped)),
            "fewest_products_non_zero_for_a_non_zero_factor": _least_weight(U),
            "multiplications_from_the_products": {"pairs": PAIRS, "wrong": int(wrong_pairs), "seed": SAMPLE_SEED},
            "circuit_from_its_gates_alone": {"reproduces_the_tensor": True, **account},
            "circuit_run_bit_by_bit": {"states": STATES, "wrong": int(wrong_states), "seed": SAMPLE_SEED},
        },
        "circuit": {"computes": "(x, y, z) -> (x, y, z + x*y)", "qubits": account["qubits"], "qubits_beyond_the_three_registers": 0,
                    "toffolis": account["toffolis"], "cnots": account["cnots"], "cnots_minimised": False,
                    "layout": "x on 0..11, y on 12..23, z on 24..35; 'C control target', 'T control control target'",
                    "gates_sha256": hashlib.sha256("\n".join(text).encode()).hexdigest(), "gates": text},
        "beside_it": {
            "published_circuit": PUBLISHED,
            "the_repository_s_54_through_the_same_builder": _through_the_builder(build_54_reference(), tensor),
            "karatsuba_63_through_the_same_builder": _through_the_builder(build_karatsuba_reference(), tensor),
        },
        "least_possible": "not known. At least 31 by the classical argument that the products' forms make a binary code of "
                          "dimension 12 and distance 12, with the Griesmer bound (this repository's arithmetic, not a quoted "
                          "figure). Nothing below 42 was found published for n = 12; that is a search, not a proof of absence.",
        "projection_onto_the_decoder_benchmark": projection(tower.RANK),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=RECORD)
    parser.add_argument("--check", action="store_true", help="rebuild and compare with the stored record")
    args = parser.parse_args(argv)
    fresh = json.loads(json.dumps(build()))
    if args.check:
        stored = json.loads(args.out.read_text())
        differing = sorted(key for key in set(stored) | set(fresh) if stored.get(key) != fresh.get(key))
        for key in differing:
            print(f"FAIL {key} differs from the stored record")
        if not differing:
            print(f"OK {args.out.name}: {fresh['decomposition']['rank']} products, {fresh['circuit']['toffolis']} Toffolis, "
                  f"{fresh['circuit']['cnots']} CNOTs, {fresh['circuit']['qubits']} qubits")
        return 1 if differing else 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fresh, indent=1) + "\n")
    checks = fresh["checks"]
    print(f"{fresh['decomposition']['rank']} products; tensor entries that differ: {checks['tensor_entries_that_differ_of_1728']}; "
          f"circuit {fresh['circuit']['toffolis']} Toffolis, {fresh['circuit']['cnots']} CNOTs, {fresh['circuit']['qubits']} qubits; "
          f"wrong states {checks['circuit_run_bit_by_bit']['wrong']} of {STATES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
