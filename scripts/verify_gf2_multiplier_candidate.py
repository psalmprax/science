#!/usr/bin/env python3
"""What can and cannot be certified about a sub-51 GF(2^12) multiplier today.

Status: audit-only

A bilinear decomposition is a *polynomial identity*: if the 1,728 coefficients reconstruct,
the multiplier is correct on all 2^24 input pairs, with no shots and no statistics.  That makes
the algebraic gate stronger than any verifier receipt -- and it is also the only gate that can
be closed today, because the field engine prices a decomposition but never emits a gate
netlist, and a bare multiplier is not a competition circuit.

The four gates this script reports, in order:

1. ``algebraic`` — every coefficient of ``T`` reconstructed from ``U``, ``V``, ``W``; plus an
   independent ``W`` solved from ``U`` and ``V`` alone (so the shipped ``W`` is not trusted).
2. ``bit_order_crosscheck`` — the decomposition rebuilds ``gf_mul`` for sampled input pairs,
   which catches a bit-order convention error the tensor identity would still pass.
3. ``reversible_netlist`` — **MISSING**: ``compile_reversible_multiplier`` returns counts
   (Toffolis, CNOTs, ancillas, depth), not a gate list, so nothing can be simulated end to end.
4. ``embedded_benchmark_receipt`` — **MISSING**: only a scored circuit gets an official PASS;
   a component rank is not a submission.

    .venv/bin/python scripts/verify_gf2_multiplier_candidate.py --seed karatsuba-54
    .venv/bin/python scripts/verify_gf2_multiplier_candidate.py --out /tmp/cert.json --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from apedp_dyn.quantum.field_tensor_search import (  # noqa: E402
    DEGREE,
    MODULUS_INT,
    build_54_reference,
    build_karatsuba_reference,
    build_multiplication_tensor,
    build_schoolbook_reference,
    compile_reversible_multiplier,
    gf_mul,
    load_published_51_reference,
    verify_tensor_decomposition,
)
from apedp_dyn.quantum.gf2_rank_search import solve_weights  # noqa: E402

SEEDS = {
    "schoolbook": build_schoolbook_reference,
    "karatsuba-63": build_karatsuba_reference,
    "karatsuba-54": build_54_reference,
    "published-51": load_published_51_reference,
}
FIELD_ORDER = 1 << DEGREE
INCUMBENT_51 = 51
DQI_PRODUCT_CALL_TOFFOLI = 63  # the measured `c_gf_iadd` product call in the DQI projection


def _parity(value: int) -> int:
    return bin(value).count("1") & 1


def _digest(array) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _bit_order_crosscheck(decomp, tensor, samples: int, seed: int) -> dict[str, object]:
    """Rebuild ``gf_mul`` from the decomposition on sampled pairs (bit-order sanity)."""
    rng = random.Random(seed)
    bad = 0
    for _ in range(samples):
        a = rng.randrange(FIELD_ORDER)
        b = rng.randrange(FIELD_ORDER)
        expected = gf_mul(a, b, MODULUS_INT)
        for k in range(DEGREE):
            total = 0
            for index in range(decomp.rank):
                if not decomp.W[index, k]:
                    continue
                if _parity(int(a) & sum(1 << i for i in range(DEGREE) if decomp.U[index, i])) and _parity(
                    int(b) & sum(1 << j for j in range(DEGREE) if decomp.V[index, j])
                ):
                    total ^= 1
            if total != ((expected >> k) & 1):
                bad += 1
    return {"pairs": samples, "coefficient_mismatches": bad, "status": "PASS" if bad == 0 else "FAIL"}


def certificate(seed_name: str, samples: int, sample_seed: int) -> dict[str, object]:
    tensor = build_multiplication_tensor()
    decomp = SEEDS[seed_name]()
    exact, mismatches = verify_tensor_decomposition(tensor, decomp)
    solved, W_solved = solve_weights(tensor, decomp.U, decomp.V)
    solved_exact = False
    if solved and W_solved is not None:
        from apedp_dyn.quantum.field_tensor_search import TensorDecomposition

        solved_exact = verify_tensor_decomposition(
            tensor,
            TensorDecomposition(rank=decomp.rank, U=decomp.U, V=decomp.V, W=W_solved, name="solved-W"),
        )[0]

    circuit = compile_reversible_multiplier(decomp)
    controlled = compile_reversible_multiplier(decomp, controlled=True)
    crosscheck = _bit_order_crosscheck(decomp, tensor, samples, sample_seed)

    gates = {
        "algebraic_all_1728_coefficients": "PASS" if exact else "FAIL",
        "independent_W_from_U_and_V": "PASS" if solved_exact else "FAIL",
        "bit_order_crosscheck": crosscheck["status"],
        "reversible_netlist": "MISSING",
        "embedded_benchmark_receipt": "MISSING",
    }
    usable = all(gates[name] == "PASS" for name in ("algebraic_all_1728_coefficients", "independent_W_from_U_and_V"))

    return {
        "what": "GF(2^12) multiplier certificate: what is proved, and which submission gates are still absent.",
        "seed": seed_name,
        "status": "ALGEBRAIC_CERTIFICATE_ONLY" if usable else "REJECTED_BY_EXACT_SCREEN",
        "rank": decomp.rank,
        "coefficient_mismatches": len(mismatches),
        "tensor_sha256": _digest(tensor),
        "U_sha256": _digest(decomp.U),
        "V_sha256": _digest(decomp.V),
        "W_sha256": _digest(decomp.W),
        "toffoli_uncontrolled": circuit.toffoli_count,
        "toffoli_controlled": controlled.toffoli_count,
        "cnot_count": circuit.cnot_count,
        "ancilla_count": circuit.ancilla_count,
        "gates": gates,
        "bit_order_crosscheck": crosscheck,
        "delta_vs_incumbent_51": circuit.toffoli_count - INCUMBENT_51,
        "and_savings_vs_dqi_product_call": DQI_PRODUCT_CALL_TOFFOLI - circuit.toffoli_count,
        "promotion": (
            "none available: a bare multiplier has no verifier contract, and the field engine "
            "emits counts rather than a gate netlist. The next gate is a netlist emitter, then "
            "embedding the product call in a scored benchmark circuit and re-verifying there."
            if usable
            else "none: the decomposition does not reconstruct, so there is nothing to submit."
        ),
    }


def check(path: pathlib.Path) -> int:
    stored = json.loads(path.read_text())
    fresh = certificate(stored["seed"], samples=stored["bit_order_crosscheck"]["pairs"], sample_seed=0)
    keys = ("status", "rank", "toffoli_uncontrolled", "toffoli_controlled", "tensor_sha256", "U_sha256", "W_sha256")
    failures = [f"{key}: {stored.get(key)!r} != {fresh.get(key)!r}" for key in keys if stored.get(key) != fresh.get(key)]
    for failure in failures:
        print(f"FAIL {failure}")
    if not failures:
        print(f"OK {path.name}: {stored['seed']} {stored['status']} rank={stored['rank']} toffoli={stored['toffoli_uncontrolled']}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", choices=sorted(SEEDS), default="karatsuba-54")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--check", action="store_true", help="re-derive and compare a stored certificate")
    args = parser.parse_args(argv)

    if args.check:
        if args.out is None:
            parser.error("--check needs --out PATH")
        return check(args.out)

    record = certificate(args.seed, args.samples, args.sample_seed)
    text = json.dumps(record, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)

    print(f"{record['seed']}: {record['status']} rank={record['rank']} toffoli={record['toffoli_uncontrolled']} "
          f"(controlled {record['toffoli_controlled']}) mismatches={record['coefficient_mismatches']}")
    for gate, status in record["gates"].items():
        print(f"  {gate:<36} {status}")
    print(f"  promotion: {record['promotion']}")
    return 0 if record["status"] == "ALGEBRAIC_CERTIFICATE_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
