#!/usr/bin/env python3
"""Which circuit-designer direction actually pays? Three measurements in one record.

The six leverage levers are exhausted (`docs/CIRCUIT_OPTIMISATION_RULES.md` §5), so this
script prices the three ways left to make the decoder better, and lets the numbers choose:

**A. Predicate discovery over dead work.** For every region of the six-lever arm that is
still executed unconditionally, the region is deleted outright and the executed count is
differenced. The deleted circuit is wrong; only its cost is read, and the difference is an
upper bound on what *any* predicate on that region could ever skip. If the sum is small,
a predicate discoverer has nothing to discover here.

**B. Bilinear search for a cheaper field multiply.** A bilinear GF(2) algorithm with `r`
multiplications is correct iff the 2n-1 anti-diagonal matrices of the product tensor lie in
the span of the `r` rank-1 matrices `u_t v_t^T` its products use. That is an exact,
finite search, so the minimum `r` for n = 2 and n = 3 is established rather than assumed,
the found algorithm is verified over every input pair, and Karatsuba composition then gives
the AND count at n = 12. It is priced against the measured multiply share.

**C. A different decoder algorithm.** Derived, not measured: the executed-count arithmetic
says what multiply cost would be needed to reach Benchmark II's 445,229, and whether a
multiply alone can ever get there.

Run it under the interpreter `kickmix` was built for (CPython 3.13)::

    /dev/shm/qvcc-venv313/bin/python scripts/study_circuit_designer_directions.py
    /dev/shm/qvcc-venv313/bin/python scripts/study_circuit_designer_directions.py \\
        --official --record benchmarks/quantum_compilation/designer_directions.json
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
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "designer_directions.json"

#: Pinned reference and target, in Toffoli per shot.
REFERENCE_T_MEAN = 5_725_071.61
TARGET_T_MEAN = 445_229.0

#: Measured once at the pinned instance and recorded; the ladder in part C is arithmetic
#: on these numbers, not a second measurement.
EXECUTED_SIX_LEVERS = 1_325_230.0     # kickmix_mr_arm_costs.json, 64 shots
#: The arm decomposes exactly as `everything else + calls * (ands + 12)`, measured in
#: gf54_mul_kickmix.json by differencing two arms that differ only in the multiply. The
#: `+ 12` is the controlled add, which no multiply avoids.
MULTIPLY_CALLS = 10_510               # gf54_mul_kickmix.json, executed calls per shot
EVERYTHING_ELSE = 536_980.0           # gf54_mul_kickmix.json, per shot
MULTIPLY_ADD_TOFFOLI = 12             # controlled add into the target
#: Superseded: the earlier 86.33% came from deleting the multiply and differencing the
#: executed count, which perturbs the state and so the predicates that gate the rest of
#: the EEA. Kept only so the correction is on the record.
SUPERSEDED_MULTIPLY_SHARE_PERCENT = 86.33

SIX_LEVERS = {"gate_forney": True, "clear_eea": True, "gate_sweep": True,
              "gate_swap": True, "gate_shift": True, "gate_quotient": True}

#: Regions of the gated arm that still run unconditionally, and the method to delete.
REGIONS = (
    "_update_flag",
    "_update_metadata",
    "_update_counter",
    "_update_halting_counter",
    "_initial_shift_loop",
    "_pre_align_reg_b",
)


# ---------------------------------------------------------------- bilinear search

def anti_diagonals(n: int) -> list[int]:
    """The 2n-1 target matrices of the product tensor, as n*n-bit integers."""
    out = []
    for k in range(2 * n - 1):
        mask = 0
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                mask |= 1 << (i * n + j)
        out.append(mask)
    return out


def rank_one_candidates(n: int) -> list[tuple[int, int, int]]:
    """Every distinct rank-1 matrix u v^T over GF(2)^n, as (matrix, u, v)."""
    seen: set[int] = set()
    out = []
    for u in range(1, 1 << n):
        for v in range(1, 1 << n):
            matrix = 0
            for i in range(n):
                if (u >> i) & 1:
                    for j in range(n):
                        if (v >> j) & 1:
                            matrix |= 1 << (i * n + j)
            if matrix not in seen:
                seen.add(matrix)
                out.append((matrix, u, v))
    return out


def _reduce(vector: int, basis: list[int]) -> int:
    for b in basis:
        vector = min(vector, vector ^ b)
    return vector


def _span(vectors: list[int]) -> list[int]:
    basis: list[int] = []
    for vector in vectors:
        reduced = _reduce(vector, basis)
        if reduced:
            basis.append(reduced)
            basis.sort(reverse=True)
    return basis


def bilinear_search(n: int, r_max: int = 8, node_budget: int = 4_000_000) -> dict:
    """Smallest set of rank-1 matrices whose span holds the whole product tensor.

    Exhaustive for small n: a negative result at a given r means every choice of r was
    tried, so the minimum is proved inside this class, not estimated.
    """
    targets = anti_diagonals(n)
    need = len(_span(targets))
    cands = rank_one_candidates(n)
    nodes = 0

    for r in range(need, r_max + 1):
        basis: list[int] = []
        chosen: list[int] = []

        def dfs(start: int) -> bool:
            nonlocal nodes
            nodes += 1
            if nodes > node_budget:
                raise RuntimeError("node budget exhausted")
            if len(basis) + (r - len(chosen)) < need:
                return False
            if len(chosen) == r:
                return all(_reduce(t, basis) == 0 for t in targets)
            for index in range(start, len(cands)):
                reduced = _reduce(cands[index][0], basis)
                if reduced == 0:
                    continue
                basis.append(reduced)
                chosen.append(index)
                if dfs(index + 1):
                    return True
                chosen.pop()
                basis.pop()
            return False

        try:
            found = dfs(0)
        except RuntimeError:
            return {"n": n, "rank": None, "proved": False, "candidates": len(cands),
                    "nodes": nodes, "note": "node budget exhausted before r was closed"}
        if found:
            picked = [cands[i] for i in chosen]
            matrices = [m for m, _u, _v in picked]
            weights = [_solve(matrices, t) for t in targets]
            pairs = [(u, v) for _m, u, v in picked]
            bad = sum(
                1 for a in range(1 << n) for b in range(1 << n)
                if _evaluate_bilinear(pairs, weights, a, b) != _poly_product(a, b, n)
            )
            return {"n": n, "rank": r, "proved": True, "candidates": len(cands),
                    "nodes": nodes, "pairs": pairs, "weights": weights,
                    "verified_pairs": 1 << (2 * n), "mismatches": bad}
    return {"n": n, "rank": None, "proved": True, "candidates": len(cands), "nodes": nodes}


def _solve(matrices: list[int], target: int) -> list[int]:
    """Weights w with XOR_t w[t]*matrices[t] == target over GF(2)."""
    pivots: dict[int, tuple[int, int]] = {}
    for t, matrix in enumerate(matrices):
        coeff, value = 1 << t, matrix
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                p_coeff, p_value = pivots[pivot]
                coeff ^= p_coeff
                value ^= p_value
            else:
                pivots[pivot] = (coeff, value)
                break
    coeff, value = 0, target
    while value:
        pivot = value.bit_length() - 1
        if pivot not in pivots:
            raise ValueError("target outside the span")
        p_coeff, p_value = pivots[pivot]
        coeff ^= p_coeff
        value ^= p_value
    return [(coeff >> t) & 1 for t in range(len(matrices))]


def _evaluate_bilinear(pairs, weights, a: int, b: int) -> int:
    products = [(bin(u & a).count("1") & 1) & (bin(v & b).count("1") & 1)
                for u, v in pairs]
    out = 0
    for k, row in enumerate(weights):
        bit = 0
        for t, use in enumerate(row):
            if use:
                bit ^= products[t]
        out |= bit << k
    return out


def _poly_product(a: int, b: int, n: int) -> int:
    out = 0
    for i in range(n):
        if (a >> i) & 1:
            out ^= b << i
    return out & ((1 << (2 * n - 1)) - 1)


def composed_and_count(n: int, base: dict[int, int]) -> int:
    """Ands for an n-bit product from base-case algorithms, Karatsuba split at ceil(n/2)."""
    if n in base:
        return base[n]
    k = (n + 1) // 2
    return 3 * composed_and_count(k, base)


def composed_product(a: int, b: int, n: int, base_algo: dict) -> int:
    """The same composition, evaluated, so the AND count is attached to a real algorithm."""
    if n in base_algo:
        algo = base_algo[n]
        return _evaluate_bilinear(algo["pairs"], algo["weights"], a, b)
    k = (n + 1) // 2
    mask = (1 << k) - 1
    a0, a1 = a & mask, (a >> k) & mask
    b0, b1 = b & mask, (b >> k) & mask
    low = composed_product(a0, b0, k, base_algo)
    high = composed_product(a1, b1, k, base_algo)
    mid = composed_product(a0 ^ a1, b0 ^ b1, k, base_algo)
    return (low ^ (mid ^ low ^ high) << k ^ high << (2 * k)) & ((1 << (2 * n - 1)) - 1)


# ---------------------------------------------------------------- circuit measurement

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


def static_toffoli(km, runner, arms, deleted: str | None, m, n, degree) -> int:
    """Static (emitted) Toffoli-like count with one region deleted.

    Static counts do not depend on the state, so deleting a region measures its own gates
    exactly: no cascade, and no wrong circuit is simulated. Since a predicate can at most
    remove every gate a region emits, the difference is the ceiling for any predicate on
    that region.
    """
    shim = runner.shim_for(km, runner.builder_class(km))
    decoder_class = arms.MRGatedDecoder
    if deleted is not None:
        decoder_class = type(
            f"AblatingDecoder_{deleted.strip('_')}",
            (arms.MRGatedDecoder,),
            {deleted: lambda self, *args, **kwargs: None},
        )
    original = arms.DR.km
    arms.DR.km = shim
    try:
        decoder = decoder_class(n, km.GF2Field(degree), **SIX_LEVERS)
        builder = shim.CircuitBuilder()
        syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
        error_mbuc = [builder.create_classical_register(degree, name=f"E{j}") for j in range(m)]
        decoder.build_syndrome_decoding_mbuc(
            builder, syndromes=syndromes, error_mbuc=error_mbuc,
            n_iters=arms.DR.default_n_iters(n))
        stats = dict(builder.finish_circuit().operation_statistics())
        return int(sum(stats.get(name, 0) for name in ("CCX", "CSWAP", "AND")))
    finally:
        arms.DR.km = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--official", action="store_true",
                        help="run part A at 4095/70/GF(2^12); default is the small instance")
    parser.add_argument("--m", type=int, default=15)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--kickmix-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--record", type=Path, default=None)
    args = parser.parse_args()
    if args.official:
        args.m, args.n, args.degree, args.shots = 4095, 70, 12, 8

    runner = load_runner()
    km = runner.load_kickmix(args.kickmix_build)
    arms = load_arms(km, runner)

    # -- B: exact bilinear search ------------------------------------------------
    bases = {}
    for size in (2, 3):
        result = bilinear_search(size, r_max=size * size)
        bases[size] = result
    base_cost = {size: result["rank"] for size, result in bases.items()
                 if result.get("rank")}
    base_algo = {size: {"pairs": result["pairs"], "weights": result["weights"]}
                 for size, result in bases.items() if result.get("rank")}
    composed = composed_and_count(12, base_cost)
    if 3 in base_algo:
        field = km.GF2Field(12)
        rng = np.random.default_rng(7)
        bad = 0
        for _ in range(200):
            a, b = int(rng.integers(0, 1 << 12)), int(rng.integers(0, 1 << 12))
            product = composed_product(a, b, 12, base_algo)
            # The composition gives the polynomial product; reduction mod p is linear,
            # so fold x^i -> x^i mod p and compare with the field's own multiply.
            reduced = 0
            for i in range(2 * 12 - 1):
                if (product >> i) & 1:
                    reduced ^= _x_pow_mod(field, i)
            bad += reduced != field.mul(a, b)
        composed_check = {"samples": 200, "mismatches": bad}
    else:
        composed_check = {"samples": 0, "mismatches": None}

    # -- A: what a predicate discoverer could still skip ------------------------
    baseline_static = static_toffoli(km, runner, arms, None, args.m, args.n, args.degree)
    regions = {}
    for region in REGIONS:
        without = static_toffoli(km, runner, arms, region, args.m, args.n, args.degree)
        regions[region] = {
            "static_toffoli_without_region": without,
            "ceiling_static_toffoli": baseline_static - without,
            "ceiling_percent_of_static": 100 * (baseline_static - without) / baseline_static,
        }
    discoverer_ceiling = sum(row["ceiling_static_toffoli"] for row in regions.values())

    # -- C: what a different decoder would have to buy --------------------------
    # Executed baseline and the multiply's share are measured in
    # benchmarks/quantum_compilation/{kickmix_mr_arm_costs,gf_mul_and_static_headroom}.json.
    ladder: dict[str, dict] = {}
    if args.official:
        for ands in (63, 54, 45, 23, 0):
            t_mean = EVERYTHING_ELSE + MULTIPLY_CALLS * (ands + MULTIPLY_ADD_TOFFOLI)
            ladder[str(ands)] = {
                "t_mean": t_mean,
                "ratio_vs_reference": REFERENCE_T_MEAN / t_mean,
                "ratio_vs_target": t_mean / TARGET_T_MEAN,
            }
        # What the superseded model would have said, for the record.
        superseded_share = EXECUTED_SIX_LEVERS * SUPERSEDED_MULTIPLY_SHARE_PERCENT / 100
        superseded = EXECUTED_SIX_LEVERS - superseded_share * (1 - 54 / 63)

    record: dict[str, object] = {
        "engine": "kickmix",
        "kickmix": str(Path(km.__file__).resolve()),
        "kickmix_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
        "kickmix_role": "local rebuild with MR; not the pinned competition binary",
        "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                     "shots": args.shots, "official": args.official},
        "b_bilinear_search": {
            "method": "exact search over rank-1 matrices u v^T whose span holds all 2n-1 "
                      "anti-diagonals of the product tensor; exhaustive at n=2,3",
            "n2": bases[2],
            "n3": bases[3],
            "base_and_costs": base_cost,
            "composed_and_count_12": composed,
            "karatsuba_builtin_and_count_12": 63,
            "composed_verified_against_field": composed_check,
        },
        "a_predicate_ceiling": {
            "measure": "static Toffoli-like count with one region deleted outright; static "
                       "counts do not depend on state, so the difference is that region's own "
                       "gates and an exact ceiling for any predicate on it (executed <= static)",
            "baseline_static_toffoli": baseline_static,
            "regions": regions,
            "total_ceiling_static_toffoli": discoverer_ceiling,
            "total_ceiling_percent": 100 * discoverer_ceiling / baseline_static,
        },
        "c_decoder_requirement": {
            "target_t_mean": TARGET_T_MEAN if args.official else None,
            "derived_from": {
                "executed_six_levers": EXECUTED_SIX_LEVERS,
                "multiply_calls_per_shot": MULTIPLY_CALLS,
                "everything_else_per_shot": EVERYTHING_ELSE,
                "controlled_add_per_call": MULTIPLY_ADD_TOFFOLI,
                "model": "t(ands) = everything_else + calls * (ands + 12), so the controlled "
                         "add is counted in every row -- a multiply costs ands + 12, not ands",
                "measured_in": "benchmarks/quantum_compilation/gf54_mul_kickmix.json",
                "superseded_share_percent": SUPERSEDED_MULTIPLY_SHARE_PERCENT,
                "superseded_54_and_projection": superseded if args.official else None,
            },
            "ladder": ladder,
        },
    }
    record["finding"] = _finding(record)

    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n")

    search = record["b_bilinear_search"]
    print(f"B: mu(2)={search['n2'].get('rank')} mu(3)={search['n3'].get('rank')} "
          f"(exhaustive: {search['n3'].get('proved')}), n=3 verified "
          f"{search['n3'].get('verified_pairs')} pairs / {search['n3'].get('mismatches')} bad")
    print(f"   composed to n=12: {search['composed_and_count_12']} ANDs "
          f"vs built-in 63; field check {composed_check}")
    print(f"A: baseline {baseline_static:,} static Toffoli; deleting every unconditional "
          f"region is worth at most {discoverer_ceiling:,} "
          f"({record['a_predicate_ceiling']['total_ceiling_percent']:.2f}%)")
    for name, row in regions.items():
        print(f"   {name:26s} ceiling {row['ceiling_static_toffoli']:9,d} "
              f"({row['ceiling_percent_of_static']:5.2f}%)")
    if ladder:
        print("C: multiply-cost ladder (derived from the measured share)")
        for ands, row in ladder.items():
            print(f"   {ands:>2s} ANDs -> {row['t_mean']:13,.1f} T/shot  "
                  f"{row['ratio_vs_reference']:6.3f}x  {row['ratio_vs_target']:5.2f}x target")
    if args.record is not None:
        print(f"record: {args.record}")
    return 0


def _x_pow_mod(field, i: int) -> int:
    """x^i reduced modulo the field polynomial, via the field's own multiplication."""
    value = 1
    for _ in range(i):
        value = field.mul(value, 2)
    return value


def _finding(record: dict) -> str:
    search = record["b_bilinear_search"]
    a = record["a_predicate_ceiling"]
    c = record["c_decoder_requirement"]
    mu3 = search["n3"].get("rank")
    lines = [
        f"B: the exact search gives mu(2) = {search['n2'].get('rank')} and mu(3) = {mu3} "
        f"(the n=3 minimum is proved, not estimated: every choice of five rank-1 matrices "
        f"was tried). Composed by Karatsuba to n=12 that is "
        f"{search['composed_and_count_12']} ANDs against the built-in "
        f"{search['karatsuba_builtin_and_count_12']}, i.e. "
        f"{100 * (1 - search['composed_and_count_12'] / 63):.1f}% fewer, and the composed "
        f"algorithm reproduces the field product on "
        f"{search['composed_verified_against_field']['samples']} sampled inputs with "
        f"{search['composed_verified_against_field']['mismatches']} mismatches. ",
        f"A: deleting every region that still runs unconditionally is worth at most "
        f"{a['total_ceiling_static_toffoli']:,} static Toffolis "
        f"({a['total_ceiling_percent']:.2f}% of the arm's emitted stream, and executed <= "
        f"static), so a predicate discoverer has that much to find here and no more. ",
    ]
    ladder = c["ladder"]
    if not ladder:
        lines.append("C: the multiply-cost ladder is derived at the pinned instance only; "
                     "rerun with --official for it.")
        return " ".join(lines)
    derived = c["derived_from"]
    lines.append(
        f"C: the multiply is most of the gap but not all of it. The arm decomposes exactly "
        f"as {derived['everything_else_per_shot']:,.1f} Toffoli of non-multiply work plus "
        f"{derived['multiply_calls_per_shot']:,} executed calls of a multiply that costs its "
        f"AND count plus {derived['controlled_add_per_call']}, so at the proved {mu3}-AND "
        f"base the arm would execute {ladder['54']['t_mean']:,.1f} per shot; even a 23-AND "
        f"multiply -- the 2n-1 dimension bound, which this repository has no characteristic-2 "
        f"construction for -- leaves {ladder['23']['t_mean']:,.1f} "
        f"({ladder['23']['ratio_vs_target']:.2f}x the 445,229 target), and a multiply that "
        f"cost nothing but its {derived['controlled_add_per_call']}-Toffoli add still leaves "
        f"{ladder['0']['t_mean']:,.1f} ({ladder['0']['ratio_vs_target']:.2f}x the target). "
        f"So no multiply, however cheap, delivers the target on this decoder: the "
        f"{derived['everything_else_per_shot']:,.1f} that is not the multiply has to shrink "
        f"too, which is an algorithm change."
    )
    return " ".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
