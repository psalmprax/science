#!/usr/bin/env python3
"""Mixed-polarity (non-Davio) search for the four-output restoration PDDD.

The incumbent plan built by `scripts/test_shared_pddd_circuit.py` uses a
fixed-polarity shared Davio recursion: every AND node is `x_v AND g` with the
positive literal, and each output's remainder is always the low cofactor
`f(x_v = 0)`. This script searches the generalisation in which each node picks
its literal independently:

    f = f_p XOR (x_v XOR p) AND (f_0 XOR f_1),      p in {0, 1}

`g = f_0 XOR f_1` (the derivative) is polarity-free; the freedom is which
cofactor `f_p` becomes the recursively shared remainder, and whether the node is
`x_v AND g` or `NOT x_v AND g`. Node identity is `(variable, p, g-expression)`,
so the positive and negative literal are different wires and only genuinely
identical terms are shared.

What this script reports:
1. That the harness reproduces the 36-node Davio plan at the known order and
   polarity (otherwise it is searching a different problem).
2. That every plan it scores is functionally correct, by evaluating the AND/XOR
   DAG over all 256 address assignments and comparing with the truth tables.
3. The best joint node count found (reported as a lower bound only — a joint plan
   needs one monolithic restore, which costs 22 qubits), the best *usable* two-phase
   count (coarse6 before `outer`, curve4..6 after), the search scope actually visited,
   and the verified-score value of any saving.

It is a search, not an exhaustive optimum: the order/polarity space covered is
printed, and the negative result it reports holds for that space.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/invsqrt"))

import signed_digit_invsqrt_v3 as core
import scheduled_arithmetic90 as H
import test_shared_pddd_circuit as builder

RECEIPT = REPO_ROOT / "artifacts/quantum_compilation/invsqrt/invsqrt_89q_shared_pddd_full_receipt.json"
SHARED_PDDD = REPO_ROOT / "src/shared_davio_results.json"
COARSE_CHOICES = REPO_ROOT / "src/boolean_choices90.json"

N_VARS = 8
ORDER4 = (7, 4, 3, 2, 1, 0, 6, 5)   # joint order of the incumbent plan
POLARITY4 = 32                     # joint global polarity of the incumbent plan


# ---------------------------------------------------------------------------
# Truth tables
# ---------------------------------------------------------------------------

def permuted_table(values: np.ndarray, order: tuple[int, ...], polarity: int) -> int:
    """Packs `values` into the bit order the recursion uses (same as build_shared_plan)."""
    indices = np.arange(1 << N_VARS)
    original = sum(((indices >> j) & 1) << order[j] for j in range(N_VARS)) ^ polarity
    packed = np.packbits(np.asarray(values, dtype=np.uint8)[original], bitorder="little")
    return int.from_bytes(packed.tobytes(), "little")


# ---------------------------------------------------------------------------
# Generalised (per-node mixed-polarity) planner
# ---------------------------------------------------------------------------

def plan_mixed(tables: dict[str, int], order: tuple[int, ...],
               patterns: dict[str, int], polarity: int = 0) -> dict:
    """Builds a shared AND/XOR plan with per-node literal polarity.

    `patterns[name]` is a bitmask: bit (level-1) is the polarity of the node that
    splits on variable `order[level-1]` while decomposing `name`. Zero everywhere
    reproduces the incumbent positive-Davio recursion.

    Returns {"nodes": [(variable_level, polarity, right_expr), ...],
             "results": {name: (constant, frozenset(wire_ids))}}.
    Wire ids 0..n-1 are the variables (variable `j` = address bit `order[j]`),
    ids >= n are the AND nodes.
    """
    nodes: list[tuple[int, int, tuple[int, frozenset]]] = []
    ids: dict[tuple[int, int, tuple[int, frozenset]], int] = {}

    def rec(t: int, level: int, pat: int) -> tuple[int, frozenset]:
        one = (1 << (1 << level)) - 1
        if t == 0:
            return (0, frozenset())
        if t == one:
            return (1, frozenset())
        half = 1 << (level - 1)
        mask = (1 << half) - 1
        lo = t & mask
        hi = t >> half
        pol = (pat >> (level - 1)) & 1
        remainder = lo if pol == 0 else hi
        derivative = lo ^ hi

        left = rec(remainder, level - 1, pat)
        right = rec(derivative, level - 1, pat)

        if not right[1]:
            # The derivative is a constant, so no AND node is needed: the term is
            # the literal itself, `x_v` for pol=0 and `NOT x_v` (x_v XOR 1) for pol=1.
            if not right[0]:
                return left
            constant = left[0] ^ pol
            return (constant, left[1] ^ {level - 1})

        key = (level - 1, pol, right)
        if key not in ids:
            ids[key] = N_VARS + len(nodes)
            nodes.append(key)
        return (left[0], left[1] ^ {ids[key]})

    results = {name: rec(t, N_VARS, patterns.get(name, 0)) for name, t in tables.items()}
    # `polarity` is carried so a circuit builder can flip the address lanes; it does
    # not affect node identity, which is why the search ignores it.
    return {"nodes": nodes, "results": results, "order": order, "polarity": polarity}


# ---------------------------------------------------------------------------
# Validation: evaluate the plan classically over the whole domain
# ---------------------------------------------------------------------------

def evaluate(plan: dict, wire_assignment: int) -> dict[str, int]:
    """Evaluates every output of `plan` for one assignment of the permuted variables.

    Bit j of `wire_assignment` is the value of wire j, and wire j is the variable
    the recursion splits on at level j (address bit `order[j]` in the circuit).
    Working in the permuted index space keeps this check independent of the
    address-register permutation, which the circuit builder applies explicitly.
    """
    bits = [(wire_assignment >> j) & 1 for j in range(N_VARS)]
    values: list[int] = list(bits)

    def expr_value(expr: tuple[int, frozenset]) -> int:
        v = expr[0] & 1
        for wid in expr[1]:
            v ^= values[wid]
        return v

    for variable, pol, right in plan["nodes"]:
        literal = bits[variable] ^ pol
        values.append(literal & expr_value(right))

    return {name: expr_value(expr) for name, expr in plan["results"].items()}


def validate(plan: dict, permuted_tables: dict[str, int], name: str) -> None:
    """Raises if any output disagrees with its (permuted) truth table on 256/256 inputs."""
    for assignment in range(1 << N_VARS):
        expected = {k: (t >> assignment) & 1 for k, t in permuted_tables.items()}
        got = evaluate(plan, assignment)
        for key in expected:
            if expected[key] != got[key]:
                raise AssertionError(
                    f"{name} invalid at wire assignment {assignment}: {key} expected "
                    f"{expected[key]}, plan gives {got[key]}"
                )


# ---------------------------------------------------------------------------
# Joint objective and search
# ---------------------------------------------------------------------------

def joint_count(tables: dict[str, int], order: tuple[int, ...], polarity: int,
                patterns: dict[str, int]) -> int:
    permuted = {k: permuted_table(v, order, polarity) for k, v in tables.items()}
    return len(plan_mixed(permuted, order, patterns)["nodes"])


def coordinate_descent(tables, order, polarity, start, sweeps: int = 3, budget=None):
    """Greedy per-output, per-level polarity descent on the joint node count."""
    patterns = dict(start)
    best = joint_count(tables, order, polarity, patterns)
    names = list(tables)
    for _ in range(sweeps):
        improved = False
        for name in names:
            for level in range(1, N_VARS):
                trial = dict(patterns)
                trial[name] ^= 1 << (level - 1)
                count = joint_count(tables, order, polarity, trial)
                if count < best:
                    best, patterns, improved = count, trial, True
                if budget is not None and budget.exhausted():
                    return best, patterns, True
        if not improved:
            break
    return best, patterns, False


class Budget:
    def __init__(self, seconds: float):
        self.deadline = time.perf_counter() + seconds
        self.hits = 0

    def exhausted(self) -> bool:
        if time.perf_counter() > self.deadline:
            self.hits += 1
            return True
        return False


def candidate_orders(curve_bias: bool, limit: int = 300) -> list[tuple[tuple[int, ...], int, str]]:
    """Order/polarity seeds: the joint order, the coarse6 order, the curve search
    results, then random permutations (seeded, so the run is repeatable)."""
    cands: list[tuple[tuple[int, ...], int, str]] = [(ORDER4, POLARITY4, "joint order")]
    coarse = json.loads(COARSE_CHOICES.read_text())["coarse6"]
    cands.append((tuple(coarse["order"]), int(coarse["polarity"]), "coarse6 order"))
    if curve_bias:
        for cand in json.loads(SHARED_PDDD.read_text())["candidates"][:140]:
            cands.append((tuple(cand["order"]), int(cand["polarity"]), "curve candidate"))
    rng = random.Random(20261028)
    seen = {c[0] for c in cands}
    while len(cands) < limit:
        perm = list(range(N_VARS))
        rng.shuffle(perm)
        perm = tuple(perm)
        if perm in seen:
            continue
        seen.add(perm)
        cands.append((perm, rng.randrange(256), "random order"))
    return cands


def search_best(tables: dict[str, int], order: tuple[int, ...], polarity: int,
                budget: Budget, label: str, limit: int = 300):
    """Coordinate-descent mixed-polarity search for one set of outputs."""
    best = None
    best_key = (order, polarity)
    best_pat: dict[str, int] = {}
    visited = 0
    for o, p, _label in candidate_orders(curve_bias=len(tables) > 1, limit=limit):
        if budget.exhausted():
            break
        visited += 1
        seed: dict[str, int] = {}
        for name in tables:
            table = permuted_table(tables[name], o, p)
            counts = [len(plan_mixed({name: table}, o, {name: q})["nodes"])
                      for q in range(1 << (N_VARS - 1))]
            seed[name] = counts.index(min(counts))
        count, pats, _ = coordinate_descent(tables, o, p, seed, budget=budget)
        if best is None or count < best:
            best, best_key, best_pat = count, (o, p), pats
            print(f"   [{label}] new best {count} nodes at order={o} polarity={p}")
    if best is None:
        raise SystemExit(f"[{label}] time budget expired before any order was evaluated")
    print(f"   [{label}] orders visited: {visited}, best {best} nodes")
    return best, best_key, best_pat


def main() -> int:
    start = time.perf_counter()
    cfg, tables_raw, _ = core.load_or_build_tables()
    split, _ = H.prepare_derivative_split(H, cfg, tables_raw)

    cv_curve = np.array([tables_raw[(i // 128, i % 128)][0] for i in range(256)], dtype=np.int64)
    gray = cv_curve ^ (cv_curve >> 1)
    cv = np.array([split["coarse"][(i // 128, i % 128)] for i in range(256)], dtype=np.int64)
    functions = {
        "coarse6": ((cv ^ (cv >> 1)) >> 6) & 1,
        "curve4": (gray >> 4) & 1,
        "curve5": (gray >> 5) & 1,
        "curve6": (gray >> 6) & 1,
    }

    order4, polarity4 = ORDER4, POLARITY4

    print("=" * 78)
    print("MIXED-POLARITY (NON-DAVIO) SEARCH FOR THE 4-OUTPUT RESTORATION")
    print("=" * 78)

    # 1. Harness check: all-positive polarity must reproduce the incumbent 36 nodes.
    permuted = {k: permuted_table(v, order4, polarity4) for k, v in functions.items()}
    zero = {k: 0 for k in functions}
    ref_plan = plan_mixed(permuted, order4, zero)
    ref_count = len(ref_plan["nodes"])
    incumbent = len(builder.build_shared_plan(functions, order=order4, polarity=polarity4)["nodes"])
    print(f"\n1. Harness check at order={order4} polarity={polarity4}")
    print(f"   all-positive mixed-polarity planner: {ref_count} nodes")
    print(f"   scripts/test_shared_pddd_circuit.build_shared_plan: {incumbent} nodes")
    if ref_count != incumbent:
        print("   MISMATCH — the search is not reproducing the incumbent; aborting.")
        return 1
    validate(ref_plan, permuted, "incumbent")
    print("   plan validated: 256/256 addresses agree with the truth tables")

    # 2. Per-output minima under the joint order, polarity patterns exhaustively.
    print("\n2. Per-output minimum node count over all 2^7 polarity patterns")
    print("   (single output in isolation, same order/polarity as the joint plan)")
    single_positive = {}
    single_best = {}
    for name, table in permuted.items():
        counts = []
        for pat in range(1 << (N_VARS - 1)):
            plan = plan_mixed({name: table}, order4, {name: pat})
            counts.append(len(plan["nodes"]))
        single_positive[name] = counts[0]
        single_best[name] = min(counts)
        improved = single_positive[name] - single_best[name]
        print(f"   {name:<9s} positive={single_positive[name]:3d}  "
              f"best mixed={single_best[name]:3d}  ({improved:+d} nodes)")
    print(f"   summed singles: positive={sum(single_positive.values())} "
          f"mixed={sum(single_best.values())}")

    # 3. Joint four-output search. Reported, but NOT schedulable: one restore of all
    # four outputs forces the monolithic schedule, which test_viable_optimization_path
    # measures at 111 qubits (+22) and +38,291.57 score units. The usable objective is
    # section 4, where the two phases are restored in separate calls.
    print("\n3. Joint four-output search (lower bound only; not schedulable)")
    budget = Budget(seconds=60.0)
    candidates: list[tuple[tuple[int, ...], int, str]] = [(order4, polarity4, "incumbent order")]
    coarse = json.loads(COARSE_CHOICES.read_text())["coarse6"]
    candidates.append((tuple(coarse["order"]), int(coarse["polarity"]), "coarse6 order"))
    for cand in json.loads(SHARED_PDDD.read_text())["candidates"][:140]:
        candidates.append((tuple(cand["order"]), int(cand["polarity"]), "curve search candidate"))
    rng = random.Random(20261028)
    seen = {c[0] for c in candidates}
    while len(candidates) < 400:
        perm = list(range(N_VARS))
        rng.shuffle(perm)
        perm = tuple(perm)
        if perm in seen:
            continue
        seen.add(perm)
        candidates.append((perm, rng.randrange(256), "random order"))

    best = incumbent
    best_desc = "incumbent (order 7,4,3,2,1,0,6,5 / polarity 32, positive Davio)"
    best_patterns = dict(zero)
    best_order, best_polarity = order4, polarity4
    visited = 0
    for order, polarity, label in candidates:
        if budget.exhausted():
            break
        visited += 1
        # Start each order from the mixed-polarity patterns that minimise the singles,
        # which is cheap and gives the descent a better basin than all-zero.
        seed = {}
        for name in functions:
            table = permuted_table(functions[name], order, polarity)
            counts = [len(plan_mixed({name: table}, order, {name: p})["nodes"])
                      for p in range(1 << (N_VARS - 1))]
            seed[name] = counts.index(min(counts))
        count, patterns, _ = coordinate_descent(functions, order, polarity, seed, budget=budget)
        if count < best:
            best, best_patterns, best_order, best_polarity = count, patterns, order, polarity
            best_desc = f"{label}: order={order} polarity={polarity} patterns={patterns}"
            print(f"   new best: {best} nodes  [{best_desc}]")

    print(f"\n   orders visited: {visited} of {len(candidates)} offered"
          f"{' (time budget hit)' if budget.hits else ''}")
    print(f"   best joint count found: {best} nodes")
    print(f"   {best_desc}")

    permuted_best = {k: permuted_table(v, best_order, best_polarity) for k, v in functions.items()}
    final = plan_mixed(permuted_best, best_order, best_patterns)
    if len(final["nodes"]) != best:
        raise AssertionError(f"rebuilt plan has {len(final['nodes'])} nodes, expected {best}")
    validate(final, permuted_best, "joint best plan")
    print(f"   joint plan validated: 256/256 wire assignments agree, {len(final['nodes'])} nodes")
    print(f"   joint count {best} vs incumbent joint {incumbent}: not usable, and the")
    print(f"   monolithic schedule it requires costs +22 qubits (+38,291.57 units).")

    # 4. Usable two-phase search: coarse6 before `outer`, curve4..6 after it.
    print("\n4. Usable two-phase search (the schedule the circuit can run)")
    phase_budget = Budget(seconds=150.0)
    coarse = json.loads(COARSE_CHOICES.read_text())["coarse6"]
    phase1, key1, pat1 = search_best(
        {"coarse6": functions["coarse6"]}, tuple(coarse["order"]), int(coarse["polarity"]),
        Budget(seconds=60.0), label="coarse6",
    )
    phase2, key2, pat2 = search_best(
        {k: functions[k] for k in ("curve4", "curve5", "curve6")}, order4, polarity4,
        phase_budget, label="curves",
    )
    print(f"   phase 1 (coarse6):   {phase1} nodes  [order={key1[0]} polarity={key1[1]}]")
    print(f"   phase 2 (curve4..6): {phase2} nodes  [order={key2[0]} polarity={key2[1]}]")
    total = phase1 + phase2
    print(f"   two-phase total: {total} nodes (incumbent 11 + 28 = 39)")

    # 5. Validate both phase plans and price the usable saving.
    print("\n5. Validation and pricing")
    p1_perm = {"coarse6": permuted_table(functions["coarse6"], key1[0], key1[1])}
    p1_plan = plan_mixed(p1_perm, key1[0], pat1)
    validate(p1_plan, p1_perm, "phase 1 plan")
    curves = {k: functions[k] for k in ("curve4", "curve5", "curve6")}
    p2_perm = {k: permuted_table(v, key2[0], key2[1]) for k, v in curves.items()}
    p2_plan = plan_mixed(p2_perm, key2[0], pat2)
    validate(p2_plan, p2_perm, "phase 2 plan")
    print(f"   both plans validated: 256/256 assignments each, "
          f"{len(p1_plan['nodes'])} + {len(p2_plan['nodes'])} nodes")

    receipt = json.loads(RECEIPT.read_text())
    q = int(receipt["num_qubits"])
    score = float(receipt["score"])
    delta_nodes = 39 - total
    delta_toffoli = 2 * delta_nodes
    delta_score = q * delta_toffoli
    print(f"   incumbent 39 nodes -> mixed two-phase {total}: {delta_nodes:+d} nodes"
          f" = {delta_toffoli:+d} Toffolis")
    print(f"   score effect at Q={q}: {delta_score:+.1f} units"
          f" ({100 * delta_score / score:+.4f}%)")
    if delta_nodes > 0:
        print(f"   A saving of {delta_nodes} node(s) is worth {delta_score:.1f} score units.")
        print("   Not a result yet: per-node polarity needs an X on the control lane around")
        print("   each init_and/del_and, and the plan must be built, compiled and verified.")
    elif delta_nodes == 0:
        print("   No usable saving: mixed polarity does not beat the positive-Davio")
        print("   two-phase plan in the searched space, so it buys nothing.")
    else:
        print("   The mixed-polarity two-phase best is worse than the incumbent here.")

    elapsed = time.perf_counter() - start
    print(f"\n   search wall-clock: {elapsed:.1f} s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
