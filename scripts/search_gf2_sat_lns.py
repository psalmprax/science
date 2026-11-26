#!/usr/bin/env python3
"""GF(2^12) below 42 by large-neighbourhood SAT: replace k products of a decomposition by k - 1 (development).

Status: audit-only

**Development, not a result.** The third of the four approaches the owner asked on 26 November to build
and test, after Heule, Kauers and Seidl ("Local search for fast matrix multiplication", SAT 2019), who
found thousands of new 3x3 schemes by SAT with streamlining. Here: take a decomposition, free a set of k
of its products, and ask CryptoMiniSat (XOR clauses natively) for k - 1 products with the same sum; if one
exists the decomposition is one shorter. The equations: for every entry (i, j, l) of the freed products'
sum R, the XOR over the new products of a_i b_j c_l equals R_ijl, with a_i b_j and (a_i b_j) c_l as
auxiliary variables. Streamlining: ``symmetric`` asks for a = b in every new product. The neighbourhoods:
``random`` (k products at random) or ``blocks`` (whole blocks of a construction: the products of one or more
GF(8) points of a tower, or the three products of a GF(4) product). Every instance has a time limit; an
undecided instance is counted as such, not as a no.

Nothing counts until the shorter decomposition is checked against every entry of its tensor and, for
GF(2^12), by ``field_tensor_search.verify_tensor_decomposition``.

    python scripts/search_gf2_sat_lns.py run --field 12 --start ref --k 12 --trials 20 --limit 120 --out DIR
    python scripts/search_gf2_sat_lns.py collect --out DIR
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gf2_towers as towers  # noqa: E402

TAG = "DEV-GF2-12-SAT-LNS-2026-11"
KNOWN = {7: 22, 8: 24, 9: 30, 10: 33, 12: 42}


def start(n: int, spec: str) -> list[tuple[int, int, int]]:
    import search_gf2_and_walk as gates
    return [(a, b >> n, c) for a, b, c in gates.start(n, spec)]


def tensor_of(triples: list[tuple[int, int, int]], n: int) -> set[tuple[int, int, int]]:
    out: set[tuple[int, int, int]] = set()
    for a, b, c in triples:
        for i in range(n):
            if a >> i & 1:
                for j in range(n):
                    if b >> j & 1:
                        for k in range(n):
                            if c >> k & 1:
                                out ^= {(i, j, k)}
    return out


def verified(triples: list[tuple[int, int, int]], n: int) -> bool:
    if not triples or not towers.exact(triples, n):
        return False
    if n != 12:
        return True
    from apedp_dyn.quantum import flip_search
    return flip_search.verified(triples)


def solve(residual: set[tuple[int, int, int]], r: int, n: int, symmetric: bool, limit: float, seed: int = 0) -> tuple[str, list[tuple[int, int, int]]]:
    """Is the residual a sum of r products? 'sat' with them, 'unsat', or 'unknown' when the limit ran out."""
    import pycryptosat
    solver = pycryptosat.Solver(time_limit=limit, threads=1)
    next_var = [0]

    def new() -> int:
        next_var[0] += 1
        return next_var[0]
    a = [[new() for _ in range(n)] for _ in range(r)]
    b = a if symmetric else [[new() for _ in range(n)] for _ in range(r)]
    c = [[new() for _ in range(n)] for _ in range(r)]
    ab: dict[tuple[int, int, int], int] = {}
    for t in range(r):
        for i in range(n):
            for j in range(n):
                if symmetric and j < i:
                    ab[(t, i, j)] = ab[(t, j, i)]
                    continue
                v = new()
                ab[(t, i, j)] = v
                solver.add_clause([-v, a[t][i]])
                solver.add_clause([-v, b[t][j]])
                solver.add_clause([v, -a[t][i], -b[t][j]])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                vs = []
                for t in range(r):
                    v = new()
                    p = ab[(t, i, j)]
                    solver.add_clause([-v, p])
                    solver.add_clause([-v, c[t][k]])
                    solver.add_clause([v, -p, -c[t][k]])
                    vs.append(v)
                solver.add_xor_clause(vs, (i, j, k) in residual)
    for t in range(r):                                       # no zero factor; and order the products (symmetry)
        solver.add_clause(a[t])
        solver.add_clause(c[t])
        if not symmetric:
            solver.add_clause(b[t])
    sat, model = solver.solve()
    if sat is None:
        return "unknown", []
    if not sat:
        return "unsat", []

    def val(var: int) -> int:
        return 1 if model[var] else 0
    out = []
    for t in range(r):
        av = sum(val(a[t][i]) << i for i in range(n))
        bv = sum(val(b[t][i]) << i for i in range(n))
        cv = sum(val(c[t][i]) << i for i in range(n))
        out.append((av, bv, cv))
    if tensor_of(out, n) != residual:
        raise RuntimeError("the solver's products do not add up to the residual")
    return "sat", out


def blocks(triples: list[tuple[int, int, int]], n: int) -> list[list[int]]:
    """Groups of products that belong together: products whose forms lie in one small subspace (a GF(8) point of a
    tower spans three dimensions, a GF(4) product two), found by greedy closure of spans of dimension at most 3."""
    def rank(vs: list[int]) -> int:
        basis: list[int] = []
        for v in vs:
            for bv in basis:
                v = min(v, v ^ bv)
            if v:
                basis.append(v)
        return len(basis)
    left = list(range(len(triples)))
    groups = []
    while left:
        g = [left.pop(0)]
        for t in list(left):
            if rank([triples[s][0] for s in g] + [triples[t][0]]) <= 3:
                g.append(t)
                left.remove(t)
        groups.append(g)
    return groups


def neighbourhood(triples: list[tuple[int, int, int]], n: int, kind: str, k: int, rng: random.Random) -> list[int]:
    if kind == "random":
        return sorted(rng.sample(range(len(triples)), k))
    groups = blocks(triples, n)
    rng.shuffle(groups)
    chosen: list[int] = []
    for g in groups:
        if len(chosen) + len(g) > k:
            continue
        chosen += g
    return sorted(chosen)


def run(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    current = start(args.field, args.start)
    if not verified(current, args.field):
        raise ValueError("start is not exact")
    rows = []
    began = time.time()
    for trial in range(args.trials):
        if time.time() - began > args.budget:
            break
        k = rng.choice(args.k)
        chosen = neighbourhood(current, args.field, args.kind, k, rng)
        residual = tensor_of([current[t] for t in chosen], args.field)
        t0 = time.time()
        outcome, found = solve(residual, len(chosen) - 1, args.field, args.symmetric, args.limit, rng.getrandbits(30))
        row: dict[str, Any] = {"trial": trial, "k": len(chosen), "kind": args.kind, "symmetric": args.symmetric,
                               "outcome": outcome, "seconds": round(time.time() - t0, 2)}
        if outcome == "sat":
            new = [t for i, t in enumerate(current) if i not in chosen] + found
            ok = verified(new, args.field)
            row["exact"] = ok
            if ok:
                current = new
                row["products_now"] = len(current)
                row["triples"] = [list(t) for t in current]
        rows.append(row)
        print(json.dumps({k2: v for k2, v in row.items() if k2 != "triples"}), flush=True)
        if len(current) <= KNOWN.get(args.field, 0) - 1 and args.field == 12:
            break
    record = {"field": args.field, "start": args.start, "kind": args.kind, "symmetric": args.symmetric, "seed": args.seed,
              "limit": args.limit, "starting_products": len(start(args.field, args.start)), "best": len(current),
              "trials": rows}
    if len(current) < len(start(args.field, args.start)):
        record["best_triples"] = [list(t) for t in current]
    name = f"lns_f{args.field}_{args.start.replace(':', '-')}_{args.kind}_{'sym' if args.symmetric else 'gen'}_seed{args.seed}.json"
    (args.out / name).write_text(json.dumps(record) + "\n")
    return 0


def collect(args: argparse.Namespace) -> int:
    runs = [json.loads(p.read_text()) for p in sorted(args.out.glob("lns_*.json"))]
    for r in runs:
        if "best_triples" in r:
            found = [tuple(t) for t in r["best_triples"]]
            r["best_exact"] = len(found) == r["best"] and verified(found, r["field"])
    summary: dict[str, Any] = {}
    for n in sorted({r["field"] for r in runs}):
        some = [r for r in runs if r["field"] == n]
        trials = [t for r in some for t in r["trials"]]
        exact_best = [r["best"] for r in some if r.get("best_exact", "best_triples" not in r)]
        summary[str(n)] = {"runs": len(some), "trials": len(trials),
                           "sat": sum(t["outcome"] == "sat" for t in trials),
                           "unsat": sum(t["outcome"] == "unsat" for t in trials),
                           "unknown": sum(t["outcome"] == "unknown" for t in trials),
                           "by_k": {str(k): {o: sum(t["outcome"] == o for t in trials if t["k"] == k) for o in ("sat", "unsat", "unknown")}
                                    for k in sorted({t["k"] for t in trials})},
                           "known": KNOWN.get(n), "best_reached_exactly": min(exact_best, default=None),
                           "core_hours": round(sum(t["seconds"] for t in trials) / 3600, 2)}
    twelve = summary.get("12", {})
    record = {"tag": TAG, "development_not_a_result": True, "target": 41,
              "target_met": bool(twelve.get("best_reached_exactly") is not None and twelve["best_reached_exactly"] <= 41),
              "by_field": summary, "runs": runs}
    (args.out / "sat_lns.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "runs"}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("run", "collect"))
    parser.add_argument("--field", type=int, default=12)
    parser.add_argument("--start", default="ref")
    parser.add_argument("--kind", choices=("random", "blocks"), default="random")
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--k", type=int, nargs="+", default=[10])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--limit", type=float, default=60.0)
    parser.add_argument("--budget", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
