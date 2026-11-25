#!/usr/bin/env python3
"""The stronger baseline with no language model: a flip-graph search for multiplication in GF(2^12).

Status: audit-only

**Development, not a result.** The owner's decision 4 of 25 November sets the target at a decomposition
of multiplication in GF(2^12) with 41 products or fewer, the first baseline being the exact search the
repository already had (nothing found from the 42: ``benchmarks/gf2_multiplication/rank_search_from_42.json``).
That search cannot leave its starting point. This script runs one that can
(``apedp_dyn/quantum/flip_search.py``: random walks of flips, reductions and plus transitions), with no
model in it, on GitHub's machines, owner's word of 25 November ("go ahead and build it").

Two kinds of job:

- **Calibration**, on the smaller fields GF(2^7) to GF(2^10), where a flip-graph search of the same kind
  has published counts (Khoruzhii, Gelß and Pokutta, "Tensor decomposition for non-Clifford gate
  minimization", arXiv:2602.15285, Table 2, column "Bilinear": 22, 26, 32, 39). Each walk stops when it
  reaches the published count. A search that cannot reach those counts says little when it fails at
  GF(2^12).
- **The field itself**, GF(2^12), from three starts: the verified 42 products
  (``tower_reference.build_tower_42_reference``), the 54 products of the composed Karatsuba reference
  (``field_tensor_search.build_54_reference``) and the 144 products of the schoolbook. Walks run for the
  time given and keep the shortest decomposition they reach.

Nothing is counted that is not exact. ``collect`` checks every decomposition it reports against every
entry of its tensor (for GF(2^12) also by ``field_tensor_search.verify_tensor_decomposition``) and against
``MULTIPLICATIONS`` products of random pairs of field elements computed directly.

    python scripts/search_gf2_mul_flip.py run --field 12 --start tower42 --seconds 600 --seeds 0 1 --out DIR
    python scripts/search_gf2_mul_flip.py collect --out DIR
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.quantum import flip_search, tower_reference  # noqa: E402
from apedp_dyn.quantum.field_tensor_search import (  # noqa: E402
    MODULUS_INT,
    build_54_reference,
    gf_mul,
)

TAG = "DEV-GF2-12-FLIP-SEARCH-NO-MODEL-2026-11"
TARGET = 41
STARTING_POINT = 42
#: irreducible moduli of the calibration fields (bit k is the coefficient of u^k), and GF(2^12)'s own
MODULI = {7: 0b10000011, 8: 0b100011011, 9: 0b1000010001, 10: 0b10000001001, 12: MODULUS_INT}
#: published counts of a flip-graph search on the bilinear formulation (arXiv:2602.15285, Table 2, "Bilinear")
PUBLISHED = {7: 22, 8: 26, 9: 32, 10: 39}
STARTS = ("tower42", "karatsuba54", "schoolbook")
MULTIPLICATIONS = 4096


def field_tensor(n: int) -> np.ndarray:
    """The multiplication tensor of GF(2^n) in the polynomial basis: ``T[i, j, k]`` is bit ``k`` of ``u^i u^j``."""
    modulus = MODULI[n]
    T = np.zeros((n, n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            product = gf_mul(1 << i, 1 << j, modulus)
            for k in range(n):
                T[i, j, k] = (product >> k) & 1
    return T


def starting_scheme(n: int, start: str) -> flip_search.Scheme:
    if start == "schoolbook":
        return flip_search.schoolbook(field_tensor(n))
    if n != 12:
        raise ValueError(f"the start {start!r} exists for GF(2^12) only")
    if start == "tower42":
        return flip_search.from_decomposition(tower_reference.build_tower_42_reference())
    if start == "karatsuba54":
        return flip_search.from_decomposition(build_54_reference())
    raise ValueError(f"unknown start {start!r}")


def multiplications_right(triples: list[tuple[int, int, int]], n: int, pairs: int = MULTIPLICATIONS, seed: int = 20261125) -> int:
    """How many of ``pairs`` random products the decomposition gets wrong, computed from its products alone."""
    rng = random.Random(seed)
    wrong = 0
    for _ in range(pairs):
        x, y = rng.randrange(1 << n), rng.randrange(1 << n)
        z = 0
        for a, b, c in triples:
            if (bin(a & x).count("1") & 1) and (bin(b & y).count("1") & 1):
                z ^= c
        wrong += z != gf_mul(x, y, MODULI[n])
    return wrong


def one_walk(job: dict[str, Any]) -> dict[str, Any]:
    n, start, seed = job["field"], job["start"], job["seed"]
    tensor = field_tensor(n)
    scheme = starting_scheme(n, start)
    result = flip_search.walk(scheme, job["seconds"], seed=seed, target=job.get("target"), slack=job["slack"])
    summary = result.summary()
    return {"field": n, "start": start, "starting_products": summary.pop("start"), "seed": seed, "slack": job["slack"],
            "seconds_given": job["seconds"], "target": job.get("target"), **summary,
            "triples": [list(t) for t in result.best_triples], "exact": flip_search.verified(result.best_triples, tensor)}


def run(args: argparse.Namespace) -> int:
    target = PUBLISHED.get(args.field) if args.field != 12 else None
    jobs = [{"field": args.field, "start": args.start, "seed": seed, "seconds": args.seconds, "target": target,
             "slack": args.slack} for seed in args.seeds]
    with mp.get_context("spawn").Pool(min(args.processes, len(jobs))) as pool:
        for row in pool.imap_unordered(one_walk, jobs):
            name = f"walk_f{row['field']}_{row['start']}_s{row['slack']}_seed{row['seed']}.json"
            (args.out / name).write_text(json.dumps(row) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k not in ("triples", "reached")}), flush=True)
    return 0


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calibration = {}
    for n in sorted(PUBLISHED):
        some = [r for r in rows if r["field"] == n]
        if some:
            calibration[str(n)] = {"published": PUBLISHED[n], "walks": len(some), "best_reached": min(r["best"] for r in some),
                                   "walks_that_reached_the_published_count": sum(r["best"] <= PUBLISHED[n] for r in some),
                                   "seconds_to_reach_it": sorted(r["reached"][-1]["seconds"] for r in some if r["best"] <= PUBLISHED[n]),
                                   "all_exact": all(r["exact"] for r in some)}
    field = [r for r in rows if r["field"] == 12]
    by_start = {}
    for start in STARTS:
        some = [r for r in field if r["start"] == start]
        if some:
            by_start[start] = {"walks": len(some), "starting_products": some[0]["starting_products"],
                               "best_reached": min(r["best"] for r in some),
                               "bests": sorted(r["best"] for r in some), "flips_in_all": sum(r["flips"] for r in some),
                               "machine_hours": round(sum(r["seconds"] for r in some) / 3600, 2)}
    exact = [r for r in field if r["exact"]]
    best = min((r["best"] for r in exact), default=None)
    return {"target": TARGET, "starting_point": STARTING_POINT, "calibration": calibration, "gf2_12_by_start": by_start,
            "fewest_products_reached_exactly": best,
            "shorter_than_42": bool(best is not None and best < STARTING_POINT),
            "target_met": bool(best is not None and best <= TARGET),
            "walks_not_exact": sum(not r["exact"] for r in rows)}


def collect(args: argparse.Namespace) -> int:
    rows = [json.loads(path.read_text()) for path in sorted(args.out.glob("walk_*.json"))]
    for row in rows:                       # every decomposition checked again here, not taken from the walk's word
        triples = [tuple(t) for t in row["triples"]]
        row["exact"] = flip_search.verified(triples, field_tensor(row["field"]))
        row["products_counted"] = len(triples)
        row["exact"] = row["exact"] and len(triples) == row["best"]
        row["wrong_of_random_multiplications"] = multiplications_right(triples, row["field"])
        row["exact"] = row["exact"] and row["wrong_of_random_multiplications"] == 0
    summary = summarise(rows)
    shortest = None
    exact12 = sorted((r for r in rows if r["field"] == 12 and r["exact"]), key=lambda r: r["best"])
    if exact12:
        shortest = {"products": exact12[0]["best"], "start": exact12[0]["start"], "seed": exact12[0]["seed"],
                    "triples_bit_k_is_coefficient_k": exact12[0]["triples"]}
    record = {"tag": TAG, "development_not_a_result": True,
              "what": "a flip-graph search with no language model for multiplication in GF(2^12), with calibration on "
                      "GF(2^7) to GF(2^10) against published counts; every decomposition checked exactly",
              "search": "apedp_dyn/quantum/flip_search.py (flips, three reductions, plus transitions)",
              "settings": {"plateau": flip_search.PLATEAU, "reset": flip_search.RESET, "kept": flip_search.KEPT,
                           "random_multiplications_checked": MULTIPLICATIONS},
              "published_counts_source": "Khoruzhii, Gelß, Pokutta, arXiv:2602.15285 (2026), Table 2, column 'Bilinear'",
              "summary": summary, "shortest_for_gf2_12": shortest,
              "walks": [{k: v for k, v in r.items() if k != "triples" or r["field"] != 12 or r["best"] < STARTING_POINT}
                        for r in rows]}
    (args.out / "flip_search.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("run", "collect"))
    parser.add_argument("--field", type=int, choices=sorted(MODULI), default=12)
    parser.add_argument("--start", choices=STARTS, default="tower42")
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--slack", type=int, default=flip_search.SLACK)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    return run(args) if args.command == "run" else collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
