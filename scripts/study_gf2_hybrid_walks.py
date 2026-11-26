#!/usr/bin/env python3
"""Hybrids of the GF(2^n) searchers already tried, at equal time (development).

Status: audit-only

**Development, not a result.** On 26 November the owner asked for hybrids of the approaches tested to be
tried too. Each searcher alone has been measured: the plain flip walk (``scripts/gf2_flip_walk.c``), the
walk that keeps a Frobenius symmetry (``scripts/gf2_orbit_walk.c``, fast to the best known counts on
GF(2^8) and GF(2^9)) and SAT on neighbourhoods (``scripts/search_gf2_sat_lns.py``). Here they are chained,
each arm given the same total time:

* ``plain``: the plain walk from the schoolbook;
* ``plain-normal``: the plain walk from the schoolbook in a normal basis (the symmetric walk's start, as a
  control: is any gain the start's or the symmetry's?);
* ``orbit``: the symmetric walk from the normal-basis schoolbook;
* ``orbit+plain`` (hybrid): the symmetric walk for half the time, then the plain walk from its best, which
  may break the symmetry;
* ``plain+sat`` (hybrid): the plain walk for half the time, then SAT on random neighbourhoods of its best
  (6 to 8 products asked for one fewer) for the rest;
* ``orbit+sat`` (hybrid): the symmetric walk for half the time, then SAT on its best.

Every result is checked against all n^3 entries of the multiplication tensor before it is counted.

    python scripts/study_gf2_hybrid_walks.py --fields 8:4 9:3 10:2 --seconds 40 --seeds 3 --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gf2_towers as towers  # noqa: E402
import search_gf2_mul_flip_c as plain_driver  # noqa: E402
import search_gf2_orbit_walk as orbit_driver  # noqa: E402
import search_gf2_sat_lns as sat  # noqa: E402

TAG = "DEV-GF2-HYBRID-WALKS-2026-11"
KNOWN = {4: 9, 5: 13, 6: 15, **orbit_driver.KNOWN}   #: least counts known (4 to 7 proven)
ARMS = ("plain", "plain-normal", "orbit", "orbit+plain", "plain+sat", "orbit+sat")
Triples = list[tuple[int, int, int]]


def _parse(out: str) -> tuple[int, Triples]:
    lines = out.strip().split("\n")
    best = int(lines[0].split()[0])
    return best, [tuple(int(v) for v in line.split()) for line in lines[1:] if line.strip()]  # type: ignore[misc]


def plain_walk(binary: Path, n: int, triples: Triples, seconds: float, seed: int) -> Triples:
    text = f"{n} {len(triples)}\n" + "".join(f"{a} {b} {c}\n" for a, b, c in triples)
    done = subprocess.run([str(binary), str(seconds), str(seed), "100", "2", "300000", "0"], input=text,  # noqa: S603 - built here
                          capture_output=True, text=True, timeout=seconds + 600, check=True)
    best, found = _parse(done.stdout)
    return found if len(found) == best else triples


def orbit_walk(binary: Path, n: int, step: int, triples: Triples, seconds: float, seed: int) -> Triples:
    text = f"{n} {step} {towers.MODULI[n]} {len(triples)}\n" + "".join(f"{a} {b} {c}\n" for a, b, c in triples)
    done = subprocess.run([str(binary), str(seconds), str(seed), "100", "2", "1000000", "0"], input=text,  # noqa: S603 - built here
                          capture_output=True, text=True, timeout=seconds + 600, check=True)
    best, found = _parse(done.stdout)
    return found if len(found) == best else triples


def sat_polish(n: int, triples: Triples, seconds: float, seed: int, ks: tuple[int, ...] = (6, 7, 8),
               limit: float = 5.0) -> tuple[Triples, dict[str, int]]:
    """SAT on random neighbourhoods: k products of the decomposition asked for k - 1 with the same sum."""
    rng = random.Random(seed)
    current = list(triples)
    tally = {"sat": 0, "unsat": 0, "unknown": 0}
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds and len(current) > max(ks):
        k = rng.choice(ks)
        chosen = sat.neighbourhood(current, n, "random", k, rng)
        residual = sat.tensor_of([current[t] for t in chosen], n)
        outcome, found = sat.solve(residual, k - 1, n, False, min(limit, max(0.5, seconds - (time.monotonic() - t0))),
                                   rng.getrandbits(30))
        tally[outcome] = tally.get(outcome, 0) + 1
        if outcome == "sat":
            new = [t for i, t in enumerate(current) if i not in chosen] + found
            if towers.exact(new, n):
                current = new
    return current, tally


def run_arm(arm: str, n: int, step: int, seconds: float, seed: int, binaries: dict[str, Path]) -> dict[str, Any]:
    school = plain_driver.start(n, "schoolbook")
    normal = orbit_driver.normal_schoolbook(n)
    half = seconds / 2
    tally: dict[str, int] = {}
    if arm == "plain":
        out = plain_walk(binaries["plain"], n, school, seconds, seed)
    elif arm == "plain-normal":
        out = plain_walk(binaries["plain"], n, normal, seconds, seed)
    elif arm == "orbit":
        out = orbit_walk(binaries["orbit"], n, step, normal, seconds, seed)
    elif arm == "orbit+plain":
        out = plain_walk(binaries["plain"], n, orbit_walk(binaries["orbit"], n, step, normal, half, seed), half, seed)
    elif arm == "plain+sat":
        out, tally = sat_polish(n, plain_walk(binaries["plain"], n, school, half, seed), half, seed)
    elif arm == "orbit+sat":
        out, tally = sat_polish(n, orbit_walk(binaries["orbit"], n, step, normal, half, seed), half, seed)
    else:
        raise ValueError(arm)
    return {"arm": arm, "field": n, "step": step, "seed": seed, "seconds": seconds, "best": len(out),
            "exact": towers.exact(out, n), "sat_tally": tally}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--fields", nargs="+", default=["8:4", "9:3", "10:2"], help="field:step of the symmetry")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        binaries = {"plain": plain_driver.build(Path(tmp) / "plain"), "orbit": orbit_driver.build(Path(tmp) / "orbit")}
        jobs = [(arm, int(f.split(":")[0]), int(f.split(":")[1]), seed) for f in args.fields
                for seed in range(args.seeds) for arm in args.arms]
        with ThreadPoolExecutor(max_workers=args.processes) as pool:
            rows = list(pool.map(lambda j: run_arm(j[0], j[1], j[2], args.seconds, j[3], binaries), jobs))
    summary: dict[str, Any] = {}
    for f in args.fields:
        n = int(f.split(":")[0])
        summary[f"GF(2^{n})"] = {"known": KNOWN.get(n), **{arm: sorted(r["best"] for r in rows if r["field"] == n and r["arm"] == arm)
                                                          for arm in args.arms}}
    record = {"tag": TAG, "development_not_a_result": True, "seconds_per_arm": args.seconds,
              "all_exact": all(r["exact"] for r in rows), "summary": summary, "rows": rows}
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
