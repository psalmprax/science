#!/usr/bin/env python3
"""GF(2^n) multiplication with few AND gates: a flip walk over quadratic circuits (development).

Status: deprecated (retired 27 Nov 2026 on the owner's word; kept to replicate its record)

**Development, not a result, and not the bilinear target.** The owner asked on 26 November to build and
test four unconventional approaches to the GF(2^12) target; this is the first: count AND gates rather than
bilinear products. A gate may multiply two XORs of bits of both inputs, (a.z)(b.z) with z = (x, y); what
it contributes beyond XORs is the 2-form a ^ b (Mirwald and Schnorr). This is a larger space than bilinear
decompositions (those are the gates with a on x and b on y), and each gate has three interchangeable
vectors in its plane, so decompositions that are dead ends for the bilinear walk have neighbours here.
**A circuit of 41 AND gates found here is not a bilinear decomposition of 41 products and does not meet
decision 4 unless the owner says it should.** The record says which it is.

``scripts/gf2_and_walk.c`` walks; starts are bilinear decompositions written as gates (the repository's
42, the GF(4)-type 42s, towers, the schoolbook). Every result is checked here twice: its 2-forms against
those of the product, output by output, and as a circuit (gates, then the linear corrections) on
4,096 random pairs, or on every pair for n <= 8.

    python scripts/search_gf2_and_walk.py run --field 12 --starts ref gf4:0 --seconds 600 --out DIR
    python scripts/search_gf2_and_walk.py collect --out DIR
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gf2_towers as towers  # noqa: E402

TAG = "DEV-GF2-12-AND-WALK-2026-11"
SOURCE = ROOT / "scripts" / "gf2_and_walk.c"
BILINEAR_KNOWN = {2: 3, 3: 6, 4: 9, 5: 13, 6: 15, 7: 22, 8: 24, 9: 30, 10: 33, 12: 42}   #: bilinear (2-4, 6, 7 proven)


def gates_from_bilinear(triples: list[tuple[int, int, int]], n: int) -> list[tuple[int, int, int]]:
    return [(a, b << n, c) for a, b, c in triples]


def start(n: int, spec: str) -> list[tuple[int, int, int]]:
    """``schoolbook``, ``ref`` (the repository's 42), ``gf4:K`` (the GF(4)-type 42s) or ``tower:k:variant``."""
    if spec == "schoolbook":
        mod = towers.MODULI[n]
        bil = [(1 << i, 1 << j, c) for i in range(n) for j in range(n) if (c := towers.gf_mul(1 << i, 1 << j, mod))]
    elif spec == "ref":
        from apedp_dyn.quantum import flip_search, tower_reference
        bil = flip_search.from_decomposition(tower_reference.build_tower_42_reference()).triples()
    elif spec.startswith("gf4:"):
        import search_gf2_orbit_walk as orbit
        bil = orbit.start(12, 2, spec)
    elif spec.startswith("tower:"):
        _, k, variant = spec.split(":")
        vs = towers.variants(n, int(k))
        pts, theta, base = vs[int(variant) % len(vs)]
        bil = towers.tower(n, int(k), pts, theta, base)
    else:
        raise ValueError(spec)
    return gates_from_bilinear(bil, n)


def _wedge(a: int, b: int, m: int) -> int:
    """The 2-form a ^ b on m variables, bit p*m+q for p < q."""
    out = 0
    for p in range(m):
        for q in range(p + 1, m):
            if ((a >> p) & (b >> q) & 1) ^ ((a >> q) & (b >> p) & 1):
                out |= 1 << (p * m + q)
    return out


def two_forms_exact(gates: list[tuple[int, int, int]], n: int) -> bool:
    """Output by output, the 2-forms of the gates add up to those of the product."""
    mod, m = towers.MODULI[n], 2 * n
    want = [0] * n
    for i in range(n):
        for j in range(n):
            c = towers.gf_mul(1 << i, 1 << j, mod)
            for k in range(n):
                if c >> k & 1:
                    want[k] ^= 1 << (i * m + n + j)
    got = [0] * n
    for a, b, c in gates:
        w = _wedge(a, b, m)
        for k in range(n):
            if c >> k & 1:
                got[k] ^= w
    return got == want


def circuit_exact(gates: list[tuple[int, int, int]], n: int, samples: int = 4096, seed: int = 0) -> bool:
    """As a circuit: each output is the XOR of its gates and of their diagonal (linear) parts; checked on every
    pair for n <= 8, else on ``samples`` random pairs."""
    mod = towers.MODULI[n]
    lin = [0] * n
    for a, b, c in gates:
        d = a & b
        for k in range(n):
            if c >> k & 1:
                lin[k] ^= d
    if n <= 8:
        pairs = [(x, y) for x in range(1 << n) for y in range(1 << n)]
    else:
        rng = random.Random(seed)
        pairs = [(rng.getrandbits(n), rng.getrandbits(n)) for _ in range(samples)]
    for x, y in pairs:
        z = x | y << n
        out = 0
        for a, b, c in gates:
            if bin(a & z).count("1") & 1 and bin(b & z).count("1") & 1:
                out ^= c
        for k in range(n):
            out ^= (bin(lin[k] & z).count("1") & 1) << k
        if out != towers.gf_mul(x, y, mod):
            return False
    return True


def verified(gates: list[tuple[int, int, int]], n: int) -> bool:
    return bool(gates) and two_forms_exact(gates, n) and circuit_exact(gates, n)


def is_bilinear(gates: list[tuple[int, int, int]], n: int) -> bool:
    """Every gate's plane has a basis with one vector on x and one on y (then it is a bilinear product)."""
    xmask, ymask = (1 << n) - 1, ((1 << n) - 1) << n
    for a, b, _ in gates:
        vs = (a, b, a ^ b)
        if not (any(v and not v & ymask for v in vs) and any(v and not v & xmask for v in vs)):
            return False
    return True


def build(binary: Path) -> Path:
    subprocess.run(["gcc", "-O3", "-march=native", "-o", str(binary), str(SOURCE)], check=True)  # noqa: S603,S607 - the repository's own source
    return binary


def walk(binary: Path, n: int, spec: str, seed: int, seconds: float, slack: int, plateau: int, reset: int,
         target: int) -> dict[str, Any]:
    gates = start(n, spec)
    if not verified(gates, n):
        raise ValueError(f"start {spec} is not exact")
    text = f"{n} {len(gates)}\n" + "".join(f"{a} {b} {c}\n" for a, b, c in gates)
    done = subprocess.run([str(binary), str(seconds), str(seed), str(plateau), str(slack), str(reset), str(target)],  # noqa: S603 - built here
                          input=text, capture_output=True, text=True, timeout=seconds + 600)
    if done.returncode:
        raise RuntimeError(f"walk failed ({done.returncode}): {done.stderr[-400:]}")
    lines = done.stdout.strip().split("\n")
    best, flips, pluses, resets, took = lines[0].split()
    found = [tuple(int(v) for v in line.split()) for line in lines[1:] if line.strip()]
    ok = len(found) == int(best) and verified(found, n)            # type: ignore[arg-type]
    row: dict[str, Any] = {"field": n, "start": spec, "starting_gates": len(gates), "seed": seed, "slack": slack,
                           "plateau": plateau, "reset": reset, "seconds": float(took), "best_and_gates": int(best),
                           "flips": int(flips), "plus_transitions": int(pluses), "resets": int(resets), "exact": ok,
                           "best_is_bilinear": bool(ok and is_bilinear(found, n)),            # type: ignore[arg-type]
                           "progress": [line for line in done.stderr.split("\n") if line.startswith("best")][-6:]}
    if ok and int(best) < BILINEAR_KNOWN.get(n, 10**6):
        row["gates"] = [list(t) for t in found]
    return row


def run(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        binary = build(Path(tmp) / "walk")
        jobs = list(zip(args.starts, args.seeds, args.slack * len(args.starts), strict=False))
        with ThreadPoolExecutor(max_workers=args.processes) as pool:
            futures = [pool.submit(walk, binary, args.field, spec, seed, args.seconds, slack, args.plateau, args.reset,
                                   args.target) for spec, seed, slack in jobs]
            for future in futures:
                row = future.result()
                name = f"and_f{row['field']}_{row['start'].replace(':', '-')}_seed{row['seed']}_s{row['slack']}.json"
                (args.out / name).write_text(json.dumps(row) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k != "gates"}), flush=True)
    return 0


def collect(args: argparse.Namespace) -> int:
    rows = [json.loads(p.read_text()) for p in sorted(args.out.glob("and_*.json"))]
    for row in rows:
        if "gates" in row:
            found = [tuple(t) for t in row["gates"]]
            row["exact"] = len(found) == row["best_and_gates"] and verified(found, row["field"])
    by_field: dict[str, Any] = {}
    for n in sorted({r["field"] for r in rows}):
        some = [r for r in rows if r["field"] == n]
        exact = [r for r in some if r["exact"]]
        best = min((r["best_and_gates"] for r in exact), default=None)
        by_field[str(n)] = {"walks": len(some), "bilinear_known": BILINEAR_KNOWN.get(n), "least_and_gates_exactly": best,
                            "below_the_bilinear_count": bool(best is not None and best < BILINEAR_KNOWN.get(n, 10**6)),
                            "walks_not_exact": len(some) - len(exact), "flips_in_all": sum(r["flips"] for r in some),
                            "core_hours": round(sum(r["seconds"] for r in some) / 3600, 1)}
    record = {"tag": TAG, "development_not_a_result": True,
              "what_it_counts": "AND gates of a quadratic circuit, not bilinear products; not decision 4 unless the owner says so",
              "by_field": by_field, "walks": rows}
    (args.out / "and_walk.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "walks"}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("run", "collect"))
    parser.add_argument("--field", type=int, default=12)
    parser.add_argument("--starts", nargs="+", default=["ref"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--slack", type=int, nargs="+", default=[2])
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--plateau", type=int, default=100)
    parser.add_argument("--reset", type=int, default=1_000_000)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
