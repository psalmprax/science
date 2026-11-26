#!/usr/bin/env python3
"""GF(4^6) multiplication over GF(4) in 13 products or fewer: the flip walk over GF(4) (development).

Status: deprecated (retired 27 Nov 2026 on the owner's word; kept to replicate its record)

**Development, not a result.** The owner asked on 26 November for the GF(2^12) target (41 products over
GF(2) or fewer) to be pursued until it is solved. GF(2^12) is GF(4^6); a decomposition of its
multiplication over GF(4) in r products gives 3r over GF(2) (each GF(4) product three, Karatsuba). The
best published over GF(4) is 14 (42 over GF(2)); 13 would give 39. This runs ``scripts/gf4_flip_walk.c``
(flips with a GF(4) scalar, merges, dependent groups, plus transitions) from the schoolbook and from the
decompositions of 14 that the invariant search found (``benchmarks/gf2_multiplication/
invariant_search_gf4_x16.json``), with GF(4^4) and GF(4^5) (published 8 and 11) as calibration; locally
the walk reaches 8 and 11 from the schoolbook in under a minute.

The field is GF(2^(2m)) with the modulus of ``gf2_towers.MODULI``, ω an element of order 3, and the
GF(4) basis 1, θ, ..., θ^(m-1) for a primitive θ. A GF(4) element b0 + b1 ω is coded b0 + 2 b1, a
vector of m of them in 2m bits. Nothing counts until it is checked here: every entry of the GF(4)
tensor, then its 3r products over GF(2) against every entry of the GF(2) tensor, and for GF(2^12) by
``field_tensor_search.verify_tensor_decomposition``.

    python scripts/search_gf4_mul_flip.py run --degree 6 --starts schoolbook f14:0 --seconds 600 --out DIR
    python scripts/search_gf4_mul_flip.py collect --out DIR
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gf2_invariant_search as inv  # noqa: E402
import gf2_towers as towers  # noqa: E402

TAG = "DEV-GF4-6-FLIP-WALK-2026-11"
SOURCE = ROOT / "scripts" / "gf4_flip_walk.c"
KNOWN = {4: 8, 5: 11, 6: 14}          #: best published over GF(4) (Ballet et al., survey, table 2)
TARGET = 13
INVARIANT = ROOT / "benchmarks" / "gf2_multiplication" / "invariant_search_gf4_x16.json"
MUL4 = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]


class Field4:
    """GF(2^(2m)) seen as GF(4^m): coordinates in the basis theta^i, GF(4) elements coded b0 + 2 b1."""

    def __init__(self, m: int) -> None:
        self.m, self.n = m, 2 * m
        self.field = inv.Field(self.n, 4)
        theta = towers.primitive_elements(self.n)[0]
        self.basis = [self.field.pow(theta, i) for i in range(m)]
        self.omega = self.field.omega
        cols = []
        for b in self.basis:
            cols += [b, self.field.mul(self.omega, b)]
        self.solve = _inverse(cols, self.n)            # GF(2) coordinates in the basis (beta_i, omega beta_i)

    def code(self, z: int) -> int:
        """The GF(4) coordinates of z, packed two bits each."""
        bits = self.solve(z)
        return sum(((bits >> (2 * i) & 1) | (bits >> (2 * i + 1) & 1) << 1) << (2 * i) for i in range(self.m))

    def element(self, coded: int) -> int:
        z = 0
        for i in range(self.m):
            e = coded >> (2 * i) & 3
            if e & 1:
                z ^= self.basis[i]
            if e & 2:
                z ^= self.field.mul(self.omega, self.basis[i])
        return z

    def scalar(self, t: int) -> int:
        """Code of a GF(4) element given as an element of the big field."""
        return {0: 0, 1: 1, self.omega: 2, self.omega ^ 1: 3}[t]

    def tensor(self) -> dict[tuple[int, int], int]:
        return {(i, j): self.code(self.field.mul(bi, bj)) for i, bi in enumerate(self.basis) for j, bj in enumerate(self.basis)}

    def form(self, lam: int) -> int:
        """The coded GF(4) linear form x -> Tr_{GF(4)}(lam x) in the coordinates of the basis."""
        return sum(self.scalar(self.field.trace(self.field.mul(lam, b))) << (2 * i) for i, b in enumerate(self.basis))


def _inverse(cols: list[int], n: int):
    """A function giving the coordinates of z in the GF(2) basis cols (n vectors of n bits)."""
    piv: dict[int, tuple[int, int]] = {}
    for k, col in enumerate(cols):
        v, tag = col, 1 << k
        for bit in sorted(piv, reverse=True):
            if v >> bit & 1:
                v ^= piv[bit][0]
                tag ^= piv[bit][1]
        if not v:
            raise ValueError("not a basis")
        piv[v.bit_length() - 1] = (v, tag)

    def solve(z: int) -> int:
        tag = 0
        for bit in sorted(piv, reverse=True):
            if z >> bit & 1:
                z ^= piv[bit][0]
                tag ^= piv[bit][1]
        if z:
            raise ValueError("outside the span")
        return tag
    return solve


@cache
def field4(m: int) -> Field4:
    return Field4(m)


def exact4(terms: list[tuple[int, int, int]], m: int) -> bool:
    """Every entry of the GF(4) tensor."""
    T = field4(m).tensor()
    for i in range(m):
        for j in range(m):
            for k in range(m):
                acc = 0
                for a, b, c in terms:
                    acc ^= MUL4[MUL4[a >> (2 * i) & 3][b >> (2 * j) & 3]][c >> (2 * k) & 3]
                if acc != T[(i, j)] >> (2 * k) & 3:
                    return False
    return True


def over_gf2(terms: list[tuple[int, int, int]], m: int) -> list[tuple[int, int, int]]:
    """Three GF(2) products for each GF(4) product, on the coefficient bits of the polynomial basis."""
    f = field4(m)
    coords = [f.code(1 << p) for p in range(f.n)]          # GF(4) coordinates of each GF(2) basis element

    def halves(form: int) -> tuple[int, int]:
        u0 = u1 = 0
        for p, x in enumerate(coords):
            t = 0
            for i in range(m):
                t ^= MUL4[form >> (2 * i) & 3][x >> (2 * i) & 3]
            u0 |= (t & 1) << p
            u1 |= (t >> 1 & 1) << p
        return u0, u1

    out = []
    w = f.omega
    for a, b, c in terms:
        (u0, u1), (v0, v1) = halves(a), halves(b)
        z = f.element(c)
        for x, y, o in ((u0, v0, f.field.mul(1 ^ w, z)), (u1, v1, z), (u0 ^ u1, v0 ^ v1, f.field.mul(w, z))):
            if x and y and o:
                out.append((x, y, o))
    return out


def verified(terms: list[tuple[int, int, int]], m: int) -> bool:
    if not terms or not exact4(terms, m):
        return False
    binary = over_gf2(terms, m)
    if not towers.exact(binary, 2 * m):
        return False
    if 2 * m != 12:
        return True
    from apedp_dyn.quantum import flip_search
    return flip_search.verified(binary)


def start(m: int, spec: str) -> list[tuple[int, int, int]]:
    """``schoolbook`` or ``f14:K`` (the K-th decomposition of 14 the invariant search recorded, GF(4^6) only)."""
    if spec == "schoolbook":
        T = field4(m).tensor()
        return [(1 << (2 * i), 1 << (2 * j), T[(i, j)]) for i in range(m) for j in range(m) if T[(i, j)]]
    kind, k = spec.split(":")
    if kind != "f14" or m != 6:
        raise ValueError(spec)
    record = json.loads(INVARIANT.read_text())
    reps = record["classes"]["x4.bin rank 14"]["reps"][int(k)]
    f = field4(m)
    # the record's step 1 counts powers of x -> x^4 (it ran before steps were made absolute): absolute step 2
    terms = inv.gf_terms(f.field, 2, reps)
    return [(f.form(lam), f.form(lam), f.code(c)) for lam, c in terms]


def build(binary: Path) -> Path:
    subprocess.run(["gcc", "-O3", "-march=native", "-o", str(binary), str(SOURCE)], check=True)  # noqa: S603,S607 - the repository's own source
    return binary


def walk(binary: Path, m: int, spec: str, seed: int, seconds: float, slack: int, plateau: int, reset: int, target: int) -> dict[str, Any]:
    terms = start(m, spec)
    if not verified(terms, m):
        raise ValueError(f"start {spec} is not exact")
    text = f"{m} {len(terms)}\n" + "".join(f"{a} {b} {c}\n" for a, b, c in terms)
    done = subprocess.run([str(binary), str(seconds), str(seed), str(plateau), str(slack), str(reset), str(target)],  # noqa: S603 - built here
                          input=text, capture_output=True, text=True, timeout=seconds + 600)
    lines = done.stdout.strip().split("\n")
    best, flips, pluses, resets, took = lines[0].split()
    found = [tuple(int(v) for v in line.split()) for line in lines[1:] if line.strip()]
    ok = len(found) == int(best) and verified(found, m)
    row: dict[str, Any] = {"degree": m, "start": spec, "starting_products": len(terms), "seed": seed, "slack": slack,
                           "plateau": plateau, "reset": reset, "seconds": float(took), "best": int(best), "flips": int(flips),
                           "plus_transitions": int(pluses), "resets": int(resets), "exact": ok,
                           "progress": [line for line in done.stderr.split("\n") if line.startswith("best")][-6:]}
    if ok and int(best) <= KNOWN.get(m, 10**6):
        row["terms"] = [list(t) for t in found]
    return row


def run(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.target if args.target is not None else (TARGET if args.degree == 6 else KNOWN.get(args.degree, 0))
    with tempfile.TemporaryDirectory() as tmp:
        binary = build(Path(tmp) / "walk")
        jobs = list(zip(args.starts, args.seeds, args.slack * len(args.starts), strict=False))
        with ThreadPoolExecutor(max_workers=args.processes) as pool:
            futures = [pool.submit(walk, binary, args.degree, spec, seed, args.seconds, slack, args.plateau, args.reset, target)
                       for spec, seed, slack in jobs]
            for future in futures:
                row = future.result()
                name = f"walk4_m{row['degree']}_{row['start'].replace(':', '-')}_seed{row['seed']}_s{row['slack']}.json"
                (args.out / name).write_text(json.dumps(row) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k != "terms"}), flush=True)
    return 0


def collect(args: argparse.Namespace) -> int:
    rows = [json.loads(p.read_text()) for p in sorted(args.out.glob("walk4_*.json"))]
    for row in rows:                                         # checked again, not taken from the job's word
        if "terms" in row:
            found = [tuple(t) for t in row["terms"]]
            row["exact"] = len(found) == row["best"] and verified(found, row["degree"])
    by_degree: dict[str, Any] = {}
    for m in sorted({r["degree"] for r in rows}):
        some = [r for r in rows if r["degree"] == m]
        exact = [r for r in some if r["exact"]]
        best = min((r["best"] for r in exact), default=None)
        by_degree[str(m)] = {"walks": len(some), "known": KNOWN.get(m), "best_reached_exactly": best,
                             "below_known": bool(best is not None and best < KNOWN.get(m, 10**6)),
                             "walks_not_exact": len(some) - len(exact),
                             "by_start_kind": {kind: min((r["best"] for r in exact if r["start"].startswith(kind)), default=None)
                                               for kind in ("schoolbook", "f14")},
                             "flips_in_all": sum(r["flips"] for r in some),
                             "core_hours": round(sum(r["seconds"] for r in some) / 3600, 1)}
    six = by_degree.get("6", {})
    record = {"tag": TAG, "development_not_a_result": True, "target_over_gf4": TARGET, "target_over_gf2": 3 * TARGET,
              "target_met": bool(six.get("best_reached_exactly") is not None and six["best_reached_exactly"] <= TARGET),
              "by_degree": by_degree, "walks": rows}
    (args.out / "gf4_flip_walk.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "walks"}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("run", "collect"))
    parser.add_argument("--degree", type=int, default=6)
    parser.add_argument("--starts", nargs="+", default=["schoolbook"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--slack", type=int, nargs="+", default=[2])
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--plateau", type=int, default=100)
    parser.add_argument("--reset", type=int, default=1_000_000)
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
