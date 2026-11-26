#!/usr/bin/env python3
"""GF(2^12) below 42 products: the flip walk that keeps a Frobenius symmetry (development).

Status: audit-only

**Development, not a result.** The owner asked on 26 November for the GF(2^12) target (41 products or
fewer, decision 4) to be pursued until it is solved, and on the same day to build and test four
unconventional approaches; this is the second: flip graphs with symmetry (Moosbauer and Poole, ISSAC
2025, who found new matrix multiplication schemes this way). A decomposition invariant under
phi: x -> x^(2^step) is a union of orbits of tau(a, b, c) = (P^T a, P^T b, F c); ``scripts/gf2_orbit_walk.c``
applies every flip, merge and plus transition to whole orbits, and collapses an orbit whose
representative has two factors fixed by a smaller power of tau. The space is far smaller than the whole
flip graph, and the complete searches of this repository covered only its symmetric part (a = b).

Starts: the six decompositions of 14 GF(4) products found by the invariant search (``gf4:K``, 42 over
GF(2), invariant under x -> x^4 and so under x -> x^16 and x -> x^64); for calibration, a GF(16) tower
of GF(2^8) (27) invariant under x -> x^16, where the best known 24 lies in the class (a symmetric
decomposition of 24 invariant under x -> x^2 exists). Nothing counts until it is checked here against
every entry of its tensor, for GF(2^12) also by ``field_tensor_search.verify_tensor_decomposition``, and
found invariant.

    python scripts/search_gf2_orbit_walk.py run --field 12 --step 2 --starts gf4:0 gf4:1 --seconds 600 --out DIR
    python scripts/search_gf2_orbit_walk.py collect --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gf2_invariant_search as inv  # noqa: E402
import gf2_towers as towers  # noqa: E402

TAG = "DEV-GF2-12-ORBIT-WALK-2026-11"
SOURCE = ROOT / "scripts" / "gf2_orbit_walk.c"
KNOWN = {7: 22, 8: 24, 9: 30, 10: 33, 12: 42}   #: best published (7: proven)
TARGET = 41
#: the six choices of orbits (x -> x^4, labels 406 403 402 402 401) the invariant search finds at 14 GF(4) products
GF4_CHOICES = [[103, 131, 138, 195, 1], [122, 130, 884, 1970, 678], [127, 197, 884, 1970, 678],
               [166, 131, 822, 2036, 459], [303, 197, 138, 195, 1], [495, 130, 822, 2036, 459]]
GF4_LABELS = [406, 403, 402, 402, 401]


@cache
def tau_tables(n: int, step: int) -> tuple[list[int], list[int]]:
    """tau on forms (P^T a) and on outputs (F c) for phi: x -> x^(2^step)."""
    mod = towers.MODULI[n]

    def frob(v: int, k: int) -> int:
        for _ in range(k % n):
            v = towers.gf_mul(v, v, mod)
        return v
    outs = [frob(v, step) for v in range(1 << n)]
    back = [frob(1 << j, n - step) for j in range(n)]
    forms = [sum((bin(v & back[j]).count("1") & 1) << j for j in range(n)) for v in range(1 << n)]
    return forms, outs


def invariant(triples: list[tuple[int, int, int]], n: int, step: int) -> bool:
    forms, outs = tau_tables(n, step)
    return Counter(triples) == Counter((forms[a], forms[b], outs[c]) for a, b, c in triples)


def start(field: int, step: int, spec: str) -> list[tuple[int, int, int]]:
    """``gf4:K`` (GF(2^12)), ``normal`` (the schoolbook in a normal basis, invariant under every step) or
    ``tower-inv:k`` (the first tower over GF(2^k) invariant under the step)."""
    kind, _, arg = spec.partition(":")
    if kind == "gf4":
        if field != 12:
            raise ValueError("the GF(4) starts are for GF(2^12)")
        triples, _ = inv.extract_mixed(2, GF4_LABELS, GF4_CHOICES[int(arg)])
        return [tuple(t) for t in triples]                     # type: ignore[misc]
    if kind == "normal":
        return normal_schoolbook(field)
    if kind == "tower-inv":
        k = int(arg)
        for pts, theta, base in towers.variants(field, k):
            t = towers.tower(field, k, pts, theta, base)
            if invariant(t, field, step):
                return t
        raise ValueError(f"no tower over GF(2^{k}) of GF(2^{field}) is invariant under step {step}")
    raise ValueError(spec)


def normal_schoolbook(n: int) -> list[tuple[int, int, int]]:
    """The schoolbook in a normal basis: (x_i)(y_j) beta_i beta_j with x_i = Tr(gamma_i x), gamma the dual basis.
    The Frobenius shifts a normal basis cyclically, so this start is invariant under every power of it."""
    mod = towers.MODULI[n]

    def frob(v: int) -> int:
        return towers.gf_mul(v, v, mod)

    def trace(v: int) -> int:
        acc, t = 0, v
        for _ in range(n):
            acc ^= t
            t = frob(t)
        return acc & 1
    for beta in range(1, 1 << n):
        conj = [beta]
        for _ in range(n - 1):
            conj.append(frob(conj[-1]))
        gram = [[trace(towers.gf_mul(a, b, mod)) for b in conj] for a in conj]
        inverse = _invert_gf2(gram)
        if inverse is not None:
            break
    dual = []
    for i in range(n):
        g = 0
        for j in range(n):
            if inverse[i][j]:
                g ^= conj[j]
        dual.append(g)
    forms = [sum(trace(towers.gf_mul(g, 1 << q, mod)) << q for q in range(n)) for g in dual]
    out = []
    for i in range(n):
        for j in range(n):
            c = towers.gf_mul(conj[i], conj[j], mod)
            if c:
                out.append((forms[i], forms[j], c))
    return out


def _invert_gf2(matrix: list[list[int]]) -> list[list[int]] | None:
    size = len(matrix)
    a = [row[:] + [int(i == j) for j in range(size)] for i, row in enumerate(matrix)]
    for col in range(size):
        piv = next((i for i in range(col, size) if a[i][col]), None)
        if piv is None:
            return None
        a[col], a[piv] = a[piv], a[col]
        for i in range(size):
            if i != col and a[i][col]:
                a[i] = [x ^ y for x, y in zip(a[i], a[col], strict=True)]
    return [row[size:] for row in a]


def verified(triples: list[tuple[int, int, int]], field: int, step: int) -> bool:
    if not triples or not towers.exact(triples, field) or not invariant(triples, field, step):
        return False
    if field != 12:
        return True
    from apedp_dyn.quantum import flip_search
    return flip_search.verified(triples)


def build(binary: Path) -> Path:
    subprocess.run(["gcc", "-O3", "-march=native", "-o", str(binary), str(SOURCE)], check=True)  # noqa: S603,S607 - the repository's own source
    return binary


def walk(binary: Path, field: int, step: int, spec: str, seed: int, seconds: float, slack: int, plateau: int,
         reset: int, target: int, check: bool = False) -> dict[str, Any]:
    """One walk; with ``check`` the C program recomputes the whole tensor after every move and stops if it changed."""
    triples = start(field, step, spec)
    if not verified(triples, field, step):
        raise ValueError(f"start {spec} is not exact and invariant")
    text = f"{field} {step} {towers.MODULI[field]} {len(triples)}\n" + "".join(f"{a} {b} {c}\n" for a, b, c in triples)
    env = dict(os.environ, ORBIT_CHECK="1") if check else {k: v for k, v in os.environ.items() if k != "ORBIT_CHECK"}
    done = subprocess.run([str(binary), str(seconds), str(seed), str(plateau), str(slack), str(reset), str(target)],  # noqa: S603 - built here
                          input=text, capture_output=True, text=True, timeout=seconds + 600, env=env)
    if done.returncode:
        raise RuntimeError(f"walk failed ({done.returncode}): {done.stderr[-400:]}")
    lines = done.stdout.strip().split("\n")
    best, flips, pluses, resets, took = lines[0].split()
    found = [tuple(int(v) for v in line.split()) for line in lines[1:] if line.strip()]
    ok = len(found) == int(best) and verified(found, field, step)        # type: ignore[arg-type]
    row: dict[str, Any] = {"field": field, "step": step, "start": spec, "starting_products": len(triples), "seed": seed,
                           "slack": slack, "plateau": plateau, "reset": reset, "seconds": float(took), "best": int(best),
                           "flips": int(flips), "plus_transitions": int(pluses), "resets": int(resets), "exact": ok,
                           "progress": [line for line in done.stderr.split("\n") if line.startswith("best")][-6:]}
    if ok and int(best) <= KNOWN.get(field, 10**6):
        row["triples"] = [list(t) for t in found]
    return row


def run(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.target if args.target is not None else (TARGET if args.field == 12 else KNOWN.get(args.field, 0))
    with tempfile.TemporaryDirectory() as tmp:
        binary = build(Path(tmp) / "walk")
        jobs = list(zip(args.starts, args.seeds, args.slack * len(args.starts), strict=False))
        with ThreadPoolExecutor(max_workers=args.processes) as pool:
            futures = [pool.submit(walk, binary, args.field, args.step, spec, seed, args.seconds, slack, args.plateau,
                                   args.reset, target) for spec, seed, slack in jobs]
            for future in futures:
                row = future.result()
                name = f"orbit_f{row['field']}_st{row['step']}_{row['start'].replace(':', '-')}_seed{row['seed']}_s{row['slack']}.json"
                (args.out / name).write_text(json.dumps(row) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k != "triples"}), flush=True)
    return 0


def collect(args: argparse.Namespace) -> int:
    rows = [json.loads(p.read_text()) for p in sorted(args.out.glob("orbit_*.json"))]
    for row in rows:
        if "triples" in row:
            found = [tuple(t) for t in row["triples"]]
            row["exact"] = len(found) == row["best"] and verified(found, row["field"], row["step"])
    by_class: dict[str, Any] = {}
    for key in sorted({(r["field"], r["step"]) for r in rows}):
        some = [r for r in rows if (r["field"], r["step"]) == key]
        exact = [r for r in some if r["exact"]]
        best = min((r["best"] for r in exact), default=None)
        by_class[f"GF(2^{key[0]}) x -> x^(2^{key[1]})"] = {
            "walks": len(some), "known": KNOWN.get(key[0]), "best_reached_exactly": best,
            "below_known": bool(best is not None and best < KNOWN.get(key[0], 10**6)),
            "walks_not_exact": len(some) - len(exact), "flips_in_all": sum(r["flips"] for r in some),
            "core_hours": round(sum(r["seconds"] for r in some) / 3600, 1)}
    twelve = [v for k, v in by_class.items() if k.startswith("GF(2^12)")]
    met = any(v["best_reached_exactly"] is not None and v["best_reached_exactly"] <= TARGET for v in twelve)
    record = {"tag": TAG, "development_not_a_result": True, "target": TARGET, "target_met": met,
              "by_class": by_class, "walks": rows}
    (args.out / "orbit_walk.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "walks"}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("run", "collect"))
    parser.add_argument("--field", type=int, default=12)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--starts", nargs="+", default=["gf4:0"])
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
