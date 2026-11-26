#!/usr/bin/env python3
"""GF(2^12) below 42 products: the flip walk in C from many tower decompositions (development).

Status: audit-only

**Development, not a result.** The owner asked on 26 November for the GF(2^12) target (41 products or
fewer, decision 4) to be pursued until it is solved, by unconventional means where needed. What is
known: the 42 is a tower (seven products over GF(8), each of six over GF(2)); the best published
counts for GF(2^8), GF(2^9) and GF(2^10) are towers too; a Chudnovsky-type construction over GF(2)
cannot go below 43 on curves of genus 3 or less by the counts of their places (``/tmp`` working, see the
record); the Python walk could not leave the 42 and fell short of published counts from the
schoolbook beyond GF(2^7). This runs ``scripts/gf2_flip_walk.c`` (the same moves, about a hundred
times faster) from many towers that differ by construction (``scripts/gf2_towers.py``: the
interpolation points, the generator over the subfield, the subfield's decomposition), from the
repository's own 42, and, as calibration, from the GF(2^9) towers of 30 and the GF(2^8) schoolbook.

Nothing counts until it is checked here against every entry of its tensor; for GF(2^12) also by
``field_tensor_search.verify_tensor_decomposition``.

    python scripts/search_gf2_mul_flip_c.py run --field 12 --starts tower:3:0 tower:3:1 ref --seconds 600 --out DIR
    python scripts/search_gf2_mul_flip_c.py collect --out DIR
"""
from __future__ import annotations

import argparse
import json
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

TAG = "DEV-GF2-12-FLIP-WALK-C-2026-11"
SOURCE = ROOT / "scripts" / "gf2_flip_walk.c"
KNOWN = {7: 22, 8: 24, 9: 30, 10: 33, 12: 42}   #: least counts known (7: proven, Barbulescu et al.; others published upper bounds)
TARGET = 41


def build(binary: Path) -> Path:
    subprocess.run(["gcc", "-O3", "-march=native", "-o", str(binary), str(SOURCE)], check=True)  # noqa: S603,S607 - the repository's own source
    return binary


def start(field: int, spec: str) -> list[tuple[int, int, int]]:
    """``schoolbook``, ``ref`` (the repository's 42) or ``tower:k:variant``."""
    if spec == "schoolbook":
        mod = towers.MODULI[field]
        return [(1 << i, 1 << j, c) for i in range(field) for j in range(field)
                if (c := towers.gf_mul(1 << i, 1 << j, mod))]
    if spec == "ref":
        if field != 12:
            raise ValueError("the repository's reference is GF(2^12)")
        from apedp_dyn.quantum import flip_search, tower_reference
        return flip_search.from_decomposition(tower_reference.build_tower_42_reference()).triples()
    kind, k, variant = spec.split(":")
    if kind != "tower":
        raise ValueError(spec)
    pts, theta, base = towers.variants(field, int(k))[int(variant) % len(towers.variants(field, int(k)))]
    return towers.tower(field, int(k), pts, theta, base)


def verified(triples: list[tuple[int, int, int]], field: int) -> bool:
    if not towers.exact(triples, field):
        return False
    if field != 12:
        return True
    from apedp_dyn.quantum import flip_search
    return flip_search.verified(triples)


def walk(binary: Path, field: int, spec: str, seed: int, seconds: float, slack: int, plateau: int, reset: int) -> dict[str, Any]:
    triples = start(field, spec)
    text = f"{field} {len(triples)}\n" + "".join(f"{a} {b} {c}\n" for a, b, c in triples)
    done = subprocess.run([str(binary), str(seconds), str(seed), str(plateau), str(slack), str(reset), "0"],  # noqa: S603 - built here
                          input=text, capture_output=True, text=True, timeout=seconds + 600)
    lines = done.stdout.strip().split("\n")
    best, flips, pluses, resets, took = lines[0].split()
    found = [tuple(int(v) for v in line.split()) for line in lines[1:] if line.strip()]
    ok = len(found) == int(best) and verified(found, field)
    row = {"field": field, "start": spec, "starting_products": len(triples), "seed": seed, "slack": slack,
           "plateau": plateau, "reset": reset, "seconds": float(took), "best": int(best), "flips": int(flips),
           "plus_transitions": int(pluses), "resets": int(resets), "exact": ok,
           "progress": [line for line in done.stderr.split("\n") if line.startswith("best")][-6:]}
    if ok and int(best) < KNOWN.get(field, 10**6):
        row["triples"] = [list(t) for t in found]          # anything below what is known is kept whole
    return row


def run(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        binary = build(Path(tmp) / "walk")
        jobs = [(spec, seed, slack) for spec, seed, slack in zip(args.starts, args.seeds, args.slack * len(args.starts), strict=False)]
        with ThreadPoolExecutor(max_workers=args.processes) as pool:
            futures = [pool.submit(walk, binary, args.field, spec, seed, args.seconds, slack, args.plateau, args.reset)
                       for spec, seed, slack in jobs]
            for future in futures:
                row = future.result()
                name = f"walk_f{row['field']}_{row['start'].replace(':', '-')}_seed{row['seed']}_s{row['slack']}.json"
                (args.out / name).write_text(json.dumps(row) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k != "triples"}), flush=True)
    return 0


def collect(args: argparse.Namespace) -> int:
    rows = [json.loads(p.read_text()) for p in sorted(args.out.glob("walk_*.json"))]
    for row in rows:                                         # checked again, not taken from the job's word
        if "triples" in row:
            found = [tuple(t) for t in row["triples"]]
            row["exact"] = len(found) == row["best"] and verified(found, row["field"])
    by_field: dict[str, Any] = {}
    for field in sorted({r["field"] for r in rows}):
        some = [r for r in rows if r["field"] == field]
        exact = [r for r in some if r["exact"]]
        best = min((r["best"] for r in exact), default=None)
        by_field[str(field)] = {"walks": len(some), "known": KNOWN.get(field), "best_reached_exactly": best,
                                "below_known": bool(best is not None and best < KNOWN.get(field, 10**6)),
                                "walks_not_exact": len(some) - len(exact),
                                "by_start_kind": {kind: min((r["best"] for r in exact if r["start"].startswith(kind)), default=None)
                                                  for kind in ("tower", "ref", "schoolbook")},
                                "flips_in_all": sum(r["flips"] for r in some),
                                "core_hours": round(sum(r["seconds"] for r in some) / 3600, 1)}
    twelve = by_field.get("12", {})
    record = {"tag": TAG, "development_not_a_result": True, "target": TARGET,
              "target_met": bool(twelve.get("best_reached_exactly") is not None and twelve["best_reached_exactly"] <= TARGET),
              "by_field": by_field, "walks": rows}
    (args.out / "flip_walk_c.json").write_text(json.dumps(record, indent=1) + "\n")
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
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
