#!/usr/bin/env python3
"""The baseline with no language model: what the existing exact search reaches from the 42 products of GF(2^12).

Status: audit-only

**Development, not a result.** The owner's decision 4 of 25 November sets the first experiment's target
at a decomposition of multiplication in GF(2^12) with 41 products or fewer, and asks first for a
baseline: what the search the repository already has reaches from the verified 42
(``apedp_dyn/quantum/tower_reference.py``) with no model in it, so that any coding model has
something to beat. This is that baseline.

The search is ``apedp_dyn/quantum/gf2_rank_search.py``, unchanged: *deflation* (drop a product whose
removal still leaves every output spanned) and the *k -> k-1 rewrite* (replace ``k`` products by
``k-1``), each exhaustive over its family. One job is run for each ``k``.

**How much a negative says.** A rewrite exists only where the outputs, taken modulo the products kept,
leave a space of at most ``k-1`` dimensions all of whose elements are single products. That is a narrow
family, and the search walks it exhaustively rather than exploring: it cannot leave the neighbourhood of
the 42 it is given. "No rewrite up to k" is a statement about this family from this starting point. It
is not evidence that 41 is out of reach; a search that moves (a flip graph, as used for matrix
multiplication) is the stronger baseline and is not built yet.

    python scripts/search_gf2_mul_rank_from_42.py run --k 2 3 4 --out DIR
    python scripts/search_gf2_mul_rank_from_42.py collect --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.quantum import gf2_rank_search as search  # noqa: E402
from apedp_dyn.quantum import tower_reference  # noqa: E402
from apedp_dyn.quantum.field_tensor_search import build_multiplication_tensor  # noqa: E402

TAG = "DEV-GF2-12-BASELINE-NO-MODEL-2026-11"
TARGET = 41


def start() -> tuple[list[int], list[int], int]:
    """The 42 products as the search holds them, the outputs they must span, and the field's degree."""
    reference = tower_reference.build_tower_42_reference()
    slice_masks, degree = search.build_slices(build_multiplication_tensor())
    atoms = [atom for atom in search.atom_masks(reference.U, reference.V) if atom]
    return atoms, slice_masks, degree


def deflation() -> dict[str, Any]:
    atoms, slice_masks, _ = start()
    began = time.perf_counter()
    kept, dropped = search.deflate(atoms, slice_masks)
    return {"move": "deflation", "products_at_the_start": len(atoms), "products_dropped": dropped,
            "products_left": len(kept), "seconds": round(time.perf_counter() - began, 2)}


def rewrite(k: int) -> dict[str, Any]:
    """Every way to replace ``k`` of the 42 products by ``k-1``, by the search's own exhaustive step."""
    atoms, slice_masks, degree = start()
    began = time.perf_counter()
    shorter, tried = search.reduce_once(atoms, slice_masks, degree, k)
    out: dict[str, Any] = {"move": f"{k} -> {k - 1} rewrite", "k": k, "products_at_the_start": len(atoms),
                           "sets_of_products_in_the_family": math.comb(len(atoms), k), "sets_tried": tried,
                           "every_set_tried": shorter is None and tried == math.comb(len(atoms), k),
                           "found": shorter is not None, "seconds": round(time.perf_counter() - began, 1)}
    if shorter is not None:               # a shorter decomposition: say what it is, checked against every coefficient
        U, V, W = search._verified(build_multiplication_tensor(), shorter, degree)
        out.update({"products_after": len(shorter), "reconstruction_exact": U is not None,
                    "U": None if U is None else U.tolist(), "V": None if V is None else V.tolist(),
                    "W": None if W is None else W.tolist()})
    return out


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewrites = sorted((r for r in rows if "k" in r), key=lambda r: r["k"])
    found = [r for r in rewrites if r["found"]]
    exhausted = [r["k"] for r in rewrites if r["every_set_tried"]]
    reached = min((r["products_after"] for r in found if r.get("reconstruction_exact")), default=None)
    return {"target": TARGET, "products_at_the_start": 42,
            "products_dropped_by_deflation": next((r["products_dropped"] for r in rows if r["move"] == "deflation"), None),
            "rewrites_exhausted_for_k": exhausted, "sets_tried_in_all": sum(r["sets_tried"] for r in rewrites),
            "a_rewrite_was_found": bool(found), "fewest_products_reached": reached if reached is not None else 42,
            "target_met": bool(reached is not None and reached <= TARGET)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("run", "collect"))
    parser.add_argument("--k", type=int, nargs="*", default=[])
    parser.add_argument("--deflation", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        if args.deflation:
            row = deflation()
            (args.out / "move_deflation.json").write_text(json.dumps(row) + "\n")
            print(json.dumps(row))
        for k in args.k:
            row = rewrite(k)
            (args.out / f"move_k{k}.json").write_text(json.dumps(row) + "\n")
            print(json.dumps({key: value for key, value in row.items() if key not in ("U", "V", "W")}), flush=True)
        return 0
    rows = [json.loads(path.read_text()) for path in sorted(args.out.glob("move_*.json"))]
    record = {"tag": TAG, "development_not_a_result": True,
              "what": "the existing exact search from the verified 42 products, with no language model: deflation and "
                      "every k -> k-1 rewrite; a negative is a statement about this family from this starting point only",
              "search": "apedp_dyn/quantum/gf2_rank_search.py (deflate, reduce_once), unchanged",
              "start": "apedp_dyn/quantum/tower_reference.build_tower_42_reference()",
              "summary": summarise(rows), "moves": rows}
    (args.out / "rank_search_from_42.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
