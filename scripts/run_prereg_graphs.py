"""PREREG-GRAPHS-2026-11: lagged causal graphs recovered from simulated series with a known graph.

    python scripts/run_prereg_graphs.py --out preregistration/RESULT-GRAPHS-2026-11

Runs once. Refuses to run if a frozen file changed, or if the output directory exists.
Protocol and expectations: ``preregistration/PREREG-GRAPHS-2026-11.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import causal_graph_worlds as worlds  # noqa: E402

from apedp_dyn.proposers import causal_graph as cg  # noqa: E402

PREREGISTRATION_ID = "PREREG-GRAPHS-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-GRAPHS.json"
FROZEN = ("preregistration/PREREG-GRAPHS-2026-11.md", "scripts/run_prereg_graphs.py", "scripts/causal_graph_worlds.py",
          "apedp_dyn/proposers/causal_graph.py", "apedp_dyn/proposers/pcmci_proposer.py")
SEEDS = (3000, 3001, 3002, 3003, 3004)
SIZES = ((5, 300), (5, 1000), (10, 300), (10, 1000))
METHODS = {"pcmci": cg.discover_graph, "pcmci_even": cg.discover_graph_with_even_links,
           "var": cg.var_links, "correlation": cg.correlation_links}


def one(kind: str, n_vars: int, length: int, seed: int) -> dict[str, Any]:
    world = worlds.make(kind, n_vars, length, seed)
    even = {link for link, fn in world["function_of"].items() if fn == "even_square"}
    row: dict[str, Any] = {"kind": kind, "n_vars": n_vars, "length": length, "seed": seed,
                           "true_links": len(world["truth"]), "even_links": len(even)}
    for name, method in METHODS.items():
        started = time.perf_counter()
        found = cg.as_set(method(world["frame"]))
        row[name] = {**cg.score(found, world["truth"]), "even_found": len(found & even), "seconds": time.perf_counter() - started}
    return row


def pooled(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    hit = sum(r[method]["hit"] for r in rows)
    found = sum(r[method]["found"] for r in rows)
    true = sum(r[method]["true"] for r in rows)
    precision = hit / found if found else 1.0
    recall = hit / true if true else 1.0
    return {"datasets": len(rows), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "even_links": sum(r["even_links"] for r in rows), "even_found": sum(r[method]["even_found"] for r in rows)}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = {"all": rows,
                                              "linear_and_heavy_tail": [r for r in rows if r["kind"] != "nonlinear"]}
    for kind in worlds.KINDS:
        groups[kind] = [r for r in rows if r["kind"] == kind]
        for n_vars, length in SIZES:
            groups[f"{kind}|N={n_vars}|T={length}"] = [r for r in rows if (r["kind"], r["n_vars"], r["length"]) == (kind, n_vars, length)]
    for label, sel in groups.items():
        out[label] = {method: pooled(sel, method) for method in METHODS}
    return out


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} changed since it was frozen")
    args.out.mkdir(parents=True, exist_ok=False)
    warnings.simplefilter("ignore")
    rows = [one(kind, n_vars, length, seed) for kind in worlds.KINDS for n_vars, length in SIZES for seed in SEEDS]
    results = {"preregistration": PREREGISTRATION_ID, "frozen": frozen_digests(), "summary": summarise(rows)}
    (args.out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "runs.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in results["summary"].items() if "|" not in k}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
