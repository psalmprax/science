#!/usr/bin/env python3
"""Do the recorded recoveries still hold on a range twice as wide as the benchmark sampled? (audit M-23)

Status: audit-only

The benchmark's criterion refits a reported law's constants to the true law on the test rows and
accepts a relative error below 1e-8. A form that only follows the law on the sampled range can
pass that narrowly: the cosine of a cosine plus a cubic passed for Planck's law. This script puts
every recovery of the recorded SRBench result sets through the same criterion again, on rows drawn
from a range twice as wide in every variable (not crossing zero where the sampled range does not).
A law that is the law passes there too. An approximation tuned to the sampled range does not.

Many of these laws have a pole (``1/(1 - v/c)``), and a wider range can reach it. Rows on which the
true law is more than ``CAP`` times its largest magnitude on the sampled range are left out, so
that the refit is not decided by a handful of rows next to a singularity.

Nothing is rerun and no recorded result is changed: the reported laws are read from the committed
rows, the true laws and ranges from ``benchmarks/srbench/manifest.json``.

    python scripts/check_recoveries_out_of_range.py            # write benchmarks/recoveries_out_of_range.json
    python scripts/check_recoveries_out_of_range.py --check    # fail if the committed file differs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_srbench_baseline as base  # noqa: E402

OUT = ROOT / "benchmarks" / "recoveries_out_of_range.json"
RESULT_SETS = ("RESULT-MAIN-SEARCH-2026-11", "RESULT-REFINED-SEARCH-2026-11", "RESULT-LLM-CONTROLLER-2026-11",
               "RESULT-PORTFOLIO-SEARCH-2026-11")
WIDEN = 0.5        # each side of a variable's range moves out by this share of the range's width
DRAWS = 2000       # rows drawn from the wide range
MIN_ROWS = 200     # fewer rows on which the true law is finite: the run is not testable this way
BY_FORM = 1e-10    # "recovered by form" in the preregistered runs that use it
CAP = 10.0         # wide rows keep the true law within this multiple of its largest size on the sampled range
TAG = "recoveries-out-of-range-2026-11"


def wide_range(low: float, high: float, widen: float = WIDEN) -> tuple[float, float]:
    """The range with each side moved out by ``widen`` widths, kept on its side of zero."""
    width = high - low
    new_low, new_high = low - widen * width, high + widen * width
    if low > 0:
        new_low = max(new_low, low / 2.0)
    if high < 0:
        new_high = min(new_high, high / 2.0)
    return new_low, new_high


def sampled_ranges(entry: dict[str, Any]) -> list[tuple[float, float]]:
    """Each feature's sampled range: the published one, or the data's own where none is published."""
    published = entry.get("ranges") or {}
    if all(feature in published for feature in entry["features"]):
        return [(float(published[f][0]), float(published[f][1])) for f in entry["features"]]
    frame = base.srb.load_dataset(entry)       # the Strogatz problems: rows from trajectories, no stated range
    return [(float(frame[f].min()), float(frame[f].max())) for f in entry["features"]]


def wide_rows(entry: dict[str, Any], truth: sp.Expr, draws: int = DRAWS) -> np.ndarray:
    """Rows from the wide range on which the true law is finite, in the order of the problem's features."""
    names = [f"x{i}" for i in range(len(entry["features"]))]
    seed = int.from_bytes(hashlib.sha256(f"{TAG}|{entry['name']}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    sampled = sampled_ranges(entry)
    function = sp.lambdify([sp.Symbol(n) for n in names], truth, "numpy", dummify=True)

    def law(X: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            value = np.asarray(function(*X.T), dtype=complex) * np.ones(len(X))
        real: np.ndarray = np.where(np.isfinite(value.real) & (np.abs(value.imag) == 0), value.real, np.nan)
        return real

    inside = law(np.column_stack([rng.uniform(low, high, draws) for low, high in sampled]))
    largest = float(np.nanmax(np.abs(inside))) if np.any(np.isfinite(inside)) else np.inf
    X: np.ndarray = np.column_stack([rng.uniform(*wide_range(low, high), draws) for low, high in sampled])
    value = law(X)
    with np.errstate(invalid="ignore"):
        keep = np.isfinite(value) & (np.abs(value) <= CAP * largest)
    rows: np.ndarray = X[keep]
    return rows


def judge(model_text: str, entry: dict[str, Any]) -> dict[str, Any]:
    """The benchmark's criterion for one reported law on the wide rows."""
    truth = base.truth_expr(entry)
    names = [f"x{i}" for i in range(len(entry["features"]))]
    rows = wide_rows(entry, truth)
    if len(rows) < MIN_ROWS:
        return {"testable": False, "rows": len(rows)}
    model = base.parse_formula(model_text)
    outcome = base.symbolic_recovery(model, truth, names, rows)
    return {"testable": True, "rows": len(rows), "holds": bool(outcome["recovered"]),
            "rel_rmse": outcome["rel_rmse"], "reason": outcome["reason"]}


def recovered_rows(result_set: str) -> list[dict[str, Any]]:
    folder = ROOT / "preregistration" / result_set
    rows = [json.loads(line) for path in sorted(folder.glob("runs-shard-*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("status") == "ok" and r.get("symbolic") is True and r.get("model") not in (None, "None")]


def build() -> dict[str, Any]:
    entries = {e["name"]: e for e in base.load_manifest()}
    out: dict[str, Any] = {"widen": WIDEN, "draws": DRAWS, "min_rows": MIN_ROWS, "cap": CAP, "tolerance": base.SYMBOLIC_TOL,
                           "result_sets": {}}
    for result_set in RESULT_SETS:
        cell: dict[str, Any] = {"recovered": 0, "by_form": 0, "testable": 0, "hold": 0, "hold_by_form": 0,
                                "do_not_hold": [], "not_testable": []}
        for row in recovered_rows(result_set):
            entry = entries[row["problem"]]
            verdict = judge(str(row["model"]), entry)
            inside = row.get("symbolic_rel_rmse")
            by_form = inside is not None and float(inside) <= BY_FORM
            cell["recovered"] += 1
            cell["by_form"] += int(by_form)
            label = {"problem": row["problem"], "noise": row["noise"], "engine": row.get("engine")}
            if not verdict["testable"]:
                cell["not_testable"].append({**label, "rows": verdict["rows"]})
                continue
            cell["testable"] += 1
            if verdict["holds"]:
                cell["hold"] += 1
                cell["hold_by_form"] += int(by_form)
            else:
                cell["do_not_hold"].append({**label, "by_form": bool(by_form), "rel_rmse_sampled_range": inside,
                                            "rel_rmse_wide_range": verdict["rel_rmse"], "reason": verdict["reason"],
                                            "model": str(row["model"])[:300]})
        out["result_sets"][result_set] = cell
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="compare with the committed file instead of writing it")
    args = parser.parse_args(argv)
    made = build()
    text = json.dumps(made, indent=1, sort_keys=True) + "\n"
    for name, cell in made["result_sets"].items():
        print(f"{name}: recovered {cell['recovered']} (by form {cell['by_form']}), testable {cell['testable']}, "
              f"hold on the wide range {cell['hold']}, do not {len(cell['do_not_hold'])}, "
              f"not testable {len(cell['not_testable'])}")
    if args.check:
        same = OUT.exists() and json.loads(OUT.read_text(encoding="utf-8")) == json.loads(text)
        print("matches the committed file" if same else "differs from the committed file")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
