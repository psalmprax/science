#!/usr/bin/env python3
"""Development: the table route in force beyond the conditions it was measured in: more inputs, more noise.

**Development, not a result.** The route's reports were measured with at most three inputs and noise up to 5 %
of the target's size (``apedp_dyn/discovery_l2/measured_range.py``). This runs the product's own reading
(``ui.route_endpoint.route_reading``, with the search stage in force) on invented laws of three, four and five
inputs at noise of 1, 5, 10 and 20 %, 1,000 rows each, and counts per cell what it reports, how many reports
are wrong (not accurate on 500 clean rows), how many laws are recovered by form, and what the measured-range
mark says. Three inputs at 1 and 5 % are inside the measured range and are the control.

Laws come from the invented-law grammar, one block per input and one more; structures of the benchmark are
left out by fingerprint, as everywhere, and no held-back law is read.

    python scripts/run_wider_conditions.py run --out DIR --searches 90 [--shard 0 --shards 4]
    python scripts/run_wider_conditions.py collect --out DIR
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_srbench_baseline as base  # noqa: E402

from apedp_dyn.recursion import curriculum  # noqa: E402
from apedp_dyn.ui import route_endpoint  # noqa: E402

SEED = 2026112402
INPUTS = (3, 4, 5)
NOISES = (0.01, 0.05, 0.10, 0.20)
LAWS_PER_INPUTS = 10
ROWS, TEST_ROWS = 1000, 500
STALL_REPORT_S = 1800.0


def draw(rng: np.random.Generator, inputs: int) -> sp.Expr:
    """One block per input and one more, joined left to right: the invented-law grammar at any number of inputs."""
    symbols = [sp.Symbol(f"x{i}") for i in range(inputs)]
    order = [*symbols, symbols[int(rng.integers(0, inputs))]]
    rng.shuffle(order)
    expr = sp.Float(float(rng.choice(curriculum.CONSTANTS))) * curriculum._block(rng, order[0])
    for symbol in order[1:]:
        term, scale = curriculum._block(rng, symbol), sp.Float(float(rng.choice(curriculum.CONSTANTS)))
        kind = int(rng.integers(0, 4))
        if kind == 0:
            expr = expr + scale * term
        elif kind == 1:
            expr = expr - scale * term
        elif kind == 2:
            expr = expr * term
        else:
            expr = expr / (scale + term)
    return expr


def draw_laws(seed: int = SEED) -> list[dict[str, Any]]:
    excluded = curriculum.benchmark_fingerprints()
    rng = np.random.default_rng(seed)
    laws: list[dict[str, Any]] = []
    seen: set[str] = set()
    for inputs in INPUTS:
        found = 0
        for _ in range(200_000):
            if found == LAWS_PER_INPUTS:
                break
            expr = draw(rng, inputs)
            shape = curriculum.structure(expr)
            if shape in seen or curriculum.fingerprint(expr) in excluded or not curriculum.acceptable(expr, inputs, rng):
                continue
            seen.add(shape)
            found += 1
            for noise in NOISES:
                laws.append({"index": len(laws), "law": len(seen) - 1, "text": str(expr), "inputs": inputs, "noise": noise})
        if found != LAWS_PER_INPUTS:
            raise SystemExit(f"only {found} laws of {inputs} inputs")
    return laws


def one(job: dict[str, Any], searches: float) -> dict[str, Any]:
    names = [f"x{i}" for i in range(job["inputs"])]
    symbols = [sp.Symbol(n) for n in names]
    truth = sp.sympify(job["text"], locals=dict(zip(names, symbols, strict=True)))
    key = hashlib.sha256(f"wider-conditions|{job['index']}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(key, "big"))
    X = rng.uniform(curriculum.LOW, curriculum.HIGH, (ROWS + TEST_ROWS, job["inputs"]))
    clean = np.asarray(sp.lambdify(symbols, truth, "numpy")(*X.T), dtype=float) * np.ones(len(X))
    y = clean[:ROWS] + job["noise"] * float(np.sqrt(np.mean(clean[:ROWS] ** 2))) * rng.standard_normal(ROWS)
    started = time.perf_counter()
    # a reading that takes this long has stopped somewhere: say where, in the job's log, every STALL_REPORT_S seconds
    faulthandler.dump_traceback_later(STALL_REPORT_S, repeat=True, file=sys.stderr)
    try:
        got = route_endpoint.route_reading({n: X[:ROWS, i] for i, n in enumerate(names)}, y, noise="absolute",
                                           search_seconds=searches, seed=job["index"])
    finally:
        faulthandler.cancel_dump_traceback_later()
    row: dict[str, Any] = {k: job[k] for k in ("index", "law", "inputs", "noise")}
    row.update({"action": got.get("action"), "reported": bool(got.get("reported")), "tentative": bool(got.get("tentative")),
                "range": (got.get("measured_range") or {}).get("state"), "vouched": bool((got.get("vouching") or {}).get("applied")),
                "search_stage": got.get("search_stage"), "error": got.get("error"),
                "seconds": round(time.perf_counter() - started, 1), "accurate": False, "recovered": False})
    law = got.get("law_as_confirmed")
    if law:
        try:
            model = sp.sympify(str(law), locals=dict(zip(names, symbols, strict=True)))
            with np.errstate(all="ignore"):
                pred = np.asarray(sp.lambdify(symbols, model, "numpy")(*X[ROWS:].T), dtype=float) * np.ones(TEST_ROWS)
            score = base._r2(clean[ROWS:], pred)
            row["accurate"] = bool(score is not None and score >= base.R2_ACCURACY)
            row["recovered"] = bool(base.symbolic_recovery(model, truth, names, X[ROWS:])["recovered"])
        except Exception as exc:  # noqa: BLE001 - a law that cannot be scored is counted as neither accurate nor recovered
            row["scoring_error"] = f"{type(exc).__name__}: {exc}"[:120]
    return row


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = {}
    for inputs in INPUTS:
        for noise in NOISES:
            sel = [r for r in rows if r["inputs"] == inputs and r["noise"] == noise]
            reported = [r for r in sel if r["reported"]]
            cells[f"{inputs} inputs, noise {noise:g}"] = {
                "runs": len(sel), "reported": len(reported), "wrong_reports": sum(not r["accurate"] for r in reported),
                "reported_recovered": sum(r["recovered"] for r in reported),
                "recovered_reported_or_tentative": sum(r["recovered"] for r in sel),
                "accurate_reported_or_tentative": sum(r["accurate"] for r in sel),
                "range": {state: sum(r["range"] == state for r in sel) for state in ("inside", "edge", "outside", "unknown")},
                "not_run": sum(r["action"] == "not_run" for r in sel),
                "inside_the_measured_range": inputs <= 3 and noise <= 0.05}
    return cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "collect", "laws"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--searches", type=float, default=90.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--inputs", default="", help="only tables with these numbers of inputs, e.g. 3,4 (default: all)")
    parser.add_argument("--noises", default="", help="only tables at these noise levels, e.g. 0.01,0.05 (default: all)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "collect":
        rows = [json.loads(line) for path in sorted(args.out.glob("rows_*.jsonl")) for line in path.read_text().splitlines()]
        rows.sort(key=lambda r: r["index"])
        record = {"what": "development: the table route in force beyond its measured conditions; not a result",
                  "runs": len(rows), "cells": summarise(rows), "rows": rows}
        (args.out / "wider_conditions.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(record["cells"], indent=1))
        return 0
    jobs = draw_laws()
    only_inputs = {int(value) for value in args.inputs.split(",") if value.strip()}
    only_noises = {float(value) for value in args.noises.split(",") if value.strip()}
    jobs = [job for job in jobs if (not only_inputs or job["inputs"] in only_inputs) and (not only_noises or job["noise"] in only_noises)]
    if args.command == "laws":
        print(json.dumps([{k: j[k] for k in ("index", "inputs", "noise")} for j in jobs]))
        return 0
    if args.searches > 0:
        try:
            import pysr  # noqa: F401 - the reading falls back to the first pass alone without it, which is not what is measured here
        except Exception as exc:  # noqa: BLE001 - whatever stops PySR loading ends the run with its reason
            raise SystemExit(f"PySR cannot be imported ({type(exc).__name__}: {exc}). Nothing was run.") from exc
    with (args.out / f"rows_{args.shard}.jsonl").open("w", encoding="utf-8") as handle:
        for job in jobs[args.shard::args.shards]:
            handle.write(json.dumps(one(job, args.searches), sort_keys=True) + "\n")
            handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
