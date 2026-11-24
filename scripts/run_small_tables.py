#!/usr/bin/env python3
"""Development: the small-table reading against the route in force, on generated tables of 8 to 30 rows.

**Development, not a result.** ``apedp_dyn/discovery_l2/small_table.py`` reads tables of tens of rows. Real tables
that small are few and most are held back (``benchmarks/real_data/registry.json``), so the reading is developed
on generated ones: laws drawn from the invented-law grammar at one or two inputs (structures of the benchmark
left out by fingerprint, as everywhere), ``ROWS`` rows each, noise of 0, 1 or 5 % of the target's size.

Two arms on the same rows:

* ``small``: ``small_table.read`` (nothing known beforehand: invented laws have no theory);
* ``direct``: the route in force called as the product would call it, 70 % of the rows to work on and 30 % to
  confirm on (the product itself refuses fewer than 24 rows).

Each law an arm gives (reported, or tentative) is scored on 400 clean rows: accurate at the line that defines a
wrong law, and whether it has the true law's form (``run_srbench_baseline.symbolic_recovery``). A *wrong report*
is a reported law that is not accurate.

    python scripts/run_small_tables.py dev --out DIR [--laws 120] [--searches 90] [--shard 0 --shards 4]
    python scripts/run_small_tables.py collect --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
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

from apedp_dyn.discovery_l2 import small_table, vouched_route_v2  # noqa: E402
from apedp_dyn.recursion import curriculum  # noqa: E402
from apedp_dyn.ui import route_endpoint  # noqa: E402

SEED = 2026112401
ROWS = (8, 12, 20, 30)
NOISES = (0.0, 0.01, 0.05)
LEVELS = (1, 2, 3)            # one input with two blocks; two inputs with two and three
TEST_ROWS = 400
ARMS = ("small", "direct")
GROSS_ERROR = 10.0            # a table with one gross error has one row off by this many of its noise's standard deviations


def draw_laws(count: int, seed: int = SEED, leave_out: frozenset[str] = frozenset(),
              levels: tuple[int, ...] = LEVELS) -> list[dict[str, Any]]:
    """Invented laws, none with a structure of the benchmark or of ``leave_out``, each with its table's size and noise."""
    excluded = curriculum.benchmark_fingerprints()
    rng = np.random.default_rng(seed)
    laws: list[dict[str, Any]] = []
    seen: set[str] = set(leave_out)
    for _ in range(200_000):
        if len(laws) == count:
            break
        level = levels[len(laws) % len(levels)]
        expr, inputs = curriculum.draw(rng, level)
        shape = curriculum.structure(expr)
        if shape in seen or curriculum.fingerprint(expr) in excluded or not curriculum.acceptable(expr, inputs, rng):
            continue
        seen.add(shape)
        index = len(laws)
        laws.append({"index": index, "text": str(expr), "inputs": inputs, "level": level,
                     "rows": ROWS[(index // len(levels)) % len(ROWS)],
                     "noise": NOISES[(index // (len(levels) * len(ROWS))) % len(NOISES)]})
    if len(laws) != count:
        raise SystemExit(f"only {len(laws)} laws")
    return laws


def table_of(law: dict[str, Any]) -> dict[str, Any]:
    names = [f"x{i}" for i in range(law["inputs"])]
    symbols = [sp.Symbol(n) for n in names]
    truth = sp.sympify(law["text"], locals=dict(zip(names, symbols, strict=True)))
    key = hashlib.sha256(f"{law.get('tag', 'small-tables-dev')}|{law['index']}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(key, "big"))
    X = rng.uniform(curriculum.LOW, curriculum.HIGH, (law["rows"] + TEST_ROWS, law["inputs"]))
    clean = np.asarray(sp.lambdify(symbols, truth, "numpy")(*X.T), dtype=float) * np.ones(len(X))
    n = law["rows"]
    size = float(np.sqrt(np.mean(clean[:n] ** 2)))
    draws = rng.standard_normal(n)
    if law.get("trap") == "relative_noise":                  # each row off in proportion to its own value
        y = clean[:n] * (1.0 + law["noise"] * draws)
    else:
        y = clean[:n] + law["noise"] * size * draws
    if law.get("trap") == "one_gross_error":
        row = int(rng.integers(n))
        y[row] += GROSS_ERROR * law["noise"] * size * float(rng.choice([-1.0, 1.0]))
    return {"names": names, "truth": truth, "X": X[:n], "y": y, "test_X": X[n:], "test_clean": clean[n:]}


def judged(law: str | None, table: dict[str, Any]) -> dict[str, Any]:
    if not law:
        return {"law": None, "accurate": False, "true_form": False}
    names = table["names"]
    symbols = [sp.Symbol(n) for n in names]
    try:
        model = sp.sympify(str(law), locals=dict(zip(names, symbols, strict=True)))
        with np.errstate(all="ignore"):
            pred = np.asarray(sp.lambdify(symbols, model, "numpy")(*table["test_X"].T), dtype=float) * np.ones(TEST_ROWS)
        score = base._r2(table["test_clean"], pred)
        form = bool(base.symbolic_recovery(model, table["truth"], names, table["test_X"])["recovered"])
    except Exception as exc:  # noqa: BLE001 - a law that cannot be scored is counted as neither accurate nor the true form
        return {"law": str(law), "accurate": False, "true_form": False, "error": f"{type(exc).__name__}: {exc}"[:120]}
    return {"law": str(law), "accurate": bool(score is not None and score >= base.R2_ACCURACY), "true_form": form,
            "r2_clean": None if score is None else round(float(score), 6)}


def direct(table: dict[str, Any], seed: int, searches: float) -> dict[str, Any]:
    """The route in force with the product's own split, on a table the product would refuse below 24 rows."""
    names, X, y = table["names"], table["X"], table["y"]
    order = np.random.default_rng(seed).permutation(len(y))
    cut = round((1.0 - route_endpoint.POOL_SHARE) * len(y))
    dev, pool = order[:cut], order[cut:]
    pool_y = y[pool]
    try:
        proposer, _stage = route_endpoint._proposer(searches)
        got = vouched_route_v2.discover_law({n: X[dev, i] for i, n in enumerate(names)}, y[dev], X[pool],
                                            lambda rows: pool_y[rows], ledger=None, search_id="small-tables-dev",
                                            proposer=proposer, noise="absolute", seed=seed)
        return {"law": got["law_as_confirmed"], "reported": bool(got["reported"]), "action": got["action"]}
    except Exception as exc:  # noqa: BLE001 - what the route does on a table this small is part of the comparison
        return {"law": None, "reported": False, "action": "not_run", "error": f"{type(exc).__name__}: {exc}"[:200]}


def one(job: tuple[dict[str, Any], float]) -> dict[str, Any]:
    law, searches = job
    table = table_of(law)
    row: dict[str, Any] = {k: law[k] for k in ("index", "inputs", "level", "rows", "noise", "trap") if k in law}
    started = time.perf_counter()
    proposer = route_endpoint._proposer(searches)[0] if searches > 0 else None
    reading = small_table.read({n: table["X"][:, i] for i, n in enumerate(table["names"])}, table["y"], proposer=proposer,
                               seed=law["index"])
    row["small"] = {"reported": bool(reading["reported"]), "action": reading["action"], **judged(reading.get("law"), table),
                    "stable": (reading.get("stability") or {}).get("stable"), "class_size": reading.get("class_size"),
                    "source": (reading.get("chosen") or {}).get("source"), "reasons": reading.get("tentative_reasons"),
                    "constants": (reading.get("chosen") or {}).get("constants"),
                    "errors": reading.get("errors"), "error": reading.get("error"),
                    "seconds": round(time.perf_counter() - started, 1)}
    started = time.perf_counter()
    got = direct(table, law["index"], searches)
    row["direct"] = {"reported": got["reported"], "action": got["action"], **judged(got.get("law"), table),
                     "error": got.get("error"), "seconds": round(time.perf_counter() - started, 1)}
    return row


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def cell(sel: list[dict[str, Any]], arm: str) -> dict[str, Any]:
        got = [r[arm] for r in sel]
        return {"tables": len(got), "law_given": sum(bool(g.get("law")) for g in got),
                "true_form": sum(bool(g.get("true_form")) for g in got), "accurate": sum(bool(g.get("accurate")) for g in got),
                "reported": sum(bool(g.get("reported")) for g in got),
                "reported_true_form": sum(bool(g.get("reported")) and bool(g.get("true_form")) for g in got),
                "wrong_reports": sum(bool(g.get("reported")) and not g.get("accurate") for g in got),
                "not_run": sum(g.get("action") == "not_run" for g in got)}

    out: dict[str, Any] = {arm: {"all": cell(rows, arm)} for arm in ARMS}
    for arm in ARMS:
        out[arm]["by_rows"] = {str(n): cell([r for r in rows if r["rows"] == n], arm) for n in ROWS}
        out[arm]["by_noise"] = {str(z): cell([r for r in rows if r["noise"] == z], arm) for z in NOISES}
    out["paired"] = {"true_form_gained_by_small": sum(bool(r["small"].get("true_form")) and not r["direct"].get("true_form") for r in rows),
                     "true_form_lost_by_small": sum(bool(r["direct"].get("true_form")) and not r["small"].get("true_form") for r in rows)}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("dev", "collect"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--laws", type=int, default=120)
    parser.add_argument("--searches", type=float, default=0.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "collect":
        rows = [json.loads(line) for path in sorted(args.out.glob("rows_*.jsonl")) for line in path.read_text().splitlines()]
        rows.sort(key=lambda r: r["index"])
        record = {"what": "development: the small-table reading against the route in force on generated tables; not a result",
                  "tables": len(rows), "summary": summarise(rows)}
        (args.out / "small_tables_dev.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(record["summary"], indent=1))
        return 0
    laws = draw_laws(args.laws)[args.shard::args.shards]
    workers = 1 if args.searches > 0 else max(args.workers, 1)
    path = args.out / f"rows_{args.shard}.jsonl"
    started = time.perf_counter()
    with path.open("w", encoding="utf-8") as handle:
        if workers == 1:
            for law in laws:
                handle.write(json.dumps(one((law, args.searches)), sort_keys=True) + "\n")
                handle.flush()
        else:
            with mp.get_context("fork").Pool(workers, maxtasksperchild=8) as pool:
                for row in pool.imap(one, [(law, args.searches) for law in laws], chunksize=1):
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    print(json.dumps({"tables": len(rows), "seconds": round(time.perf_counter() - started, 1), **summarise(rows)["paired"],
                      **{arm: summarise(rows)[arm]["all"] for arm in ARMS}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
