#!/usr/bin/env python3
"""Development: are the laws of wide tables found when a table is searched by its parts?

**Development, not a result.** The table route in force has recovered no law of four or five inputs
by form. ``apedp_dyn/proposers/split_search.py`` reads how a table's law is put together
(``table_shape.py``), searches each part on its own with PySR, and puts the parts' laws together.
This study runs that proposer alone, outside the route, and scores what it proposes the way the
benchmark runs score a law: *accurate* (R² of ``R2_ACCURACY`` on 500 rows without noise) and
*recovered* (the law itself once its constants are refitted, ``symbolic_recovery``).

What it is and is not:

- It measures a **list**: whether the law is among the proposals. Nothing here is confirmed, refused
  or reported by the path; what the route would report with this proposer in its stage is another
  measurement.
- The proposer is given the first ``DEV_ROWS`` rows of each table, as a proposer in the route is
  given the development rows and not the pool.
- ``--set first`` reads the tables of the wider-conditions runs, so each row can be set beside what
  the route in force reported for the same table (``benchmarks/wider_conditions/dev_run4.json``).
  The shape reading was worked out looking at three of these 30 laws. ``--set second`` reads 30 laws
  drawn after that reading was fixed.
- The parts' searches together get ``--seconds`` (90 by default: the budget of each of the route's
  four searches in the run it is set beside; the route's searches run side by side on four cores,
  these one after another on one).

    python scripts/study_split_search.py run --set first --out DIR [--seconds 90] [--shard 0 --shards 8]
    python scripts/study_split_search.py collect --set first --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_srbench_baseline as base  # noqa: E402
import run_wider_conditions as wider  # noqa: E402
import study_table_split as shapes  # noqa: E402

from apedp_dyn.grammar.proposer_interface import ProposerBase  # noqa: E402
from apedp_dyn.proposers import split_search  # noqa: E402
from apedp_dyn.recursion import curriculum  # noqa: E402

TAG = "DEV-SPLIT-SEARCH-2026-11"
DEV_ROWS = 700            #: rows the proposer is given: the share of a table the route gives a proposer
SECONDS = 90.0
TABLE_LIMIT_S = 1200.0    #: a table whose reading and searches take longer is counted as proposing nothing
ROUTE_RECORD = ROOT / "benchmarks" / "wider_conditions" / "dev_run4.json"
Inner = Callable[[float], ProposerBase]


class _TooLong(Exception):
    pass


def _alarm(signum: int, frame: Any) -> None:
    raise _TooLong


def pysr_inner(seconds: float) -> ProposerBase:
    """PySR for one part, with the settings of the benchmark runs and this part's share of the seconds."""
    from apedp_dyn.proposers.pysr_proposer import PySRProposer

    return PySRProposer(budget_seconds=seconds, max_proposals=split_search.PER_PART)


def warm_up() -> None:
    """One short search on a table of no interest, so that the first real part is not charged for PySR starting."""
    x = np.linspace(1.0, 5.0, 60)
    pysr_inner(3.0).fit_and_propose(pd.DataFrame({"x": x, "target": 2.0 * x + 1.0}), target_col="target")


def clean_rows(job: dict[str, Any], which: str) -> tuple[np.ndarray, np.ndarray, sp.Expr, list[str]]:
    """The 500 rows without noise a law of this table is scored on: those of the wider-conditions run for the first set."""
    names = [f"x{i}" for i in range(job["inputs"])]
    symbols = [sp.Symbol(n) for n in names]
    law = sp.sympify(job["text"], locals=dict(zip(names, symbols, strict=True)))
    name = "wider-conditions" if which == "first" else f"table-split-{which}"
    key = hashlib.sha256(f"{name}|{job['index']}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(key, "big"))
    X = rng.uniform(curriculum.LOW, curriculum.HIGH, (wider.ROWS + wider.TEST_ROWS, job["inputs"]))
    clean = np.asarray(sp.lambdify(symbols, law, "numpy")(*X.T), dtype=float) * np.ones(len(X))
    return X[wider.ROWS:], clean[wider.ROWS:], law, names


def scored(formula: str, law: sp.Expr, names: list[str], X: np.ndarray, clean: np.ndarray) -> dict[str, Any]:
    """A proposal against the law on rows without noise: accurate, and the law itself or not."""
    out: dict[str, Any] = {"formula": formula[:600], "r2": None, "accurate": False, "recovered": False}
    try:
        symbols = [sp.Symbol(n) for n in names]
        model = sp.sympify(formula, locals=dict(zip(names, symbols, strict=True)))
        with np.errstate(all="ignore"):
            pred = np.asarray(sp.lambdify(symbols, model, "numpy")(*X.T), dtype=float) * np.ones(len(X))
        r2 = base._r2(clean, pred)
        out["r2"] = None if r2 is None else float(r2)
        out["accurate"] = bool(r2 is not None and r2 >= base.R2_ACCURACY)
        out["recovered"] = bool(base.symbolic_recovery(model, law, names, X)["recovered"])
    except Exception as exc:  # noqa: BLE001 - a proposal that cannot be scored is counted as neither accurate nor the law
        out["scoring_error"] = f"{type(exc).__name__}: {exc}"[:120]
    return out


def one(job: dict[str, Any], which: str, inner: Inner, seconds: float = SECONDS, limit: float = TABLE_LIMIT_S) -> dict[str, Any]:
    X, y = shapes.table(job, which)
    test_X, clean, law, names = clean_rows(job, which)
    frame = pd.DataFrame(X[:DEV_ROWS], columns=names)
    frame["target"] = y[:DEV_ROWS]
    proposer = split_search.SplitSearchProposer(inner, seconds=seconds)
    row: dict[str, Any] = {**{k: job[k] for k in ("index", "law", "inputs", "noise")}, "text": job["text"], "error": None,
                           "proposals": []}
    started = time.perf_counter()
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        proposals = proposer.fit_and_propose(frame, target_col="target")
        row["proposals"] = [scored(str(p.formula), law, names, test_X, clean) for p in proposals]
    except _TooLong:
        row["error"] = f"longer than {limit:g} s"
    except Exception as exc:  # noqa: BLE001 - one table failing is a row of the record, not the end of the shard
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    run = proposer.last_run
    row.update({"seconds": round(time.perf_counter() - started, 1), "read": run.get("read"), "shape": run.get("shape"),
                "parts": run.get("parts"), "widest": run.get("widest"), "part_laws": run.get("part_laws"),
                "stand_ins": sum(bool(note.get("stand_in")) for note in run.get("part_laws") or []),
                "a_proposal_accurate": any(p["accurate"] for p in row["proposals"]),
                "a_proposal_the_law": any(p["recovered"] for p in row["proposals"]),
                "first_accurate": bool(row["proposals"] and row["proposals"][0]["accurate"]),
                "first_the_law": bool(row["proposals"] and row["proposals"][0]["recovered"])})
    return row


def summarise(rows: list[dict[str, Any]], route: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    def block(some: list[dict[str, Any]]) -> dict[str, Any]:
        out = {"tables": len(some), "split": sum((r.get("parts") or 1) > 1 for r in some),
               "proposed": sum(bool(r["proposals"]) for r in some),
               "a_proposal_accurate": sum(r["a_proposal_accurate"] for r in some),
               "a_proposal_the_law": sum(r["a_proposal_the_law"] for r in some),
               "first_accurate": sum(r["first_accurate"] for r in some), "first_the_law": sum(r["first_the_law"] for r in some),
               "errors": sum(bool(r["error"]) for r in some),
               "parts_stood_in_for": sum(r.get("stand_ins") or 0 for r in some),
               "median_seconds": float(np.median([r["seconds"] for r in some])) if some else None}
        if route is not None:
            beside = [route[r["index"]] for r in some if r["index"] in route]
            out["the_route_in_force"] = {"tables": len(beside), "reported": sum(b["reported"] for b in beside),
                                         "accurate": sum(b["accurate"] for b in beside),
                                         "the_law": sum(b["recovered"] for b in beside)}
        return out

    cells = {}
    for inputs in sorted({r["inputs"] for r in rows}):
        for noise in sorted({r["noise"] for r in rows}):
            cells[f"{inputs} inputs, noise {noise:g}"] = block([r for r in rows if r["inputs"] == inputs and r["noise"] == noise])
    by_inputs = {str(inputs): block([r for r in rows if r["inputs"] == inputs]) for inputs in sorted({r["inputs"] for r in rows})}
    return {"all": block(rows), "by_inputs": by_inputs, "cells": cells}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("run", "collect"))
    parser.add_argument("--set", dest="which", choices=("first", "second"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=SECONDS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.command == "run":
        todo = [j for j in shapes.jobs(args.which) if j["index"] % args.shards == args.shard]
        if args.limit:
            todo = todo[:args.limit]
        warm_up()
        rows = []
        for job in todo:
            rows.append(one(job, args.which, pysr_inner, args.seconds))
            r = rows[-1]
            print(f"{r['index']:4d} law {r['law']:2d} inputs {r['inputs']} noise {r['noise']:.2f}  parts {r['parts']}  "
                  f"accurate {int(r['a_proposal_accurate'])} the law {int(r['a_proposal_the_law'])}  {r['seconds']} s  {r['error'] or ''}",
                  flush=True)
            (args.out / f"shard_{args.shard}.json").write_text(json.dumps({"set": args.which, "seconds": args.seconds, "rows": rows}) + "\n")
        return 0

    parts = [json.loads(path.read_text()) for path in sorted(args.out.glob("shard_*.json"))]
    rows = sorted((r for part in parts for r in part["rows"]), key=lambda r: r["index"])
    route = None
    if args.which == "first" and ROUTE_RECORD.is_file():
        route = {r["index"]: r for r in json.loads(ROUTE_RECORD.read_text())["rows"]}
    record = {"tag": TAG, "development_not_a_result": True, "set": args.which,
              "what": "a list: whether the law is among what the split-search proposer proposes; nothing is confirmed or reported",
              "settings": {"rows_given": DEV_ROWS, "seconds_for_all_parts": parts[0]["seconds"] if parts else None,
                           "laws_a_part": split_search.PER_PART, "proposals": split_search.MAX_PROPOSALS,
                           "table_limit_s": TABLE_LIMIT_S, "accurate_r2": base.R2_ACCURACY},
              "tables_read": len(rows), "tables_of_the_set": len(shapes.jobs(args.which)),
              "summary": summarise(rows, route), "rows": rows}
    (args.out / "split_search.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["summary"]["by_inputs"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
