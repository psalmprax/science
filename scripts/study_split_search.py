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
- ``--inner pysr`` (the default) searches each part with PySR; ``--inner blocks`` with the block search
  of ``apedp_dyn/proposers/block_search.py`` first and PySR in the seconds left, both lists together
  (the idea of look-ahead taken from the owner's non-neural program-search studies). The two are run on
  the same tables with the same seconds, so they can be set side by side table by table (``compare``).
- ``--set third`` reads 30 laws of the development grammar drawn after the block search was fixed;
  ``--set other`` 30 laws of another generator (``draw_other``), with the same joins and other blocks of
  one input, three of them outside the block search's vocabulary (AGENTS.md 1.6: a test of the block
  search with traps it was not developed against). Neither set is drawn by any test.

    python scripts/study_split_search.py run --set first --out DIR [--seconds 90] [--shard 0 --shards 8] [--inner blocks]
    python scripts/study_split_search.py collect --set first --out DIR
    python scripts/study_split_search.py compare --a RECORD --b RECORD
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
from apedp_dyn.proposers import block_search, split_search  # noqa: E402
from apedp_dyn.recursion import curriculum  # noqa: E402

TAG = "DEV-SPLIT-SEARCH-2026-11"
DEV_ROWS = 700            #: rows the proposer is given: the share of a table the route gives a proposer
SECONDS = 90.0
TABLE_LIMIT_S = 1200.0    #: a table whose reading and searches take longer is counted as proposing nothing
KEPT_LAWS = 6             #: laws of each part kept in the record, shortest first; the laws chosen are kept whole beside them
PER_PART = 2 * split_search.PER_PART   #: laws a part may hand on: PySR gives at most 30, the block search 40 and PySR's beside them
ROUTE_RECORD = ROOT / "benchmarks" / "wider_conditions" / "dev_run4.json"
THIRD_SEED = 2026112603   #: drawn after the block search's vocabulary was fixed
OTHER_SEED = 2026112604
SETS = ("first", "second", "third", "other")
INNERS = ("pysr", "blocks")
Inner = Callable[[float], ProposerBase]


def _other_block(rng: np.random.Generator, x: sp.Symbol) -> sp.Expr:
    """Blocks of one input of another generator. Outside the block search's vocabulary: ``atan(a x)``,
    ``exp(-a/x)`` and ``1/sqrt(x + a)``; reachable only as two of its blocks: ``x exp(-a x)`` and
    ``x**1.5``; in it, written otherwise: ``cos(a x)``, ``log(x + a)``, ``x**2/(a + x**2)``, ``1/(1 + a x**2)``."""
    a = sp.Float(float(rng.choice(curriculum.CONSTANTS)))
    kind = int(rng.integers(0, 9))
    if kind == 0:
        return sp.atan(a * x)
    if kind == 1:
        return sp.exp(-a / x)
    if kind == 2:
        return 1 / sp.sqrt(x + a)
    if kind == 3:
        return x * sp.exp(-a * x)
    if kind == 4:
        return x ** sp.Float(1.5)
    if kind == 5:
        return sp.cos(a * x)
    if kind == 6:
        return sp.log(x + a)
    if kind == 7:
        return x**2 / (a + x**2)
    return 1 / (1 + a * x**2)


def draw_other(rng: np.random.Generator, inputs: int) -> sp.Expr:
    """One block per input and one more, joined left to right as ``run_wider_conditions.draw`` joins them."""
    symbols = [sp.Symbol(f"x{i}") for i in range(inputs)]
    order = [*symbols, symbols[int(rng.integers(0, inputs))]]
    rng.shuffle(order)
    expr = sp.Float(float(rng.choice(curriculum.CONSTANTS))) * _other_block(rng, order[0])
    for symbol in order[1:]:
        term, scale = _other_block(rng, symbol), sp.Float(float(rng.choice(curriculum.CONSTANTS)))
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


def draw_other_laws(seed: int = OTHER_SEED) -> list[dict[str, Any]]:
    """Ten laws of three, four and five inputs from the other generator, each at the four levels of noise."""
    rng = np.random.default_rng(seed)
    laws: list[dict[str, Any]] = []
    seen: set[str] = set()
    for inputs in wider.INPUTS:
        found = 0
        for _ in range(200_000):
            if found == wider.LAWS_PER_INPUTS:
                break
            expr = draw_other(rng, inputs)
            shape = curriculum.structure(expr)
            if shape in seen or not curriculum.acceptable(expr, inputs, rng):
                continue
            seen.add(shape)
            found += 1
            for noise in wider.NOISES:
                laws.append({"index": len(laws), "law": len(seen) - 1, "text": str(expr), "inputs": inputs, "noise": noise})
        if found != wider.LAWS_PER_INPUTS:
            raise SystemExit(f"only {found} laws of {inputs} inputs")
    return laws


def jobs(which: str) -> list[dict[str, Any]]:
    if which == "third":
        return wider.draw_laws(THIRD_SEED)
    if which == "other":
        return draw_other_laws()
    return shapes.jobs(which)


class _TooLong(Exception):
    pass


def _alarm(signum: int, frame: Any) -> None:
    raise _TooLong


def pysr_inner(seconds: float) -> ProposerBase:
    """PySR for one part, with the settings of the benchmark runs and this part's share of the seconds."""
    from apedp_dyn.proposers.pysr_proposer import PySRProposer

    return PySRProposer(budget_seconds=seconds, max_proposals=split_search.PER_PART)


def blocks_inner(seconds: float) -> ProposerBase:
    """The block search first, then PySR in the seconds left; the two lists together, best fit to the part first."""
    return block_search.BlocksFirst(seconds, pysr_inner)


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
    proposer = split_search.SplitSearchProposer(inner, seconds=seconds, per_part=PER_PART)
    row: dict[str, Any] = {**{k: job[k] for k in ("index", "law", "inputs", "noise")}, "text": job["text"], "error": None,
                           "proposals": []}
    started = time.perf_counter()
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        proposals = proposer.fit_and_propose(frame, target_col="target")
        row["proposals"] = [{"whole": (p.metadata or {}).get("whole"), **scored(str(p.formula), law, names, test_X, clean)}
                            for p in proposals]
    except _TooLong:
        row["error"] = f"longer than {limit:g} s"
    except Exception as exc:  # noqa: BLE001 - one table failing is a row of the record, not the end of the shard
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    run = proposer.last_run
    kept = [{**note, "laws_found": len(note.get("laws") or []), "laws": (note.get("laws") or [])[:KEPT_LAWS]}
            for note in run.get("part_laws") or []]
    row.update({"seconds": round(time.perf_counter() - started, 1), "read": run.get("read"), "shape": run.get("shape"),
                "parts": run.get("parts"), "widest": run.get("widest"), "part_laws": kept, "wholes": run.get("wholes"),
                "wholes_refitted": run.get("wholes_refitted"),
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


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Two searches of the parts on the same tables, table by table: where each proposed the law or an accurate one."""
    if a["set"] != b["set"]:
        raise ValueError("the two records read different sets")
    left = {r["index"]: r for r in a["rows"]}
    right = {r["index"]: r for r in b["rows"]}
    both = sorted(set(left) & set(right))
    out: dict[str, Any] = {"set": a["set"], "a": a.get("inner", "pysr"), "b": b.get("inner", "pysr"), "tables": len(both)}
    for key in ("a_proposal_the_law", "a_proposal_accurate"):
        only_a = [i for i in both if left[i][key] and not right[i][key]]
        only_b = [i for i in both if right[i][key] and not left[i][key]]
        out[key] = {"a": sum(left[i][key] for i in both), "b": sum(right[i][key] for i in both), "both": sum(left[i][key] and right[i][key] for i in both),
                    "only_a": len(only_a), "only_b": len(only_b), "sign_test_p": _sign_test(len(only_a), len(only_b))}
        out[key]["by_inputs"] = {str(n): {"a": sum(left[i][key] for i in both if left[i]["inputs"] == n),
                                          "b": sum(right[i][key] for i in both if right[i]["inputs"] == n)}
                                 for n in sorted({left[i]["inputs"] for i in both})}
    return out


def _sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test on the tables where only one of the two did it."""
    n = wins + losses
    if n == 0:
        return 1.0
    from math import comb
    tail = sum(comb(n, k) for k in range(0, min(wins, losses) + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("run", "collect", "compare"))
    parser.add_argument("--set", dest="which", choices=SETS)
    parser.add_argument("--inner", choices=INNERS, default="pysr")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--a", type=Path, help="compare: the record of one search of the parts")
    parser.add_argument("--b", type=Path, help="compare: the record of the other, on the same tables")
    parser.add_argument("--seconds", type=float, default=SECONDS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "compare":
        print(json.dumps(compare(json.loads(args.a.read_text()), json.loads(args.b.read_text())), indent=1))
        return 0
    if args.which is None or args.out is None:
        parser.error("run and collect need --set and --out")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.command == "run":
        todo = [j for j in jobs(args.which) if j["index"] % args.shards == args.shard]
        if args.limit:
            todo = todo[:args.limit]
        warm_up()
        inner = blocks_inner if args.inner == "blocks" else pysr_inner
        rows = []
        for job in todo:
            rows.append(one(job, args.which, inner, args.seconds))
            r = rows[-1]
            print(f"{r['index']:4d} law {r['law']:2d} inputs {r['inputs']} noise {r['noise']:.2f}  parts {r['parts']}  "
                  f"accurate {int(r['a_proposal_accurate'])} the law {int(r['a_proposal_the_law'])}  {r['seconds']} s  {r['error'] or ''}",
                  flush=True)
            (args.out / f"shard_{args.shard}.json").write_text(json.dumps({"set": args.which, "inner": args.inner,
                                                                           "seconds": args.seconds, "rows": rows}) + "\n")
        return 0

    parts = [json.loads(path.read_text()) for path in sorted(args.out.glob("shard_*.json"))]
    rows = sorted((r for part in parts for r in part["rows"]), key=lambda r: r["index"])
    route = None
    if args.which == "first" and ROUTE_RECORD.is_file():
        route = {r["index"]: r for r in json.loads(ROUTE_RECORD.read_text())["rows"]}
    inner = parts[0].get("inner", "pysr") if parts else args.inner
    record = {"tag": TAG, "development_not_a_result": True, "set": args.which, "inner": inner,
              "what": "a list: whether the law is among what the split-search proposer proposes; nothing is confirmed or reported",
              "settings": {"rows_given": DEV_ROWS, "seconds_for_all_parts": parts[0]["seconds"] if parts else None,
                           "laws_a_part": split_search.PER_PART, "proposals": split_search.MAX_PROPOSALS,
                           "rows_set_aside": split_search.HELD_SHARE, "tolerance": split_search.TOLERANCE,
                           "table_limit_s": TABLE_LIMIT_S, "accurate_r2": base.R2_ACCURACY},
              "tables_read": len(rows), "tables_of_the_set": len(jobs(args.which)),
              "summary": summarise(rows, route), "rows": rows}
    (args.out / "split_search.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["summary"]["by_inputs"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
