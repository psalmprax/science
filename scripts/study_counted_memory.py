#!/usr/bin/env python3
"""Development: does the memory of confirmed laws propose the same laws on a slower machine?

Status: audit-only

The memory's growth stops at a clock (5 seconds in the small-table reading), and its remembered
whole laws at another (15 seconds). This study measures, on development tables of the small-table
reading, how often those clocks are what stops the work, whether the proposals change when the
clocks are halved (a machine half as fast), and whether a variant that counts fits instead
(``apedp_dyn/proposers/structure_memory_counted.py``) proposes the same laws twice and finds the
true form as often. Nothing here is in force and nothing is decided by it.

    python scripts/study_counted_memory.py --out DIR [--tables 60] [--workers 2]
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_small_tables as tables  # noqa: E402

from apedp_dyn.discovery_l2 import memory_stage, small_table  # noqa: E402
from apedp_dyn.proposers import structure_memory as timed  # noqa: E402
from apedp_dyn.proposers import structure_memory_counted as counted  # noqa: E402

TARGET = "__target__"
WHOLE_SECONDS = 15.0            # ``remembered_whole``'s own clock, as the proposer calls it


def proposals_timed(memory: timed.Memory, frame: pd.DataFrame, share: float) -> tuple[list[str], dict[str, Any], float]:
    """The timed proposer's laws with both of its clocks at ``share`` of their length (1: as in the reading)."""
    whole = timed.remembered_whole
    # the proposer calls remembered_whole with its default clock; a slower machine is that clock made shorter
    timed.remembered_whole = functools.partial(whole, seconds=WHOLE_SECONDS * share)  # type: ignore[assignment]
    try:
        proposer = timed.StructureMemoryProposer(memory, max_proposals=3, seconds=small_table.MEMORY_SECONDS * share,
                                                 target_col=TARGET)
        started = time.monotonic()
        laws = [str(p.formula) for p in proposer.fit_and_propose(frame, target_col=TARGET)]
        return laws, dict(proposer.last_run), time.monotonic() - started
    finally:
        timed.remembered_whole = whole  # type: ignore[assignment]


def proposals_counted(memory: timed.Memory, frame: pd.DataFrame) -> tuple[list[str], dict[str, Any], float]:
    proposer = counted.CountedMemoryProposer(memory, max_proposals=3, target_col=TARGET)
    started = time.monotonic()
    laws = [str(p.formula) for p in proposer.fit_and_propose(frame, target_col=TARGET)]
    return laws, dict(proposer.last_run), time.monotonic() - started


def one(law: dict[str, Any]) -> dict[str, Any]:
    table = tables.table_of(law)
    memory = memory_stage.recorded_memory()
    frame = pd.DataFrame({**{n: table["X"][:, i] for i, n in enumerate(table["names"])}, TARGET: table["y"]})

    def true_form(laws: list[str]) -> bool:
        return any(tables.judged(text, table)["true_form"] for text in laws)

    full, full_run, full_seconds = proposals_timed(memory, frame, 1.0)
    half, _half_run, _ = proposals_timed(memory, frame, 0.5)
    first, first_run, first_seconds = proposals_counted(memory, frame)
    second, _second_run, _ = proposals_counted(memory, frame)
    return {"index": law["index"], "inputs": law["inputs"], "rows": law["rows"], "noise": law["noise"],
            "timed": {"laws": full, "stopped": full_run.get("stopped"), "fits": full_run.get("fits"),
                      "seconds": round(full_seconds, 1), "true_form": true_form(full)},
            "timed_at_half": {"laws": half, "same_as_timed": half == full, "true_form": true_form(half)},
            "counted": {"laws": first, "stopped": first_run.get("stopped"), "fits": first_run.get("fits"),
                        "whole": first_run.get("whole"), "seconds": round(first_seconds, 1), "true_form": true_form(first),
                        "same_twice": first == second, "same_as_timed": first == full}}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def count(test: Any) -> int:
        return sum(1 for row in rows if test(row))

    seconds = sorted(row["counted"]["seconds"] for row in rows)
    return {
        "tables": len(rows),
        "timed_stopped_by_its_clock": count(lambda r: r["timed"]["stopped"] == "out of time"),
        "timed_ran_into_both_of_its_clocks": count(
            lambda r: r["timed"]["seconds"] >= small_table.MEMORY_SECONDS + WHOLE_SECONDS),
        "proposals_differ_with_the_clocks_halved": count(lambda r: not r["timed_at_half"]["same_as_timed"]),
        "true_form_among_the_proposals": {"timed": count(lambda r: r["timed"]["true_form"]),
                                          "timed_with_the_clocks_halved": count(lambda r: r["timed_at_half"]["true_form"]),
                                          "counted": count(lambda r: r["counted"]["true_form"])},
        "true_form_gained_by_counted": count(lambda r: r["counted"]["true_form"] and not r["timed"]["true_form"]),
        "true_form_lost_by_counted": count(lambda r: r["timed"]["true_form"] and not r["counted"]["true_form"]),
        "counted_proposes_the_same_twice": count(lambda r: r["counted"]["same_twice"]),
        "counted_proposes_what_timed_proposes": count(lambda r: r["counted"]["same_as_timed"]),
        "counted_seconds": {"median": seconds[len(seconds) // 2], "largest": seconds[-1]},
        "counted_fits_allowed": {"growth": counted.GROW_FITS, "remembered_whole_laws": counted.WHOLE_FITS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tables", type=int, default=60)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    laws = tables.draw_laws(120)[:: max(120 // args.tables, 1)][: args.tables]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(one, laws):
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("index", "inputs", "rows")} | {
                "timed": row["timed"]["stopped"], "half_same": row["timed_at_half"]["same_as_timed"],
                "counted_twice": row["counted"]["same_twice"]}), flush=True)
            (args.out / "rows.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    record = {"what": "development: the memory's proposals with a clock and with a count of fits, on development tables "
                      "of the small-table reading (scripts/study_counted_memory.py)",
              "machine": "the development workspace, two workers side by side on two cores",
              "summary": summarise(rows), "rows": rows}
    (args.out / "counted_memory.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(record["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
