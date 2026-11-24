#!/usr/bin/env python3
"""The forecast filed with PREREG-COUNTERFEIT-2026-11 against what the run showed.

    python scripts/score_counterfeit_forecast.py            # writes benchmarks/counterfeit_forecast_scored.json
    python scripts/score_counterfeit_forecast.py --check    # the file is what the records give

Three comparisons, none of them a criterion of the run:

* every method's totals (right, wrong, score) with the forecast's 80 % intervals;
* the chance the forecast gave each of the five expectations with whether it held, as a Brier
  score: 0 is a forecast that was sure and right, 0.25 is what "even odds" scores whatever
  happens, 1 is sure and wrong;
* cell by cell (noise level, law offered or withheld), the share of cases that ended in each
  outcome with the share the forecast's source gave: the campaign's randomly drawn worlds of
  that kind, and for the checked route what the replay of the checks made of the route's reports
  there. This is where a forecast's source and the run it stood in for can be seen to differ.

Both inputs are records: the forecast was frozen with the preregistration and the result is
committed as collected. Nothing here is fitted to the result.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FORECAST = ROOT / "preregistration" / "FORECAST-COUNTERFEIT-2026-11.json"
RESULT = ROOT / "preregistration" / "RESULT-COUNTERFEIT-2026-11"
OUT = ROOT / "benchmarks" / "counterfeit_forecast_scored.json"
OUTCOMES = ("the law", "accurate, not the law", "wrong", "nothing")
METHODS = ("route", "vouched", "best_fit", "validated")
EVEN_ODDS = 0.25


def outcome(row: dict[str, Any], method: str) -> str:
    """How a method ended on one case, by the run's scoring. A case that crashed or ran over reported nothing."""
    if row.get("status") != "ok" or not row[method]["reported"]:
        return "nothing"
    if row[method]["symbolic"]:
        return "the law"
    return "wrong" if row[method]["wrong"] else "accurate, not the law"


def _shares(counts: list[float]) -> list[float]:
    total = float(sum(counts))
    return [round(count / total, 4) if total else 0.0 for count in counts]


def expected_shares(cell: dict[str, Any], method: str) -> list[float]:
    """The share of cases the forecast's counts put on each outcome, for one cell and method."""
    if method != "vouched":
        return _shares(cell[method])
    route = _shares(cell["route"])
    out = [0.0] * len(OUTCOMES)
    out[-1] += route[-1]                                    # nothing reported: nothing to check
    for before, name in enumerate(OUTCOMES[:-1]):
        after = cell["after"][name]
        passed = _shares(after) if sum(after) else [float(i == before) for i in range(len(OUTCOMES))]
        for i, share in enumerate(passed):
            out[i] += route[before] * share
    return [round(value, 4) for value in out]


def score() -> dict[str, Any]:
    filed = json.loads(FORECAST.read_text(encoding="utf-8"))
    summary = json.loads((RESULT / "results.json").read_text(encoding="utf-8"))["summary"]
    with gzip.open(RESULT / "runs.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    totals: dict[str, Any] = {}
    inside = 0
    for method in METHODS:
        totals[method] = {}
        for name in ("right", "wrong", "score"):
            said, seen = filed["totals"][method][name], summary[f"{method}|all"][name]
            within = bool(said["interval"][0] <= seen <= said["interval"][1])
            inside += within
            totals[method][name] = {"expected": said["expected"], "interval": said["interval"], "level": said["level"],
                                    "seen": round(float(seen), 4), "inside": within}
    expectations: dict[str, Any] = {}
    for key in "12345":
        chance = float(filed["chance_that_an_expectation_holds"][key])
        held = bool(summary["expectations"][f"{key}_holds"])
        expectations[key] = {"chance": chance, "held": held, "brier": round((chance - float(held)) ** 2, 4)}
    cells: dict[str, Any] = {}
    for key, counted in filed["counts"].items():
        noise, kind = key.split("|")
        mine = [row for row in rows if f"{float(row['world']['noise']):g}" == noise
                and bool(row["world"]["truth_listed"]) == (kind == "offered")]
        cells[key] = {"cases": len(mine), "outcomes": list(OUTCOMES)}
        for method in METHODS:
            seen_counts = [sum(outcome(row, method) == name for row in mine) for name in OUTCOMES]
            cells[key][method] = {"forecast_share": expected_shares(counted, method), "seen": seen_counts,
                                  "seen_share": _shares([float(count) for count in seen_counts])}
    briers = [entry["brier"] for entry in expectations.values()]
    return {"what": "The forecast filed before PREREG-COUNTERFEIT-2026-11 against the run. Totals against their 80 % "
                    "intervals, the chance given to each expectation against whether it held, and each cell's outcomes "
                    "against the shares of the forecast's source.",
            "forecast": str(FORECAST.relative_to(ROOT)), "result": str(RESULT.relative_to(ROOT)),
            "totals": totals, "totals_inside_their_intervals": [inside, len(METHODS) * 3],
            "expectations": expectations, "brier": round(sum(briers) / len(briers), 4), "brier_of_even_odds": EVEN_ODDS,
            "cells": cells}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="compare the file with what the records give")
    args = parser.parse_args(argv)
    text = json.dumps(score(), indent=1, sort_keys=True) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT.relative_to(ROOT)} is not what the records give")
            return 1
        print(f"{OUT.relative_to(ROOT)}: as scored")
        return 0
    OUT.write_text(text, encoding="utf-8")
    scored = json.loads(text)
    print(f"totals inside their intervals: {scored['totals_inside_their_intervals']}; Brier {scored['brier']} "
          f"(even odds {EVEN_ODDS})")
    for key, entry in scored["expectations"].items():
        print(f"  expectation {key}: chance {entry['chance']:.3f}, held {entry['held']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
