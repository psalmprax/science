#!/usr/bin/env python3
"""The system's forecast of its own runs: learn from the runs on record, backtest, and file a forecast.

    python scripts/forecast_runs.py backtest                 # each recorded run forecast from the others
    python scripts/forecast_runs.py file --problems FILE --route ROUTE --group GROUP --out FILE
    python scripts/forecast_runs.py score --out FILE         # every forecast filed before a run, against that run

``backtest`` holds one recorded run out at a time, learns from the other runs of the same group
of routes, and compares the forecast with what happened and with the forecast that only knows
the other runs' averages. **A backtest is not a test:** every outcome was known when the
forecaster was written. The test is a forecast filed before a run (``file``) and scored after it.

``file`` reads problems (``[{"name", "law", "inputs", "noise"}, ...]``) and writes, for one route,
each problem's chances and the totals with their intervals. The route may be new; it is then
forecast as ``--like`` an existing one, and the file says so.

``score`` compares each filed forecast (``FILED``) with the rows of the run it was filed for: the
totals against their intervals and against what the earlier runs' averages alone would have said,
and, run by run, the Brier score against those averages and against the rate seen afterwards, and
how well the chances rank the runs (0.5 is chance).

Groups of routes that are learned together (a route is an indicator inside its group):

- ``tables``: the path-based routes on SRBench (main search, refined, four searches, controlled);
- ``trajectories``: the path-based routes on ODEBench and on the blind chaotic systems;
- ``sparse``: sparse regression on a cubic library, the reference of the trajectory runs.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_srbench_baseline as base  # noqa: E402

from apedp_dyn.discovery_l2 import outcome_forecast as of  # noqa: E402

RESULTS = ROOT / "preregistration"
#: group -> {result directory: {engine in its rows: route name here}}
GROUPS: dict[str, dict[str, dict[str, str]]] = {
    "tables": {"RESULT-MAIN-SEARCH-2026-11": {"main_search": "main_search"},
               "RESULT-REFINED-SEARCH-2026-11": {"refined": "refined"},
               "RESULT-PORTFOLIO-SEARCH-2026-11": {"refined_portfolio": "refined_portfolio"},
               "RESULT-LLM-CONTROLLER-2026-11": {"refined_controlled": "refined_controlled"}},
    "trajectories": {"RESULT-ODE-MAIN-SEARCH-2026-11": {"main_search": "ode_main_search"},
                     "RESULT-ODE-REFINED-2026-11": {"refined_ode": "ode_refined"},
                     "RESULT-ODE-BLIND-2026-11": {"main_search": "ode_main_search", "repaired": "ode_repaired"}},
    "sparse": {"RESULT-ODE-MAIN-SEARCH-2026-11": {"sparse_poly3": "sparse_poly3"},
               "RESULT-ODE-BLIND-2026-11": {"sparse_poly3": "sparse_poly3"}},
}


def _rows(result: str) -> list[dict[str, Any]]:
    plain, packed = RESULTS / result / "runs.jsonl", RESULTS / result / "runs.jsonl.gz"
    text = gzip.decompress(packed.read_bytes()).decode() if packed.exists() else plain.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def recorded(group: str) -> list[of.Run]:
    """Every run on record of one group of routes, with its law and what came of it."""
    manifest = {entry["name"]: entry for entry in base.load_manifest()}
    runs = []
    for result, engines in GROUPS[group].items():
        for row in _rows(result):
            key = row.get("engine") or row.get("arm")
            if key not in engines or row.get("status", "ok") != "ok":
                continue
            if "truth" in row:
                law, inputs = str(row["truth"]), int(row["dim"])
                name = f"{row.get('system')}:{row.get('component')}"
            else:
                entry = manifest[row["problem"]]
                law, inputs, name = str(base.truth_expr(entry)), len(entry["features"]), row["problem"]
            reported = row.get("model") not in (None, "None")
            runs.append(of.Run(law, inputs, float(row["noise"]), engines[key], bool(row.get("symbolic")), reported,
                               bool(reported and not row.get("symbolic") and not row.get("accuracy_solution")),
                               source=result, name=f"{name}@{row['noise']:g}"))
    return runs


def backtest(group: str) -> list[dict[str, Any]]:
    """Each (result, route) of the group forecast from every other result of the group."""
    runs = recorded(group)
    out = []
    for source in dict.fromkeys(r.source for r in runs):
        rest = [r for r in runs if r.source != source]
        if not rest:
            continue
        model = of.OutcomeForecaster().fit(rest)
        for route in dict.fromkeys(r.route for r in runs if r.source == source):
            held = [r for r in runs if r.source == source and r.route == route]
            forecast = model.forecast(held)
            plain = of.baseline(rest, len(held), route)
            scored, flat = of.score(forecast, held), of.score(plain, held)
            out.append({"group": group, "held_out": source, "route": route, "runs": len(held),
                        "learned_from": len(rest), "route_seen_before": route in {r.route for r in rest},
                        "forecast": scored, "averages_only": flat})
    return out


def between_runs_spread(rows: list[dict[str, Any]], outcome: str = "recovered") -> float:
    """How far whole runs have differed from their forecasts: the root mean square of (seen - forecast) / runs."""
    errors = [(row["forecast"][outcome]["seen"] - row["forecast"][outcome]["expected"]) / row["runs"] for row in rows]
    return float((sum(e * e for e in errors) / len(errors)) ** 0.5) if errors else 0.0


BACKTEST = ROOT / "benchmarks" / "outcome_forecast_backtest.json"
#: forecasts filed before a run: (forecast file, result directory, the arm it was filed for)
FILED: tuple[tuple[str, str, str], ...] = (
    ("FORECAST-ODE-ADDITIONS-2026-11.json", "RESULT-ODE-ADDITIONS-2026-11", "default"),
    ("FORECAST-MEMORY-2026-11.json", "RESULT-MEMORY-2026-11", "default"),
)


def _name(row: dict[str, Any]) -> str:
    """A run's name as the forecast files write it: a law at a noise level, or a system's component at one."""
    if "problem" in row:
        return f"{row['problem']}@{float(row['noise']):g}"
    return f"{row['name']}:{row['component']}@{float(row['noise']):g}"


def _happened(row: dict[str, Any]) -> dict[str, bool]:
    reported = bool(row.get("formula"))
    return {"recovered": bool(row.get("symbolic")), "reported": reported,
            "wrong": reported and not row.get("symbolic") and not row.get("accuracy_solution")}


def score(forecast_file: str, result: str, arm: str) -> dict[str, Any]:
    """One filed forecast against the run it was filed for."""
    filed = json.loads((RESULTS / forecast_file).read_text(encoding="utf-8"))
    with gzip.open(RESULTS / result / "runs.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = {_name(row): row for row in map(json.loads, handle) if row.get("arm") == arm}
    if set(rows) != set(filed["names"]):
        raise ValueError(f"{forecast_file} and {result} do not name the same runs")
    outcomes: dict[str, Any] = {}
    for outcome in of.OUTCOMES:
        seen = [float(_happened(rows[name])[outcome]) for name in filed["names"]]
        chances = [float(c) for c in filed["chances"][outcome]]
        average = float(filed["averages_only"][outcome])
        rate = sum(seen) / len(seen)
        hits = [c for c, y in zip(chances, seen, strict=True) if y]
        misses = [c for c, y in zip(chances, seen, strict=True) if not y]
        ranks = (sum((a > b) + 0.5 * (a == b) for a in hits for b in misses) / (len(hits) * len(misses))
                 if hits and misses else None)
        total = filed["totals"][outcome]
        outcomes[outcome] = {
            "seen": int(sum(seen)), "expected": total["expected"], "interval": total["interval"], "level": total["level"],
            "inside": bool(total["interval"][0] <= sum(seen) <= total["interval"][1]),
            "averages_only_expected": round(average * len(seen), 1),
            "brier": round(sum((c - y) ** 2 for c, y in zip(chances, seen, strict=True)) / len(seen), 4),
            "brier_averages_only": round(sum((average - y) ** 2 for y in seen) / len(seen), 4),
            "brier_of_the_rate_seen_afterwards": round(rate * (1.0 - rate), 4),
            "ranks_the_runs": None if ranks is None else round(ranks, 3)}
    return {"forecast": forecast_file, "result": result, "arm": arm, "runs": len(rows), "forecast_as": filed["forecast_as"],
            "outcomes": outcomes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("backtest", "file", "score"))
    parser.add_argument("--group", choices=sorted(GROUPS), default=None)
    parser.add_argument("--problems", type=Path)
    parser.add_argument("--route", default="")
    parser.add_argument("--like", default="", help="an existing route the new one is forecast as")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "backtest":
        rows = [row for group in ([args.group] if args.group else sorted(GROUPS)) for row in backtest(group)]
        for row in rows:
            f, a = row["forecast"]["recovered"], row["averages_only"]["recovered"]
            print(f"{row['group']:12s} {row['held_out'][7:]:28s} {row['route']:18s} n={row['runs']:4d}  recovered: seen {f['seen']:4d}, "
                  f"forecast {f['expected']:6.1f} {f['interval']}, averages {a['expected']:6.1f}; Brier {f['brier']:.3f} "
                  f"against {a['brier']:.3f}")
        spreads = {o: round(between_runs_spread(rows, o), 4) for o in of.OUTCOMES}
        beats = sum(row["forecast"]["recovered"]["brier"] < row["averages_only"]["recovered"]["brier"] for row in rows)
        inside = sum(row["forecast"]["recovered"]["inside"] for row in rows)
        print(f"per problem, the forecast beats the averages in {beats} of {len(rows)} held-out runs; the total falls "
              f"inside the unwidened interval in {inside}; between-run spread {spreads}")
        if args.out:
            record = {"what": "Each recorded run forecast from the other runs of its group. A backtest, not a test: "
                              "every outcome was known when the forecaster was written.",
                      "between_runs_spread": spreads, "forecast_beats_averages": [beats, len(rows)],
                      "total_inside_unwidened_interval": [inside, len(rows)], "held_out_runs": rows}
            args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    if args.command == "score":
        scored = [score(*entry) for entry in FILED]
        for entry in scored:
            for outcome, s in entry["outcomes"].items():
                print(f"{entry['result'][7:]:26s} {outcome:9s} seen {s['seen']:4d}, forecast {s['expected']:6.1f} {s['interval']} "
                      f"({'inside' if s['inside'] else 'OUTSIDE'}), averages {s['averages_only_expected']:6.1f}; Brier "
                      f"{s['brier']:.3f} against {s['brier_averages_only']:.3f}; ranks {s['ranks_the_runs']}")
        totals = [s for entry in scored for s in entry["outcomes"].values()]
        if args.out:
            record = {"what": "Forecasts filed before a run, against that run. Each was frozen with its preregistration.",
                      "totals_inside_their_intervals": [sum(s["inside"] for s in totals), len(totals)], "forecasts": scored}
            args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    if not (args.problems and args.out and args.group and args.route):
        parser.error("file needs --problems, --route, --group and --out")
    problems = json.loads(args.problems.read_text(encoding="utf-8"))
    runs = recorded(args.group)
    model = of.OutcomeForecaster().fit(runs)
    known = sorted({r.route for r in runs})
    as_route = args.like or args.route
    if as_route not in known:
        parser.error(f"--like has to name one of {known}")
    forecast = model.forecast([of.Run(p["law"], int(p["inputs"]), float(p["noise"]), as_route, name=p["name"])
                               for p in problems], route=as_route)
    spread = float(json.loads(BACKTEST.read_text(encoding="utf-8"))["between_runs_spread"]["recovered"])
    plain = of.baseline([r for r in runs if r.route == as_route], len(problems), as_route)
    filed: dict[str, Any] = {"route": args.route, "forecast_as": as_route, "group": args.group, "learned_from_runs": len(runs),
              "learned_from_results": sorted({r.source for r in runs}),
              "note": "Filed before the run. Chances come from the runs on record; the features are read from the "
                      "true laws, which the benchmark knows and the route does not. Intervals are widened by the "
                      "spread between whole runs and their forecasts in the committed backtest.",
              "averages_only": {o: round(plain.chances[o][0], 4) for o in of.OUTCOMES},
              **{k: v for k, v in forecast.as_dict(between_runs=spread).items() if k != "route"}}
    args.out.write_text(json.dumps(filed, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(problems)} problems, route {args.route} forecast as {as_route}")
    for outcome, total in filed["totals"].items():
        print(f"  {outcome}: {total['expected']} expected, {total['interval']} at {int(100 * total['level'])} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
