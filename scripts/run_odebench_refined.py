#!/usr/bin/env python3
"""ODEBench through the refined path, for PREREG-ODE-REFINED-2026-11.

The same systems, trajectories, noise realisations, rows, search budget and scoring as
PREREG-ODE-MAIN-SEARCH-2026-11, whose recorded runs are the comparison (80 and 69 of 117 components
recovered; 9 of 184 reported laws wrong). One thing differs: each component goes through
``apedp_dyn.ode.refined_ode`` (polynomial library as a candidate, confirmed-and-pruned report, a law
nothing challenged not reported) instead of ``apedp_dyn.ode.main_search_ode``.

    python scripts/run_odebench_refined.py smoke   --out DIR                       # toy systems
    python scripts/run_odebench_refined.py develop --out DIR [--budget S] [--shard k/N]   # development systems only
    python scripts/run_odebench_refined.py run     --out DIR --shard k/N
    python scripts/run_odebench_refined.py collect --out DIR

``--budget 0`` runs the first passes alone (no PySR), which is what can be run where Julia is not
installed; it is not the measured path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import odebench_worlds as worlds  # noqa: E402
import run_odebench_main_search as ode_run  # noqa: E402

PREREGISTRATION_ID = "PREREG-ODE-REFINED-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-ODE-REFINED.json"
FROZEN = ("preregistration/PREREG-ODE-REFINED-2026-11.md", "scripts/run_odebench_refined.py",
          "scripts/run_odebench_main_search.py", "scripts/odebench_worlds.py", "apedp_dyn/ode/window_rows.py",
          "apedp_dyn/ode/main_search_ode.py", "apedp_dyn/ode/refined_ode.py",
          "apedp_dyn/discovery_l2/refined_search.py", "apedp_dyn/proposers/composed.py",
          "apedp_dyn/discovery_l2/main_search.py", "apedp_dyn/proposers/pysr_proposer.py",
          "apedp_dyn/discovery_l2/contrast_confirmation.py", "apedp_dyn/discovery_l2/contrast_resolution.py",
          "apedp_dyn/discovery_l2/confirmed_stop.py", "apedp_dyn/discovery_l2/guided_search.py",
          "apedp_dyn/discovery_l2/exhaustive_search.py", "apedp_mmlm/law_corpora.py")

ENGINE = "refined_ode"
NOISE_LEVELS = ode_run.NOISE_LEVELS
DEVELOPMENT_TAG = "dev-ode-refined|"
GROUPS = ("all", "held_out", "development")


def used_in_development(system_id: int) -> bool:
    """About a quarter of the systems, fixed by their id: the ones this module was built on."""
    return int(hashlib.sha256(f"{DEVELOPMENT_TAG}{system_id}".encode()).hexdigest(), 16) % 4 == 0


def run_refined(world: dict[str, Any], component: int, budget: float = ode_run.SEARCH_BUDGET_S) -> dict[str, Any]:
    from apedp_dyn.ode import refined_ode
    from apedp_dyn.ode.main_search_ode import state_names

    proposer = None
    if budget > 0:
        from apedp_dyn.proposers.pysr_proposer import PySRProposer

        proposer = PySRProposer(budget_seconds=budget, seed=ode_run.SEED)
    dev, pool = world["dev"], world["pool"]
    names = state_names(dev.states.shape[1])
    result = refined_ode.discover_component(dev, pool, component, proposer=proposer, seed=ode_run.SEED)
    symbols = {n: sp.Symbol(n) for n in names}
    path = result.path
    return {"model": sp.sympify(result.law, locals=symbols) if result.law else None, "formula": result.law,
            "extra": {"action": result.action, "tentative_law": result.tentative_law,
                      "dropped_terms": [str(t) for t in result.dropped_terms],
                      "proposer_failures": list(result.proposer_failures), "search_stage": proposer is not None,
                      "found_by": path.found_by if path else None, "path_law_text": path.law if path else None,
                      "rejected_law": path.rejected_law if path else None, "adequacy": path.adequacy if path else None,
                      "rows_read": path.rows_read if path else None, "search": path.search if path else None,
                      "candidates": path.candidates if path else None,
                      "equivalent": len(path.contrast.equivalent) if path and path.contrast else 0}}


ode_run.RUNNERS[ENGINE] = run_refined


def run_one(job: dict[str, Any], system: dict[str, Any], world: dict[str, Any], budget: float = ode_run.SEARCH_BUDGET_S,
            time_limit: float = ode_run.TIME_LIMIT_S) -> dict[str, Any]:
    """The frozen runner's ``run_world``, plus a score for a tentative law (reported apart, never counted)."""
    row: dict[str, Any] = ode_run.run_world(job, system, world, budget, time_limit)
    row["used_in_development"] = used_in_development(system["id"])
    tentative = row.get("tentative_law")
    if row["status"] == "ok" and tentative:
        component = job["component"]
        names = [f"x{i}" for i in range(system["dim"])]
        scored = ode_run.score({**world, "test_rates_component": world["test_rates"][:, component]},
                               system["components"][component],
                               {"model": sp.sympify(tentative, locals={n: sp.Symbol(n) for n in names}), "formula": tentative})
        row.update({"tentative_symbolic": scored["symbolic"], "tentative_accurate": scored["accuracy_solution"],
                    "tentative_test_r2": scored["test_r2"]})
    return row


def all_jobs() -> list[dict[str, Any]]:
    return [{"system": s["id"], "component": k, "noise": noise, "engine": ENGINE}
            for s in worlds.systems() for k in range(s["dim"]) for noise in NOISE_LEVELS]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in GROUPS:
        for noise in (*NOISE_LEVELS, "all"):
            sel = [r for r in rows if noise in (r["noise"], "all") and (
                group == "all" or (group == "development") == used_in_development(r["system"]))]
            ok = [r for r in sel if r["status"] == "ok"]
            reported = [r for r in ok if r.get("model") not in (None, "None")]
            tentative = [r for r in ok if r.get("tentative_law")]
            by_system: dict[tuple[int, float], list[dict[str, Any]]] = {}
            for r in sel:
                by_system.setdefault((r["system"], r["noise"]), []).append(r)
            out[f"{group}|{noise}"] = {
                "n": len(sel), "excluded": sum(r["status"] == "excluded" for r in sel),
                "errors": sum(r["status"] == "error" for r in sel), "timeouts": sum(r["status"] == "timeout" for r in sel),
                "median_seconds": float(np.median([r["seconds"] for r in sel])) if sel else None,
                "symbolic": sum(bool(r["symbolic"]) for r in ok), "accuracy_solution": sum(bool(r["accuracy_solution"]) for r in ok),
                "reported": len(reported), "reported_symbolic": sum(bool(r["symbolic"]) for r in reported),
                "reported_not_accurate": sum(not r["accuracy_solution"] for r in reported),
                "reported_wrong": sum(not r["accuracy_solution"] and not r["symbolic"] for r in reported),
                "tentative": len(tentative), "tentative_symbolic": sum(bool(r.get("tentative_symbolic")) for r in tentative),
                "tentative_not_accurate": sum(not r.get("tentative_accurate") for r in tentative),
                "terms_dropped": sum(bool(r.get("dropped_terms")) for r in ok),
                "proposer_failures": sum(bool(r.get("proposer_failures")) for r in ok),
                "systems": len(by_system),
                "systems_symbolic": sum(all(r["status"] == "ok" and r["symbolic"] for r in g) for g in by_system.values()),
                "actions": {a: sum(r.get("action") == a for r in ok)
                            for a in ("confirmed", "equivalent_class", "unchallenged", "widen_search")},
                "found_by": {s: sum(r.get("found_by") == s for r in reported) for s in ("first_pass", "search")},
            }
    return out


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def check_frozen() -> None:
    if not DIGEST.exists() or json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} is not frozen, or changed since it was frozen")


def _printable(row: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in row.items() if k not in ("candidates", "trace", "first_pass")}, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("run", "collect", "smoke", "develop"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1", help="k/N: this shard's runs")
    parser.add_argument("--budget", type=float, default=None, help="smoke and develop only: PySR seconds; 0 = first passes alone")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    k, n = (int(v) for v in args.shard.split("/"))
    if args.command in ("smoke", "develop"):
        if args.command == "smoke":
            systems = ode_run.toy_systems()
            budget = float(os.environ.get("SMOKE_BUDGET_SECONDS", "30")) if args.budget is None else args.budget
            tag = "smoke"
        else:  # the development systems only; never a held-out one
            systems = [s for s in worlds.systems() if used_in_development(s["id"])]
            budget = ode_run.SEARCH_BUDGET_S if args.budget is None else args.budget
            tag = "PREREG-ODE-MAIN-SEARCH-2026-11"  # the recorded run's rows, for a run-for-run comparison
        failed, index = 0, 0
        for system in systems:
            for noise in NOISE_LEVELS:
                made: dict[str, Any] | None = None
                for component in range(system["dim"]):
                    index += 1
                    if (index - 1) % n != k:
                        continue
                    made = made if made is not None else worlds.world(system, noise, tag=tag)
                    row = run_one({"system": system["id"], "component": component, "noise": noise, "engine": ENGINE},
                                  system, made, budget)
                    print("ROW " + _printable(row), flush=True)
                    if row["status"] not in ("ok", "excluded"):
                        print(row.get("trace", ""), flush=True)
                        failed += 1
        return 1 if failed and args.command == "smoke" else 0
    check_frozen()
    if args.command == "run":
        systems_by_id = {s["id"]: s for s in worlds.systems()}
        made_worlds: dict[tuple[int, float], dict[str, Any]] = {}
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for job in ode_run.shard(all_jobs(), k, n):
                key = (job["system"], job["noise"])
                if key not in made_worlds:
                    made_worlds[key] = worlds.world(systems_by_id[job["system"]], job["noise"])
                handle.write(json.dumps(run_one(job, systems_by_id[job["system"]], made_worlds[key]), sort_keys=True, default=str) + "\n")
                handle.flush()
        return 0
    rows = [json.loads(line) for path in sorted(args.out.glob("runs-shard-*.jsonl")) for line in path.read_text().splitlines()]
    expected = len(all_jobs())
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} runs, found {len(rows)}")
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (args.out / "results.json").write_text(json.dumps(
        {"preregistration": PREREGISTRATION_ID, "frozen": frozen_digests(), "summary": summarise(rows)},
        indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"collected {len(rows)} runs into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
