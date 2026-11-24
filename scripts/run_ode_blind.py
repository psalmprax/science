#!/usr/bin/env python3
"""The repaired refined ODE route on chaotic systems it was not built on, for PREREG-ODE-BLIND-2026-11.

PREREG-ODE-REFINED-2026-11 measured the refined path with four searches on ODEBench: it recovered
more than the recorded ODE route and reported one wrong law too many, so it was not adopted. Two
repairs followed, both written with that run's outcomes known: a guard that drops a proposal the
path cannot fit (audit M-24) and term pruning to the rows' own precision (audit M-15). Neither has
been measured on systems they were not written on. This is that measurement.

    python scripts/run_ode_blind.py smoke   --out DIR
    python scripts/run_ode_blind.py run     --out DIR --shard k/N
    python scripts/run_ode_blind.py collect --out DIR

The systems are ``benchmarks/dysts_polynomial_systems.json`` (``scripts/build_dysts_polynomial_systems.py``).
Rows, scoring, the child process and the two engines of comparison are the frozen ODEBench
runner's (``scripts/run_odebench_main_search.py``, ``scripts/odebench_worlds.py``).

Three engines per component and noise level:

- ``repaired``: the refined path, four PySR searches side by side behind the fit guard, and the
  reported law pruned to the rows' precision. The law before pruning is scored too.
- ``main_search``: the recorded ODE route, as run on ODEBench.
- ``sparse_poly3``: sparse regression on a cubic library, the reference of that run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import odebench_worlds as worlds  # noqa: E402
import run_odebench_main_search as ode_run  # noqa: E402

PREREGISTRATION_ID = "PREREG-ODE-BLIND-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-ODE-BLIND.json"
SYSTEMS_FILE = ROOT / "benchmarks" / "dysts_polynomial_systems.json"
FROZEN = ("preregistration/PREREG-ODE-BLIND-2026-11.md", "scripts/run_ode_blind.py",
          "benchmarks/dysts_polynomial_systems.json", "scripts/build_dysts_polynomial_systems.py",
          "scripts/run_odebench_main_search.py", "scripts/odebench_worlds.py",
          "apedp_dyn/ode/refined_ode.py", "apedp_dyn/ode/main_search_ode.py", "apedp_dyn/ode/window_rows.py",
          "apedp_dyn/discovery_l2/refined_search.py", "apedp_dyn/discovery_l2/main_search.py",
          "apedp_dyn/discovery_l2/precision_pruning.py", "apedp_dyn/proposers/fit_guard.py",
          "apedp_dyn/proposers/portfolio.py", "apedp_dyn/proposers/pysr_proposer.py")

REPAIRED = "repaired"
ENGINES = (REPAIRED, "main_search", "sparse_poly3")
NOISE_LEVELS = ode_run.NOISE_LEVELS
MEMBERS = 4                 # four searches side by side, seeds SEED .. SEED + 3, as in PREREG-ODE-REFINED
PORTFOLIO_EXTRA_S = 120.0   # added to the time limit of a repaired run: four Julia runtimes have to start
WORLD_TAG = PREREGISTRATION_ID
WRONG_LIMIT = 0.05          # expectation 1: the share of reported laws that are wrong


def systems() -> list[dict[str, Any]]:
    """The fresh systems in the frozen world builder's form: id, dim, components, inits.

    Systems whose rows the world builder cannot produce in time (screened when the file was built,
    with no discovery run) are left out.
    """
    record = json.loads(SYSTEMS_FILE.read_text(encoding="utf-8"))
    skip = set(record.get("not_buildable", {}))
    return [{**s, "components": [sp.sympify(text) for text in s["texts"]]}
            for s in record["systems"] if s["name"] not in skip]


def search_stage(budget: float) -> Any:
    """Four PySR searches side by side, behind the fit guard; nothing without a budget."""
    if budget <= 0:
        return None
    from apedp_dyn.proposers.fit_guard import FitGuardProposer
    from apedp_dyn.proposers.portfolio import PortfolioProposer, pysr_members

    return FitGuardProposer(PortfolioProposer(
        pysr_members(budget, tuple(ode_run.SEED + k for k in range(MEMBERS))),
        max_proposals=10, parallel=True, timeout=budget + 300.0))


#: Replaced in tests by a stage that needs no search engine.
STAGE: Callable[[float], Any] = search_stage


def run_repaired(world: dict[str, Any], component: int, budget: float = ode_run.SEARCH_BUDGET_S) -> dict[str, Any]:
    from apedp_dyn.discovery_l2.precision_pruning import prune_to_precision
    from apedp_dyn.ode import refined_ode
    from apedp_dyn.ode.main_search_ode import state_names

    proposer = STAGE(budget)
    dev, pool = world["dev"], world["pool"]
    names = state_names(dev.states.shape[1])
    result = refined_ode.discover_component(dev, pool, component, proposer=proposer, seed=ode_run.SEED)
    law, pruned, dropped, prune_error = result.law, result.law, [], None
    if law:
        try:
            expression, terms = prune_to_precision(law, names, dev.states, dev.rates[:, component],
                                                   float(dev.rate_sd[component]), noise="absolute")
            pruned, dropped = str(expression), [str(t) for t in terms]
        except Exception as exc:  # noqa: BLE001 - a law the pruning cannot handle is reported unpruned, with the reason
            prune_error = f"{type(exc).__name__}: {exc}"[:200]
    symbols = {n: sp.Symbol(n) for n in names}
    path = result.path
    return {"model": sp.sympify(pruned, locals=symbols) if pruned else None, "formula": pruned,
            "extra": {"action": result.action, "tentative_law": result.tentative_law, "unpruned_law": law,
                      "precision_dropped": dropped, "prune_error": prune_error,
                      "dropped_terms": [str(t) for t in result.dropped_terms],
                      "proposer_failures": list(result.proposer_failures), "search_stage": proposer is not None,
                      "fit_guard_dropped": [list(d) for d in getattr(proposer, "dropped", [])],
                      "portfolio": dict(getattr(proposer, "last_run", {}) or {}),
                      "found_by": path.found_by if path else None, "rejected_law": path.rejected_law if path else None,
                      "adequacy": path.adequacy if path else None, "rows_read": path.rows_read if path else None,
                      "candidates": path.candidates if path else None,
                      "equivalent": len(path.contrast.equivalent) if path and path.contrast else 0}}


ode_run.RUNNERS[REPAIRED] = run_repaired


def run_one(job: dict[str, Any], system: dict[str, Any], world: dict[str, Any], budget: float = ode_run.SEARCH_BUDGET_S,
            time_limit: float | None = None) -> dict[str, Any]:
    """The frozen runner's ``run_world``; for a repaired run, the law before pruning is scored as well."""
    repaired = job["engine"] == REPAIRED
    limit = time_limit if time_limit is not None else ode_run.TIME_LIMIT_S + (PORTFOLIO_EXTRA_S if repaired else 0.0)
    row: dict[str, Any] = ode_run.run_world(job, system, world, budget, limit)
    row["name"] = system.get("name")
    unpruned = row.get("unpruned_law")
    if repaired and row["status"] == "ok" and unpruned:
        component = job["component"]
        names = [f"x{i}" for i in range(system["dim"])]
        scored = ode_run.score({**world, "test_rates_component": world["test_rates"][:, component]},
                               system["components"][component],
                               {"model": sp.sympify(unpruned, locals={n: sp.Symbol(n) for n in names}), "formula": unpruned})
        row.update({"unpruned_symbolic": scored["symbolic"], "unpruned_accurate": scored["accuracy_solution"],
                    "unpruned_test_r2": scored["test_r2"]})
    return row


def all_jobs() -> list[dict[str, Any]]:
    return [{"system": s["id"], "component": k, "noise": noise, "engine": engine}
            for s in systems() for k in range(s["dim"]) for noise in NOISE_LEVELS for engine in ENGINES]


def _wrong(row: dict[str, Any], prefix: str = "") -> bool:
    """A reported law that is neither accurate on the test rows nor the law: the frozen runs' definition."""
    accurate = row["accuracy_solution"] if not prefix else row.get("unpruned_accurate")
    symbolic = row["symbolic"] if not prefix else row.get("unpruned_symbolic")
    return not accurate and not symbolic


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per engine and noise level; then the paired numbers the expectations are judged on."""
    out: dict[str, Any] = {}
    for engine in ENGINES:
        for noise in (*NOISE_LEVELS, "all"):
            sel = [r for r in rows if r["engine"] == engine and noise in (r["noise"], "all")]
            ok = [r for r in sel if r["status"] == "ok"]
            reported = [r for r in ok if r.get("model") not in (None, "None")]
            by_system: dict[tuple[int, float], list[dict[str, Any]]] = {}
            for r in sel:
                by_system.setdefault((r["system"], r["noise"]), []).append(r)
            cell: dict[str, Any] = {
                "n": len(sel), "excluded": sum(r["status"] == "excluded" for r in sel),
                "errors": sum(r["status"] == "error" for r in sel), "timeouts": sum(r["status"] == "timeout" for r in sel),
                "median_seconds": float(np.median([r["seconds"] for r in sel])) if sel else None,
                "symbolic": sum(bool(r["symbolic"]) for r in ok),
                "reported": len(reported), "reported_symbolic": sum(bool(r["symbolic"]) for r in reported),
                "reported_not_accurate": sum(not r["accuracy_solution"] for r in reported),
                "reported_wrong": sum(_wrong(r) for r in reported),
                "systems": len(by_system),
                "systems_symbolic": sum(all(r["status"] == "ok" and r["symbolic"] for r in g) for g in by_system.values()),
            }
            if engine == REPAIRED:
                before = [r for r in ok if r.get("unpruned_law")]
                cell.update({
                    "pruning_changed": sum(bool(r.get("precision_dropped")) for r in ok),
                    "pruning_errors": sum(bool(r.get("prune_error")) for r in ok),
                    "unpruned_symbolic": sum(bool(r.get("unpruned_symbolic")) for r in before),
                    "unpruned_wrong": sum(_wrong(r, "unpruned") for r in before),
                    "recoveries_lost_by_pruning": sum(bool(r.get("unpruned_symbolic")) and not r["symbolic"] for r in before),
                    "recoveries_gained_by_pruning": sum(not r.get("unpruned_symbolic") and bool(r["symbolic"]) for r in before),
                    "runs_with_a_proposal_dropped_by_the_guard": sum(bool(r.get("fit_guard_dropped")) for r in ok),
                    "proposals_dropped_by_the_guard": sum(len(r.get("fit_guard_dropped") or []) for r in ok),
                    "member_failures": sum(len((r.get("portfolio") or {}).get("failures", [])) for r in ok),
                    "tentative": sum(bool(r.get("tentative_law")) for r in ok),
                    "actions": {a: sum(r.get("action") == a for r in ok)
                                for a in ("confirmed", "equivalent_class", "unchallenged", "widen_search")},
                })
            out[f"{engine}|{noise}"] = cell
    repaired, recorded = out[f"{REPAIRED}|all"], out["main_search|all"]
    share = repaired["reported_wrong"] / repaired["reported"] if repaired["reported"] else None
    out["expectations"] = {
        "1_share_of_reported_laws_wrong": share, "1_limit": WRONG_LIMIT,
        "1_holds": bool(share is not None and share <= WRONG_LIMIT),
        "2_recovered_without_noise": [out[f"{REPAIRED}|0.0"]["symbolic"], out["main_search|0.0"]["symbolic"]],
        "2_holds": bool(out[f"{REPAIRED}|0.0"]["symbolic"] >= out["main_search|0.0"]["symbolic"]),
        "3_recovered_with_noise": [out[f"{REPAIRED}|0.01"]["symbolic"], out["main_search|0.01"]["symbolic"]],
        "3_holds": bool(out[f"{REPAIRED}|0.01"]["symbolic"] >= out["main_search|0.01"]["symbolic"]),
        "4_recoveries_lost_by_pruning": repaired["recoveries_lost_by_pruning"],
        "4_wrong_with_and_without_pruning": [repaired["reported_wrong"], repaired["unpruned_wrong"]],
        "4_holds": bool(repaired["recoveries_lost_by_pruning"] == 0
                        and repaired["reported_wrong"] <= repaired["unpruned_wrong"]),
        "recorded_route_share_wrong": (recorded["reported_wrong"] / recorded["reported"]) if recorded["reported"] else None,
    }
    return out


def frozen_digests() -> dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FROZEN}


def check_frozen() -> None:
    if not DIGEST.exists():
        raise SystemExit(f"{DIGEST.name} is missing: the preregistration is not frozen")
    recorded = json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"]
    changed = sorted(f for f, digest in frozen_digests().items() if recorded.get(f) != digest)
    if changed:
        raise SystemExit(f"frozen files differ from {DIGEST.name}: {changed}")


def _printable(row: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in row.items() if k not in ("candidates", "trace", "first_pass", "portfolio")},
                      default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("run", "collect", "smoke"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1", help="k/N: this shard's runs")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "smoke":
        # the frozen runner's toy systems: none is a system of the blind set
        budget = float(os.environ.get("SMOKE_BUDGET_SECONDS", "30"))
        engines = [e for e in ENGINES if e in os.environ.get("SMOKE_ENGINES", ",".join(ENGINES)).split(",")]
        failed = 0
        for system in ode_run.toy_systems():
            for noise in NOISE_LEVELS:
                world = worlds.world(system, noise, tag="smoke")
                for component in range(system["dim"]):
                    for engine in engines:
                        row = run_one({"system": system["id"], "component": component, "noise": noise, "engine": engine},
                                      system, world, budget)
                        print(_printable(row), flush=True)
                        if row["status"] != "ok":
                            print(row.get("trace", ""), flush=True)
                            failed += 1
        return 1 if failed else 0
    check_frozen()
    if args.command == "run":
        k, n = (int(v) for v in args.shard.split("/"))
        by_id = {s["id"]: s for s in systems()}
        made: dict[tuple[int, float], dict[str, Any]] = {}
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for job in ode_run.shard(all_jobs(), k, n):
                key = (job["system"], job["noise"])
                if key not in made:
                    made[key] = worlds.world(by_id[job["system"]], job["noise"], tag=WORLD_TAG)
                handle.write(json.dumps(run_one(job, by_id[job["system"]], made[key]), sort_keys=True, default=str) + "\n")
                handle.flush()
        return 0
    rows = [json.loads(line) for path in sorted(args.out.glob("runs-shard-*.jsonl")) for line in path.read_text().splitlines()]
    expected = len(all_jobs())
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} runs, found {len(rows)}")
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    summary = summarise(rows)
    (args.out / "results.json").write_text(json.dumps(
        {"preregistration": PREREGISTRATION_ID, "frozen": frozen_digests(), "summary": summary},
        indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"collected {len(rows)} runs into {args.out}")
    print(json.dumps(summary["expectations"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
