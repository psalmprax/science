#!/usr/bin/env python3
"""ODEBench through the main search path, for PREREG-ODE-MAIN-SEARCH-2026-11.

One run is one component of one system at one noise level. Rows come from
``scripts/odebench_worlds.py``; the path is ``apedp_dyn.ode.main_search_ode``
with PySR as the search stage, run on every component. Beside it runs a
polynomial sparse regression on the same rows, as the reference a reader will
know.

    python scripts/run_odebench_main_search.py smoke   --out DIR   # toy systems, not benchmark data
    python scripts/run_odebench_main_search.py run     --out DIR --shard k/N
    python scripts/run_odebench_main_search.py collect --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import odebench_worlds as worlds  # noqa: E402
import run_srbench_baseline as base  # noqa: E402

PREREGISTRATION_ID = "PREREG-ODE-MAIN-SEARCH-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-ODE-MAIN-SEARCH.json"
FROZEN = ("preregistration/PREREG-ODE-MAIN-SEARCH-2026-11.md", "scripts/run_odebench_main_search.py",
          "scripts/odebench_worlds.py", "apedp_dyn/ode/window_rows.py", "apedp_dyn/ode/main_search_ode.py",
          "apedp_dyn/discovery_l2/main_search.py", "apedp_dyn/proposers/pysr_proposer.py",
          "apedp_dyn/discovery_l2/contrast_confirmation.py", "apedp_dyn/discovery_l2/contrast_resolution.py",
          "apedp_dyn/discovery_l2/confirmed_stop.py", "apedp_dyn/discovery_l2/guided_search.py",
          "apedp_dyn/discovery_l2/exhaustive_search.py", "apedp_mmlm/law_corpora.py")

ENGINES = ("main_search", "sparse_poly3")
NOISE_LEVELS = (0.0, 0.01)
SEARCH_BUDGET_S = 240.0
TIME_LIMIT_S = 480.0
SEED = 20261119
REPORTED = ("confirmed", "equivalent_class", "unchallenged")
THRESHOLDS = tuple(float(v) for v in np.logspace(-3, 0.5, 15))  # the sparse regression's grid, chosen by BIC


def run_main_search(world: dict[str, Any], component: int, budget: float = SEARCH_BUDGET_S) -> dict[str, Any]:
    from apedp_dyn.ode import main_search_ode
    from apedp_dyn.proposers.pysr_proposer import PySRProposer

    dev, pool = world["dev"], world["pool"]
    names = main_search_ode.state_names(dev.states.shape[1])
    result = main_search_ode.discover_component(dev, pool, component, proposer=PySRProposer(budget_seconds=budget, seed=SEED),
                                                seed=SEED, always_search=True)
    model = main_search_ode.confirmed_form(result.law, names, dev.states, dev.rates[:, component]) if result.law else None
    return {"model": model, "formula": result.law,
            "extra": {"action": result.action, "found_by": result.found_by, "rejected_law": result.rejected_law,
                      "adequacy": result.adequacy, "rows_read": result.rows_read, "first_pass": result.first_pass,
                      "search": result.search, "candidates": result.candidates,
                      "equivalent": len(result.contrast.equivalent) if result.contrast else 0}}


def run_sparse_poly3(world: dict[str, Any], component: int, budget: float = SEARCH_BUDGET_S) -> dict[str, Any]:
    """Sequentially thresholded least squares over all monomials up to degree three; threshold chosen by BIC."""
    from apedp_dyn.ode.main_search_ode import state_names
    from apedp_dyn.ode.weak_form import stlsq

    X, y = world["dev"].states, world["dev"].rates[:, component]
    names = state_names(X.shape[1])
    symbols = [sp.Symbol(n) for n in names]
    powers = [p for p in itertools.product(range(4), repeat=len(names)) if sum(p) <= 3]
    library = np.column_stack([np.prod(X ** np.array(p), axis=1) for p in powers])
    best: tuple[float, np.ndarray] | None = None
    floor = (1e-6 * float(np.sqrt(np.mean(y ** 2)))) ** 2  # exact data: fits below this are equally good
    for threshold in THRESHOLDS:
        coef: np.ndarray = stlsq(library, y, threshold)[:, 0]
        rss = float(np.sum((library @ coef - y) ** 2))
        bic = len(y) * float(np.log(max(rss / len(y), floor, 1e-300))) + int(np.count_nonzero(coef)) * float(np.log(len(y)))
        if best is None or bic < best[0]:
            best = (bic, coef)
    assert best is not None
    model = sp.Add(*[sp.Float(float(c)) * sp.Mul(*[s ** k for s, k in zip(symbols, p, strict=True)])
                     for c, p in zip(best[1], powers, strict=True) if c != 0.0])
    return {"model": model if model.free_symbols else None, "formula": str(model), "extra": {"terms": int(np.count_nonzero(best[1]))}}


RUNNERS = {"main_search": run_main_search, "sparse_poly3": run_sparse_poly3}


def score(world: dict[str, Any], truth: sp.Expr, raw: dict[str, Any]) -> dict[str, Any]:
    names = [f"x{i}" for i in range(world["test_states"].shape[1])]
    model, X, target = raw.get("model"), world["test_states"], world["test_rates_component"]
    row: dict[str, Any] = {"model": str(model) if model is not None else None, "formula": raw.get("formula")}
    if model is None:
        return {**row, "symbolic": False, "symbolic_note": "no model", "test_r2": None, "accuracy_solution": False}
    try:
        sym = base.symbolic_recovery(model, truth, names, X)
    except Exception as exc:  # noqa: BLE001 - a model the scorer cannot handle is scored as not recovered, with the reason
        sym = {"recovered": False, "rel_rmse": None, "reason": f"scoring failed: {type(exc).__name__}"}
    fn = sp.lambdify([sp.Symbol(n) for n in names], model, ["scipy", "numpy"], dummify=True)
    with np.errstate(all="ignore"):
        pred = np.asarray(fn(*X.T), dtype=float) * np.ones(len(X))
    r2 = base._r2(target, pred) if np.all(np.isfinite(pred)) else None
    return {**row, "symbolic": bool(sym["recovered"]), "symbolic_rel_rmse": sym["rel_rmse"], "symbolic_note": sym["reason"],
            "test_r2": r2, "accuracy_solution": bool(r2 is not None and r2 >= base.R2_ACCURACY)}


def _child(conn: Any, engine: str, world: dict[str, Any], truth: sp.Expr, component: int, budget: float) -> None:
    try:
        raw = RUNNERS[engine](world, component, budget)
        conn.send(("ok", {**score(world, truth, raw), **raw.get("extra", {})}))
    except Exception as exc:  # noqa: BLE001 - an engine crash is a scored outcome, reported with its class and message
        conn.send(("error", {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-1500:]}))
    finally:
        conn.close()


def run_world(job: dict[str, Any], system: dict[str, Any], world: dict[str, Any], budget: float = SEARCH_BUDGET_S,
              time_limit: float = TIME_LIMIT_S) -> dict[str, Any]:
    component = job["component"]
    head = {**job, "dim": system["dim"], "truth": str(system["components"][component])}
    if "excluded" in world:
        return {**head, "status": "excluded", "reason": world["excluded"], "seconds": 0.0}
    world = {**world, "test_rates_component": world["test_rates"][:, component]}
    head.update({"n_dev": len(world["dev"].states), "n_pool": len(world["pool"].states),
                 "window_degree": int(world["dev"].degree[component]), "rate_sd": float(world["dev"].rate_sd[component])})
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    started = time.perf_counter()
    proc = ctx.Process(target=_child, args=(child, job["engine"], world, system["components"][component], component, budget))
    proc.start()
    child.close()
    outcome: tuple[str, dict[str, Any]] = ("timeout", {})
    if parent.poll(time_limit):
        try:
            outcome = parent.recv()
        except EOFError:
            outcome = ("error", {"error": "engine process died without a result"})
    proc.join(1.0)
    if proc.is_alive():
        proc.kill()
        proc.join()
    return {**head, "status": outcome[0], "seconds": round(time.perf_counter() - started, 2), **outcome[1]}


def shard(jobs: list[dict[str, Any]], k: int, n: int) -> list[dict[str, Any]]:
    """Shard by component, so that long systems are spread over the shards."""
    units = sorted({(job["system"], job["component"]) for job in jobs})
    mine = {unit for i, unit in enumerate(units) if i % n == k}
    return [job for job in jobs if (job["system"], job["component"]) in mine]


def all_jobs() -> list[dict[str, Any]]:
    return [{"system": s["id"], "component": k, "noise": noise, "engine": engine}
            for s in worlds.systems() for k in range(s["dim"]) for noise in NOISE_LEVELS for engine in ENGINES]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for engine in ENGINES:
        for noise in (*NOISE_LEVELS, "all"):
            sel = [r for r in rows if r["engine"] == engine and noise in (r["noise"], "all")]
            ok = [r for r in sel if r["status"] == "ok"]
            reported = [r for r in ok if r.get("model") not in (None, "None")]
            cell: dict[str, Any] = {
                "n": len(sel), "excluded": sum(r["status"] == "excluded" for r in sel),
                "errors": sum(r["status"] == "error" for r in sel), "timeouts": sum(r["status"] == "timeout" for r in sel),
                "median_seconds": float(np.median([r["seconds"] for r in sel])) if sel else None,
                "symbolic": sum(bool(r["symbolic"]) for r in ok), "accuracy_solution": sum(bool(r["accuracy_solution"]) for r in ok),
                "reported": len(reported), "reported_symbolic": sum(bool(r["symbolic"]) for r in reported),
                "reported_accurate": sum(bool(r["accuracy_solution"]) for r in reported),
                "reported_not_accurate": sum(not r["accuracy_solution"] for r in reported),
            }
            # a system counts when every one of its components does
            by_system: dict[tuple[int, float], list[dict[str, Any]]] = {}
            for r in sel:
                by_system.setdefault((r["system"], r["noise"]), []).append(r)
            cell["systems"] = len(by_system)
            cell["systems_symbolic"] = sum(all(r["status"] == "ok" and r["symbolic"] for r in group) for group in by_system.values())
            if engine == "main_search":
                cell["actions"] = {a: sum(r.get("action") == a for r in ok) for a in (*REPORTED, "widen_search")}
                cell["found_by"] = {s: sum(r.get("found_by") == s for r in reported) for s in ("first_pass", "search")}
            out[f"{engine}|{noise}"] = cell
    return out


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def check_frozen() -> None:
    if not DIGEST.exists() or json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} is not frozen, or changed since it was frozen")


TOY_SYSTEMS: tuple[dict[str, Any], ...] = (  # development systems; none is an ODEBench entry with its published parameters
    {"id": -1, "dim": 2, "texts": ["x1", "-0.3*x1 - 1.2*x0 - 0.8*x0**3"], "inits": [[1.5, 0.0], [-0.7, 1.1]]},
    {"id": -2, "dim": 2, "texts": ["1 - x0*x1", "x0 - 0.5*x1"], "inits": [[0.3, 2.0], [2.5, 0.4]]},
    {"id": -3, "dim": 1, "texts": ["0.8*x0*(1 - x0/3) - 0.2"], "inits": [[0.6], [4.0]]},
)


def toy_systems() -> list[dict[str, Any]]:
    return [{**s, "components": [sp.sympify(text) for text in s["texts"]]} for s in TOY_SYSTEMS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("run", "collect", "smoke"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1", help="k/N: this shard's runs")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "smoke":
        budget = float(os.environ.get("SMOKE_BUDGET_SECONDS", "30"))
        engines = [e for e in ENGINES if e in os.environ.get("SMOKE_ENGINES", ",".join(ENGINES)).split(",")]
        failed = 0
        for system in toy_systems():
            for noise in NOISE_LEVELS:
                world = worlds.world(system, noise, tag="smoke")
                for component in range(system["dim"]):
                    for engine in engines:
                        row = run_world({"system": system["id"], "component": component, "noise": noise, "engine": engine},
                                        system, world, budget)
                        print(json.dumps({k: v for k, v in row.items() if k not in ("candidates", "trace", "first_pass")}, default=str), flush=True)
                        if row["status"] != "ok":
                            print(row.get("trace", ""), flush=True)
                            failed += 1
        return 1 if failed else 0
    check_frozen()
    if args.command == "run":
        k, n = (int(v) for v in args.shard.split("/"))
        systems = {s["id"]: s for s in worlds.systems()}
        made: dict[tuple[int, float], dict[str, Any]] = {}
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for job in shard(all_jobs(), k, n):
                key = (job["system"], job["noise"])
                if key not in made:
                    made[key] = worlds.world(systems[job["system"]], job["noise"])
                handle.write(json.dumps(run_world(job, systems[job["system"]], made[key]), sort_keys=True, default=str) + "\n")
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
