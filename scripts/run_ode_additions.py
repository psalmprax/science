#!/usr/bin/env python3
"""Three additions to the ODE route, each against the route in force, with the same candidates for all.

    python scripts/run_ode_additions.py smoke   --out DIR                # two toy components, a short real search
    python scripts/run_ode_additions.py replay  --out DIR --shard k/N    # development: the first blind set, replayed
    python scripts/run_ode_additions.py run     --out DIR --shard k/N    # PREREG-ODE-ADDITIONS-2026-11, once
    python scripts/run_ode_additions.py collect --out DIR [--replay]

For every component and noise level the four searches of the default route run **once**
(``run``), or the candidates of a recorded run stand in for them (``replay``). Every arm is then
given those same candidates, so the arms differ only in what they do with them:

- ``default``: the route in force (``ode/default_route.py``);
- ``second_engine``: sparse regression on a cubic library beside the searches;
- ``precision``: a law refused at the adequacy check is reinstated if its residual is within the
  rows' own precision;
- ``both``: the two together;
- ``experiment``: ``both``, then one follow-up experiment: the system is restarted where the
  reported law and its own shorter versions disagree most, and a term is dropped if the shorter
  law predicts those restarts better (``ode/experiment_pruning.py``);
- ``experiment_random``: the same with the restarts at random states of the same box: equal
  data, no choice.

An earlier form of the experiment arms (the experiment's rows added to the pool and everything
confirmed again) was replayed on 2026-11-21 and made the route worse; it is recorded in
``benchmarks/ode_additions_dev_replay.json`` and is not an arm any more.

``sparse_poly3`` is the frozen runner's reference, reported beside them.

Rows, noise and scoring are the frozen ODEBench runner's (``odebench_worlds.py``,
``run_odebench_main_search.py``). ``replay`` runs on the 98 systems of PREREG-ODE-BLIND-2026-11,
whose outcomes are known: it is development, and what it prints is not a result.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import odebench_worlds as worlds  # noqa: E402
import run_ode_blind as first_blind  # noqa: E402
import run_odebench_main_search as ode_run  # noqa: E402

PREREGISTRATION_ID = "PREREG-ODE-ADDITIONS-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-ODE-ADDITIONS.json"
SYSTEMS_FILE = ROOT / "benchmarks" / "ode_second_blind_systems.json"
RECORD = ROOT / "preregistration" / "RESULT-ODE-BLIND-2026-11" / "runs.jsonl.gz"
FROZEN = (
    "preregistration/PREREG-ODE-ADDITIONS-2026-11.md", "preregistration/FORECAST-ODE-ADDITIONS-2026-11.json",
    "scripts/run_ode_additions.py", "scripts/build_ode_second_blind_systems.py", "benchmarks/ode_second_blind_systems.json",
    "scripts/odebench_worlds.py", "scripts/run_odebench_main_search.py", "scripts/run_ode_blind.py",
    "scripts/build_dysts_polynomial_systems.py",
    "apedp_dyn/ode/extended_route.py", "apedp_dyn/ode/second_engine.py", "apedp_dyn/ode/precision_adequacy.py",
    "apedp_dyn/ode/chosen_experiment.py", "apedp_dyn/ode/experiment_pruning.py", "apedp_dyn/ode/default_route.py",
    "apedp_dyn/ode/refined_ode.py",
    "apedp_dyn/ode/main_search_ode.py", "apedp_dyn/ode/window_rows.py", "apedp_dyn/ode/weak_form.py",
    "apedp_dyn/discovery_l2/deciding_measurement.py", "apedp_dyn/discovery_l2/evolved_stage.py",
    "apedp_dyn/discovery_l2/precision_pruning.py", "apedp_dyn/discovery_l2/refined_search.py",
    "apedp_dyn/discovery_l2/main_search.py", "apedp_dyn/proposers/sparse_library.py", "apedp_dyn/proposers/fit_guard.py",
    "apedp_dyn/proposers/portfolio.py", "apedp_dyn/proposers/pysr_proposer.py",
)
ARMS = ("default", "second_engine", "precision", "both", "experiment", "experiment_random")
REFERENCE = "sparse_poly3"
NOISE_LEVELS = (0.0, 0.01)
WRONG_LIMIT = 0.05
TIME_LIMIT_S = 1800.0           # one component at one noise level: the searches once, then every arm
REPLAY_TAG = first_blind.WORLD_TAG
#: arm -> (second engine, adequacy at the rows' precision)
SETTINGS = {"default": (False, False), "second_engine": (True, False), "precision": (False, True), "both": (True, True)}


def systems(seen: bool = False) -> list[dict[str, Any]]:
    """The second blind set (or, for a replay, the first one), in the frozen world builder's form."""
    if seen:
        return [{**s, "part": "first blind set"} for s in first_blind.systems()]
    record = json.loads(SYSTEMS_FILE.read_text(encoding="utf-8"))
    skip = set(record.get("not_buildable", {}))
    return [{**s, "components": [sp.sympify(text) for text in s["texts"]]} for s in record["systems"] if s["name"] not in skip]


def recorded_candidates() -> dict[tuple[float, int, int], list[str]]:
    """What the four searches handed over in PREREG-ODE-BLIND-2026-11, per noise level, system and component."""
    rows = [json.loads(line) for line in gzip.decompress(RECORD.read_bytes()).decode().splitlines()]
    return {(float(r["noise"]), int(r["system"]), int(r["component"])):
            [c["formula"] for c in (r.get("candidates") or []) if c.get("source") == "search"]
            for r in rows if r["engine"] == first_blind.REPAIRED}


def _scored(world: dict[str, Any], truth: sp.Expr, found: Any, names: list[str]) -> dict[str, Any]:
    law = found.law
    model = sp.sympify(law, locals={n: sp.Symbol(n) for n in names}) if law else None
    row = ode_run.score(world, truth, {"model": model, "formula": law})
    path = getattr(found.result, "path", None)
    return {**row, "action": found.action, "tentative_law": found.tentative_law, "unpruned_law": found.unpruned_law,
            "precision_dropped": found.precision_dropped, "prune_error": found.prune_error,
            "fit_guard_dropped": len(found.fit_guard_dropped),
            "found_by": path.found_by if path else None, "rejected_law": path.rejected_law if path else None,
            "adequacy": path.adequacy if path else None, "rows_read": path.rows_read if path else None}


def arms_of(system: dict[str, Any], world: dict[str, Any], component: int, noise: float, searched: list[str] | None,
            budget: float) -> list[dict[str, Any]]:
    """Every arm on one component: one row each, the reference last."""
    from apedp_dyn.ode import default_route, experiment_pruning, extended_route

    names = [f"x{i}" for i in range(system["dim"])]
    truth = system["components"][component]
    dev, pool = world["dev"], world["pool"]
    world = {**world, "test_rates_component": world["test_rates"][:, component]}
    rate = worlds.vector_field(system)
    rows: list[dict[str, Any]] = []
    found: dict[str, Any] = {}

    def record(arm: str, law: Any, extra: dict[str, Any], started: float) -> None:
        found[arm] = law
        rows.append({"arm": arm, "status": "ok", "seconds": round(time.perf_counter() - started, 2),
                     **_scored(world, truth, law, names), **extra})

    started = time.perf_counter()
    if searched is None:                                    # the searches run once, here
        stage = extended_route.Recording(default_route.search_stage(budget, ode_run.SEED)) if budget > 0 else None
        law, done = extended_route.discover_component(dev, pool, component, proposer=stage, seed=ode_run.SEED)
        searched = list(stage.formulas) if stage is not None else []
        done["search"] = dict(stage.last_run) if stage is not None else {}
    else:
        law, done = extended_route.discover_component(dev, pool, component, proposer=extended_route.replayed(searched),
                                                      seed=ode_run.SEED)
    record("default", law, {**done, "searched": searched}, started)
    for arm in ("second_engine", "precision", "both"):
        started = time.perf_counter()
        second, precise = SETTINGS[arm]
        law, done = extended_route.discover_component(dev, pool, component, proposer=extended_route.replayed(searched),
                                                      seed=ode_run.SEED, second_engine=second, at_precision=precise)
        record(arm, law, done, started)
    both_done = dict(done)
    for arm, chosen in (("experiment", True), ("experiment_random", False)):
        started = time.perf_counter()
        law, done = experiment_pruning.decided(found["both"], dev, component, rate, level=noise, spread=world["spread"],
                                               seed=ode_run.SEED, chosen=chosen)
        record(arm, law, {**both_done, **done}, started)
    started = time.perf_counter()
    raw = ode_run.run_sparse_poly3(world, component)
    rows.append({"arm": REFERENCE, "status": "ok", "seconds": round(time.perf_counter() - started, 2),
                 **ode_run.score(world, truth, raw), **raw.get("extra", {})})
    return rows


def _child(conn: Any, system: dict[str, Any], world: dict[str, Any], component: int, noise: float,
           searched: list[str] | None, budget: float) -> None:
    try:
        conn.send(("ok", arms_of(system, world, component, noise, searched, budget)))
    except Exception as exc:  # noqa: BLE001 - a crash of the unit is a scored outcome, reported with its class and message
        conn.send(("error", {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-2000:]}))
    finally:
        conn.close()


def run_unit(system: dict[str, Any], world: dict[str, Any], component: int, noise: float, searched: list[str] | None,
             budget: float = ode_run.SEARCH_BUDGET_S, time_limit: float = TIME_LIMIT_S) -> list[dict[str, Any]]:
    """One component at one noise level: a row per arm. A crash or a timeout gives every arm that status."""
    head = {"system": system["id"], "name": system.get("name"), "part": system.get("part"), "component": component,
            "noise": noise, "dim": system["dim"], "truth": str(system["components"][component])}
    if "excluded" in world:
        return [{**head, "arm": arm, "status": "excluded", "reason": world["excluded"]} for arm in (*ARMS, REFERENCE)]
    head.update({"n_dev": len(world["dev"].states), "n_pool": len(world["pool"].states),
                 "rate_sd": float(world["dev"].rate_sd[component])})
    context = mp.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    started = time.perf_counter()
    process = context.Process(target=_child, args=(child, system, world, component, noise, searched, budget))
    process.start()
    child.close()
    outcome: tuple[str, Any] = ("timeout", {})
    if parent.poll(time_limit):
        try:
            outcome = parent.recv()
        except EOFError:
            outcome = ("error", {"error": "the unit's process died without a result"})
    process.join(1.0)
    if process.is_alive():
        process.kill()
        process.join()
    if outcome[0] == "ok":
        return [{**head, **row} for row in outcome[1]]
    seconds = round(time.perf_counter() - started, 2)
    return [{**head, "arm": arm, "status": outcome[0], "seconds": seconds, **outcome[1]} for arm in (*ARMS, REFERENCE)]


def units(seen: bool = False) -> list[tuple[dict[str, Any], int, float]]:
    return [(system, component, noise) for system in systems(seen) for component in range(system["dim"])
            for noise in NOISE_LEVELS]


def _reported(row: dict[str, Any]) -> bool:
    return row["status"] == "ok" and row.get("model") not in (None, "None")


def _wrong(row: dict[str, Any]) -> bool:
    return _reported(row) and not row.get("symbolic") and not row.get("accuracy_solution")


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per arm, noise level and part: runs, reports, wrong laws, recoveries; then the expectations as written."""
    out: dict[str, Any] = {}
    parts = ["all", *dict.fromkeys(str(r.get("part")) for r in rows)]
    for arm in (*ARMS, REFERENCE):
        for noise in (*NOISE_LEVELS, "all"):
            for part in parts:
                sel = [r for r in rows if r["arm"] == arm and noise in (r["noise"], "all") and part in (str(r.get("part")), "all")]
                if not sel:
                    continue
                ok = [r for r in sel if r["status"] == "ok"]
                reported = [r for r in ok if _reported(r)]
                out[f"{arm}|{noise}|{part}"] = {
                    "runs": len(sel), "errors": sum(r["status"] == "error" for r in sel),
                    "timeouts": sum(r["status"] == "timeout" for r in sel), "reported": len(reported),
                    "recovered": sum(bool(r.get("symbolic")) for r in ok), "wrong": sum(_wrong(r) for r in ok),
                    "reported_accurate_not_the_law": sum(bool(r.get("accuracy_solution")) and not r.get("symbolic") for r in reported),
                    "reinstated": sum(bool((r.get("precision") or {}).get("reinstated")) for r in ok),
                    "experiments_run": sum(bool((r.get("experiment") or {}).get("restarts_used")) for r in ok),
                    "laws_shortened_by_the_experiment": sum(bool((r.get("experiment") or {}).get("dropped")) for r in ok),
                }
    recovered = {(r["arm"], r["noise"], r["system"], r["component"]) for r in rows if r["status"] == "ok" and r.get("symbolic")}
    cells = sorted({(r["noise"], r["system"], r["component"]) for r in rows})
    paired = {}
    for a, b in (("both", "default"), ("second_engine", "default"), ("precision", "default"), ("experiment", "both"),
                 ("experiment_random", "both"), ("experiment", "experiment_random")):
        for noise in NOISE_LEVELS:
            mine = [c for c in cells if c[0] == noise]
            paired[f"{a} against {b}|{noise}"] = {"gained": sum((a, *c) in recovered and (b, *c) not in recovered for c in mine),
                                                  "lost": sum((a, *c) not in recovered and (b, *c) in recovered for c in mine)}
    out["run_for_run"] = paired

    def count(arm: str, noise: Any, key: str) -> int:
        return int(out.get(f"{arm}|{noise}|all", {}).get(key, 0))

    def share_wrong(arm: str) -> float:
        return count(arm, "all", "wrong") / max(count(arm, "all", "reported"), 1)

    def recovered_by(arm: str) -> dict[str, int]:
        return {str(n): count(arm, n, "recovered") for n in NOISE_LEVELS}

    out["expectations"] = {
        "1_share_of_reported_laws_wrong": {arm: share_wrong(arm) for arm in ("both", "experiment")},
        "1_limit": WRONG_LIMIT,
        "1_holds": all(share_wrong(arm) <= WRONG_LIMIT for arm in ("both", "experiment")),
        "2_recovered_both": recovered_by("both"), "2_recovered_default": recovered_by("default"),
        "2_holds": all(count("both", n, "recovered") >= count("default", n, "recovered") for n in NOISE_LEVELS),
        "3_recovered_experiment": recovered_by("experiment"),
        "3_wrong_experiment_and_both": [count("experiment", "all", "wrong"), count("both", "all", "wrong")],
        "3_holds": all(count("experiment", n, "recovered") >= count("both", n, "recovered") for n in NOISE_LEVELS)
        and count("experiment", "all", "recovered") > count("both", "all", "recovered")
        and count("experiment", "all", "wrong") <= count("both", "all", "wrong"),
        "4_recovered_experiment_random": recovered_by("experiment_random"),
        "4_holds": count("experiment", "all", "recovered") > count("experiment_random", "all", "recovered"),
        "default_share_wrong": share_wrong("default"),
    }
    return out


def frozen_digests() -> dict[str, str]:
    return {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in FROZEN}


def check_frozen() -> None:
    recorded = json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"]
    now = frozen_digests()
    changed = sorted(rel for rel in FROZEN if recorded.get(rel) != now[rel])
    if changed or set(recorded) != set(FROZEN):
        raise SystemExit(f"frozen files differ from {DIGEST.name}: {changed or sorted(set(recorded) ^ set(FROZEN))}")


def _printable(row: dict[str, Any]) -> str:
    keep = ("name", "component", "noise", "arm", "status", "symbolic", "accuracy_solution", "action", "formula", "seconds", "error")
    return json.dumps({k: row.get(k) for k in keep}, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("smoke", "replay", "run", "collect"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--replay", action="store_true", help="for collect: the rows are a replay (development)")
    parser.add_argument("--every", type=int, default=1, help="for replay: only every Nth unit")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    k, n = (int(v) for v in args.shard.split("/"))
    if args.command == "smoke":
        failed = 0
        for system in ode_run.toy_systems()[1:3]:
            system = {**system, "part": "toy"}
            world = worlds.world(system, 0.01, tag="smoke")
            for row in run_unit(system, world, 0, 0.01, None, budget=20.0, time_limit=900.0):
                print(_printable(row), flush=True)
                if row["status"] != "ok":
                    print(row.get("trace", ""), flush=True)
                    failed += 1
        return 1 if failed else 0
    if args.command in ("replay", "run"):
        seen = args.command == "replay"
        if not seen:
            check_frozen()
        candidates = recorded_candidates() if seen else {}
        tag = REPLAY_TAG if seen else PREREGISTRATION_ID
        mine = [unit for i, unit in enumerate(units(seen)) if i % args.every == 0 and (i // args.every) % n == k]
        path = args.out / f"{'replay' if seen else 'runs'}-shard-{k:03d}-of-{n:03d}.jsonl"
        built: dict[tuple[int, float], dict[str, Any]] = {}
        with path.open("w", encoding="utf-8") as handle:
            for system, component, noise in mine:
                key = (system["id"], noise)
                if key not in built:
                    built = {key: worlds.world(system, noise, tag=tag)}
                searched = candidates[(noise, system["id"], component)] if seen else None
                for row in run_unit(system, built[key], component, noise, searched):
                    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                handle.flush()
        return 0
    prefix = "replay" if args.replay else "runs"
    rows = [json.loads(line) for path in sorted(args.out.glob(f"{prefix}-shard-*.jsonl")) for line in path.read_text().splitlines()]
    (args.out / f"{prefix}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    summary = summarise(rows)
    record = {"preregistration": None if args.replay else PREREGISTRATION_ID,
              "what": "A replay on systems whose outcomes are known: development, not a result." if args.replay
              else "The preregistered run.", "rows": len(rows), "summary": summary}
    if not args.replay:
        record["frozen"] = frozen_digests()
    (args.out / ("replay_results.json" if args.replay else "results.json")).write_text(
        json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"collected {len(rows)} rows into {args.out}")
    print(json.dumps({key: value for key, value in summary.items() if key.endswith("|all") or "|" not in key}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
