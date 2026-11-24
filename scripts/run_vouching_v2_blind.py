#!/usr/bin/env python3
"""The second version of the last checks on laws nobody has run: PREREG-VOUCHING-V2-2026-11.

    python scripts/run_vouching_v2_blind.py laws [--check]           # benchmarks/vouching_v2_blind_laws.json
    python scripts/run_vouching_v2_blind.py smoke
    python scripts/run_vouching_v2_blind.py run --part handed|searched --shard k/N --out DIR
    python scripts/run_vouching_v2_blind.py collect --out DIR

Two parts, each on laws drawn here with their own seed and with no structure used anywhere
before (the invented-laws benchmark by fingerprint; the red-team campaigns' laws, the memory's
development and test laws, the counterfeit benchmark's):

* **handed**: PREREG-COUNTERFEIT's design on 60 new laws. Each case hands the route the law's
  counterfeits as candidates, with the law among them or with only wrong ones, at noise 0, 1 %
  and 5 %; no search engine runs. Five reports per case: the route, the route after the first
  version of the checks (``first``), after the second (``second``), the best fit and the best
  fit confirmed on fresh rows.
* **searched**: 40 new laws of the invented-law grammar behind the search stage in force for
  tables (four searches side by side and the memory of reported laws), at noise 0, 1 % and 5 %.
  Three reports per run, from one search: the route, ``first`` and ``second``.

``run_counterfeit_benchmark.py``, ``run_red_team.py`` and the routes are frozen by earlier
preregistrations and are used as they are.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_invented_laws as grammar  # noqa: E402
import run_counterfeit_benchmark as bench  # noqa: E402
import run_red_team as red_team  # noqa: E402
import run_srbench_baseline as base  # noqa: E402
import run_srbench_main_search as path_run  # noqa: E402

from apedp_dyn.conf.canonical_spec import REFEREE_MAX_FPR, REFEREE_MIN_VRR  # noqa: E402
from apedp_dyn.discovery_l2 import vouching, vouching_v2  # noqa: E402
from apedp_dyn.redteam import worlds as rt  # noqa: E402

PREREGISTRATION_ID = "PREREG-VOUCHING-V2-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-VOUCHING-V2.json"
LAWS_FILE = ROOT / "benchmarks" / "vouching_v2_blind_laws.json"
FROZEN = (
    "preregistration/PREREG-VOUCHING-V2-2026-11.md", "scripts/run_vouching_v2_blind.py", "benchmarks/vouching_v2_blind_laws.json",
    "apedp_dyn/discovery_l2/vouching_v2.py", "apedp_dyn/discovery_l2/vouching.py", "scripts/run_counterfeit_benchmark.py",
    "scripts/run_red_team.py", "scripts/build_invented_laws.py", "scripts/run_srbench_baseline.py",
    "scripts/run_srbench_main_search.py", "apedp_dyn/redteam/counterfeit.py", "apedp_dyn/redteam/worlds.py",
    "apedp_dyn/redteam/plain_selection.py", "apedp_dyn/discovery_l2/refined_search.py", "apedp_dyn/discovery_l2/main_search.py",
    "apedp_dyn/discovery_l2/contrast_confirmation.py", "apedp_dyn/discovery_l2/stage_in_force.py",
    "apedp_dyn/discovery_l2/memory_stage.py", "apedp_dyn/discovery_l2/search_stage.py", "apedp_dyn/proposers/portfolio.py",
    "apedp_dyn/proposers/fit_guard.py", "apedp_dyn/proposers/structure_memory.py", "benchmarks/structure_memory.json",
)
HANDED_SEED, HANDED_COUNT, HANDED_DRAWN = 20261203, 60, 90
SEARCHED_SEED, SEARCHED_COUNT = 20261204, 40
HANDED_ID, SEARCHED_ID = 8000, 8500
NOISE_LEVELS = (0.0, 0.01, 0.05)
SEARCH_BUDGET_S = 240.0
TIME_LIMIT_S = 1500.0
HANDED_METHODS = ("route", "first", "second", "best_fit", "validated")
SEARCHED_ARMS = ("route", "first", "second")
WRONG_LIMIT = REFEREE_MAX_FPR
KEPT_SHARE = 1.0 - REFEREE_MAX_FPR        # up to 1 % noise: withhold a right law no more often than a referee may be wrong
KEPT_SHARE_NOISY = REFEREE_MIN_VRR         # at 5 % noise: keep the canonical share of valid results a referee must retain
QUIET = 0.01                               # the noise up to which the stricter share applies


# ---------------------------------------------------------------- laws
def used() -> set[str]:
    """Every structure used before: the counterfeit benchmark's own exclusions, its laws, the second campaign's laws."""
    taken: set[str] = set(bench._taken())
    taken |= {grammar.structure(sp.sympify(law["law"])) for law in json.loads(bench.OUT.read_text(encoding="utf-8"))["laws"]}
    campaign2 = ROOT / "benchmarks" / "red_team_checks_laws.json"
    taken |= {grammar.structure(sp.sympify(law["law"])) for law in json.loads(campaign2.read_text(encoding="utf-8"))["laws"]}
    return taken


def build_laws() -> dict[str, Any]:
    taken = used()
    drawn = bench.build(HANDED_DRAWN, HANDED_SEED)
    handed = [law for law in drawn["laws"] if grammar.structure(sp.sympify(law["law"])) not in taken][:HANDED_COUNT]
    if len(handed) != HANDED_COUNT:
        raise SystemExit(f"only {len(handed)} handed laws used nowhere else")
    renumber = {law["id"]: HANDED_ID + k for k, law in enumerate(handed)}
    cases = [{**case, "law_id": renumber[case["law_id"]], "case": f"{renumber[case['law_id']]}-" + case["case"].split("-", 1)[1]}
             for case in drawn["cases"] if case["law_id"] in renumber]
    for law in handed:
        law["id"] = renumber[law["id"]]
    taken |= {grammar.structure(sp.sympify(law["law"])) for law in handed}
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    rng = np.random.default_rng(SEARCHED_SEED)
    searched: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(grammar.MAX_DRAWS):
        if len(searched) == SEARCHED_COUNT:
            break
        inputs = int(rng.integers(1, 4))
        expr = grammar.draw(rng, inputs)
        key = grammar.structure(expr)
        if key in seen or not grammar.acceptable(expr, inputs, rng):
            continue
        seen.add(key)
        if grammar.fingerprint(str(expr)) in in_benchmark or key in taken:
            continue
        searched.append({"id": SEARCHED_ID + len(searched), "inputs": inputs, "text": str(expr)})
    if len(searched) != SEARCHED_COUNT:
        raise SystemExit(f"only {len(searched)} searched laws")
    return {"what": "Laws for PREREG-VOUCHING-V2-2026-11, drawn after the second version of the checks was written. "
                    "No route and no check has run on them.",
            "seeds": {"handed": HANDED_SEED, "searched": SEARCHED_SEED}, "domain": [grammar.LOW, grammar.HIGH],
            "noise_levels": list(NOISE_LEVELS), "handed": {"laws": handed, "cases": cases}, "searched": searched}


def record() -> dict[str, Any]:
    out: dict[str, Any] = json.loads(LAWS_FILE.read_text(encoding="utf-8"))
    return out


# ---------------------------------------------------------------- the handed part
def checked_row(world: rt.World, route: dict[str, Any], module: Any) -> dict[str, Any]:
    """The route's report after one version of the checks, scored as the counterfeit benchmark scores."""
    if not route.get("reported") or not route.get("law"):
        nothing: dict[str, Any] = bench.silent("the route reported nothing")
        return nothing
    names = [f"x{i}" for i in range(world.inputs)]
    truth = sp.sympify(world.law, locals={n: sp.Symbol(n) for n in names})
    rows = rt.rows_of(world)
    order = np.random.default_rng([world.seed, 4]).permutation(world.rows)
    cut = round(path_run.DEV_SHARE * world.rows)
    dev, pool = order[:cut], order[cut:]
    try:
        done = module.vouch(route["law"], list(route.get("equivalent") or []), names, rows.X[dev], rows.y[dev], rows.X[pool])
    except Exception as exc:  # noqa: BLE001 - checks that crash vouch for nothing; recorded with the error's class
        return {**bench.silent(f"the checks failed: {type(exc).__name__}: {exc}"[:200]), "failed": True}
    return {**red_team._scored(done.law, truth, names, rows), "reason": done.reason, "switched": done.switched,
            "power": done.power}


def handed_rows(cases: list[dict[str, Any]], cores: int) -> list[dict[str, Any]]:
    worlds = [rt.World.from_dict(case["world"]) for case in cases]
    rows: list[dict[str, Any]] = red_team.evaluate(worlds, cores, bench.TIME_LIMIT_S)
    for case, world, row in zip(cases, worlds, rows, strict=True):
        row.update({"case": case["case"], "law_id": case["law_id"]})
        if row.get("status") == "ok":
            row["first"] = checked_row(world, row["route"], vouching)
            row["second"] = checked_row(world, row["route"], vouching_v2)
    return rows


def tally_handed(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    offered = [row for row in rows if row["world"]["truth_listed"]]
    right = sum(bool(row[method]["reported"] and row[method]["symbolic"]) for row in ok)
    wrong = sum(bool(row[method]["wrong"]) for row in ok)
    reported = sum(bool(row[method]["reported"]) for row in ok)
    return {"cases": len(rows), "errors": sum(row.get("status") == "error" for row in rows),
            "timeouts": sum(row.get("status") == "timeout" for row in rows), "reported": reported, "right": right,
            "wrong": wrong, "wrong_share": wrong / reported if reported else None,
            "score": (right - bench.PENALTY * wrong) / len(offered) if offered else None}


# ---------------------------------------------------------------- the searched part
def entry_of(law: dict[str, Any], domain: list[float]) -> dict[str, Any]:
    return {"name": f"vouching_v2_{law['id']}", "family": "invented", "features": [f"x{i}" for i in range(law["inputs"])],
            "formula": law["text"], "units": {}, "target_units": {}, "domain": domain}


def frame_of(entry: dict[str, Any]) -> pd.DataFrame:
    low, high = entry["domain"]
    rng = np.random.default_rng(base._seed(PREREGISTRATION_ID, entry["name"], "rows"))
    X = rng.uniform(low, high, (base.N_TRAIN + base.N_TEST, len(entry["features"])))
    fn = sp.lambdify([sp.Symbol(f"x{i}") for i in range(X.shape[1])], base.truth_expr(entry), "numpy")
    frame = pd.DataFrame(X, columns=entry["features"])
    frame["target"] = np.asarray(fn(*X.T), dtype=float) * np.ones(len(X))
    return frame


def _scored(model_text: str | None, entry: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    names = split["names"]
    model = sp.sympify(model_text, locals={n: sp.Symbol(n) for n in names}) if model_text else None
    raw = {"model": model, "pred": path_run._predict(model, names, split["X_test"]) if model is not None else None,
           "formula": model_text}
    try:
        scored: dict[str, Any] = base.score_run("vouching_v2", entry, split, raw)
    except Exception as exc:  # noqa: BLE001 - a model the scorer cannot handle is scored as not recovered, with the reason
        r2 = base._r2(split["y_test"], raw["pred"]) if raw["pred"] is not None else None
        scored = {"symbolic": False, "symbolic_note": f"scoring failed: {type(exc).__name__}", "test_r2": r2,
                  "accuracy_solution": bool(r2 is not None and r2 >= base.R2_ACCURACY), "model": model_text, "formula": model_text}
    return {**scored, "reported": model_text is not None}


def searched_unit(entry: dict[str, Any], noise: float, budget: float) -> dict[str, Any]:
    from apedp_dyn.discovery_l2 import refined_search, stage_in_force

    split = base.make_split(entry, frame_of(entry), noise)
    names, X, y = split["names"], split["X_train"], split["y_train"]
    order = np.random.default_rng(base._seed(PREREGISTRATION_ID, entry["name"], "pool")).permutation(len(y))
    cut = round(path_run.DEV_SHARE * len(y))
    dev, pool = order[:cut], order[cut:]
    pool_y = y[pool]
    stage = stage_in_force.search_stage(budget, seed=base.SEED)[0] if budget > 0 else None
    started = time.perf_counter()
    result = refined_search.discover_refined({n: X[dev, i] for i, n in enumerate(names)}, y[dev], X[pool],
                                             lambda picked: pool_y[picked], proposer=stage, noise="absolute", seed=base.SEED)
    path = result.path
    assert path is not None
    equivalent = list(path.contrast.equivalent) if path.contrast is not None else []
    out: dict[str, Any] = {"action": result.action, "found_by": path.found_by, "equivalent": equivalent,
                           "search_seconds": round(time.perf_counter() - started, 2), "route": _scored(result.law, entry, split)}
    for arm, module in (("first", vouching), ("second", vouching_v2)):
        if result.law is None:
            out[arm] = {**_scored(None, entry, split), "reason": "the route reported nothing"}
            continue
        try:
            done = module.vouch(result.law, equivalent, names, X[dev], y[dev], X[pool])
            out[arm] = {**_scored(done.law, entry, split), "reason": done.reason, "switched": done.switched}
        except Exception as exc:  # noqa: BLE001 - checks that crash vouch for nothing
            out[arm] = {**_scored(None, entry, split), "reason": f"the checks failed: {type(exc).__name__}", "failed": True}
    return out


def _child(conn: Any, entry: dict[str, Any], noise: float, budget: float) -> None:
    try:
        conn.send(("ok", searched_unit(entry, noise, budget)))
    except Exception as exc:  # noqa: BLE001 - a crash of the unit is a scored outcome
        conn.send(("error", {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-1500:]}))
    finally:
        conn.close()


def run_searched(entry: dict[str, Any], noise: float, budget: float = SEARCH_BUDGET_S) -> dict[str, Any]:
    head = {"problem": entry["name"], "noise": noise, "inputs": len(entry["features"]), "truth": entry["formula"]}
    context = mp.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    started = time.perf_counter()
    process = context.Process(target=_child, args=(child, entry, noise, budget))
    process.start()
    child.close()
    outcome: tuple[str, Any] = ("timeout", {})
    if parent.poll(TIME_LIMIT_S):
        try:
            outcome = parent.recv()
        except EOFError:
            outcome = ("error", {"error": "the unit's process died without a result"})
    process.join(1.0)
    if process.is_alive():
        process.kill()
        process.join()
    return {**head, "status": outcome[0], "seconds": round(time.perf_counter() - started, 2), **outcome[1]}


def tally_searched(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    reported = [row for row in ok if row[arm]["reported"]]
    return {"runs": len(rows), "errors": sum(row["status"] == "error" for row in rows),
            "timeouts": sum(row["status"] == "timeout" for row in rows), "reported": len(reported),
            "recovered": sum(bool(row[arm]["symbolic"]) for row in reported),
            "not_accurate": sum(not row[arm]["accuracy_solution"] for row in reported),
            "not_accurate_share": (sum(not row[arm]["accuracy_solution"] for row in reported) / len(reported)) if reported else None}


# ---------------------------------------------------------------- the expectations
def summarise(handed: list[dict[str, Any]], searched: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"handed": {}, "searched": {}}
    for method in HANDED_METHODS:
        out["handed"][f"{method}|all"] = tally_handed(handed, method)
        for noise in NOISE_LEVELS:
            out["handed"][f"{method}|{noise:g}"] = tally_handed([r for r in handed if float(r["world"]["noise"]) == noise], method)
    for arm in SEARCHED_ARMS:
        out["searched"][f"{arm}|all"] = tally_searched(searched, arm)
        for noise in NOISE_LEVELS:
            out["searched"][f"{arm}|{noise:g}"] = tally_searched([r for r in searched if float(r["noise"]) == noise], arm)
    def kept(rows: list[dict[str, Any]], noise_of: Any, quiet: bool, arm: str) -> list[int]:
        right = [row for row in rows if row.get("status") == "ok" and row["route"]["reported"] and row["route"]["symbolic"]
                 and (float(noise_of(row)) <= QUIET) == quiet]
        return [sum(bool(row[arm]["reported"] and row[arm]["symbolic"]) for row in right), len(right)]

    def holds(pair: list[int], share: float) -> bool:
        return pair[1] == 0 or pair[0] >= share * pair[1]

    h_noise, s_noise = (lambda row: row["world"]["noise"]), (lambda row: row["noise"])
    kept_h = {(arm, quiet): kept(handed, h_noise, quiet, arm) for arm in ("first", "second") for quiet in (True, False)}
    kept_s = {(arm, quiet): kept(searched, s_noise, quiet, arm) for arm in ("first", "second") for quiet in (True, False)}
    shares = {f"{noise:g}": out["handed"][f"second|{noise:g}"]["wrong_share"] for noise in NOISE_LEVELS}
    second_s = out["searched"]["second|all"]
    out["expectations"] = {
        "1_wrong_share_of_second_by_noise": shares, "1_limit": WRONG_LIMIT,
        "1_holds": all((share or 0.0) <= WRONG_LIMIT for share in shares.values()),
        "2_scores": {method: out["handed"][f"{method}|all"]["score"] for method in HANDED_METHODS},
        "2_holds": (out["handed"]["second|all"]["score"] or -1e9) > (out["handed"]["route|all"]["score"] or -1e9),
        "3_right_laws_of_the_route_kept_handed": {"up_to_1_percent": kept_h[("second", True)], "at_5_percent": kept_h[("second", False)]},
        "3_limits": {"up_to_1_percent": KEPT_SHARE, "at_5_percent": KEPT_SHARE_NOISY},
        "3_holds": holds(kept_h[("second", True)], KEPT_SHARE) and holds(kept_h[("second", False)], KEPT_SHARE_NOISY),
        "4_recovered_laws_of_the_route_kept_searched": {"up_to_1_percent": kept_s[("second", True)], "at_5_percent": kept_s[("second", False)]},
        "4_limits": {"up_to_1_percent": KEPT_SHARE, "at_5_percent": KEPT_SHARE_NOISY},
        "4_holds": holds(kept_s[("second", True)], KEPT_SHARE) and holds(kept_s[("second", False)], KEPT_SHARE_NOISY),
        "5_not_accurate_share_of_second_searched": second_s["not_accurate_share"], "5_limit": WRONG_LIMIT,
        "5_holds": (second_s["not_accurate_share"] or 0.0) <= WRONG_LIMIT,
        "first_version_for_comparison": {"kept_handed": {"up_to_1_percent": kept_h[("first", True)], "at_5_percent": kept_h[("first", False)]},
                                         "kept_searched": {"up_to_1_percent": kept_s[("first", True)], "at_5_percent": kept_s[("first", False)]}},
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("laws", "smoke", "run", "collect"))
    parser.add_argument("--part", choices=("handed", "searched"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "laws":
        text = json.dumps(build_laws(), indent=1, sort_keys=True) + "\n"
        if args.check:
            ok = LAWS_FILE.read_text(encoding="utf-8") == text
            print(f"{LAWS_FILE.relative_to(ROOT)}: {'as generated' if ok else 'not what the generator gives here'}")
            return 0 if ok else 1
        LAWS_FILE.write_text(text, encoding="utf-8")
        made = json.loads(text)
        print(f"wrote {LAWS_FILE.relative_to(ROOT)}: {len(made['handed']['laws'])} handed laws, "
              f"{len(made['handed']['cases'])} cases, {len(made['searched'])} searched laws")
        return 0
    if args.command == "smoke":      # a law that is in no benchmark: four handed cases, and one searched run with a short search
        rows = handed_rows(bench.smoke_cases(), red_team._cores())
        for row in rows:
            if row["status"] != "ok":
                print(row["status"], row.get("error"), row.get("trace", ""))
                return 1
            print(row["case"], {m: (row[m]["reported"], row[m]["symbolic"], row[m]["wrong"]) for m in HANDED_METHODS})
        smoke = {"id": 0, "inputs": 2, "text": "0.7*x0**2 + 1.3/x1"}
        row = run_searched(entry_of(smoke, [grammar.LOW, grammar.HIGH]), 0.01, budget=20.0)
        print("searched smoke:", row["status"], {arm: row.get(arm, {}).get("formula") for arm in SEARCHED_ARMS}, row.get("error", ""))
        return 0 if row["status"] == "ok" else 1
    if args.out is None:
        parser.error("--out is required")
    args.out.mkdir(parents=True, exist_ok=True)
    k, n = (int(v) for v in args.shard.split("/"))
    if args.command == "run":
        check_frozen()
        made = record()
        if args.part == "handed":
            mine = [case for i, case in enumerate(made["handed"]["cases"]) if i % n == k]
            rows = handed_rows(mine, red_team._cores())
        else:
            units = [(law, noise) for law in made["searched"] for noise in NOISE_LEVELS]
            rows = [run_searched(entry_of(law, made["domain"]), noise) for i, (law, noise) in enumerate(units) if i % n == k]
        path = args.out / f"{args.part}-shard-{k:03d}-of-{n:03d}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows), encoding="utf-8")
        return 0
    handed = [json.loads(line) for path in sorted(args.out.glob("handed-shard-*.jsonl")) for line in path.read_text().splitlines()]
    searched = [json.loads(line) for path in sorted(args.out.glob("searched-shard-*.jsonl")) for line in path.read_text().splitlines()]
    made = record()
    if len(handed) != len(made["handed"]["cases"]) or len(searched) != len(made["searched"]) * len(NOISE_LEVELS):
        raise SystemExit(f"expected {len(made['handed']['cases'])} handed and {len(made['searched']) * len(NOISE_LEVELS)} "
                         f"searched rows, found {len(handed)} and {len(searched)}")
    for name, rows in (("handed", handed), ("searched", searched)):
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    summary = summarise(handed, searched)
    (args.out / "results.json").write_text(json.dumps({"preregistration": PREREGISTRATION_ID, "frozen": frozen_digests(),
                                                       "summary": summary}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["expectations"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
