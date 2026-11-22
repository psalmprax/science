#!/usr/bin/env python3
"""Does a memory of confirmed laws help the default route on laws it has never seen?

    python scripts/run_memory_laws.py laws [--check]              # benchmarks/memory_blind_laws.json
    python scripts/run_memory_laws.py smoke   --out DIR           # two laws, a short real search
    python scripts/run_memory_laws.py run     --out DIR --shard k/N   # PREREG-MEMORY-2026-11, once
    python scripts/run_memory_laws.py collect --out DIR

Two arms on every law and noise level, **given the same searched candidates**: the default stage's
searches run once, and

- ``default`` hands the path those candidates (the route in force);
- ``memory`` hands it the same candidates with laws grown from the memory beside them
  (``discovery_l2/memory_stage.py``, ``benchmarks/structure_memory.json``).

So the arms differ in the memory and in nothing a second search happened to find.

The laws are drawn here, after the memory file was written, from the invented-law grammar with
their own seed; they leave out every structure of the invented-laws benchmark (by fingerprint,
without reading a held-back law) and of the eighty laws the memory proposer was developed on
(``run_search_evolution.draw_laws``). Rows, noise, the split and the scoring are the invented-laws
runner's and through it the SRBench runner's.
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
import run_search_evolution as development  # noqa: E402
import run_srbench_baseline as base  # noqa: E402
import run_srbench_main_search as path_run  # noqa: E402

from apedp_dyn.conf.canonical_spec import REFEREE_MAX_FPR  # noqa: E402

PREREGISTRATION_ID = "PREREG-MEMORY-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-MEMORY.json"
LAWS_FILE = ROOT / "benchmarks" / "memory_blind_laws.json"
FROZEN = (
    "preregistration/PREREG-MEMORY-2026-11.md", "preregistration/FORECAST-MEMORY-2026-11.json",
    "scripts/run_memory_laws.py", "benchmarks/memory_blind_laws.json", "benchmarks/structure_memory.json",
    "scripts/build_structure_memory.py", "scripts/build_invented_laws.py", "scripts/run_srbench_baseline.py",
    "scripts/run_srbench_main_search.py",
    "apedp_dyn/proposers/structure_memory.py", "apedp_dyn/discovery_l2/memory_stage.py",
    "apedp_dyn/discovery_l2/evolved_stage.py", "apedp_dyn/discovery_l2/search_stage.py",
    "apedp_dyn/discovery_l2/refined_search.py", "apedp_dyn/discovery_l2/main_search.py",
    "apedp_dyn/ode/extended_route.py", "apedp_dyn/proposers/fit_guard.py", "apedp_dyn/proposers/portfolio.py",
    "apedp_dyn/proposers/pysr_proposer.py",
)
LAW_SEED = 20261127
LAW_COUNT = 40
ARMS = ("default", "memory")
NOISE_LEVELS = (0.0, 0.01)
WRONG_LIMIT = REFEREE_MAX_FPR            # expectation 1: the share of reported laws that are wrong
SEARCH_BUDGET_S = 240.0
MEMORY_S = 60.0
TIME_LIMIT_S = 1500.0                    # one law at one noise level: the searches once, then both arms
#: two laws that are in no set, for the smoke run
SMOKE_LAWS = tuple({"name": name, "family": "invented", "features": features, "formula": formula, "units": {},
                    "target_units": {}, "domain": [grammar.LOW, grammar.HIGH]}
                   for name, features, formula in (("smoke_sum", ["x0", "x1"], "0.7*x0**2 + 1.3/x1"),
                                                   ("smoke_product", ["x0"], "1.7*x0*sin(1.3*x0)")))


# ---------------------------------------------------------------- laws
def build_laws(count: int = LAW_COUNT, seed: int = LAW_SEED) -> dict[str, Any]:
    """``count`` laws of the invented-law grammar that are in neither the benchmark nor the development set."""
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    developed_on = {grammar.structure(sp.sympify(law["text"])) for law in development.draw_laws(development.TRAIN + development.VALIDATION)}
    rng = np.random.default_rng(seed)
    laws: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(grammar.MAX_DRAWS):
        if len(laws) == count:
            break
        inputs = int(rng.integers(1, 4))
        expr = grammar.draw(rng, inputs)
        key = grammar.structure(expr)
        if key in seen or not grammar.acceptable(expr, inputs, rng):
            continue
        seen.add(key)
        if grammar.fingerprint(str(expr)) in in_benchmark or key in developed_on:
            continue
        laws.append({"id": 4000 + len(laws), "inputs": inputs, "text": str(expr)})
    if len(laws) != count:
        raise SystemExit(f"only {len(laws)} laws after {grammar.MAX_DRAWS} draws")
    return {"what": "Laws for one preregistered comparison of the default route with and without a memory of confirmed "
                    "laws. Drawn after the memory was written; no route has run on them.",
            "seed": seed, "domain": [grammar.LOW, grammar.HIGH], "laws": laws}


def entries() -> dict[str, dict[str, Any]]:
    """The laws in the SRBench runs' form."""
    record = json.loads(LAWS_FILE.read_text(encoding="utf-8"))
    out = {}
    for law in record["laws"]:
        name = f"memory_{law['id']}"
        out[name] = {"name": name, "family": "invented", "features": [f"x{i}" for i in range(law["inputs"])],
                     "formula": law["text"], "units": {}, "target_units": {}, "domain": record["domain"]}
    return out


def frame_of(entry: dict[str, Any]) -> pd.DataFrame:
    """Rows drawn uniformly on the law's domain, with the law's own value as the target (no noise yet)."""
    low, high = entry["domain"]
    rng = np.random.default_rng(base._seed(PREREGISTRATION_ID, entry["name"], "rows"))
    X = rng.uniform(low, high, (base.N_TRAIN + base.N_TEST, len(entry["features"])))
    truth = base.truth_expr(entry)
    fn = sp.lambdify([sp.Symbol(f"x{i}") for i in range(X.shape[1])], truth, "numpy")
    frame = pd.DataFrame(X, columns=entry["features"])
    frame["target"] = np.asarray(fn(*X.T), dtype=float)
    return frame


# ---------------------------------------------------------------- one law at one noise level
def arms_of(entry: dict[str, Any], split: dict[str, Any], budget: float, memory_seconds: float) -> list[dict[str, Any]]:
    """Both arms on one law: a row each. The searches run once, in the default arm."""
    from apedp_dyn.discovery_l2 import memory_stage, refined_search, search_stage
    from apedp_dyn.ode.extended_route import Recording, replayed

    names, X, y = split["names"], split["X_train"], split["y_train"]
    order = np.random.default_rng(base._seed(PREREGISTRATION_ID, entry["name"], "pool")).permutation(len(y))
    cut = round(path_run.DEV_SHARE * len(y))
    dev, pool = order[:cut], order[cut:]
    pool_y = y[pool]
    truth = base.truth_expr(entry)
    memory = memory_stage.recorded_memory()
    rows: list[dict[str, Any]] = []
    searched: list[str] = []
    for arm in ARMS:
        started = time.perf_counter()
        if arm == "default":
            stage: Any = Recording(search_stage.default_search_stage(budget, seed=base.SEED)[0]) if budget > 0 else None
        else:
            stage = memory_stage.with_memory(replayed(searched), memory, seconds=memory_seconds)
        result = refined_search.discover_refined({n: X[dev, i] for i, n in enumerate(names)}, y[dev], X[pool],
                                                 lambda picked: pool_y[picked], proposer=stage, noise="absolute", seed=base.SEED)
        if arm == "default":
            searched = list(stage.formulas) if stage is not None else []
        law = sp.sympify(result.law, locals={n: sp.Symbol(n) for n in names}) if result.law else None
        path = result.path
        assert path is not None
        raw = {"model": law, "pred": path_run._predict(law, names, split["X_test"]) if law is not None else None,
               "formula": result.law}
        extra: dict[str, Any] = {"action": result.action, "tentative_law": result.tentative_law, "found_by": path.found_by,
                                 "rejected_law": path.rejected_law, "adequacy": path.adequacy, "rows_read": path.rows_read,
                                 "candidates": path.candidates, "searched": searched, "dropped_terms": result.dropped_terms,
                                 "proposer_failures": result.proposer_failures}
        if arm == "memory":
            grown = dict(getattr(stage, "last_run", {}) or {})
            texts = [str(c["formula"]) for c in path.candidates if c["source"] == "search"]
            mine = [t for t in texts if t not in set(searched)]
            extra["memory"] = {**grown.get("generator", grown), "handed_over": len(mine),
                               "the_law_among_them": any(base.symbolic_recovery(
                                   sp.sympify(t, locals={n: sp.Symbol(n) for n in names}), truth, names, split["X_test"])["recovered"]
                                   for t in mine)}
        rows.append({"arm": arm, "status": "ok", "seconds": round(time.perf_counter() - started, 2),
                     **base.score_run("memory", entry, split, raw), **extra})
    return rows


def _child(conn: Any, entry: dict[str, Any], split: dict[str, Any], budget: float, memory_seconds: float) -> None:
    try:
        conn.send(("ok", arms_of(entry, split, budget, memory_seconds)))
    except Exception as exc:  # noqa: BLE001 - a crash of the unit is a scored outcome, reported with its class and message
        conn.send(("error", {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-2000:]}))
    finally:
        conn.close()


def run_unit(entry: dict[str, Any], noise: float, budget: float = SEARCH_BUDGET_S, memory_seconds: float = MEMORY_S,
             time_limit: float = TIME_LIMIT_S) -> list[dict[str, Any]]:
    """One law at one noise level: a row per arm. A crash or a timeout gives both arms that status."""
    head = {"problem": entry["name"], "noise": noise, "inputs": len(entry["features"]), "truth": entry["formula"]}
    split = base.make_split(entry, frame_of(entry), noise)
    context = mp.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    started = time.perf_counter()
    process = context.Process(target=_child, args=(child, entry, split, budget, memory_seconds))
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
    return [{**head, "arm": arm, "status": outcome[0], "seconds": seconds, **outcome[1]} for arm in ARMS]


def units() -> list[tuple[dict[str, Any], float]]:
    return [(entry, noise) for entry in entries().values() for noise in NOISE_LEVELS]


# ---------------------------------------------------------------- summary
def _reported(row: dict[str, Any]) -> bool:
    return row["status"] == "ok" and row.get("model") not in (None, "None")


def _wrong(row: dict[str, Any]) -> bool:
    return _reported(row) and not row.get("symbolic") and not row.get("accuracy_solution")


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per arm and noise level: runs, reports, wrong laws, recoveries; then the expectations as written."""
    out: dict[str, Any] = {}
    for arm in ARMS:
        for noise in (*NOISE_LEVELS, "all"):
            sel = [r for r in rows if r["arm"] == arm and noise in (r["noise"], "all")]
            ok = [r for r in sel if r["status"] == "ok"]
            reported = [r for r in ok if _reported(r)]
            grown = [r.get("memory") or {} for r in ok]
            out[f"{arm}|{noise}"] = {
                "runs": len(sel), "errors": sum(r["status"] == "error" for r in sel),
                "timeouts": sum(r["status"] == "timeout" for r in sel), "reported": len(reported),
                "recovered": sum(bool(r.get("symbolic")) for r in ok), "wrong": sum(_wrong(r) for r in ok),
                "reported_accurate_not_the_law": sum(bool(r.get("accuracy_solution")) and not r.get("symbolic") for r in reported),
                "memory_handed_over_the_law": sum(bool(g.get("the_law_among_them")) for g in grown),
                "median_seconds": float(np.median([r["seconds"] for r in sel])) if sel else None,
            }
    recovered = {(r["arm"], r["noise"], r["problem"]) for r in rows if r["status"] == "ok" and r.get("symbolic")}
    cells = sorted({(r["noise"], r["problem"]) for r in rows})
    out["run_for_run"] = {
        f"memory against default|{noise}": {
            "gained": sum(("memory", *c) in recovered and ("default", *c) not in recovered for c in cells if c[0] == noise),
            "lost": sum(("memory", *c) not in recovered and ("default", *c) in recovered for c in cells if c[0] == noise)}
        for noise in NOISE_LEVELS}

    def count(arm: str, noise: Any, key: str) -> int:
        return int(out.get(f"{arm}|{noise}", {}).get(key, 0))

    def share_wrong(arm: str) -> float:
        return count(arm, "all", "wrong") / max(count(arm, "all", "reported"), 1)

    out["expectations"] = {
        "1_share_of_reported_laws_wrong": {arm: share_wrong(arm) for arm in ARMS},
        "1_limit": WRONG_LIMIT,
        "1_holds": all(share_wrong(arm) <= WRONG_LIMIT for arm in ARMS),
        "2_recovered_memory": {str(n): count("memory", n, "recovered") for n in NOISE_LEVELS},
        "2_recovered_default": {str(n): count("default", n, "recovered") for n in NOISE_LEVELS},
        "2_holds": all(count("memory", n, "recovered") >= count("default", n, "recovered") for n in NOISE_LEVELS)
        and count("memory", "all", "recovered") > count("default", "all", "recovered"),
    }
    return out


# ---------------------------------------------------------------- command line
def frozen_digests() -> dict[str, str]:
    return {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in FROZEN}


def check_frozen() -> None:
    recorded = json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"]
    now = frozen_digests()
    changed = sorted(rel for rel in FROZEN if recorded.get(rel) != now[rel])
    if changed or set(recorded) != set(FROZEN):
        raise SystemExit(f"frozen files differ from {DIGEST.name}: {changed or sorted(set(recorded) ^ set(FROZEN))}")


def _printable(row: dict[str, Any]) -> str:
    keep = ("problem", "noise", "arm", "status", "symbolic", "accuracy_solution", "action", "formula", "seconds", "error")
    return json.dumps({k: row.get(k) for k in keep}, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("laws", "smoke", "run", "collect"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--check", action="store_true", help="for laws: compare the file with the generator")
    parser.add_argument("--budget", type=float, default=20.0, help="for smoke: seconds of search (0: no search engine)")
    args = parser.parse_args(argv)
    if args.command == "laws":
        text = json.dumps(build_laws(), indent=1, sort_keys=True) + "\n"
        if args.check:
            if LAWS_FILE.read_text(encoding="utf-8") != text:
                print(f"{LAWS_FILE.relative_to(ROOT)} is not what the generator gives")
                return 1
            print(f"{LAWS_FILE.relative_to(ROOT)}: as generated")
            return 0
        LAWS_FILE.write_text(text, encoding="utf-8")
        print(f"wrote {LAWS_FILE.relative_to(ROOT)}: {LAW_COUNT} laws")
        return 0
    if args.out is None:
        parser.error("--out is required")
    args.out.mkdir(parents=True, exist_ok=True)
    k, n = (int(v) for v in args.shard.split("/"))
    if args.command == "smoke":
        failed = 0
        for entry in SMOKE_LAWS:
            for row in run_unit(entry, 0.01, budget=args.budget, memory_seconds=20.0, time_limit=900.0):
                print(_printable(row), flush=True)
                if row["status"] != "ok":
                    print(row.get("trace", ""), flush=True)
                    failed += 1
        return 1 if failed else 0
    if args.command == "run":
        check_frozen()
        mine = [unit for i, unit in enumerate(units()) if i % n == k]
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for entry, noise in mine:
                for row in run_unit(entry, noise):
                    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                handle.flush()
        return 0
    rows = [json.loads(line) for path in sorted(args.out.glob("runs-shard-*.jsonl")) for line in path.read_text().splitlines()]
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    summary = summarise(rows)
    record = {"preregistration": PREREGISTRATION_ID, "rows": len(rows), "summary": summary, "frozen": frozen_digests()}
    (args.out / "results.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"collected {len(rows)} rows into {args.out}")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
