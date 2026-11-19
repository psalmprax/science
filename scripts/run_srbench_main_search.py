"""PREREG-MAIN-SEARCH-2026-11: the main search path, and the in-house GA/GP, on the SRBench protocol.

    python scripts/run_srbench_main_search.py run --shard 3/32 --out srbench-out
    python scripts/run_srbench_main_search.py collect --out srbench-out
    python scripts/run_srbench_main_search.py smoke --out smoke-out

Same problems, splits and noise as PREREG-SRBENCH-BASELINE (its harness builds
them), noise 0 and 1 %. Two engines:

- ``main_search``: ``apedp_dyn.discovery_l2.main_search.discover`` with PySR as
  the proposer. 70 % of the training rows are development rows; the other 30 %
  are the pool whose targets only the confirmation step may read.
- ``gagp``: ``apedp.evolutionary_search.evolve`` on the native kernel (strict:
  it refuses the Python fallback), all training rows, a configuration sized to
  about the same wall-clock time as PySR's budget.

``run`` and ``collect`` refuse to work if a frozen file changed. ``smoke`` runs
both engines on three problems with a short budget and is not part of the result.
Protocol and expectations: ``preregistration/PREREG-MAIN-SEARCH-2026-11.md``.
"""
from __future__ import annotations

import argparse
import hashlib
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

import fetch_srbench_ground_truth as srb  # noqa: E402
import run_srbench_baseline as base  # noqa: E402

PREREGISTRATION_ID = "PREREG-MAIN-SEARCH-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-MAIN-SEARCH.json"
FROZEN = ("preregistration/PREREG-MAIN-SEARCH-2026-11.md", "scripts/run_srbench_main_search.py",
          "scripts/run_srbench_baseline.py", "scripts/fetch_srbench_ground_truth.py", "benchmarks/srbench/manifest.json",
          "apedp_dyn/discovery_l2/main_search.py", "apedp_dyn/proposers/pysr_proposer.py",
          "apedp_dyn/discovery_l2/confirmed_stop.py", "apedp_dyn/discovery_l2/guided_search.py",
          "apedp_dyn/discovery_l2/exhaustive_search.py", "apedp_dyn/discovery_l2/contrast_confirmation.py",
          "apedp_dyn/discovery_l2/contrast_resolution.py", "apedp/evolutionary_search.py")
ENGINES = ("main_search", "gagp")
NOISE_LEVELS = (0.0, 0.01)
SEARCH_BUDGET_S = 240.0  # PySR's budget inside the path, as in PREREG-SRBENCH-EXTERNAL
TIME_LIMIT_S = 480.0     # when a run's process is killed
DEV_SHARE = 0.7
GA_GENERATIONS, GA_ISLANDS = 1500, 8
REPORTED = ("confirmed", "equivalent_class", "unchallenged")
_GA_NAMES = {"min": sp.Min, "max": sp.Max, "abs": sp.Abs, "cbrt": sp.cbrt, "pi": sp.pi, **base.FUNCTIONS}


def _predict(model: sp.Expr, names: list[str], X: np.ndarray) -> np.ndarray:
    fn = sp.lambdify([sp.Symbol(n) for n in names], model, "numpy", dummify=True)
    with np.errstate(all="ignore"):
        out: np.ndarray = np.asarray(fn(*X.T), dtype=float) * np.ones(len(X))
    return out


def run_main_search(split: dict[str, Any], entry: dict[str, Any], budget: float = SEARCH_BUDGET_S) -> dict[str, Any]:
    from apedp_dyn.discovery_l2 import main_search
    from apedp_dyn.proposers.pysr_proposer import PySRProposer

    names, X, y = split["names"], split["X_train"], split["y_train"]
    key = hashlib.sha256(f"{PREREGISTRATION_ID}|{entry['name']}|pool".encode()).digest()[:8]
    order = np.random.default_rng(int.from_bytes(key, "big")).permutation(len(y))
    cut = round(DEV_SHARE * len(y))
    dev, pool = order[:cut], order[cut:]
    pool_y = y[pool]
    result = main_search.discover(
        {n: X[dev, i] for i, n in enumerate(names)}, y[dev], X[pool], lambda rows: pool_y[rows],
        proposer=PySRProposer(budget_seconds=budget, seed=base.SEED), noise="absolute", seed=base.SEED)
    law = sp.sympify(result.law, locals={n: sp.Symbol(n) for n in names}) if result.law else None
    extra = {"action": result.action, "found_by": result.found_by, "rejected_law": result.rejected_law,
             "adequacy": result.adequacy, "rows_read": result.rows_read, "first_pass": result.first_pass,
             "search": result.search, "candidates": result.candidates,
             "equivalent": len(result.contrast.equivalent) if result.contrast else 0}
    return {"model": law, "pred": _predict(law, names, split["X_test"]) if law is not None else None,
            "formula": result.law, "extra": extra}


def run_gagp(split: dict[str, Any], entry: dict[str, Any], budget: float = SEARCH_BUDGET_S) -> dict[str, Any]:
    from apedp.evolutionary_search import SearchConfig, evolve

    names = split["names"]
    generations = GA_GENERATIONS if budget >= SEARCH_BUDGET_S else max(25, int(GA_GENERATIONS * budget / SEARCH_BUDGET_S))
    result = evolve({n: split["X_train"][:, i] for i, n in enumerate(names)}, split["y_train"],
                    config=SearchConfig(use_native_ga=True, strict_native_ga=True, seed=base.SEED,
                                        generations=generations, islands=GA_ISLANDS, allow_inv=True))
    text = result.denormalized_pick or result.grouped_pick
    if not text:
        return {"model": None, "pred": None, "formula": None, "extra": {"evaluations": result.n_evaluations}}
    expr = sp.sympify(text, locals={**_GA_NAMES, **{n: sp.Symbol(n) for n in names}})
    return {"model": expr, "pred": _predict(expr, names, split["X_test"]), "formula": text,
            "extra": {"evaluations": result.n_evaluations, "generations": result.generations_run}}


RUNNERS = {"main_search": run_main_search, "gagp": run_gagp}


def _child(conn: Any, engine: str, entry: dict[str, Any], split: dict[str, Any], budget: float) -> None:
    try:
        raw = RUNNERS[engine](split, entry, budget)
        try:
            row = base.score_run(engine, entry, split, raw)
        except Exception as exc:  # noqa: BLE001 - a model the scorer cannot handle is scored as not recovered, with the reason
            r2 = base._r2(split["y_test"], raw["pred"]) if raw.get("pred") is not None else None
            row = {"symbolic": False, "symbolic_rel_rmse": None, "symbolic_note": f"scoring failed: {type(exc).__name__}",
                   "test_r2": r2, "accuracy_solution": bool(r2 is not None and r2 >= base.R2_ACCURACY),
                   "model": str(raw.get("model")), "formula": raw.get("formula")}
        conn.send(("ok", {**row, **raw.get("extra", {})}))
    except Exception as exc:  # noqa: BLE001 - an engine crash is a scored outcome, reported with its class and message
        conn.send(("error", {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-1500:]}))
    finally:
        conn.close()


def run_one(job: dict[str, Any], entries: dict[str, dict[str, Any]], budget: float = SEARCH_BUDGET_S,
            time_limit: float = TIME_LIMIT_S) -> dict[str, Any]:
    entry = entries[job["problem"]]
    split = base.make_split(entry, srb.load_dataset(entry), job["noise"])
    head = {**job, "family": entry["family"], "n_train": len(split["y_train"]), "n_test": len(split["y_test"])}
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    started = time.perf_counter()
    proc = ctx.Process(target=_child, args=(child, job["engine"], entry, split, budget))
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


def all_jobs() -> list[dict[str, Any]]:
    return [{"problem": p["name"], "noise": noise, "seed": base.SEED, "engine": engine}
            for p in base.load_manifest() for noise in NOISE_LEVELS for engine in ENGINES]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for engine in ENGINES:
        for family in ("feynman", "strogatz", "all"):
            for noise in (*NOISE_LEVELS, "all"):
                sel = [r for r in rows if r["engine"] == engine and family in (r["family"], "all") and noise in (r["noise"], "all")]
                ok = [r for r in sel if r["status"] == "ok"]
                reported = [r for r in ok if r.get("model") not in (None, "None")]
                cell: dict[str, Any] = {
                    "n": len(sel), "errors": sum(r["status"] == "error" for r in sel),
                    "timeouts": sum(r["status"] == "timeout" for r in sel),
                    "median_seconds": float(np.median([r["seconds"] for r in sel])) if sel else None,
                    "symbolic": sum(bool(r["symbolic"]) for r in ok),
                    "accuracy_solution": sum(bool(r["accuracy_solution"]) for r in ok),
                }
                if engine == "main_search":
                    cell.update({
                        "reported": len(reported),
                        "reported_symbolic": sum(bool(r["symbolic"]) for r in reported),
                        "reported_accurate": sum(bool(r["accuracy_solution"]) for r in reported),
                        "reported_not_accurate": sum(not r["accuracy_solution"] for r in reported),
                        "actions": {a: sum(r.get("action") == a for r in ok) for a in (*REPORTED, "widen_search")},
                        "found_by": {s: sum(r.get("found_by") == s for r in reported) for s in ("first_pass", "search")},
                        "search_ran": sum(bool(r.get("search", {}).get("ran")) for r in ok),
                    })
                out[f"{engine}|{family}|{noise}"] = cell
    return out


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def check_frozen() -> None:
    if json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} changed since it was frozen")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("run", "collect", "smoke"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1", help="k/N: this shard's problems")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    entries = {p["name"]: p for p in base.load_manifest()}
    if args.command == "smoke":
        budget = float(os.environ.get("SMOKE_BUDGET_SECONDS", "30"))
        failed = 0
        for problem, noise in (("feynman_I_12_1", 0.01), ("strogatz_bacres1", 0.01), ("feynman_I_12_2", 0.0)):
            for engine in ENGINES:
                row = run_one({"problem": problem, "noise": noise, "seed": base.SEED, "engine": engine}, entries, budget)
                print(json.dumps({k: v for k, v in row.items() if k not in ("candidates", "trace")}, default=str), flush=True)
                if row["status"] != "ok":
                    print(row.get("trace", ""), flush=True)
                    failed += 1
        return 1 if failed else 0
    check_frozen()
    if args.command == "run":
        k, n = (int(v) for v in args.shard.split("/"))
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for job in base.shard(all_jobs(), k, n):
                handle.write(json.dumps(run_one(job, entries), sort_keys=True, default=str) + "\n")
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
