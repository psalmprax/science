#!/usr/bin/env python3
"""Planted laws in market-like series, for PREREG-MARKET-PLANTED-2026-11.

One run is one world from ``scripts/market_planted_worlds.py`` through
``apedp_dyn.discovery_l2.series_route`` with PySR as the search stage. Each run
gives two readings: what the main search path reports on its own ("ungated"),
and what the series route reports ("gated"): one candidate chosen on the
development rows, then the referee's forecast test on the later period. Laws
are scored on the last fifth of the series, which no stage reads.

    python scripts/run_market_planted_benchmark.py smoke   --out DIR   # development worlds, short budget
    python scripts/run_market_planted_benchmark.py run     --out DIR --shard k/N
    python scripts/run_market_planted_benchmark.py collect --out DIR
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

import market_planted_worlds as worlds  # noqa: E402
import run_srbench_baseline as base  # noqa: E402

PREREGISTRATION_ID = "PREREG-MARKET-PLANTED-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-MARKET-PLANTED.json"
FROZEN = ("preregistration/PREREG-MARKET-PLANTED-2026-11.md", "scripts/run_market_planted_benchmark.py",
          "scripts/market_planted_worlds.py", "apedp_dyn/discovery_l2/series_route.py",
          "apedp_dyn/discovery_l2/main_search.py", "apedp_dyn/proposers/pysr_proposer.py",
          "apedp_dyn/discovery_l2/contrast_confirmation.py", "apedp_dyn/discovery_l2/contrast_resolution.py",
          "apedp_dyn/discovery_l2/confirmed_stop.py", "apedp_dyn/discovery_l2/guided_search.py",
          "apedp_dyn/discovery_l2/exhaustive_search.py", "apedp_referee/novelty_adjudication.py")

NULL_WORLDS = 40
SEEDS_PER_CELL = 4
SEARCH_BUDGET_S = 240.0
TIME_LIMIT_S = 600.0
SEED = 20261119
SIGNAL_CAPTURED = 0.5  # a law "captures the signal" when it explains at least this share of the planted signal on the test rows


def run_route(world: dict[str, Any], budget: float = SEARCH_BUDGET_S) -> dict[str, Any]:
    from apedp_dyn.discovery_l2 import series_route
    from apedp_dyn.proposers.pysr_proposer import PySRProposer

    names = list(worlds.NAMES)
    X, y = world["X"], world["y"]
    result = series_route.discover_predictive_law(
        {n: X[world["dev"], i] for i, n in enumerate(names)}, y[world["dev"]], X[world["pool"]], y[world["pool"]],
        proposer=PySRProposer(budget_seconds=budget, seed=SEED), seed=SEED)
    selected = result.selected
    row: dict[str, Any] = {"action": result.action, "ungated_action": result.ungated_action, "law": result.law,
                           "ungated_law": result.ungated_law, "forecast_test": result.forecast_test,
                           "chosen": selected["chosen"], "no_law_validation_mse": selected["no_law_validation_mse"],
                           "validation": selected["candidates"],
                           "found_by": result.path.found_by, "first_pass": result.path.first_pass, "search": result.path.search,
                           "candidates": len(result.path.candidates), "rows_read": result.path.rows_read,
                           "contrast_verdict": result.path.contrast.verdict if result.path.contrast else None,
                           "adequacy": result.path.adequacy, "rejected_law": result.path.rejected_law}
    test_X, test_y, signal = X[world["test"]], y[world["test"]], world["signal"][world["test"]]
    dev_mean = float(y[world["dev"]].mean())
    for reading, text in (("ungated", result.ungated_law), ("gated", result.law)):
        if text is None:
            continue
        model = sp.sympify(text, locals={n: sp.Symbol(n) for n in names})
        pred = series_route.predict(model, names, test_X)
        finite = bool(np.all(np.isfinite(pred)))
        scored: dict[str, Any] = {
            "test_r2_vs_no_law": float(1.0 - np.sum((test_y - pred) ** 2) / np.sum((test_y - dev_mean) ** 2)) if finite else None}
        if world["truth"] is not None:
            centred = signal - signal.mean()
            scored["signal_r2"] = float(1.0 - np.sum((pred - pred.mean() - centred) ** 2) / np.sum(centred ** 2)) if finite else None
            scored["captures_signal"] = bool(scored["signal_r2"] is not None and scored["signal_r2"] >= SIGNAL_CAPTURED)
            try:
                sym = base.symbolic_recovery(model, world["truth"], names, test_X)
            except Exception as exc:  # noqa: BLE001 - a model the scorer cannot handle is scored as not recovered, with the reason
                sym = {"recovered": False, "reason": f"scoring failed: {type(exc).__name__}"}
            scored["symbolic"], scored["symbolic_note"] = bool(sym["recovered"]), sym["reason"]
        row[reading] = scored
    return row


def _child(conn: Any, world: dict[str, Any], budget: float) -> None:
    try:
        conn.send(("ok", run_route(world, budget)))
    except Exception as exc:  # noqa: BLE001 - a crash is a scored outcome, reported with its class and message
        conn.send(("error", {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-1500:]}))
    finally:
        conn.close()


def run_one(job: dict[str, Any], budget: float = SEARCH_BUDGET_S, time_limit: float = TIME_LIMIT_S, tag: str | None = None) -> dict[str, Any]:
    planted = job["planted"]
    world = worlds.world(planted, job["share"], job["seed"], tag) if tag else worlds.world(planted, job["share"], job["seed"])
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    started = time.perf_counter()
    proc = ctx.Process(target=_child, args=(child, world, budget))
    proc.start()
    child.close()
    outcome: tuple[str, dict[str, Any]] = ("timeout", {})
    if parent.poll(time_limit):
        try:
            outcome = parent.recv()
        except EOFError:
            outcome = ("error", {"error": "process died without a result"})
    proc.join(1.0)
    if proc.is_alive():
        proc.kill()
        proc.join()
    return {**job, "truth": worlds.LAWS.get(planted) if planted else None, "status": outcome[0],
            "seconds": round(time.perf_counter() - started, 2), **outcome[1]}


def all_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = [{"planted": None, "share": 0.0, "seed": seed} for seed in range(NULL_WORLDS)]
    jobs += [{"planted": law, "share": share, "seed": seed} for law in worlds.LAWS for share in worlds.SHARES for seed in range(SEEDS_PER_CELL)]
    return jobs


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def cell(sel: list[dict[str, Any]]) -> dict[str, Any]:
        ok = [r for r in sel if r["status"] == "ok"]
        out: dict[str, Any] = {"n": len(sel), "errors": sum(r["status"] == "error" for r in sel),
                               "timeouts": sum(r["status"] == "timeout" for r in sel)}
        for reading, key in (("ungated", "ungated_law"), ("gated", "law")):
            reported = [r for r in ok if r.get(key)]
            out[reading] = {"reported": len(reported),
                            "captures_signal": sum(bool(r[reading].get("captures_signal")) for r in reported),
                            "symbolic": sum(bool(r[reading].get("symbolic")) for r in reported),
                            "worse_than_no_law_on_test": sum((r[reading].get("test_r2_vs_no_law") or 0.0) < 0.0 for r in reported)}
        out["earned_a_test"] = sum(bool(r.get("chosen")) for r in ok)
        return out

    summary = {"null": cell([r for r in rows if r["planted"] is None])}
    for share in worlds.SHARES:
        summary[f"share_{share}"] = cell([r for r in rows if r["planted"] is not None and r["share"] == share])
    for law in worlds.LAWS:
        summary[f"law_{law}"] = cell([r for r in rows if r["planted"] == law])
    summary["planted"] = cell([r for r in rows if r["planted"] is not None])
    return summary


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def check_frozen() -> None:
    if not DIGEST.exists() or json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} is not frozen, or changed since it was frozen")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("run", "collect", "smoke", "develop"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1", help="k/N: this shard's runs")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "smoke":
        budget = float(os.environ.get("SMOKE_BUDGET_SECONDS", "30"))
        failed = 0
        for job in ({"planted": None, "share": 0.0, "seed": 0}, {"planted": "trend_with_flow", "share": 0.05, "seed": 0},
                    {"planted": "trend_fading_in_volatility", "share": 0.2, "seed": 1}):
            row = run_one(job, budget, tag="smoke")
            print(json.dumps({k: v for k, v in row.items() if k not in ("trace", "first_pass")}, default=str), flush=True)
            if row["status"] != "ok":
                print(row.get("trace", ""), flush=True)
                failed += 1
        return 1 if failed else 0
    if args.command == "develop":  # development worlds (other seeds than the run's), for looking at behaviour before freezing
        budget = float(os.environ.get("SMOKE_BUDGET_SECONDS", "60"))
        k, n = (int(v) for v in args.shard.split("/"))
        jobs = [{"planted": None, "share": 0.0, "seed": seed} for seed in range(4)]
        jobs += [{"planted": law, "share": share, "seed": seed} for law in worlds.LAWS for share in (0.05, 0.2) for seed in range(2)]
        for index, job in enumerate(jobs):
            if index % n == k:
                row = run_one(job, budget, tag="dev")
                print("DEVELOP " + json.dumps({key: v for key, v in row.items() if key not in ("trace", "first_pass")}, default=str), flush=True)
        return 0
    check_frozen()
    if args.command == "run":
        k, n = (int(v) for v in args.shard.split("/"))
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for index, job in enumerate(all_jobs()):
                if index % n == k:
                    handle.write(json.dumps(run_one(job), sort_keys=True, default=str) + "\n")
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
