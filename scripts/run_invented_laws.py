#!/usr/bin/env python3
"""The agentic variation operator against the default search stage, on laws nobody has published.

Development harness: **not a preregistered run.** It exists to find out, on the twenty development
laws of ``benchmarks/invented_laws.json``, whether a real model can drive the agent loop and what
that costs in time, so that a preregistered comparison on the forty held-back laws can be written
with its settings fixed. Nothing here reads a held-back law.

    python scripts/run_invented_laws.py smoke   --out DIR          # three development laws, short budget
    python scripts/run_invented_laws.py develop --out DIR --shard k/N
    python scripts/run_invented_laws.py collect --out DIR

Rows, split, pool, scoring and the child process are the SRBench runs' (``run_srbench_baseline``,
``run_srbench_main_search``): 1,000 training rows and 500 test rows drawn on the laws' domain,
noise on the training targets, 70 % of the training rows for development and the rest as the pool,
recovery by the benchmark's own criterion. The laws have no units, so nothing is reduced.

Arms, each through ``discovery_l2.refined_search.discover_refined``:

- ``default``: ``search_stage.default_search_stage`` with ``SEARCH_BUDGET_S`` of search;
- ``agentic``: ``agentic_stage.agentic_search_stage``: the same searches for the same budget in
  ``ROUNDS`` rounds, with an agent session after each (needs ``NVIDIA_API_KEY``);
- ``agent_alone``: the agent with no search beside it.

The agent's sessions cost time on top of the search budget (the model's latency, and the spacing
the free service's rate limit imposes). Every row records its seconds, so a comparison at equal
wall time can be designed from what this measures.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
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

import run_srbench_baseline as base  # noqa: E402
import run_srbench_main_search as path_run  # noqa: E402

LAWS_FILE = ROOT / "benchmarks" / "invented_laws.json"
TAG = "DEV-INVENTED-AGENTIC-2026-11"
ARMS = ("default", "agentic", "agent_alone")
NOISE_LEVELS = (0.0, 0.01)
SEARCH_BUDGET_S = 240.0
ROUNDS = 2
STEPS = 6
SESSION_S = 150.0          # one agent session may take this long
TIME_LIMIT_S = 1200.0      # when a run's process is killed
MODEL = os.environ.get("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
THINKING, REASONING_BUDGET = True, 2000
INTERVAL_S = float(os.environ.get("LLM_INTERVAL_SECONDS", "6.0"))   # spacing of requests; the free tier is rate limited


def entries(split: str = "development") -> dict[str, dict[str, Any]]:
    """The laws of one split in the SRBench runs' form: name, features, formula; no units."""
    record = json.loads(LAWS_FILE.read_text(encoding="utf-8"))
    out = {}
    for law in record["laws"]:
        if law["split"] != split:
            continue
        name = f"invented_{law['id']}"
        out[name] = {"name": name, "family": "invented", "features": [f"x{i}" for i in range(law["inputs"])],
                     "formula": law["text"], "units": {}, "target_units": {}, "domain": record["domain"]}
    return out


def frame_of(entry: dict[str, Any]) -> pd.DataFrame:
    """Rows drawn uniformly on the law's domain, with the law's own value as the target (no noise yet)."""
    low, high = entry["domain"]
    rng = np.random.default_rng(base._seed(TAG, entry["name"], "rows"))
    X = rng.uniform(low, high, (base.N_TRAIN + base.N_TEST, len(entry["features"])))
    truth = base.truth_expr(entry)
    fn = sp.lambdify([sp.Symbol(f"x{i}") for i in range(X.shape[1])], truth, "numpy")
    frame = pd.DataFrame(X, columns=entry["features"])
    frame["target"] = np.asarray(fn(*X.T), dtype=float)
    return frame


def _client() -> Any:
    from apedp_dyn.proposers import llm_https

    return llm_https.nvidia_client(MODEL, thinking=THINKING, reasoning_budget=REASONING_BUDGET, min_interval=INTERVAL_S,
                                   retries=3)


def run_arm(arm: str, split: dict[str, Any], entry: dict[str, Any], budget: float = SEARCH_BUDGET_S) -> dict[str, Any]:
    from apedp_dyn.discovery_l2 import agentic_stage, refined_search, search_stage
    from apedp_dyn.proposers.agentic_variation import AgenticVariationProposer
    from apedp_dyn.proposers.fit_guard import FitGuardProposer

    names, X, y = split["names"], split["X_train"], split["y_train"]
    order = np.random.default_rng(base._seed(TAG, entry["name"], "pool")).permutation(len(y))
    cut = round(path_run.DEV_SHARE * len(y))
    dev, pool = order[:cut], order[cut:]
    pool_y = y[pool]
    client = None
    if arm == "default":
        stage, described = search_stage.default_search_stage(budget, seed=base.SEED)
    elif arm == "agentic":
        client = _client()
        stage, described = agentic_stage.agentic_search_stage(client, budget, seed=base.SEED, rounds=ROUNDS, steps=STEPS,
                                                             session_seconds=SESSION_S)
    elif arm == "agent_alone":
        client = _client()
        stage = FitGuardProposer(AgenticVariationProposer(client, None, rounds=1, steps=STEPS, session_seconds=SESSION_S,
                                                          target_col="__target__", seed=base.SEED), target_col="__target__")
        described = "the agent alone"
    else:
        raise ValueError(f"unknown arm {arm!r}")
    result = refined_search.discover_refined(
        {n: X[dev, i] for i, n in enumerate(names)}, y[dev], X[pool], lambda rows: pool_y[rows],
        proposer=stage, noise="absolute", seed=base.SEED)
    law = sp.sympify(result.law, locals={n: sp.Symbol(n) for n in names}) if result.law else None
    path = result.path
    assert path is not None
    inner = getattr(stage, "_inner", None)
    extra = {"action": result.action, "tentative_law": result.tentative_law, "dropped_terms": result.dropped_terms,
             "proposer_failures": result.proposer_failures, "found_by": path.found_by, "rejected_law": path.rejected_law,
             "adequacy": path.adequacy, "rows_read": path.rows_read, "candidates": path.candidates, "stage": described,
             "agent": dict(getattr(inner, "last_run", {}) or {}) if arm != "default" else {},
             "model_calls": getattr(client, "calls", 0) if client is not None else 0}
    return {"model": law, "pred": path_run._predict(law, names, split["X_test"]) if law is not None else None,
            "formula": result.law, "extra": extra}


def _child(conn: Any, arm: str, entry: dict[str, Any], split: dict[str, Any], budget: float) -> None:
    try:
        raw = run_arm(arm, split, entry, budget)
        try:
            row = base.score_run(arm, entry, split, raw)
        except Exception as exc:  # noqa: BLE001 - a model the scorer cannot handle is scored as not recovered, with the reason
            r2 = base._r2(split["y_test"], raw["pred"]) if raw.get("pred") is not None else None
            row = {"symbolic": False, "symbolic_rel_rmse": None, "symbolic_note": f"scoring failed: {type(exc).__name__}",
                   "test_r2": r2, "accuracy_solution": bool(r2 is not None and r2 >= base.R2_ACCURACY),
                   "model": str(raw.get("model")), "formula": raw.get("formula")}
        conn.send(("ok", {**row, **raw.get("extra", {})}))
    except Exception as exc:  # noqa: BLE001 - an arm's crash is a scored outcome, reported with its class and message
        conn.send(("error", {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-1500:]}))
    finally:
        conn.close()


def run_one(job: dict[str, Any], laws: dict[str, dict[str, Any]], budget: float = SEARCH_BUDGET_S,
            time_limit: float = TIME_LIMIT_S) -> dict[str, Any]:
    entry = laws[job["problem"]]
    split = base.make_split(entry, frame_of(entry), job["noise"])
    head = {**job, "family": entry["family"], "truth": entry["formula"], "n_train": len(split["y_train"])}
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    started = time.perf_counter()
    proc = ctx.Process(target=_child, args=(child, job["arm"], entry, split, budget))
    proc.start()
    child.close()
    outcome: tuple[str, dict[str, Any]] = ("timeout", {})
    if parent.poll(time_limit):
        try:
            outcome = parent.recv()
        except EOFError:
            outcome = ("error", {"error": "the arm's process died without a result"})
    proc.join(1.0)
    if proc.is_alive():
        proc.kill()
        proc.join()
    return {**head, "status": outcome[0], "seconds": round(time.perf_counter() - started, 2), **outcome[1]}


def all_jobs(arms: tuple[str, ...] = ARMS) -> list[dict[str, Any]]:
    return [{"problem": name, "noise": noise, "arm": arm}
            for name in entries() for noise in NOISE_LEVELS for arm in arms]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per arm and noise level: runs, recoveries, reported laws, wrong ones, seconds, and what the agent did."""
    out: dict[str, Any] = {}
    for arm in dict.fromkeys(r["arm"] for r in rows):
        for noise in (*NOISE_LEVELS, "all"):
            sel = [r for r in rows if r["arm"] == arm and noise in (r["noise"], "all")]
            ok = [r for r in sel if r["status"] == "ok"]
            reported = [r for r in ok if r.get("model") not in (None, "None")]
            agents = [r.get("agent") or {} for r in ok]
            sessions = [s for a in agents for s in a.get("sessions", [])]
            out[f"{arm}|{noise}"] = {
                "n": len(sel), "errors": sum(r["status"] == "error" for r in sel),
                "timeouts": sum(r["status"] == "timeout" for r in sel),
                "median_seconds": float(np.median([r["seconds"] for r in sel])) if sel else None,
                "symbolic": sum(bool(r["symbolic"]) for r in ok), "reported": len(reported),
                "reported_wrong": sum(not r["accuracy_solution"] and not r["symbolic"] for r in reported),
                "model_calls": sum(int(r.get("model_calls") or 0) for r in ok),
                "sessions": len(sessions), "sessions_with_a_fit": sum(bool(s.get("fits")) for s in sessions),
                "sessions_submitted": sum(s.get("stopped") == "submitted" for s in sessions),
                "structures_by_agent": sum(int(a.get("by_agent") or 0) for a in agents),
                "laws_found_by": {k: sum(r.get("found_by") == k for r in reported) for k in ("first_pass", "search")},
            }
    return out


def _printable(row: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in row.items() if k not in ("candidates", "trace")}, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("smoke", "develop", "collect"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1", help="k/N: this shard's runs")
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    laws = entries()
    arms = tuple(a for a in args.arms.split(",") if a)
    if args.command == "smoke":
        budget = float(os.environ.get("SMOKE_BUDGET_SECONDS", "40"))
        failed = 0
        for name in list(laws)[:3]:
            for arm in arms:
                row = run_one({"problem": name, "noise": 0.01, "arm": arm}, laws, budget)
                print(_printable(row), flush=True)
                if row["status"] != "ok":
                    print(row.get("trace", ""), flush=True)
                    failed += 1
        return 1 if failed else 0
    if args.command == "develop":
        k, n = (int(v) for v in args.shard.split("/"))
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for position, job in enumerate(all_jobs(arms)):
                if position % n != k:
                    continue
                handle.write(json.dumps(run_one(job, laws), sort_keys=True, default=str) + "\n")
                handle.flush()
        return 0
    rows = [json.loads(line) for path in sorted(args.out.glob("runs-shard-*.jsonl")) for line in path.read_text().splitlines()]
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    summary = summarise(rows)
    (args.out / "results.json").write_text(json.dumps({"tag": TAG, "rows": len(rows), "summary": summary}, indent=1,
                                                      sort_keys=True) + "\n", encoding="utf-8")
    print(f"collected {len(rows)} development runs into {args.out}")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
