#!/usr/bin/env python3
"""The refined path with a language model configuring PySR, for PREREG-LLM-CONTROLLER-2026-11.

Identical to ``scripts/run_srbench_refined.py`` in problems, rows, pool, budget and scoring; the one
difference is the search stage inside the units wrapper: ``ControlledSearchProposer`` (the model chooses
operators, derived inputs and seeds, all checked) around the same PySR. Needs ``NVIDIA_API_KEY``.

    python scripts/run_srbench_controlled.py smoke   --out DIR
    python scripts/run_srbench_controlled.py run     --out DIR --shard k/N
    python scripts/run_srbench_controlled.py collect --out DIR
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

import run_srbench_baseline as base  # noqa: E402
import run_srbench_main_search as path_run  # noqa: E402
import run_srbench_refined as refined  # noqa: E402

PREREGISTRATION_ID = "PREREG-LLM-CONTROLLER-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-LLM-CONTROLLER.json"
FROZEN = ("preregistration/PREREG-LLM-CONTROLLER-2026-11.md", "scripts/run_srbench_controlled.py",
          *(name for name in refined.FROZEN if not name.startswith("preregistration/")),
          "apedp_dyn/proposers/llm_controller.py", "apedp_dyn/proposers/llm_clients.py", "apedp_dyn/proposers/llm_proposer.py")

ENGINE = "refined_controlled"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
THINKING, REASONING_BUDGET = True, 2000
INTERVAL_S = 3.2


def run_controlled(split: dict[str, Any], entry: dict[str, Any], budget: float = refined.SEARCH_BUDGET_S) -> dict[str, Any]:
    from apedp_dyn.discovery_l2 import refined_search
    from apedp_dyn.proposers import llm_clients
    from apedp_dyn.proposers.llm_controller import ControlledSearchProposer
    from apedp_dyn.proposers.pysr_proposer import PySRProposer

    names, X, y = split["names"], split["X_train"], split["y_train"]
    key = hashlib.sha256(f"{path_run.PREREGISTRATION_ID}|{entry['name']}|pool".encode()).digest()[:8]
    order = np.random.default_rng(int.from_bytes(key, "big")).permutation(len(y))
    cut = round(path_run.DEV_SHARE * len(y))
    dev, pool = order[:cut], order[cut:]
    pool_y = y[pool]
    units = {n: entry["units"].get(f, {}) for n, f in zip(names, entry["features"], strict=True)}
    client = llm_clients.nvidia_client(MODEL, thinking=THINKING, reasoning_budget=REASONING_BUDGET, min_interval=INTERVAL_S)
    controller = ControlledSearchProposer(client, lambda unary: PySRProposer(budget_seconds=budget, unary=unary, seed=base.SEED),
                                          target_col="__target__")
    result = refined_search.discover_refined(
        {n: X[dev, i] for i, n in enumerate(names)}, y[dev], X[pool], lambda rows: pool_y[rows],
        proposer=controller, units=units, target_units=entry["target_units"], noise="absolute", seed=base.SEED)
    law = sp.sympify(result.law, locals={n: sp.Symbol(n) for n in names}) if result.law else None
    path = result.path
    assert path is not None
    extra = {"action": result.action, "tentative_law": result.tentative_law, "dropped_terms": result.dropped_terms,
             "units": result.units, "proposer_failures": result.proposer_failures, "found_by": path.found_by,
             "path_law_text": path.law, "rejected_law": path.rejected_law, "adequacy": path.adequacy,
             "rows_read": path.rows_read, "first_pass": path.first_pass, "search": path.search, "candidates": path.candidates,
             "used_in_development": refined.used_in_development(entry["name"]),
             "controller": controller.last_run, "model_calls": getattr(client, "calls", 0)}
    return {"model": law, "pred": path_run._predict(law, names, split["X_test"]) if law is not None else None,
            "formula": result.law, "extra": extra}


path_run.RUNNERS[ENGINE] = run_controlled


def all_jobs() -> list[dict[str, Any]]:
    return [{"problem": p["name"], "noise": noise, "seed": base.SEED, "engine": ENGINE}
            for p in base.load_manifest() for noise in refined.NOISE_LEVELS]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = refined.summarise(rows)
    for key in list(out):
        group, noise = key.split("|")
        sel = [r for r in rows if r["status"] == "ok" and noise in (str(r["noise"]), "all") and (
            group == "all" or group == r["family"]
            or (group == "development" and refined.used_in_development(r["problem"]))
            or (group == "held_out" and not refined.used_in_development(r["problem"])))]
        control = [r.get("controller") or {} for r in sel]
        out[key].update({
            "model_asked": sum(bool(c.get("asked")) for c in control),
            "model_failed": sum(bool(c.get("model_failure")) for c in control),
            "reply_readable": sum(bool(c.get("readable")) for c in control),
            "operators_narrowed": sum(bool(c.get("readable")) and len(c.get("unary", [])) < 6 for c in control),
            "features_added": sum(len(c.get("features", [])) for c in control),
            "guesses_fitted": sum(len(c.get("guesses", [])) for c in control),
            "items_rejected": sum(len(c.get("rejected", [])) for c in control),
        })
    return out


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def check_frozen() -> None:
    if not DIGEST.exists() or json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} is not frozen, or changed since it was frozen")


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
        smoke = [name for name in sorted(entries) if refined.used_in_development(name)][:2]  # development problems only
        for problem, noise in zip(smoke, (0.0, 0.01), strict=True):
            row = path_run.run_one({"problem": problem, "noise": noise, "seed": base.SEED, "engine": ENGINE}, entries, budget, refined.TIME_LIMIT_S)
            print(json.dumps({k: v for k, v in row.items() if k not in ("candidates", "trace", "first_pass")}, default=str), flush=True)
            if row["status"] != "ok" or (row.get("controller") or {}).get("model_failure"):
                print(row.get("trace", ""), flush=True)
                failed += 1
        return 1 if failed else 0
    check_frozen()
    if args.command == "run":
        k, n = (int(v) for v in args.shard.split("/"))
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for job in base.shard(all_jobs(), k, n):
                handle.write(json.dumps(path_run.run_one(job, entries, refined.SEARCH_BUDGET_S, refined.TIME_LIMIT_S), sort_keys=True, default=str) + "\n")
                handle.flush()
        return 0
    rows = [json.loads(line) for path in sorted(args.out.glob("runs-shard-*.jsonl")) for line in path.read_text().splitlines()]
    expected = len(all_jobs())
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} runs, found {len(rows)}")
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (args.out / "results.json").write_text(json.dumps(
        {"preregistration": PREREGISTRATION_ID, "model": MODEL, "thinking": THINKING, "reasoning_budget": REASONING_BUDGET,
         "frozen": frozen_digests(), "summary": summarise(rows)}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"collected {len(rows)} runs into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
