#!/usr/bin/env python3
"""The refined path on the SRBench protocol, for PREREG-REFINED-SEARCH-2026-11.

The same problems, noise levels, development rows and pool as
PREREG-MAIN-SEARCH-2026-11, so the two results can be laid side by side. The
search stage is PySR inside the units wrapper, beside the polynomial library;
the law scored is the one ``refined_search.discover_refined`` reports
(confirmed against rivals, in the confirmed and pruned form).

    python scripts/run_srbench_refined.py smoke   --out DIR
    python scripts/run_srbench_refined.py run     --out DIR --shard k/N
    python scripts/run_srbench_refined.py collect --out DIR
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

PREREGISTRATION_ID = "PREREG-REFINED-SEARCH-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-REFINED-SEARCH.json"
FROZEN = ("preregistration/PREREG-REFINED-SEARCH-2026-11.md", "scripts/run_srbench_refined.py",
          "scripts/run_srbench_main_search.py", "scripts/run_srbench_baseline.py", "benchmarks/srbench/manifest.json",
          "apedp_dyn/discovery_l2/refined_search.py", "apedp_dyn/proposers/composed.py",
          "apedp_dyn/discovery_l2/main_search.py", "apedp_dyn/proposers/pysr_proposer.py",
          "apedp_dyn/discovery_l2/contrast_confirmation.py", "apedp_dyn/discovery_l2/contrast_resolution.py",
          "apedp_dyn/discovery_l2/confirmed_stop.py", "apedp_dyn/discovery_l2/guided_search.py",
          "apedp_dyn/discovery_l2/exhaustive_search.py")

ENGINE = "refined"
NOISE_LEVELS = (0.0, 0.01)
SEARCH_BUDGET_S = 240.0
TIME_LIMIT_S = 600.0
DEVELOPMENT_TAG = "dev-refined|"  # problems whose name hashes to 0 mod 5 under this tag were used in development


def used_in_development(name: str) -> bool:
    return int(hashlib.sha256((DEVELOPMENT_TAG + name).encode()).hexdigest(), 16) % 5 == 0


def run_refined(split: dict[str, Any], entry: dict[str, Any], budget: float = SEARCH_BUDGET_S) -> dict[str, Any]:
    from apedp_dyn.discovery_l2 import refined_search
    from apedp_dyn.proposers.pysr_proposer import PySRProposer

    names, X, y = split["names"], split["X_train"], split["y_train"]
    key = hashlib.sha256(f"{path_run.PREREGISTRATION_ID}|{entry['name']}|pool".encode()).digest()[:8]
    order = np.random.default_rng(int.from_bytes(key, "big")).permutation(len(y))
    cut = round(path_run.DEV_SHARE * len(y))
    dev, pool = order[:cut], order[cut:]
    pool_y = y[pool]
    units = {n: entry["units"].get(f, {}) for n, f in zip(names, entry["features"], strict=True)}
    result = refined_search.discover_refined(
        {n: X[dev, i] for i, n in enumerate(names)}, y[dev], X[pool], lambda rows: pool_y[rows],
        proposer=PySRProposer(budget_seconds=budget, seed=base.SEED), units=units, target_units=entry["target_units"],
        noise="absolute", seed=base.SEED)
    law = sp.sympify(result.law, locals={n: sp.Symbol(n) for n in names}) if result.law else None
    path = result.path
    assert path is not None
    extra = {"action": result.action, "tentative_law": result.tentative_law, "dropped_terms": result.dropped_terms,
             "units": result.units, "proposer_failures": result.proposer_failures, "found_by": path.found_by,
             "path_law_text": path.law, "rejected_law": path.rejected_law, "adequacy": path.adequacy,
             "rows_read": path.rows_read, "first_pass": path.first_pass, "search": path.search, "candidates": path.candidates,
             "used_in_development": used_in_development(entry["name"])}
    return {"model": law, "pred": path_run._predict(law, names, split["X_test"]) if law is not None else None,
            "formula": result.law, "extra": extra}


path_run.RUNNERS[ENGINE] = run_refined  # run_one isolates, times and scores any registered engine


def all_jobs() -> list[dict[str, Any]]:
    return [{"problem": p["name"], "noise": noise, "seed": base.SEED, "engine": ENGINE}
            for p in base.load_manifest() for noise in NOISE_LEVELS]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in ("all", "feynman", "strogatz", "held_out", "development"):
        for noise in (*NOISE_LEVELS, "all"):
            sel = [r for r in rows if noise in (r["noise"], "all") and (
                group == "all" or group == r["family"]
                or (group == "development" and used_in_development(r["problem"]))
                or (group == "held_out" and not used_in_development(r["problem"])))]
            ok = [r for r in sel if r["status"] == "ok"]
            reported = [r for r in ok if r.get("model") not in (None, "None")]
            out[f"{group}|{noise}"] = {
                "n": len(sel), "errors": sum(r["status"] == "error" for r in sel), "timeouts": sum(r["status"] == "timeout" for r in sel),
                "symbolic": sum(bool(r["symbolic"]) for r in ok), "accuracy_solution": sum(bool(r["accuracy_solution"]) for r in ok),
                "reported": len(reported), "reported_symbolic": sum(bool(r["symbolic"]) for r in reported),
                "reported_not_accurate": sum(not r["accuracy_solution"] for r in reported),
                "reported_wrong": sum(not r["accuracy_solution"] and not r["symbolic"] for r in reported),
                "tentative": sum(bool(r.get("tentative_law")) for r in ok),
                "units_reduced": sum(bool((r.get("units") or {}).get("reduced")) for r in ok),
                "terms_dropped": sum(bool(r.get("dropped_terms")) for r in ok),
                "proposer_failures": sum(bool(r.get("proposer_failures")) for r in ok),
                "actions": {a: sum(r.get("action") == a for r in ok) for a in ("confirmed", "equivalent_class", "unchallenged", "widen_search")},
                "median_seconds": float(np.median([r["seconds"] for r in sel])) if sel else None,
            }
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
        smoke = [name for name in sorted(entries) if used_in_development(name)][:3]  # development problems only
        for problem, noise in zip(smoke, (0.0, 0.01, 0.0), strict=True):
            row = path_run.run_one({"problem": problem, "noise": noise, "seed": base.SEED, "engine": ENGINE}, entries, budget, TIME_LIMIT_S)
            print(json.dumps({k: v for k, v in row.items() if k not in ("candidates", "trace", "first_pass")}, default=str), flush=True)
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
                handle.write(json.dumps(path_run.run_one(job, entries, SEARCH_BUDGET_S, TIME_LIMIT_S), sort_keys=True, default=str) + "\n")
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
