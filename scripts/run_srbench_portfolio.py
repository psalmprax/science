#!/usr/bin/env python3
"""The refined path with four different searches side by side, for PREREG-PORTFOLIO-SEARCH-2026-11.

Exactly PREREG-REFINED-SEARCH-2026-11's protocol (same problems, noise, rows, development and pool
split, units, 240 s per search, scoring), with one difference: the search stage is a portfolio of
four searches, each on its own core, merged before confirmation (``apedp_dyn.proposers.portfolio``).
The members are those of ``apedp_dyn.proposers.deep_pass.mixed_members``: the refined run's own PySR
search with its seed, the exhaustive pass one or two levels deeper, a PySR search with algebraic
operators only, and the default search with the next seed. The recorded refined run is the
comparison.

Development, 2026-11-20 (the 30 development problems, both noise levels, on GitHub's four-core
runners). Four seeds of the same search (``PORTFOLIO_MEMBERS=seeds``) recovered exactly the runs the
single search recovers, no more and no fewer, so that portfolio was not taken further; the mixed
portfolio is what ``run`` uses. The mixed portfolio on the same problems: 29 and 25 recovered against
27 and 25 by the benchmark's criterion; three runs gained, one lost; one of the gains passes the
tolerance narrowly and is not the law (audit M-23), so 28 and 25 by form.

    python scripts/run_srbench_portfolio.py smoke   --out DIR
    python scripts/run_srbench_portfolio.py develop --out DIR [--shard k/N]   # development problems only
    python scripts/run_srbench_portfolio.py run     --out DIR --shard k/N
    python scripts/run_srbench_portfolio.py collect --out DIR
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
import run_srbench_refined as refined_run  # noqa: E402

PREREGISTRATION_ID = "PREREG-PORTFOLIO-SEARCH-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-PORTFOLIO-SEARCH.json"
FROZEN = ("preregistration/PREREG-PORTFOLIO-SEARCH-2026-11.md", "scripts/run_srbench_portfolio.py",
          "scripts/run_srbench_refined.py", "scripts/run_srbench_main_search.py", "scripts/run_srbench_baseline.py",
          "benchmarks/srbench/manifest.json", "apedp_dyn/proposers/portfolio.py", "apedp_dyn/proposers/deep_pass.py",
          "apedp_dyn/discovery_l2/refined_search.py", "apedp_dyn/proposers/composed.py",
          "apedp_dyn/discovery_l2/main_search.py", "apedp_dyn/proposers/pysr_proposer.py",
          "apedp_dyn/discovery_l2/contrast_confirmation.py", "apedp_dyn/discovery_l2/contrast_resolution.py",
          "apedp_dyn/discovery_l2/confirmed_stop.py", "apedp_dyn/discovery_l2/guided_search.py",
          "apedp_dyn/discovery_l2/exhaustive_search.py")

ENGINE = "refined_portfolio"
NOISE_LEVELS = refined_run.NOISE_LEVELS
SEARCH_BUDGET_S = refined_run.SEARCH_BUDGET_S
MEMBERS = 4
# A recovery "by form": the benchmark's criterion (constants refitted to the clean law on the test rows,
# relative error below 1e-8) and, in addition, an error at most this. In every recorded run a real
# recovery sits below 1e-12 and a form that only follows the law on the sampled range sits above 1e-9
# (audit M-23). A reading of the score, not an acceptance threshold of the referee.
STRICT_REL_RMSE = 1e-10
MEMBER_SET = "mixed"      # what ``run`` uses; "seeds" exists for development comparisons only
MEMBER_SETS = ("mixed", "seeds")
MAX_PROPOSALS = 10        # the composite keeps eleven: these and the polynomial library's one
TIME_LIMIT_S = 720.0      # the refined run's 600 s, plus room for four Julia runtimes to start
used_in_development = refined_run.used_in_development


def member_seeds() -> tuple[int, ...]:
    """The refined run's seed first, then the next ones."""
    return tuple(base.SEED + k for k in range(MEMBERS))


def member_set() -> str:
    """``MEMBER_SET``, unless a development command asked for another one through the environment."""
    chosen = os.environ.get("PORTFOLIO_MEMBERS", MEMBER_SET)
    if chosen not in MEMBER_SETS:
        raise SystemExit(f"PORTFOLIO_MEMBERS must be one of {MEMBER_SETS}")
    return chosen


def members(budget: float) -> list[Any]:
    from apedp_dyn.proposers.deep_pass import mixed_members
    from apedp_dyn.proposers.portfolio import pysr_members

    if member_set() == "seeds":
        return list(pysr_members(budget, member_seeds()))
    return list(mixed_members(budget, base.SEED))


def run_portfolio(split: dict[str, Any], entry: dict[str, Any], budget: float = SEARCH_BUDGET_S) -> dict[str, Any]:
    from apedp_dyn.discovery_l2 import refined_search
    from apedp_dyn.proposers.portfolio import PortfolioProposer

    names, X, y = split["names"], split["X_train"], split["y_train"]
    # the refined run's own development and pool rows
    key = hashlib.sha256(f"{path_run.PREREGISTRATION_ID}|{entry['name']}|pool".encode()).digest()[:8]
    order = np.random.default_rng(int.from_bytes(key, "big")).permutation(len(y))
    cut = round(path_run.DEV_SHARE * len(y))
    dev, pool = order[:cut], order[cut:]
    pool_y = y[pool]
    units = {n: entry["units"].get(f, {}) for n, f in zip(names, entry["features"], strict=True)}
    portfolio = PortfolioProposer(members(budget), max_proposals=MAX_PROPOSALS, parallel=True, timeout=budget + 300.0)
    result = refined_search.discover_refined(
        {n: X[dev, i] for i, n in enumerate(names)}, y[dev], X[pool], lambda rows: pool_y[rows],
        proposer=portfolio, units=units, target_units=entry["target_units"], noise="absolute", seed=base.SEED)
    law = sp.sympify(result.law, locals={n: sp.Symbol(n) for n in names}) if result.law else None
    path = result.path
    assert path is not None
    extra = {"action": result.action, "tentative_law": result.tentative_law, "dropped_terms": result.dropped_terms,
             "units": result.units, "proposer_failures": result.proposer_failures, "found_by": path.found_by,
             "path_law_text": path.law, "rejected_law": path.rejected_law, "adequacy": path.adequacy,
             "rows_read": path.rows_read, "first_pass": path.first_pass, "search": path.search, "candidates": path.candidates,
             "portfolio": {**portfolio.last_run, "member_set": member_set()}, "used_in_development": used_in_development(entry["name"])}
    return {"model": law, "pred": path_run._predict(law, names, split["X_test"]) if law is not None else None,
            "formula": result.law, "extra": extra}


path_run.RUNNERS[ENGINE] = run_portfolio  # run_one isolates, times and scores any registered engine


def all_jobs() -> list[dict[str, Any]]:
    return [{"problem": p["name"], "noise": noise, "seed": base.SEED, "engine": ENGINE}
            for p in base.load_manifest() for noise in NOISE_LEVELS]


def recovered_by_form(row: dict[str, Any]) -> bool:
    """Recovered by the benchmark's criterion, and not a narrow pass of its tolerance."""
    error = row.get("symbolic_rel_rmse")
    return bool(row["status"] == "ok" and row.get("symbolic") and error is not None and float(error) <= STRICT_REL_RMSE)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The refined run's summary, plus the count by form and what the portfolio did."""
    out: dict[str, Any] = refined_run.summarise(rows)
    for key, cell in out.items():
        group, noise = key.split("|")
        sel = [r for r in rows if r["status"] == "ok" and (noise == "all" or str(r["noise"]) == noise) and (
            group == "all" or group == r["family"]
            or (group == "development" and used_in_development(r["problem"]))
            or (group == "held_out" and not used_in_development(r["problem"])))]
        cell["symbolic_by_form"] = sum(recovered_by_form(r) for r in sel)
        cell["narrow_passes"] = sorted(f"{r['problem']}|{r['noise']}" for r in sel if r.get("symbolic") and not recovered_by_form(r))
        ran = [r["portfolio"] for r in sel if (r.get("portfolio") or {}).get("members")]  # the stage is skipped when no product is left
        cell["portfolio_ran"] = len(ran)
        cell["member_failures"] = sum(len(p.get("failures", [])) for p in ran)
        cell["runs_with_a_member_failure"] = sum(bool(p.get("failures")) for p in ran)
        cell["median_distinct_proposals"] = float(np.median([p["distinct"] for p in ran])) if ran else None
        cell["median_search_seconds"] = float(np.median([p["seconds"] for p in ran])) if ran else None
    return out


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def check_frozen() -> None:
    if not DIGEST.exists() or json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} is not frozen, or changed since it was frozen")


def _printable(row: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in row.items() if k not in ("candidates", "trace", "first_pass")}, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("run", "collect", "smoke", "develop"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1", help="k/N: this shard's problems")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    entries = {p["name"]: p for p in base.load_manifest()}
    k, n = (int(v) for v in args.shard.split("/"))
    if args.command in ("smoke", "develop"):  # development problems only; never a held-out one
        names = [name for name in sorted(entries) if used_in_development(name)]
        if args.command == "smoke":
            names, levels = names[:2], (0.01,)
            budget = float(os.environ.get("SMOKE_BUDGET_SECONDS", "30"))
        else:
            levels, budget = NOISE_LEVELS, SEARCH_BUDGET_S
        failed = 0
        for index, (noise, problem) in enumerate((noise, problem) for noise in levels for problem in names):
            if index % n != k:
                continue
            row = path_run.run_one({"problem": problem, "noise": noise, "seed": base.SEED, "engine": ENGINE},
                                   entries, budget, TIME_LIMIT_S)
            print("ROW " + _printable(row), flush=True)
            if row["status"] != "ok":
                print(row.get("trace", ""), flush=True)
                failed += 1
        return 1 if failed and args.command == "smoke" else 0
    check_frozen()
    if member_set() != MEMBER_SET:
        raise SystemExit("PORTFOLIO_MEMBERS is for development commands; the run uses the frozen member set")
    if args.command == "run":
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
