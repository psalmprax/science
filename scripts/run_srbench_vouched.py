#!/usr/bin/env python3
"""What the route's last checks cost on the physics benchmark, behind the four searches: PREREG-VOUCHED-SRBENCH-2026-11.

    python scripts/run_srbench_vouched.py smoke   --out DIR
    python scripts/run_srbench_vouched.py run     --out DIR --shard k/N
    python scripts/run_srbench_vouched.py collect --out DIR

The checks of ``discovery_l2/vouching.py`` became the last step of the table route on a run in
which candidates were handed over (PREREG-COUNTERFEIT-2026-11). That run fixed in advance what
has to be said with them: not measured behind a search engine or on the physics benchmark, where
what they cost in recovered laws is unknown. This run measures it.

Exactly PREREG-PORTFOLIO-SEARCH-2026-11's protocol: the same 133 problems, noise 0 and 1 %, rows,
development and pool split, units, four searches of 240 seconds side by side, scoring. Each run
searches once and yields two reports that differ in nothing but the checks:

* ``route``: what ``refined_search.discover_refined`` reports, as in the recorded run;
* ``vouched``: that report after ``discovery_l2/vouched_route.checked``, which is what the route
  in force for tables now does.

So a law lost or gained between the two is the checks' doing and not the search's.
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
import run_srbench_portfolio as portfolio_run  # noqa: E402
import run_srbench_refined as refined_run  # noqa: E402

from apedp_dyn.conf.canonical_spec import REFEREE_MAX_FPR, REFEREE_MIN_VRR  # noqa: E402

PREREGISTRATION_ID = "PREREG-VOUCHED-SRBENCH-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-VOUCHED-SRBENCH.json"
FROZEN = ("preregistration/PREREG-VOUCHED-SRBENCH-2026-11.md", "scripts/run_srbench_vouched.py",
          "scripts/run_srbench_portfolio.py", "scripts/run_srbench_refined.py", "scripts/run_srbench_main_search.py",
          "scripts/run_srbench_baseline.py", "benchmarks/srbench/manifest.json",
          "apedp_dyn/discovery_l2/vouching.py", "apedp_dyn/discovery_l2/vouched_route.py",
          "apedp_dyn/discovery_l2/default_route.py", "apedp_dyn/discovery_l2/refined_search.py",
          "apedp_dyn/discovery_l2/main_search.py", "apedp_dyn/discovery_l2/contrast_confirmation.py",
          "apedp_dyn/proposers/portfolio.py", "apedp_dyn/proposers/deep_pass.py", "apedp_dyn/proposers/composed.py",
          "apedp_dyn/proposers/pysr_proposer.py")

ENGINE = "refined_portfolio_vouched"
ARMS = ("route", "vouched")
NOISE_LEVELS = refined_run.NOISE_LEVELS
SEARCH_BUDGET_S = refined_run.SEARCH_BUDGET_S
TIME_LIMIT_S = portfolio_run.TIME_LIMIT_S + 60.0       # the recorded run's limit, and a minute for the checks
MAX_WRONG = REFEREE_MAX_FPR                            # of the laws an arm reports, the share that may be inaccurate
MIN_KEPT = REFEREE_MIN_VRR                             # of the laws the route recovers, the share the checks must keep
KEPT_WITHOUT_COST = 1.0 - REFEREE_MAX_FPR              # the share above which the checks cost at most what a referee may get wrong


def _scored(model: sp.Expr | None, formula: str | None, entry: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    """One arm's law by the benchmark's scoring, as ``run_one`` scores an engine's; a scoring failure is not a recovery."""
    names = split["names"]
    pred = path_run._predict(model, names, split["X_test"]) if model is not None else None
    raw = {"model": model, "pred": pred, "formula": formula}
    try:
        row: dict[str, Any] = base.score_run(ENGINE, entry, split, raw)
    except Exception as exc:  # noqa: BLE001 - a model the scorer cannot handle is scored as not recovered, with the reason
        r2 = base._r2(split["y_test"], pred) if pred is not None else None
        row = {"symbolic": False, "symbolic_rel_rmse": None, "symbolic_note": f"scoring failed: {type(exc).__name__}",
               "test_r2": r2, "accuracy_solution": bool(r2 is not None and r2 >= base.R2_ACCURACY),
               "model": str(model) if model is not None else None, "formula": formula}
    return row


def run_vouched(split: dict[str, Any], entry: dict[str, Any], budget: float = SEARCH_BUDGET_S) -> dict[str, Any]:
    """One search; the route's report and the same report after the checks. ``run_one`` scores the second."""
    from apedp_dyn.discovery_l2 import default_route, refined_search, vouched_route
    from apedp_dyn.proposers.portfolio import PortfolioProposer

    names, X, y = split["names"], split["X_train"], split["y_train"]
    # the recorded runs' own development and pool rows
    key = hashlib.sha256(f"{path_run.PREREGISTRATION_ID}|{entry['name']}|pool".encode()).digest()[:8]
    order = np.random.default_rng(int.from_bytes(key, "big")).permutation(len(y))
    cut = round(path_run.DEV_SHARE * len(y))
    dev, pool = order[:cut], order[cut:]
    pool_y = y[pool]
    units = {n: entry["units"].get(f, {}) for n, f in zip(names, entry["features"], strict=True)}
    portfolio = PortfolioProposer(portfolio_run.members(budget), max_proposals=portfolio_run.MAX_PROPOSALS, parallel=True,
                                  timeout=budget + 300.0)
    columns = {n: X[dev, i] for i, n in enumerate(names)}
    result = refined_search.discover_refined(
        columns, y[dev], X[pool], lambda rows: pool_y[rows],
        proposer=portfolio, units=units, target_units=entry["target_units"], noise="absolute", seed=base.SEED)
    path = result.path
    assert path is not None
    record = {"route": default_route.REFINED, "action": result.action, "law": result.law, "reported": result.law is not None,
              "tentative_law": result.tentative_law, "tentative_reason": None,
              "law_as_confirmed": result.law or result.tentative_law, "refined": result}
    after = vouched_route.checked(record, columns, y[dev], X[pool])
    local = {n: sp.Symbol(n) for n in names}
    route_law = sp.sympify(result.law, locals=local) if result.law else None
    law = sp.sympify(after["law"], locals=local) if after["law"] else None
    contrast = path.contrast
    extra = {"action": result.action, "tentative_law": result.tentative_law, "dropped_terms": result.dropped_terms,
             "units": result.units, "proposer_failures": result.proposer_failures, "found_by": path.found_by,
             "path_law_text": path.law, "rejected_law": path.rejected_law, "adequacy": path.adequacy,
             "rows_read": path.rows_read, "first_pass": path.first_pass, "search": path.search, "candidates": path.candidates,
             "portfolio": {**portfolio.last_run, "member_set": portfolio_run.member_set()},
             "used_in_development": portfolio_run.used_in_development(entry["name"]),
             "equivalent": list(contrast.equivalent) if contrast is not None else [],
             "route": _scored(route_law, result.law, entry, split), "vouching": dict(after["vouching"])}
    return {"model": law, "pred": path_run._predict(law, names, split["X_test"]) if law is not None else None,
            "formula": after["law"], "extra": extra}


path_run.RUNNERS[ENGINE] = run_vouched  # run_one isolates, times and scores any registered engine


def all_jobs() -> list[dict[str, Any]]:
    """The recorded run's jobs, in its order, under this run's engine."""
    return [{**job, "engine": ENGINE} for job in portfolio_run.all_jobs()]


def _arm(row: dict[str, Any], arm: str) -> dict[str, Any]:
    """What an arm reported in a run: the route's law is kept under ``route``, the checked one in the row itself."""
    scored: dict[str, Any] = row["route"] if arm == "route" else row
    return scored


def _reported(row: dict[str, Any], arm: str) -> bool:
    return _arm(row, arm).get("model") not in (None, "None")


def _recovered(row: dict[str, Any], arm: str) -> bool:
    return _reported(row, arm) and bool(_arm(row, arm)["symbolic"])


def tally(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """One arm on a set of runs. A run that crashed or ran over reported nothing in either arm."""
    ok = [row for row in rows if row["status"] == "ok"]
    reported = [row for row in ok if _reported(row, arm)]
    return {"n": len(rows), "errors": sum(row["status"] == "error" for row in rows),
            "timeouts": sum(row["status"] == "timeout" for row in rows),
            "reported": len(reported), "recovered": sum(_recovered(row, arm) for row in reported),
            "recovered_by_form": sum(_recovered(row, arm) and (_arm(row, arm).get("symbolic_rel_rmse") or 0.0)
                                     <= portfolio_run.STRICT_REL_RMSE for row in reported),
            "not_accurate": sum(not _arm(row, arm)["accuracy_solution"] for row in reported),
            "not_accurate_share": (sum(not _arm(row, arm)["accuracy_solution"] for row in reported) / len(reported)
                                   if reported else None),
            "accurate_not_the_law": sum(bool(_arm(row, arm)["accuracy_solution"]) and not _recovered(row, arm)
                                        for row in reported)}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Both arms overall, by noise level, by family and on the problems never used in development; then the expectations."""
    out: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = {"all": rows}
    for noise in NOISE_LEVELS:
        groups[f"noise {noise:g}"] = [row for row in rows if float(row["noise"]) == noise]
    for family in ("feynman", "strogatz"):
        groups[family] = [row for row in rows if row["family"] == family]
    groups["held_out"] = [row for row in rows if not portfolio_run.used_in_development(row["problem"])]
    for name, sel in groups.items():
        out[name] = {arm: tally(sel, arm) for arm in ARMS}
    ok = [row for row in rows if row["status"] == "ok"]
    was_right = [row for row in ok if _recovered(row, "route")]
    kept = sum(_recovered(row, "vouched") for row in was_right)
    vouched, route = out["all"]["vouched"], out["all"]["route"]
    out["expectations"] = {
        "1_not_accurate_share_of_vouched_reports": vouched["not_accurate_share"],
        "1_limit": MAX_WRONG,
        "1_holds": (vouched["not_accurate_share"] or 0.0) <= MAX_WRONG,
        "2_recovered_laws_of_the_route_kept": [kept, len(was_right)],
        "2_limit": MIN_KEPT,
        "2_holds": bool(was_right) and kept >= MIN_KEPT * len(was_right),
        "3_limit": KEPT_WITHOUT_COST,
        "3_holds": bool(was_right) and kept >= KEPT_WITHOUT_COST * len(was_right),
        "route_not_accurate_share": route["not_accurate_share"],
    }
    states = [row.get("vouching") or {} for row in ok]
    lost = [row for row in was_right if not _recovered(row, "vouched")]
    out["checks"] = {
        "applied": sum(bool(state.get("applied")) for state in states),
        "passed_unchanged": sum(bool(state.get("applied")) and not state.get("switched") and _reported(row, "vouched")
                                for row, state in zip(ok, states, strict=True)),
        "passed_to_another_member": sum(bool(state.get("switched")) for state in states),
        "withheld": sum(bool(state.get("applied")) and not _reported(row, "vouched") for row, state in zip(ok, states, strict=True)),
        "withheld_by_reason": {},
        "recovered_laws_lost": [{"problem": row["problem"], "noise": row["noise"], "law": row["route"]["formula"],
                                 "reason": (row.get("vouching") or {}).get("reason")} for row in lost],
        "gained": [{"problem": row["problem"], "noise": row["noise"], "law": row["formula"], "route_law": row["route"]["formula"]}
                   for row in ok if _recovered(row, "vouched") and not _recovered(row, "route")],
        "not_accurate_before_and_withheld": sum(_reported(row, "route") and not row["route"]["accuracy_solution"]
                                                 and not _reported(row, "vouched") for row in ok),
        "seconds": {"median": float(np.median([row["seconds"] for row in rows])) if rows else None},
    }
    for row, state in zip(ok, states, strict=True):
        if state.get("applied") and not _reported(row, "vouched"):
            reason = str(state.get("reason", ""))[:60]
            out["checks"]["withheld_by_reason"][reason] = out["checks"]["withheld_by_reason"].get(reason, 0) + 1
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
    keep = ("problem", "noise", "status", "seconds", "symbolic", "accuracy_solution", "action", "formula", "error")
    out = {key: row.get(key) for key in keep}
    out["route"] = {key: (row.get("route") or {}).get(key) for key in ("symbolic", "accuracy_solution", "formula")}
    out["vouching"] = {key: (row.get("vouching") or {}).get(key) for key in ("applied", "reason", "switched", "why_not")}
    return json.dumps(out, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("run", "collect", "smoke"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1", help="k/N: this shard's problems")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    entries = {p["name"]: p for p in base.load_manifest()}
    k, n = (int(v) for v in args.shard.split("/"))
    if args.command == "smoke":  # two development problems, a short search; never a held-out one
        names = [name for name in sorted(entries) if portfolio_run.used_in_development(name)][:2]
        budget = float(os.environ.get("SMOKE_BUDGET_SECONDS", "30"))
        failed = 0
        for problem in names:
            row = path_run.run_one({"problem": problem, "noise": 0.01, "seed": base.SEED, "engine": ENGINE},
                                   entries, budget, TIME_LIMIT_S)
            print("ROW " + _printable(row), flush=True)
            if row["status"] != "ok":
                print(row.get("trace", ""), flush=True)
                failed += 1
        return 1 if failed else 0
    check_frozen()
    if portfolio_run.member_set() != portfolio_run.MEMBER_SET:
        raise SystemExit("PORTFOLIO_MEMBERS is for development commands; the run uses the recorded member set")
    if args.command == "run":
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for job in base.shard(all_jobs(), k, n):
                row = path_run.run_one(job, entries, SEARCH_BUDGET_S, TIME_LIMIT_S)
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                handle.flush()
        return 0
    rows = [json.loads(line) for path in sorted(args.out.glob("runs-shard-*.jsonl")) for line in path.read_text().splitlines()]
    expected = len(all_jobs())
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} runs, found {len(rows)}")
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    summary = summarise(rows)
    (args.out / "results.json").write_text(json.dumps(
        {"preregistration": PREREGISTRATION_ID, "frozen": frozen_digests(), "summary": summary},
        indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"collected {len(rows)} runs into {args.out}")
    print(json.dumps({"all": summary["all"], "expectations": summary["expectations"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
