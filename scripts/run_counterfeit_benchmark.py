#!/usr/bin/env python3
"""The counterfeit-law benchmark: a method is scored on what it refuses as much as on what it finds.

    python scripts/run_counterfeit_benchmark.py build [--check]        # benchmarks/counterfeit_laws.json
    python scripts/run_counterfeit_benchmark.py smoke
    python scripts/run_counterfeit_benchmark.py run --shard k/N --out DIR     # PREREG-COUNTERFEIT-2026-11, once
    python scripts/run_counterfeit_benchmark.py collect --out DIR

Benchmarks of law discovery ask how many laws a method finds. A method that reports its best fit
every time does well on that and cannot be relied on. This benchmark is built so that reporting
the best fit is often wrong: every case hands the method a law's *counterfeits* (other forms
fitted to the law itself, ``apedp_dyn.redteam.counterfeit``) as candidates, sometimes with the
law among them and sometimes without.

**A case** is a world of ``apedp_dyn.redteam.worlds``: rows drawn uniformly on the law's domain
with independent Gaussian noise (the conditions of the preregistered runs), split into
development rows and a pool the method may measure, and a list of candidates. For each law there
are ``len(NOISE_LEVELS)`` noise levels times two cases: the law offered beside its counterfeits,
and the law withheld.

**The score.** A law reported and right counts one. A reported law that is wrong (not the law,
and less accurate than R^2 0.999 on fresh rows without noise) costs ``(1 - a) / a`` with ``a`` the
referee's permitted share of false reports (canonical ``REFEREE_MAX_FPR``): nineteen right
answers. So a method whose reports are wrong at exactly the permitted rate scores nothing. A
report that is accurate without being the law counts nothing either way, and so does saying
nothing. The total is divided by the number of cases in which the law was offered, so 1 is
perfect and a method that reports whatever fits best goes far below zero.

**To score another method**, give ``scored`` a function of a case
(``decide(world, rows, dev, pool) -> law text or None``): see ``METHODS``.

Laws are drawn here with their own seed and leave out every structure of the invented-laws
benchmark (by fingerprint), of the red team's laws, of the laws the memory was developed on and
of the laws it is tested on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_invented_laws as grammar  # noqa: E402
import run_memory_laws as memory_run  # noqa: E402
import run_red_team as red_team  # noqa: E402
import run_search_evolution as development  # noqa: E402

from apedp_dyn.conf.canonical_spec import REFEREE_MAX_FPR  # noqa: E402
from apedp_dyn.redteam import counterfeit  # noqa: E402
from apedp_dyn.redteam import worlds as rt  # noqa: E402

PREREGISTRATION_ID = "PREREG-COUNTERFEIT-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-COUNTERFEIT.json"
OUT = ROOT / "benchmarks" / "counterfeit_laws.json"
FROZEN = (
    "preregistration/PREREG-COUNTERFEIT-2026-11.md", "preregistration/FORECAST-COUNTERFEIT-2026-11.json",
    "scripts/run_counterfeit_benchmark.py", "benchmarks/counterfeit_laws.json", "scripts/run_red_team.py",
    "scripts/build_invented_laws.py", "scripts/run_srbench_baseline.py", "scripts/run_srbench_main_search.py",
    "apedp_dyn/redteam/counterfeit.py", "apedp_dyn/redteam/worlds.py", "apedp_dyn/redteam/plain_selection.py",
    "apedp_dyn/discovery_l2/refined_search.py", "apedp_dyn/discovery_l2/main_search.py",
    "apedp_dyn/proposers/fit_guard.py", "apedp_dyn/proposers/portfolio.py",
)
SEED = 20261128
LAW_COUNT = 60
NOISE_LEVELS = (0.0, 0.01, 0.05)
ROWS = 1000
#: one counterfeit from each band of distance, where the law has one: near, middle, far (shares of the law's spread)
BANDS = ((0.001, 0.01), (0.01, 0.0316), (0.0316, 0.3))
PENALTY = (1.0 - REFEREE_MAX_FPR) / REFEREE_MAX_FPR
METHODS = ("route", "best_fit", "validated")
TIME_LIMIT_S = 600.0


def _taken() -> set[str]:
    """Structures already used somewhere: the red team's laws, the memory's development and test laws."""
    used = {grammar.structure(sp.sympify(law["law"])) for law in red_team.laws()}
    used |= {grammar.structure(sp.sympify(law["text"])) for law in development.draw_laws(development.TRAIN + development.VALIDATION)}
    used |= {grammar.structure(sp.sympify(law["text"]))
             for law in json.loads(memory_run.LAWS_FILE.read_text(encoding="utf-8"))["laws"]}
    return used


def build(count: int = LAW_COUNT, seed: int = SEED) -> dict[str, Any]:
    """The benchmark: laws with one counterfeit per band of distance, and the cases made of them."""
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    taken = _taken()
    rng = np.random.default_rng(seed)
    laws: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(grammar.MAX_DRAWS):
        if len(laws) == count:
            break
        inputs = int(rng.integers(1, 4))
        recipe = counterfeit.draw_recipe(rng, inputs, grammar.CONSTANTS)
        expr = counterfeit.law_of(recipe, inputs)
        key = grammar.structure(expr)
        if key in seen or not grammar.acceptable(expr, inputs, rng):
            continue
        seen.add(key)
        if grammar.fingerprint(str(expr)) in in_benchmark or key in taken:
            continue
        fakes = counterfeit.counterfeits(recipe, inputs, grammar.LOW, grammar.HIGH, seed=seed + len(laws))
        chosen = []
        for low, high in BANDS:
            band = [fake for fake in fakes if low <= fake.distance < high]
            if band:
                chosen.append(band[int(rng.integers(0, len(band)))])
        if len(chosen) < 2:
            continue                                    # a case needs at least two counterfeits to choose between
        laws.append({"id": 5000 + len(laws), "inputs": inputs, "law": str(expr), "counterfeits": [asdict(fake) for fake in chosen]})
    if len(laws) != count:
        raise SystemExit(f"only {len(laws)} laws after {grammar.MAX_DRAWS} draws")
    cases = []
    for law in laws:
        fakes = [fake["text"] for fake in law["counterfeits"]]
        for noise in NOISE_LEVELS:
            for offered in (True, False):
                texts = [*fakes, *([law["law"]] if offered else [])]
                order = rng.permutation(len(texts))
                world = rt.World(law=law["law"], inputs=law["inputs"], candidates=tuple(texts[int(i)] for i in order),
                                 truth_listed=offered, rows=ROWS, noise=noise, seed=int(rng.integers(0, 2 ** 31 - 1)))
                cases.append({"case": f"{law['id']}-{noise:g}-{'offered' if offered else 'withheld'}", "law_id": law["id"],
                              "world": world.as_dict()})
    return {"what": "The counterfeit-law benchmark: every case hands a method counterfeits of a law as candidates, with or "
                    "without the law. Scored on right reports minus a heavy cost for wrong ones.",
            "seed": seed, "domain": [grammar.LOW, grammar.HIGH], "noise_levels": list(NOISE_LEVELS), "rows": ROWS,
            "penalty_for_a_wrong_law": PENALTY, "laws": laws, "cases": cases}


def cases() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = json.loads(OUT.read_text(encoding="utf-8"))["cases"]
    return out


def tally(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    """One method on a set of cases: what it reported, and its score."""
    ok = [row for row in rows if row.get("status") == "ok"]
    offered = [row for row in rows if row["world"]["truth_listed"]]
    right = sum(bool(row[method]["symbolic"]) for row in ok if row[method]["reported"])
    wrong = sum(bool(row[method]["wrong"]) for row in ok)
    reported = sum(bool(row[method]["reported"]) for row in ok)
    return {"cases": len(rows), "law_offered": len(offered), "errors": sum(row.get("status") == "error" for row in rows),
            "timeouts": sum(row.get("status") == "timeout" for row in rows),
            "reported": reported, "right": right, "wrong": wrong, "accurate_not_the_law": reported - right - wrong,
            "wrong_share_of_reported": wrong / reported if reported else None,
            "law_reported_when_offered": sum(bool(row[method]["symbolic"]) for row in ok if row["world"]["truth_listed"]),
            "reported_when_withheld": sum(bool(row[method]["reported"]) for row in ok if not row["world"]["truth_listed"]),
            "score": (right - PENALTY * wrong) / len(offered) if offered else None}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every method overall, per noise level, and with the law offered or withheld; then the expectations as written."""
    out: dict[str, Any] = {}
    for method in METHODS:
        out[f"{method}|all"] = tally(rows, method)
        for noise in NOISE_LEVELS:
            out[f"{method}|{noise:g}"] = tally([row for row in rows if float(row["world"]["noise"]) == noise], method)
        for offered in (True, False):
            sel = [row for row in rows if bool(row["world"]["truth_listed"]) == offered]
            out[f"{method}|{'offered' if offered else 'withheld'}"] = tally(sel, method)
    measured = [row for row in rows if float(row["world"]["noise"]) <= 0.01]
    route_measured = tally(measured, "route")
    route = out["route|all"]
    out["expectations"] = {
        "1_wrong_share_of_the_routes_reports_up_to_1_percent_noise": route_measured["wrong_share_of_reported"],
        "1_limit": REFEREE_MAX_FPR,
        "1_holds": (route_measured["wrong_share_of_reported"] or 0.0) <= REFEREE_MAX_FPR,
        "2_scores": {method: out[f"{method}|all"]["score"] for method in METHODS},
        "2_holds": route["score"] is not None and all(route["score"] > out[f"{method}|all"]["score"]
                                                      for method in METHODS if method != "route"),
        "3_route_score": route["score"],
        "3_holds": route["score"] is not None and route["score"] > 0.0,
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
    parser.add_argument("command", choices=("build", "smoke", "run", "collect"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--check", action="store_true", help="for build: compare the file with the generator")
    args = parser.parse_args(argv)
    if args.command == "build":
        text = json.dumps(build(), indent=1, sort_keys=True) + "\n"
        if args.check:
            if OUT.read_text(encoding="utf-8") != text:
                print(f"{OUT.relative_to(ROOT)} is not what the generator gives")
                return 1
            print(f"{OUT.relative_to(ROOT)}: as generated")
            return 0
        OUT.write_text(text, encoding="utf-8")
        record = json.loads(text)
        print(f"wrote {OUT.relative_to(ROOT)}: {len(record['laws'])} laws, {len(record['cases'])} cases")
        return 0
    if args.command == "smoke":
        picked = cases()[:4]
        rows = red_team.evaluate([rt.World.from_dict(case["world"]) for case in picked], red_team._cores())
        for case, row in zip(picked, rows, strict=True):
            if row["status"] != "ok":
                print(case["case"], row["status"], row.get("error"), row.get("trace", ""))
                return 1
            print(f"{case['case']:28s} route: {row['route']['action']:17s} wrong={row['route']['wrong']!s:5s} "
                  f"best_fit wrong={row['best_fit']['wrong']!s}")
        print(json.dumps(summarise(rows)["expectations"], indent=1))
        return 0
    if args.out is None:
        parser.error("--out is required")
    args.out.mkdir(parents=True, exist_ok=True)
    k, n = (int(v) for v in args.shard.split("/"))
    if args.command == "run":
        check_frozen()
        mine = [case for i, case in enumerate(cases()) if i % n == k]
        rows = red_team.evaluate([rt.World.from_dict(case["world"]) for case in mine], red_team._cores(), TIME_LIMIT_S)
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for case, row in zip(mine, rows, strict=True):
                handle.write(json.dumps({"case": case["case"], "law_id": case["law_id"], **row}, sort_keys=True, default=str) + "\n")
        return 0
    rows = [json.loads(line) for path in sorted(args.out.glob("runs-shard-*.jsonl")) for line in path.read_text().splitlines()]
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    summary = summarise(rows)
    record = {"preregistration": PREREGISTRATION_ID, "rows": len(rows), "summary": summary, "frozen": frozen_digests()}
    (args.out / "results.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"collected {len(rows)} rows into {args.out}")
    print(json.dumps({key: value for key, value in summary.items() if key.endswith("|all") or key == "expectations"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
