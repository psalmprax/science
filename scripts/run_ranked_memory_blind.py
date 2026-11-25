#!/usr/bin/env python3
"""Does the memory's list in a learned order, beside the memory in force, make the table route report more right laws?

    python scripts/run_ranked_memory_blind.py laws [--check]              # benchmarks/ranked_memory_blind_laws.json
    python scripts/run_ranked_memory_blind.py smoke     --out DIR         # two laws, a short real search
    python scripts/run_ranked_memory_blind.py rehearsal --out DIR --shard k/N   # development laws, not the run
    python scripts/run_ranked_memory_blind.py run       --out DIR --shard k/N   # PREREG-RANKED-MEMORY-2026-11, once
    python scripts/run_ranked_memory_blind.py collect   --out DIR

Two arms on every law and noise level, through the table route in force with its last checks
(``discovery_l2/vouched_route_v2.discover_law``), **given the same candidates of the stage in force**:
the stage (four searches side by side and the memory of confirmed laws) runs once, and

- ``in force`` hands the path its candidates: what the route does now;
- ``ranked`` hands it the same candidates and, after them, up to three candidates of the memory's
  list that a learned order puts first and whose shape the stage did not hand over
  (``discovery_l2/ranked_stage.py``). The stage hands over ten; where its list is full the added
  ones take the place of its last ones.

So the arms differ in the ranked proposer's candidates and in nothing a second search happened to find.

The laws are drawn here, after the ranked proposer, its stage and its order were written, from the
invented-law grammar with their own seed. A draw is left out if its structure is in the
invented-laws benchmark (by fingerprint; no held-back law is read), among the laws the memory
proposer was developed on, among the laws of the memory's own blind test, or among the curriculum
problems the order was learned and scored on. Rows, noise, the split and the scoring are the
invented-laws runner's and through it the SRBench runner's, as in PREREG-MEMORY-2026-11.
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
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_invented_laws as grammar  # noqa: E402
import run_search_evolution as development  # noqa: E402
import run_srbench_baseline as base  # noqa: E402
import run_srbench_main_search as path_run  # noqa: E402

from apedp_dyn.discovery_l2 import report_share  # noqa: E402

PREREGISTRATION_ID = "PREREG-RANKED-MEMORY-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-RANKED-MEMORY.json"
LAWS_FILE = ROOT / "benchmarks" / "ranked_memory_blind_laws.json"
MEMORY_BLIND_LAWS = ROOT / "benchmarks" / "memory_blind_laws.json"
FROZEN = (
    "preregistration/PREREG-RANKED-MEMORY-2026-11.md", "scripts/run_ranked_memory_blind.py",
    "benchmarks/ranked_memory_blind_laws.json", "benchmarks/structure_memory.json",
    "benchmarks/order_score/learned_order_in_force.json",
    "scripts/build_invented_laws.py", "scripts/run_srbench_baseline.py", "scripts/run_srbench_main_search.py",
    "apedp_dyn/discovery_l2/ranked_stage.py", "apedp_dyn/proposers/ranked_memory.py", "apedp_dyn/proposers/learned_order.py",
    "apedp_dyn/proposers/structure_memory_counted.py", "apedp_dyn/proposers/structure_memory.py",
    "apedp_dyn/proposers/hard_exit.py", "apedp_dyn/proposers/fit_guard.py", "apedp_dyn/proposers/portfolio.py",
    "apedp_dyn/proposers/composed.py", "apedp_dyn/proposers/deep_pass.py", "apedp_dyn/proposers/pysr_proposer.py",
    "apedp_dyn/discovery_l2/memory_stage.py", "apedp_dyn/discovery_l2/stage_in_force.py",
    "apedp_dyn/discovery_l2/search_stage.py", "apedp_dyn/discovery_l2/evolved_stage.py",
    "apedp_dyn/discovery_l2/vouched_route_v2.py", "apedp_dyn/discovery_l2/vouching_v2.py",
    "apedp_dyn/discovery_l2/default_route.py", "apedp_dyn/discovery_l2/refined_search.py",
    "apedp_dyn/discovery_l2/main_search.py", "apedp_dyn/discovery_l2/contrast_confirmation.py",
    "apedp_dyn/discovery_l2/report_share.py", "apedp_dyn/ode/extended_route.py", "apedp_dyn/recursion/mdl.py",
    "specifications/discovery_checks_v1.json",
)
LAW_SEED = 20261125
LAW_COUNT = 100
ARMS = ("in force", "ranked")
NOISE_LEVELS = (0.0, 0.01)
SEARCH_BUDGET_S = 240.0                  # as in the run that put the memory beside the searches
TIME_LIMIT_S = 1800.0                    # one law at one noise level: the stage once, then both arms
REHEARSAL_LAWS = 16                      # of the laws the memory proposer was developed on; never a law of the run
#: two laws that are in no set, for the smoke run
SMOKE_LAWS = tuple({"name": name, "family": "invented", "features": features, "formula": formula, "units": {},
                    "target_units": {}, "domain": [grammar.LOW, grammar.HIGH]}
                   for name, features, formula in (("smoke_sum", ["x0", "x1"], "0.7*x0**2 + 1.3/x1"),
                                                   ("smoke_product", ["x0"], "1.7*x0*sin(1.3*x0)")))


# ---------------------------------------------------------------- laws
def _structures_seen_before() -> set[str]:
    """Structures no law of this run may have: every set an earlier development or test drew from this family."""
    import measure_order_score as order_run

    seen = {grammar.structure(sp.sympify(law["text"])) for law in development.draw_laws(development.TRAIN + development.VALIDATION)}
    seen |= {grammar.structure(sp.sympify(law["text"])) for law in json.loads(MEMORY_BLIND_LAWS.read_text(encoding="utf-8"))["laws"]}
    first = order_run.test_problems(order_run.PROBLEMS)
    for problem in (*first, *order_run.teaching_problems(max(order_run.TAUGHT), first), *order_run.fresh_problems()):
        seen.add(grammar.structure(problem.truth))
    return seen


def build_laws(count: int = LAW_COUNT, seed: int = LAW_SEED) -> dict[str, Any]:
    """``count`` laws of the invented-law grammar that no development or earlier test has met."""
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    met = _structures_seen_before()
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
        if grammar.fingerprint(str(expr)) in in_benchmark or key in met:
            continue
        laws.append({"id": 5000 + len(laws), "inputs": inputs, "text": str(expr)})
    if len(laws) != count:
        raise SystemExit(f"only {len(laws)} laws after {grammar.MAX_DRAWS} draws")
    return {"what": "Laws for one preregistered comparison of the table route in force with and without the memory's list "
                    "in a learned order. Drawn after the ranked proposer, its stage and its order were written; no route "
                    "has run on them.",
            "seed": seed, "domain": [grammar.LOW, grammar.HIGH], "laws": laws}


def _entry(name: str, inputs: int, text: str, domain: list[float]) -> dict[str, Any]:
    return {"name": name, "family": "invented", "features": [f"x{i}" for i in range(inputs)], "formula": text,
            "units": {}, "target_units": {}, "domain": domain}


def entries() -> dict[str, dict[str, Any]]:
    """The laws of the run in the SRBench runs' form."""
    record = json.loads(LAWS_FILE.read_text(encoding="utf-8"))
    return {f"ranked_{law['id']}": _entry(f"ranked_{law['id']}", law["inputs"], law["text"], record["domain"])
            for law in record["laws"]}


def rehearsal_entries(count: int = REHEARSAL_LAWS) -> dict[str, dict[str, Any]]:
    """Development laws for a dress rehearsal: laws the memory proposer was developed on, never a law of the run."""
    laws = development.draw_laws(development.TRAIN + development.VALIDATION)[:count]
    out = {}
    for index, law in enumerate(laws):
        expr = sp.sympify(law["text"])
        inputs = 1 + max(int(str(symbol)[1:]) for symbol in expr.free_symbols)
        out[f"rehearsal_{index}"] = _entry(f"rehearsal_{index}", inputs, law["text"], [grammar.LOW, grammar.HIGH])
    return out


def frame_of(entry: dict[str, Any]) -> pd.DataFrame:
    """Rows drawn uniformly on the law's domain, with the law's own value as the target (no noise yet)."""
    low, high = entry["domain"]
    rng = np.random.default_rng(base._seed(PREREGISTRATION_ID, entry["name"], "rows"))
    X = rng.uniform(low, high, (base.N_TRAIN + base.N_TEST, len(entry["features"])))
    truth = base.truth_expr(entry)
    fn = sp.lambdify([sp.Symbol(f"x{i}") for i in range(X.shape[1])], truth, "numpy")
    frame = pd.DataFrame(X, columns=entry["features"])
    frame["target"] = np.asarray(fn(*X.T), dtype=float) * np.ones(len(X))
    return frame


# ---------------------------------------------------------------- one law at one noise level
def _is_the_law(text: str, truth: sp.Expr, names: list[str], X_test: np.ndarray) -> bool:
    try:
        expr = sp.sympify(text, locals={n: sp.Symbol(n) for n in names})
        return bool(base.symbolic_recovery(expr, truth, names, X_test)["recovered"])
    except Exception:  # noqa: BLE001 - a candidate that cannot be read is not the law
        return False


def arms_of(entry: dict[str, Any], split: dict[str, Any], budget: float) -> list[dict[str, Any]]:
    """Both arms on one law: a row each. The stage in force runs once, in the first arm."""
    from apedp_dyn.discovery_l2 import default_route, memory_stage, ranked_stage, vouched_route_v2
    from apedp_dyn.ode.extended_route import Recording, replayed
    from apedp_dyn.proposers import hard_exit

    names, X, y = split["names"], split["X_train"], split["y_train"]
    order = np.random.default_rng(base._seed(PREREGISTRATION_ID, entry["name"], "pool")).permutation(len(y))
    cut = round(path_run.DEV_SHARE * len(y))
    dev, pool = order[:cut], order[cut:]
    pool_y = y[pool]
    truth = base.truth_expr(entry)
    memory, learned = memory_stage.recorded_memory(), ranked_stage.recorded_order()
    rows: list[dict[str, Any]] = []
    handed: list[str] = []
    for arm in ARMS:
        started = time.perf_counter()
        stage: Any
        if arm == "in force":
            stage = Recording(hard_exit.stage_in_force_that_cannot_hang(budget, seed=base.SEED)[0]) if budget > 0 else None
        else:
            stage = ranked_stage.with_ranked(replayed(handed), memory, learned)
        out = vouched_route_v2.discover_law({n: X[dev, i] for i, n in enumerate(names)}, y[dev], X[pool],
                                            lambda picked: pool_y[picked], ledger=None, search_id="ranked-memory",
                                            route_name=default_route.default_route(), proposer=stage, noise="absolute",
                                            seed=base.SEED)
        if arm == "in force":
            handed = list(stage.formulas) if stage is not None else []
        path = out["main_search"]
        reported = bool(out["reported"])
        text = out.get("law_as_confirmed") if reported else None
        law = sp.sympify(str(text), locals={n: sp.Symbol(n) for n in names}) if text else None
        raw = {"model": law, "pred": path_run._predict(law, names, split["X_test"]) if law is not None else None,
               "formula": str(text) if text else None}
        vouching = dict(out.get("vouching") or {})
        extra: dict[str, Any] = {
            "reported": reported, "action": out["action"], "tentative": out.get("tentative_law") is not None,
            "law_as_confirmed": out.get("law_as_confirmed"), "found_by": getattr(path, "found_by", None),
            "vouching_applied": bool(vouching.get("applied")), "vouching_reason": vouching.get("reason"),
            "candidates": len(getattr(path, "candidates", []) or []), "handed_by_the_stage": handed,
            "proposer_failures": list(getattr(stage, "failures", []) or [])}
        if arm == "ranked":
            done = dict((getattr(stage, "last_run", {}) or {}).get("ranked", {}))
            extra["ranked"] = {**done,
                               "the_law_among_the_added": any(_is_the_law(t, truth, names, split["X_test"]) for t in done.get("added", [])),
                               "the_law_among_the_displaced": any(_is_the_law(t, truth, names, split["X_test"])
                                                                  for t in done.get("displaced", []))}
        rows.append({"arm": arm, "status": "ok", "seconds": round(time.perf_counter() - started, 2),
                     **base.score_run("ranked_memory", entry, split, raw), **extra})
    return rows


def _child(conn: Any, entry: dict[str, Any], split: dict[str, Any], budget: float) -> None:
    try:
        conn.send(("ok", arms_of(entry, split, budget)))
    except Exception as exc:  # noqa: BLE001 - a crash of the unit is a scored outcome, reported with its class and message
        conn.send(("error", {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-2000:]}))
    finally:
        conn.close()


def run_unit(entry: dict[str, Any], noise: float, budget: float = SEARCH_BUDGET_S, time_limit: float = TIME_LIMIT_S
             ) -> list[dict[str, Any]]:
    """One law at one noise level: a row per arm. A crash or a timeout gives both arms that status."""
    head = {"problem": entry["name"], "noise": noise, "inputs": len(entry["features"]), "truth": entry["formula"]}
    split = base.make_split(entry, frame_of(entry), noise)
    context = mp.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    started = time.perf_counter()
    process = context.Process(target=_child, args=(child, entry, split, budget))
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


def units(which: dict[str, dict[str, Any]] | None = None) -> list[tuple[dict[str, Any], float]]:
    return [(entry, noise) for entry in (entries() if which is None else which).values() for noise in NOISE_LEVELS]


# ---------------------------------------------------------------- summary
def _reported(row: dict[str, Any]) -> bool:
    return row["status"] == "ok" and bool(row.get("reported")) and row.get("model") not in (None, "None")


def _recovered(row: dict[str, Any]) -> bool:
    return _reported(row) and bool(row.get("symbolic"))


def _right(row: dict[str, Any]) -> bool:
    """Reported, and the law itself or accurate on the scoring rows."""
    return _reported(row) and (bool(row.get("symbolic")) or bool(row.get("accuracy_solution")))


def _wrong(row: dict[str, Any]) -> bool:
    return _reported(row) and not _right(row)


def _pair(rows: list[dict[str, Any]], what: Any, noise: Any = "all") -> dict[str, Any]:
    """Run for run: units where only the ranked arm has ``what`` (gained) and where only the arm in force has it (lost)."""
    have = {(r["arm"], r["noise"], r["problem"]) for r in rows if what(r)}
    cells = sorted({(r["noise"], r["problem"]) for r in rows if noise in (r["noise"], "all")})
    gained = sum(("ranked", *c) in have and ("in force", *c) not in have for c in cells)
    lost = sum(("ranked", *c) not in have and ("in force", *c) in have for c in cells)
    return {"gained": gained, "lost": lost,
            "sign_test_p": float(binomtest(gained, gained + lost, 0.5).pvalue) if gained + lost else 1.0}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per arm and noise level: runs, reports, right and wrong laws, recoveries; then the expectations as written."""
    out: dict[str, Any] = {}
    for arm in ARMS:
        for noise in (*NOISE_LEVELS, "all"):
            sel = [r for r in rows if r["arm"] == arm and noise in (r["noise"], "all")]
            ok = [r for r in sel if r["status"] == "ok"]
            done = [r.get("ranked") or {} for r in ok]
            out[f"{arm}|{noise}"] = {
                "runs": len(sel), "errors": sum(r["status"] == "error" for r in sel),
                "timeouts": sum(r["status"] == "timeout" for r in sel), "reported": sum(_reported(r) for r in ok),
                "right": sum(_right(r) for r in ok), "recovered": sum(_recovered(r) for r in ok),
                "wrong": sum(_wrong(r) for r in ok),
                "reported_accurate_not_the_law": sum(_right(r) and not _recovered(r) for r in ok),
                "tentative": sum(bool(r.get("tentative")) for r in ok),
                "ranked_added_a_candidate": sum(bool(d.get("added")) for d in done),
                "ranked_added_the_law": sum(bool(d.get("the_law_among_the_added")) for d in done),
                "ranked_displaced_the_law": sum(bool(d.get("the_law_among_the_displaced")) for d in done),
                "median_seconds": float(np.median([r["seconds"] for r in sel])) if sel else None,
            }
    out["run_for_run"] = {f"{name}|{noise}": _pair(rows, what, noise)
                          for name, what in (("right", _right), ("recovered", _recovered), ("reported", _reported))
                          for noise in (*NOISE_LEVELS, "all")}

    def count(arm: str, noise: Any, key: str) -> int:
        return int(out.get(f"{arm}|{noise}", {}).get(key, 0))

    shares = {arm: report_share.verdict(count(arm, "all", "wrong"), count(arm, "all", "reported")) for arm in ARMS}
    out["expectations"] = {
        "1_share_of_reported_laws_wrong": shares,
        "1_holds": all(share["verdict"] == report_share.HELD for share in shares.values()),
        "2_right_ranked": {str(n): count("ranked", n, "right") for n in (*NOISE_LEVELS, "all")},
        "2_right_in_force": {str(n): count("in force", n, "right") for n in (*NOISE_LEVELS, "all")},
        "2_holds": all(count("ranked", n, "right") >= count("in force", n, "right") for n in NOISE_LEVELS)
        and count("ranked", "all", "right") > count("in force", "all", "right"),
        "3_recovered_ranked": count("ranked", "all", "recovered"),
        "3_recovered_in_force": count("in force", "all", "recovered"),
        "3_holds": count("ranked", "all", "recovered") >= count("in force", "all", "recovered"),
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
    keep = ("problem", "noise", "arm", "status", "reported", "symbolic", "accuracy_solution", "action", "formula", "seconds",
            "error")
    line = {k: row.get(k) for k in keep}
    line["ranked"] = {k: v for k, v in (row.get("ranked") or {}).items() if k != "displaced"} or None
    return json.dumps(line, default=str)


def _run_shard(which: list[tuple[dict[str, Any], float]], out: Path, k: int, n: int, budget: float) -> None:
    path = out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for entry, noise in [unit for i, unit in enumerate(which) if i % n == k]:
            for row in run_unit(entry, noise, budget=budget):
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                print(_printable(row), flush=True)
            handle.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("laws", "smoke", "rehearsal", "run", "collect"))
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
            for row in run_unit(entry, 0.01, budget=args.budget, time_limit=900.0):
                print(_printable(row), flush=True)
                if row["status"] != "ok":
                    print(row.get("trace", ""), flush=True)
                    failed += 1
        return 1 if failed else 0
    if args.command == "rehearsal":
        _run_shard(units(rehearsal_entries()), args.out, k, n, SEARCH_BUDGET_S)
        return 0
    if args.command == "run":
        check_frozen()
        _run_shard(units(), args.out, k, n, SEARCH_BUDGET_S)
        return 0
    rows = [json.loads(line) for path in sorted(args.out.glob("runs-shard-*.jsonl")) for line in path.read_text().splitlines()]
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    summary = summarise(rows)
    record = {"preregistration": PREREGISTRATION_ID, "rows": len(rows), "summary": summary,
              "frozen": frozen_digests() if DIGEST.exists() and LAWS_FILE.exists() else None}
    (args.out / "results.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"collected {len(rows)} rows into {args.out}")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
