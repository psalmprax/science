#!/usr/bin/env python3
"""PREREG-MMLM-ORDER-2026-11: the default route with and without MMLM's order of candidates, once.

Status: audit-only

On the owner's word ("go ahead to test all", 27 November 02:32), after the question whether MMLM is
reliable for APEDP-X. Two arms on every law and noise level, **given the same searched candidates**: the
default search stage runs once per unit, outside both arms, and each arm replays its candidates into the
route in force (``discovery_l2.refined_search.discover_refined``):

- ``default``: the route as it is in force, whose guided search orders candidates by size;
- ``mmlm``: the same route with MMLM's law model (``apedp_mmlm.law_language.LawLanguageModel``, fitted on
  SRBench's 133 ground-truth laws, frozen, pooled over domains) in place of the size prior. The prior does two
  things in the guided search: it orders the candidates (most plausible first, so the confirmed stop may come
  sooner) and it is their structural code length in the description length, so it can also change which
  candidate wins.

So the arms differ in the prior and in nothing a search happened to find. Each arm's guided search is prepared
once per process with every candidate priced (as the route does in use); that pricing is timed apart.

Laws: set ``invented``, 50 laws of the invented-law grammar drawn **at run time** with seed 2026112731 (the
rehearsal uses 2026112732; no other test draws from either), leaving out every structure of the invented-laws benchmark (by fingerprint, no
held-back law read), of the eighty laws the search proposers were developed on, and of the memory test's
forty; set ``other_fields``, the 24 laws from chemistry, biology, engineering, finance and geometry of
PREREG-MMLM-S (none in MMLM's training laws). Rows, noise, split and scoring are the invented-laws runner's
and through it the SRBench runner's: inputs on (1, 5), noise 0 and 1 %, 1,000 rows, 70 % for development,
30 % the pool, 500 noise-free rows for scoring.

    python scripts/run_prereg_mmlm_order.py smoke --out DIR        # two laws, a short real search, both arms
    python scripts/run_prereg_mmlm_order.py run --out DIR --shard k/N [--rehearsal]
    python scripts/run_prereg_mmlm_order.py collect --out DIR
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
import run_prereg_mmlm_track_s as track_s  # noqa: E402
import run_search_evolution as development  # noqa: E402
import run_srbench_baseline as base  # noqa: E402
import run_srbench_main_search as path_run  # noqa: E402

from apedp_dyn.conf.discovery_checks import (  # noqa: E402
    REPORTED_WRONG_SHARE_BOUND_CONFIDENCE,
    REPORTED_WRONG_SHARE_BOUND_MAX,
    REPORTED_WRONG_SHARE_MAX,
)

PREREGISTRATION_ID = "PREREG-MMLM-ORDER-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-MMLM-ORDER.json"
FROZEN = (
    "preregistration/PREREG-MMLM-ORDER-2026-11.md", "scripts/run_prereg_mmlm_order.py",
    ".github/workflows/mmlm-order.yml", "tests/test_mmlm_order_prereg.py",
    "apedp_mmlm/law_language.py", "apedp_mmlm/equation_language.py",
    "scripts/build_invented_laws.py", "scripts/run_srbench_baseline.py", "scripts/run_srbench_main_search.py",
    "scripts/run_prereg_mmlm_track_s.py", "scripts/run_search_evolution.py", "benchmarks/memory_blind_laws.json",
    "apedp_dyn/discovery_l2/search_stage.py", "apedp_dyn/discovery_l2/refined_search.py",
    "apedp_dyn/discovery_l2/main_search.py", "apedp_dyn/discovery_l2/confirmed_stop.py",
    "apedp_dyn/ode/extended_route.py", "apedp_dyn/proposers/portfolio.py", "apedp_dyn/proposers/pysr_proposer.py",
)
LAW_SEED, REHEARSAL_SEED, LAW_COUNT = 2026112731, 2026112732, 50   # no other test draws from these
ARMS = ("default", "mmlm")
NOISE_LEVELS = (0.0, 0.01)
SEARCH_BUDGET_S = 240.0
TIME_LIMIT_S = 1800.0                   # one law at one noise level: the search pass, then both arms
FASTER = 0.8                            #: expectation 2: MMLM's median route time at most this share of the default's
SMOKE_LAWS = tuple({"name": name, "family": "invented", "features": features, "formula": formula, "units": {},
                    "target_units": {}, "domain": [grammar.LOW, grammar.HIGH], "set": "smoke"}
                   for name, features, formula in (("smoke_ratio", ["x0", "x1"], "1.7*x0/(x0 + x1)"),
                                                   ("smoke_power", ["x0"], "0.9*x0**2 + 1.3")))


# ---------------------------------------------------------------- laws
def invented_laws(seed: int = LAW_SEED, count: int = LAW_COUNT) -> list[dict[str, Any]]:
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    developed_on = {grammar.structure(sp.sympify(law["text"]))
                    for law in development.draw_laws(development.TRAIN + development.VALIDATION)}
    memory = {grammar.structure(sp.sympify(law["text"]))
              for law in json.loads((ROOT / "benchmarks" / "memory_blind_laws.json").read_text())["laws"]}
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
        if grammar.fingerprint(str(expr)) in in_benchmark or key in developed_on or key in memory:
            continue
        laws.append({"id": 5000 + len(laws), "inputs": inputs, "text": str(expr)})
    if len(laws) != count:
        raise SystemExit(f"only {len(laws)} laws after {grammar.MAX_DRAWS} draws")
    return laws


def entries(seed: int = LAW_SEED) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for law in invented_laws(seed):
        name = f"order_{law['id']}"
        out[name] = {"name": name, "family": "invented", "features": [f"x{i}" for i in range(law["inputs"])],
                     "formula": law["text"], "units": {}, "target_units": {}, "domain": [grammar.LOW, grammar.HIGH],
                     "set": "invented"}
    for name, text in track_s.OTHER_FIELD_LAWS.items():
        n = len(sp.sympify(text).free_symbols)
        out[f"field_{name}"] = {"name": f"field_{name}", "family": "invented", "features": [f"x{i}" for i in range(n)],
                                "formula": text, "units": {}, "target_units": {}, "domain": [grammar.LOW, grammar.HIGH],
                                "set": "other_fields"}
    return out


def frame_of(entry: dict[str, Any]) -> pd.DataFrame:
    low, high = entry["domain"]
    rng = np.random.default_rng(base._seed(PREREGISTRATION_ID, entry["name"], "rows"))
    X = rng.uniform(low, high, (base.N_TRAIN + base.N_TEST, len(entry["features"])))
    truth = base.truth_expr(entry)
    fn = sp.lambdify([sp.Symbol(f"x{i}") for i in range(X.shape[1])], truth, "numpy")
    frame = pd.DataFrame(X, columns=entry["features"])
    frame["target"] = np.asarray(fn(*X.T), dtype=float)
    return frame


# ---------------------------------------------------------------- MMLM's prior
_PRIOR: Any = None
GUIDED: dict[str, Any] = {}             #: each arm's guided search, its orders priced once before the units run
PRICING: dict[str, float] = {}          #: seconds each arm took to price every candidate, once (not route time)


def mmlm_prior() -> Any:
    """MMLM's law model fitted on SRBench's 133 ground-truth laws (frozen, pooled over domains), as ``structure_bits``."""
    global _PRIOR
    if _PRIOR is None:
        from apedp_mmlm import equation_language as el
        from apedp_mmlm import law_language as ll

        trees = [el.to_tree(base.truth_expr(e)) for e in base.load_manifest()]
        _PRIOR = ll.search_prior(ll.LawLanguageModel().fit(trees, [None] * len(trees)))
    return _PRIOR


def prepare(inputs: set[int], max_complexity: int = 6) -> None:
    """Both arms' guided searches with every candidate priced, once, before the units fork from this process.

    The route reuses a prepared search order across data sets (``main_search.discover``'s ``guided``); pricing
    every candidate with a learned prior is the costly part, done once in use, so it is timed here apart from
    the route.
    """
    from apedp_dyn.discovery_l2.guided_search import GuidedSearch
    from apedp_dyn.discovery_l2.main_search import size_prior

    for arm, bits in (("default", size_prior), ("mmlm", mmlm_prior())):
        if arm in GUIDED:
            continue
        guided = GuidedSearch(bits)
        started = time.perf_counter()
        for n in sorted(inputs):
            guided.order(n, max_complexity)
        PRICING[arm] = round(time.perf_counter() - started, 2)
        GUIDED[arm] = guided


# ---------------------------------------------------------------- one law at one noise level
def arms_of(entry: dict[str, Any], split: dict[str, Any], budget: float) -> list[dict[str, Any]]:
    from apedp_dyn.discovery_l2 import refined_search, search_stage
    from apedp_dyn.ode.extended_route import Recording, replayed

    names, X, y = split["names"], split["X_train"], split["y_train"]
    order = np.random.default_rng(base._seed(PREREGISTRATION_ID, entry["name"], "pool")).permutation(len(y))
    cut = round(path_run.DEV_SHARE * len(y))
    dev, pool = order[:cut], order[cut:]
    pool_y = y[pool]
    searched: list[str] = []
    search_seconds = 0.0
    if budget > 0:
        # The searches, once, for both arms: a pass of the route with the recorded stage, whose result is not used.
        started = time.perf_counter()
        stage = Recording(search_stage.default_search_stage(budget, seed=base.SEED)[0])
        refined_search.discover_refined({n: X[dev, i] for i, n in enumerate(names)}, y[dev], X[pool],
                                        lambda picked: pool_y[picked], proposer=stage, noise="absolute", seed=base.SEED,
                                        guided=GUIDED["default"])
        searched = list(stage.formulas)
        search_seconds = round(time.perf_counter() - started, 2)
    rows: list[dict[str, Any]] = []
    # Which arm runs first alternates by a draw from the unit's name, so neither always pays the warm-up.
    first = int(base._seed(PREREGISTRATION_ID, entry["name"], split["noise_level"], "order")) % 2
    for place, arm in enumerate(ARMS if first == 0 else ARMS[::-1]):
        options: dict[str, Any] = {"guided": GUIDED[arm]}
        started = time.perf_counter()
        result = refined_search.discover_refined({n: X[dev, i] for i, n in enumerate(names)}, y[dev], X[pool],
                                                 lambda picked: pool_y[picked], proposer=replayed(searched),
                                                 noise="absolute", seed=base.SEED, **options)
        seconds = round(time.perf_counter() - started, 2)
        law = sp.sympify(result.law, locals={n: sp.Symbol(n) for n in names}) if result.law else None
        path = result.path
        raw = {"model": law, "pred": path_run._predict(law, names, split["X_test"]) if law is not None else None,
               "formula": result.law}
        rows.append({"arm": arm, "status": "ok", "seconds": seconds, "search_seconds": search_seconds, "ran": place,
                     "searched": len(searched), "action": result.action,
                     "candidates_fitted": len(path.candidates) if path is not None else None,
                     **base.score_run("mmlm_order", entry, split, raw)})
    return rows


def _child(conn: Any, entry: dict[str, Any], split: dict[str, Any], budget: float) -> None:
    try:
        conn.send(("ok", arms_of(entry, split, budget)))
    except Exception as exc:  # noqa: BLE001 - a crash of the unit is a scored outcome, reported with its class and message
        conn.send(("error", {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-2000:]}))
    finally:
        conn.close()


def run_unit(entry: dict[str, Any], noise: float, budget: float = SEARCH_BUDGET_S,
             time_limit: float = TIME_LIMIT_S) -> list[dict[str, Any]]:
    prepare({len(entry["features"])})
    head = {"problem": entry["name"], "set": entry["set"], "noise": noise, "inputs": len(entry["features"]),
            "truth": entry["formula"]}
    split = {**base.make_split(entry, frame_of(entry), noise), "noise_level": noise}
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


REHEARSAL_LAWS = 15                     #: the rehearsal: its own seed's first 15 invented laws, no other-field law


def units(seed: int = LAW_SEED, rehearsal: bool = False) -> list[tuple[dict[str, Any], float]]:
    chosen = list(entries(seed).values())
    if rehearsal:
        chosen = [e for e in chosen if e["set"] == "invented"][:REHEARSAL_LAWS]
    return [(entry, noise) for entry in chosen for noise in NOISE_LEVELS]


# ---------------------------------------------------------------- summary
def _reported(row: dict[str, Any]) -> bool:
    return row["status"] == "ok" and row.get("model") not in (None, "None")


def _wrong(row: dict[str, Any]) -> bool:
    return _reported(row) and not row.get("symbolic") and not row.get("accuracy_solution")


def upper_bound(wrong: int, reported: int, confidence: float = REPORTED_WRONG_SHARE_BOUND_CONFIDENCE) -> float:
    """One-sided exact (Clopper-Pearson) upper bound on the share of wrong laws among those reported."""
    from scipy.stats import beta

    if reported == 0:
        return 1.0
    if wrong >= reported:
        return 1.0
    return float(beta.ppf(confidence, wrong + 1, reported - wrong))


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ARMS:
        for group in ("invented", "other_fields", "all"):
            sel = [r for r in rows if r["arm"] == arm and group in (r["set"], "all")]
            ok = [r for r in sel if r["status"] == "ok"]
            out[f"{arm}|{group}"] = {
                "runs": len(sel), "errors": sum(r["status"] == "error" for r in sel),
                "timeouts": sum(r["status"] == "timeout" for r in sel), "reported": sum(_reported(r) for r in ok),
                "recovered": sum(bool(r.get("symbolic")) for r in ok), "wrong": sum(_wrong(r) for r in ok),
                "median_seconds": float(np.median([r["seconds"] for r in ok])) if ok else None,
                "total_seconds": round(float(sum(r["seconds"] for r in ok)), 1)}
    recovered = {(r["arm"], r["noise"], r["problem"]) for r in rows if r["status"] == "ok" and r.get("symbolic")}
    cells = sorted({(r["noise"], r["problem"], r["set"]) for r in rows})
    out["run_for_run"] = {group: {
        "gained": sum(("mmlm", n, p) in recovered and ("default", n, p) not in recovered for n, p, s in cells if s == group),
        "lost": sum(("mmlm", n, p) not in recovered and ("default", n, p) in recovered for n, p, s in cells if s == group)}
        for group in ("invented", "other_fields")}
    paired = [(a["seconds"], b["seconds"]) for a in rows for b in rows
              if a["arm"] == "mmlm" and b["arm"] == "default" and a["problem"] == b["problem"]
              and a["noise"] == b["noise"] and a["status"] == b["status"] == "ok"]

    def got(arm: str, group: str, key: str) -> Any:
        return out[f"{arm}|{group}"][key]

    def share(arm: str) -> tuple[float, float]:
        rep, wr = got(arm, "all", "reported"), got(arm, "all", "wrong")
        return (wr / rep if rep else 0.0), upper_bound(wr, rep)

    med_mmlm = float(np.median([a for a, _ in paired])) if paired else None
    med_def = float(np.median([b for _, b in paired])) if paired else None
    s_mmlm, b_mmlm = share("mmlm")
    s_def, _ = share("default")
    out["expectations"] = {
        "1 no laws lost: MMLM's order recovers at least as many on each set": {
            group: {"mmlm": got("mmlm", group, "recovered"), "default": got("default", group, "recovered")}
            for group in ("invented", "other_fields")},
        "1_holds": all(got("mmlm", g, "recovered") >= got("default", g, "recovered") for g in ("invented", "other_fields")),
        "2 faster: MMLM's median route time at most 0.8 of the default's, paired units": {
            "mmlm": med_mmlm, "default": med_def, "units": len(paired)},
        "2_holds": bool(paired) and med_mmlm is not None and med_def is not None and med_mmlm <= FASTER * med_def,
        "3 no more wrong laws: MMLM's share within the limit, its bound within the bound, not above the default's": {
            "mmlm_share": round(s_mmlm, 4), "mmlm_upper_bound": round(b_mmlm, 4), "default_share": round(s_def, 4),
            "limit": REPORTED_WRONG_SHARE_MAX, "bound_limit": REPORTED_WRONG_SHARE_BOUND_MAX},
        "3_holds": s_mmlm <= REPORTED_WRONG_SHARE_MAX and b_mmlm <= REPORTED_WRONG_SHARE_BOUND_MAX
        and got("mmlm", "all", "wrong") <= got("default", "all", "wrong"),
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
    keep = ("problem", "set", "noise", "arm", "status", "symbolic", "accuracy_solution", "action", "seconds", "error")
    return json.dumps({k: row.get(k) for k in keep}, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("smoke", "run", "collect"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--rehearsal", action="store_true",
                        help="for run: 15 invented laws of another seed, 60 s of search, no other-field law")
    parser.add_argument("--budget", type=float, default=20.0, help="for smoke: seconds of search")
    args = parser.parse_args(argv)
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
    if args.command == "run":
        if args.rehearsal:
            seed, budget = REHEARSAL_SEED, 60.0
        else:
            check_frozen()
            seed, budget = LAW_SEED, SEARCH_BUDGET_S
        mine = [unit for i, unit in enumerate(units(seed, rehearsal=args.rehearsal)) if i % n == k]
        prepare({len(entry["features"]) for entry, _ in mine} or {1})
        (args.out / f"pricing-shard-{k:03d}.json").write_text(json.dumps(PRICING) + "\n", encoding="utf-8")
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for entry, noise in mine:
                for row in run_unit(entry, noise, budget=budget):
                    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                    print(_printable(row), flush=True)
                handle.flush()
        return 0
    rows = [json.loads(line) for path in sorted(args.out.glob("runs-shard-*.jsonl")) for line in path.read_text().splitlines()]
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    pricing = [json.loads(p.read_text()) for p in sorted(args.out.glob("pricing-shard-*.json"))]
    record = {"preregistration": PREREGISTRATION_ID, "rows": len(rows), "summary": summarise(rows),
              "pricing_seconds_per_shard": pricing, "frozen": frozen_digests()}
    (args.out / "results.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
