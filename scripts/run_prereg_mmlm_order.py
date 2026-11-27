#!/usr/bin/env python3
"""PREREG-MMLM-ORDER-2026-11: MMLM's prior against the size prior in the table route in force, once.

Status: audit-only

On the owner's word ("go ahead to test all", 27 November 02:32), after the question whether MMLM is reliable for
APEDP-X. The answer given then was that it may help order a search *on familiar ground*. This test checks that
claim where it should hold: on laws MMLM has met in shape before, and a guard set where it cannot help.

The route is the table route in force (``vouched_route_v2``): the refined path (``refined_search.discover_refined``)
with the second version of the last checks (``vouched_route_v2.checked``), behind the search stage in force
(``stage_in_force.search_stage``). Two arms differ only in the prior of the route's guided search:

- ``default``: the size prior, as in force;
- ``mmlm``: MMLM-L (``apedp_mmlm.law_language.LawLanguageModel``, fitted on SRBench's 133 ground-truth laws,
  pooled over domains) through ``law_language.search_prior``, passed as the route's ``structure_bits``.

Per unit (one law at one noise level): the search stage runs once, in its own process, seeded with *both* arms'
first-pass laws; its candidates are replayed to each arm, and each arm runs in its own process forked from the same
state (so neither warms the other, and one arm's failure cannot erase the other's row). Each arm builds its guided
search per call, as the route does, so MMLM's pricing is inside its time.

    python scripts/run_prereg_mmlm_order.py smoke --out DIR            # two laws, a short real search, both arms
    python scripts/run_prereg_mmlm_order.py run --out DIR --shard k/N [--rehearsal]
    python scripts/run_prereg_mmlm_order.py collect --out DIR [--rehearsal]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import signal
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

PREREGISTRATION_ID = "PREREG-MMLM-ORDER-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-MMLM-ORDER.json"
FROZEN = (
    "preregistration/PREREG-MMLM-ORDER-2026-11.md", "scripts/run_prereg_mmlm_order.py",
    ".github/workflows/mmlm-order.yml", "tests/test_mmlm_order_prereg.py",
    "apedp_mmlm/law_language.py", "apedp_mmlm/equation_language.py", "apedp_mmlm/law_corpora.py",
    "scripts/build_invented_laws.py", "scripts/run_srbench_baseline.py", "scripts/run_srbench_main_search.py",
    "scripts/run_prereg_mmlm_track_s.py", "scripts/run_search_evolution.py", "scripts/run_blind_discovery_benchmark.py",
    "scripts/fetch_srbench_ground_truth.py", "benchmarks/srbench/manifest.json", "benchmarks/memory_blind_laws.json",
    "benchmarks/invented_laws_fingerprints.json", "specifications/discovery_checks_v1.json",
    "apedp_dyn/conf/discovery_checks.py", "apedp_dyn/discovery_l2/report_share.py",
    "apedp_dyn/discovery_l2/search_stage.py", "apedp_dyn/discovery_l2/stage_in_force.py",
    "apedp_dyn/discovery_l2/refined_search.py", "apedp_dyn/discovery_l2/main_search.py",
    "apedp_dyn/discovery_l2/guided_search.py", "apedp_dyn/discovery_l2/exhaustive_search.py",
    "apedp_dyn/discovery_l2/confirmed_stop.py", "apedp_dyn/discovery_l2/contrast_confirmation.py",
    "apedp_dyn/discovery_l2/contrast_resolution.py", "apedp_dyn/discovery_l2/default_route.py",
    "apedp_dyn/discovery_l2/vouched_route_v2.py", "apedp_dyn/discovery_l2/vouching_v2.py",
    "apedp_dyn/ode/extended_route.py", "apedp_dyn/proposers/composed.py", "apedp_dyn/proposers/fit_guard.py",
    "apedp_dyn/proposers/deep_pass.py", "apedp_dyn/proposers/portfolio.py", "apedp_dyn/proposers/pysr_proposer.py",
)
ARMS = ("default", "mmlm")
NOISE_LEVELS = (0.0, 0.01)
FAMILIAR = ("other_fields", "power_laws", "odebench")    #: where MMLM has met the shapes before
GUARD = ("guard",)                                       #: mostly out of the first pass's reach: no help possible
SETS = (*FAMILIAR, *GUARD)
GUARD_SEED, POWER_SEED = 2026112731, 2026112733          #: drawn by no other test
REHEARSAL_GUARD_SEED, REHEARSAL_POWER_SEED, REHEARSAL_LAWS = 2026112732, 2026112734, 8
GUARD_COUNT, POWER_COUNT, POWER_DRAWS = 30, 30, 3000
DUPLICATE_OF = {"reynolds_number": "ideal_gas_pressure"}  #: the same law, x0*x1/x2: one is kept
TRAINING_SHAPES = ("beer_lambert", "compound_growth", "radioactive_decay", "drag_force", "cylinder_volume")
SEARCH_BUDGET_S = 240.0
SEARCH_LIMIT_S = 1200.0           #: the search pass's process (the budget, the stage's overheads, Julia)
ARM_LIMIT_S = 900.0               #: each arm's process
SIGN_TEST_P = 0.05                #: expectation 1: one-sided exact sign test on the units where the arms differ
NOT_SLOWER = 1.1                  #: expectation 4: MMLM's median end-to-end time at most this times the default's
SEARCH_FAILED_MAX = 0.2           #: more units than this share with a failed search: the test is not read
SMOKE_LAWS = tuple({"name": name, "family": "invented", "features": features, "formula": formula, "units": {},
                    "target_units": {}, "domain": [grammar.LOW, grammar.HIGH], "set": "smoke"}
                   for name, features, formula in (("smoke_ratio", ["x0", "x1"], "1.7*x0/(x0 + x1)"),
                                                   ("smoke_power", ["x0"], "0.9*x0**2 + 1.3")))


# ---------------------------------------------------------------- laws
def _entry(name: str, text: str, which: str, **extra: Any) -> dict[str, Any]:
    n = len(sp.sympify(text).free_symbols)
    return {"name": name, "family": "invented", "features": [f"x{i}" for i in range(n)], "formula": text,
            "units": {}, "target_units": {}, "domain": [grammar.LOW, grammar.HIGH], "set": which, **extra}


def _finite_on_domain(text: str) -> bool:
    expr = sp.sympify(text)
    n = len(expr.free_symbols)
    if n == 0:
        return False
    X = np.random.default_rng(0).uniform(grammar.LOW, grammar.HIGH, (200, n))
    fn = sp.lambdify([sp.Symbol(f"x{i}") for i in range(n)], expr, "numpy")
    with np.errstate(all="ignore"):
        y = np.asarray(fn(*X.T), dtype=complex) * np.ones(200)
    return bool(np.all(np.isfinite(y)) and np.all(np.abs(y.imag) < 1e-12) and np.ptp(y.real) > 0)


def _key(text: str) -> tuple[int, int | None]:
    """The structure's place in the first pass's enumeration (up to scale and offset), or ``None`` if out of reach."""
    expr = sp.sympify(text)
    n = len(expr.free_symbols)
    return n, track_s.position({"expr": expr, "n_vars": n})


def guard_laws(seed: int = GUARD_SEED, count: int = GUARD_COUNT) -> list[dict[str, Any]]:
    """The invented-law grammar, a fresh seed, without the benchmark's, the proposers' or the memory test's shapes."""
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    developed_on = {grammar.structure(sp.sympify(law["text"]))
                    for law in development.draw_laws(development.TRAIN + development.VALIDATION)}
    memory = {grammar.structure(sp.sympify(law["text"]))
              for law in json.loads((ROOT / "benchmarks" / "memory_blind_laws.json").read_text())["laws"]}
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(grammar.MAX_DRAWS):
        if len(out) == count:
            break
        inputs = int(rng.integers(1, 4))
        expr = grammar.draw(rng, inputs)
        key = grammar.structure(expr)
        if key in seen or not grammar.acceptable(expr, inputs, rng):
            continue
        seen.add(key)
        if grammar.fingerprint(str(expr)) in in_benchmark or key in developed_on or key in memory:
            continue
        out.append(_entry(f"guard_{len(out):02d}", str(expr), "guard"))
    if len(out) != count:
        raise SystemExit(f"only {len(out)} guard laws after {grammar.MAX_DRAWS} draws")
    return out


def power_laws(seed: int = POWER_SEED, count: int = POWER_COUNT, taken: set[Any] | None = None) -> list[dict[str, Any]]:
    """The blind benchmark's generator (power laws, as in PREREG-MMLM-L), a fresh seed, reachable, one per shape."""
    import run_blind_discovery_benchmark as blind

    taken = set(taken or ())
    out: list[dict[str, Any]] = []
    for drawn in blind.invented_laws(POWER_DRAWS, seed):
        expr = drawn.sympy()
        names = sorted(expr.free_symbols, key=lambda s: list(drawn.dimensions).index(str(s)))
        expr = expr.xreplace({s: sp.Symbol(f"x{i}") for i, s in enumerate(names)})
        law = {"expr": expr, "n_vars": len(names)}
        text = str(law["expr"])
        if law["n_vars"] > 3 or not _finite_on_domain(text):
            continue
        key = _key(text)
        if key[1] is None or key in taken:
            continue
        taken.add(key)
        out.append(_entry(f"power_{len(out):02d}", text, "power_laws"))
        if len(out) == count:
            return out
    raise SystemExit(f"only {len(out)} reachable power laws in {POWER_DRAWS} draws from seed {seed}")


def other_field_laws(taken: set[Any] | None = None) -> list[dict[str, Any]]:
    out = []
    for name, text in track_s.OTHER_FIELD_LAWS.items():
        if name in DUPLICATE_OF:
            continue
        out.append(_entry(f"field_{name}", text, "other_fields", training_shape=name in TRAINING_SHAPES))
        if taken is not None:
            taken.add(_key(text))
    return out


def odebench_laws(taken: set[Any] | None = None) -> list[dict[str, Any]]:
    """ODEBench's right-hand sides with at most three inputs that the first pass reaches, one per shape."""
    from apedp_mmlm import law_corpora

    taken = set(taken or ())
    out = []
    for law in law_corpora.odebench():
        text = str(law["expr"])
        if law["n_vars"] > 3 or len(sp.sympify(text).free_symbols) != law["n_vars"] or not _finite_on_domain(text):
            continue
        key = _key(text)
        if key[1] is None or key in taken:
            continue
        taken.add(key)
        out.append(_entry(f"ode_{law['name']}", text, "odebench"))
    return out


def entries(rehearsal: bool = False) -> dict[str, dict[str, Any]]:
    if rehearsal:
        # Shapes the final sets do not have: the power-law generator has few shapes, so its rehearsal laws are
        # drawn from the ones the final run leaves out; the guard's are a fresh grammar draw.
        final = entries()
        final_guard = {grammar.structure(sp.sympify(e["formula"])) for e in final.values() if e["set"] == "guard"}
        guard = [e for e in guard_laws(REHEARSAL_GUARD_SEED, REHEARSAL_LAWS + 10)
                 if grammar.structure(sp.sympify(e["formula"])) not in final_guard][:REHEARSAL_LAWS]
        reach = {_key(e["formula"]) for e in final.values() if e["set"] in FAMILIAR}
        laws = [*guard, *power_laws(REHEARSAL_POWER_SEED, REHEARSAL_LAWS, reach)]
        return {e["name"]: {**e, "name": f"rehearsal_{e['name']}"} for e in laws}
    taken: set[Any] = set()
    fields = other_field_laws(taken)
    odes = odebench_laws(taken)
    taken |= {_key(e["formula"]) for e in odes}
    powers = power_laws(POWER_SEED, POWER_COUNT, taken)
    return {e["name"]: e for e in (*fields, *odes, *powers, *guard_laws())}


def units(rehearsal: bool = False) -> list[tuple[dict[str, Any], float]]:
    return [(e, noise) for e in entries(rehearsal).values() for noise in NOISE_LEVELS]


def shards(rehearsal: bool, n: int) -> dict[str, int]:
    """Each law's shard, dealt in turn through the sets' order: both its noise levels run on the same runner."""
    return {name: i % n for i, name in enumerate(entries(rehearsal))}


def frame_of(entry: dict[str, Any]) -> pd.DataFrame:
    low, high = entry["domain"]
    rng = np.random.default_rng(base._seed(PREREGISTRATION_ID, entry["name"], "rows"))
    X = rng.uniform(low, high, (base.N_TRAIN + base.N_TEST, len(entry["features"])))
    fn = sp.lambdify([sp.Symbol(f"x{i}") for i in range(X.shape[1])], base.truth_expr(entry), "numpy")
    frame = pd.DataFrame(X, columns=entry["features"])
    frame["target"] = np.asarray(fn(*X.T), dtype=float) * np.ones(len(X))
    return frame


# ---------------------------------------------------------------- MMLM's prior
_PRIOR: Any = None


def mmlm_prior() -> Any:
    """MMLM-L fitted on SRBench's 133 ground-truth laws (frozen, pooled over domains), as ``structure_bits``."""
    global _PRIOR
    if _PRIOR is None:
        from apedp_mmlm import equation_language as el
        from apedp_mmlm import law_language as ll

        trees = [el.to_tree(base.truth_expr(e)) for e in base.load_manifest()]
        _PRIOR = ll.search_prior(ll.LawLanguageModel().fit(trees, [None] * len(trees)))
    return _PRIOR


def _bits(arm: str) -> Any:
    from apedp_dyn.discovery_l2.main_search import size_prior

    return mmlm_prior() if arm == "mmlm" else size_prior


# ---------------------------------------------------------------- processes
def _in_child(conn: Any, fn: Any, args: tuple[Any, ...]) -> None:
    os.setpgrp()                                    # its own process group, so a time-out kills what it started
    try:
        conn.send(("ok", fn(*args)))
    except Exception as exc:  # noqa: BLE001 - a crash is a scored outcome, reported with its class and message
        conn.send(("error", {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-2000:]}))
    finally:
        conn.close()


def _run(fn: Any, args: tuple[Any, ...], limit: float) -> tuple[str, Any, float]:
    context = mp.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    started = time.perf_counter()
    process = context.Process(target=_in_child, args=(child, fn, args))
    process.start()
    child.close()
    outcome: tuple[str, Any] = ("timeout", {})
    if parent.poll(limit):
        try:
            outcome = parent.recv()
        except EOFError:
            outcome = ("error", {"error": "the process died without a result"})
    process.join(1.0)
    if process.is_alive() and process.pid is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.join()
    return outcome[0], outcome[1], round(time.perf_counter() - started, 2)


def _rows(split: dict[str, Any], entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(base._seed(PREREGISTRATION_ID, entry["name"], "pool")).permutation(len(split["y_train"]))
    cut = round(path_run.DEV_SHARE * len(order))
    return order[:cut], order[cut:]


def search_pass(entry: dict[str, Any], split: dict[str, Any], budget: float) -> dict[str, Any]:
    """Both arms' first-pass laws, then the search stage in force once, seeded with both; its candidates."""
    from apedp_dyn.discovery_l2 import exhaustive_search as es
    from apedp_dyn.discovery_l2 import search_stage, stage_in_force
    from apedp_dyn.discovery_l2.confirmed_stop import search_confirmed
    from apedp_dyn.discovery_l2.guided_search import GuidedSearch
    from apedp_dyn.ode.extended_route import Recording

    names, X, y = split["names"], split["X_train"], split["y_train"]
    dev, _pool = _rows(split, entry)
    firsts = {}
    for arm in ARMS:
        first = search_confirmed(GuidedSearch(_bits(arm)), X[dev], y[dev], names, 6 if len(names) <= 5 else 5)
        firsts[arm] = str(sp.sympify(es.model_string(first.best), locals={n: sp.Symbol(n) for n in names}))
    if budget <= 0:
        return {"formulas": [], "first_pass": firsts, "described": "no search", "failures": [], "cores": None}
    stage, described = stage_in_force.search_stage(budget, seed=base.SEED)
    recorded = Recording(stage)
    frame = pd.DataFrame({**{n: X[dev, i] for i, n in enumerate(names)}, "__target__": y[dev]})
    recorded.fit_and_propose(frame, target_col="__target__", guesses=sorted(set(firsts.values())))
    failures = [str(f)[:300] for f in (getattr(stage, "failures", None) or [])]
    return {"formulas": list(recorded.formulas), "first_pass": firsts, "described": described, "failures": failures,
            "cores": search_stage.available_cores()}


def route(entry: dict[str, Any], split: dict[str, Any], formulas: list[str], arm: str) -> dict[str, Any]:
    """One arm: the refined path with the replayed candidates and the arm's prior, then the last checks in force."""
    from apedp_dyn.discovery_l2 import default_route, refined_search, vouched_route_v2
    from apedp_dyn.ode.extended_route import replayed

    names, X, y = split["names"], split["X_train"], split["y_train"]
    dev, pool = _rows(split, entry)
    pool_y = y[pool]
    dev_X = {n: X[dev, i] for i, n in enumerate(names)}
    options: dict[str, Any] = {"structure_bits": mmlm_prior()} if arm == "mmlm" else {}
    started = time.perf_counter()
    result = refined_search.discover_refined(dev_X, y[dev], X[pool], lambda picked: pool_y[picked],
                                             proposer=replayed(formulas), noise="absolute", seed=base.SEED, **options)
    record = {"route": default_route.REFINED, "action": result.action, "law": result.law,
              "tentative_law": result.tentative_law, "law_as_confirmed": result.law or result.tentative_law,
              "reported": result.law is not None, "complete": True, "dropped_terms": list(result.dropped_terms),
              "units": dict(result.units), "proposer_failures": list(result.proposer_failures), "rounds": [],
              "refined": result, "main_search": result.path}
    out = vouched_route_v2.checked(record, dev_X, y[dev], X[pool])
    seconds = round(time.perf_counter() - started, 2)
    law_text = out.get("law")
    law = sp.sympify(law_text, locals={n: sp.Symbol(n) for n in names}) if law_text else None
    raw = {"model": law, "pred": path_run._predict(law, names, split["X_test"]) if law is not None else None,
           "formula": law_text}
    path = result.path
    vouching = out.get("vouching") or {}
    return {"seconds": seconds, "action": result.action, "route_law": result.law, "vouched": vouching.get("reason"),
            "vouching_applied": vouching.get("applied"), "candidates_fitted": len(path.candidates) if path else None,
            **base.score_run("mmlm_order", entry, split, raw)}


def run_unit(entry: dict[str, Any], noise: float, budget: float = SEARCH_BUDGET_S) -> list[dict[str, Any]]:
    mmlm_prior()                                    # fitted once in this process; the forks inherit it
    head = {"problem": entry["name"], "set": entry["set"], "noise": noise, "inputs": len(entry["features"]),
            "truth": entry["formula"], "training_shape": entry.get("training_shape")}
    split = base.make_split(entry, frame_of(entry), noise)
    state, searched, search_seconds = _run(search_pass, (entry, split, budget), SEARCH_LIMIT_S + budget)
    if state != "ok" or (budget > 0 and (not searched["formulas"] or searched["failures"])):
        why = (searched.get("error") if state == "error" else state if state != "ok"
               else "a member of the search stage failed" if searched["failures"] else "the search proposed nothing")
        return [{**head, "arm": arm, "status": "search_failed", "search_status": state, "search_seconds": search_seconds,
                 "error": why, "search_failures": (searched or {}).get("failures") if state == "ok" else None}
                for arm in ARMS]
    rows = []
    for arm in ARMS:
        status, out, wall = _run(route, (entry, split, searched["formulas"], arm), ARM_LIMIT_S)
        row = {**head, "arm": arm, "status": status, "search_seconds": search_seconds, "wall_seconds": wall,
               "searched": len(searched["formulas"]), "first_pass": searched["first_pass"][arm],
               "first_pass_differs": len(set(searched["first_pass"].values())) > 1,
               "search_described": searched["described"], "search_failures": searched["failures"],
               "cores": searched["cores"], **(out if isinstance(out, dict) else {})}
        if status == "ok":
            row["end_to_end_seconds"] = round(search_seconds + float(out["seconds"]), 2)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- summary
def _reported(row: dict[str, Any]) -> bool:
    return row["status"] == "ok" and row.get("model") not in (None, "None")


def _recovered(row: dict[str, Any]) -> bool:
    return row["status"] == "ok" and bool(row.get("symbolic"))


def _wrong(row: dict[str, Any]) -> bool:
    return _reported(row) and not row.get("symbolic") and not row.get("accuracy_solution")


def sign_test(gained: int, lost: int) -> float:
    """One-sided exact sign test: the chance of at least ``gained`` of ``gained + lost`` going MMLM's way by luck."""
    from scipy.stats import binom

    n = gained + lost
    return 1.0 if n == 0 else float(binom.sf(gained - 1, n, 0.5))


def summarise(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    from apedp_dyn.discovery_l2 import report_share

    out: dict[str, Any] = {"units_expected": expected, "units_seen": len({(r["problem"], r["noise"]) for r in rows})}
    for arm in ARMS:
        for group in (*SETS, "familiar", "all"):
            members = FAMILIAR if group == "familiar" else SETS if group == "all" else (group,)
            sel = [r for r in rows if r["arm"] == arm and r["set"] in members]
            ok = [r for r in sel if r["status"] == "ok"]
            out[f"{arm}|{group}"] = {
                "runs": len(sel), "search_failed": sum(r["status"] == "search_failed" for r in sel),
                "errors": sum(r["status"] == "error" for r in sel), "timeouts": sum(r["status"] == "timeout" for r in sel),
                "reported": sum(_reported(r) for r in ok), "recovered": sum(_recovered(r) for r in ok),
                "wrong": sum(_wrong(r) for r in ok),
                "median_route_seconds": float(np.median([r["seconds"] for r in ok])) if ok else None,
                "median_end_to_end_seconds": float(np.median([r["end_to_end_seconds"] for r in ok])) if ok else None}
    by = {(r["arm"], r["problem"], r["noise"]): r for r in rows}
    cells = sorted({(r["problem"], r["noise"], r["set"]) for r in rows})

    def both_ok(p: str, n: float) -> bool:
        return all(by.get((a, p, n), {}).get("status") == "ok" for a in ARMS)

    def differ(members: tuple[str, ...]) -> tuple[int, int]:
        gained = lost = 0
        for p, n, s in cells:
            if s not in members or not both_ok(p, n):
                continue
            m, d = _recovered(by[("mmlm", p, n)]), _recovered(by[("default", p, n)])
            gained += m and not d
            lost += d and not m
        return gained, lost

    out["run_for_run"] = {s: dict(zip(("gained", "lost"), differ((s,)), strict=True)) for s in SETS}
    fields = [(p, n) for p, n, s in cells if s == "other_fields"]
    shapes = {p: by.get(("default", p, n), {}).get("training_shape") for p, n in fields}
    out["other_fields_by_training_shape"] = {
        str(flag): {arm: sum(_recovered(by[(arm, p, n)]) for p, n in fields if shapes[p] is flag and (arm, p, n) in by)
                    for arm in ARMS} for flag in (True, False)}
    out["first_pass_differs_units"] = sum(1 for p, n, _ in cells if by.get(("default", p, n), {}).get("first_pass_differs"))
    gained, lost = differ(FAMILIAR)
    g_gained, g_lost = differ(GUARD)
    wrong = {arm: out[f"{arm}|all"]["wrong"] for arm in ARMS}
    paired = [(by[("mmlm", p, n)]["end_to_end_seconds"], by[("default", p, n)]["end_to_end_seconds"])
              for p, n, _ in cells if both_ok(p, n)]
    med_m = float(np.median([a for a, _ in paired])) if paired else None
    med_d = float(np.median([b for _, b in paired])) if paired else None
    failed = sum(1 for p, n, _ in cells if by.get(("default", p, n), {}).get("status") == "search_failed")
    complete = out["units_seen"] == expected
    read = complete and failed <= SEARCH_FAILED_MAX * expected
    p_value = sign_test(gained, lost)
    held = {
        "1": gained > lost and p_value <= SIGN_TEST_P,
        "2": g_lost <= g_gained,
        "3": wrong["mmlm"] <= wrong["default"],
        "4": bool(paired) and med_m is not None and med_d is not None and med_m <= NOT_SLOWER * med_d,
    }
    out["expectations"] = {
        "read": read,
        "why_not_read": None if read else ("units missing" if not complete else f"{failed} units with a failed search"),
        "1 a gain on familiar ground: more laws recovered run for run, one-sided exact sign test p <= 0.05": {
            "gained": gained, "lost": lost, "p": round(p_value, 4)},
        "2 no harm on the guard set: run-for-run losses no more than gains": {"gained": g_gained, "lost": g_lost},
        "3 no more wrong laws reported than the default, over all units": wrong,
        "4 not slower end to end: median at most 1.1 times the default's, paired units": {
            "mmlm": med_m, "default": med_d, "units": len(paired)},
        "held": held if read else None,
        "report_share": {arm: report_share.verdict(out[f"{arm}|all"]["wrong"], out[f"{arm}|all"]["reported"])
                         for arm in ARMS},
    }
    out["outcome"] = (
        "not read: a rerun is put to the owner" if not read else
        "all four held: propose a blind test on reachable laws from a new source before anything goes in force; "
        "MMLM stays experimental and the route is unchanged" if all(held.values()) else
        "list MMLM as the route's prior as retired (deprecated, kept for replication; nothing in force changes)")
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


def loaded_digests() -> dict[str, str]:
    """SHA-256 of every repository file this process loaded (the frozen list cannot name them all)."""
    out = {}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not name:
            continue
        path = Path(name).resolve()
        if path.suffix == ".py" and path.is_file() and ROOT in path.parents:
            out[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(out.items()))


def _printable(row: dict[str, Any]) -> str:
    keep = ("problem", "set", "noise", "arm", "status", "symbolic", "accuracy_solution", "action", "seconds", "error")
    return json.dumps({k: row.get(k) for k in keep}, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("smoke", "run", "collect"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--rehearsal", action="store_true", help="16 laws of the rehearsal's own seeds, 60 s of search")
    parser.add_argument("--budget", type=float, default=20.0, help="for smoke: seconds of search")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "smoke":
        failed = 0
        for entry in SMOKE_LAWS:
            for row in run_unit(entry, 0.01, budget=args.budget):
                print(_printable(row), flush=True)
                failed += row["status"] != "ok"
                if row["status"] != "ok":
                    print(row.get("trace", ""), flush=True)
        return 1 if failed else 0
    if args.command == "run":
        if not args.rehearsal:
            check_frozen()
        budget = 60.0 if args.rehearsal else SEARCH_BUDGET_S
        k, n = (int(v) for v in args.shard.split("/"))
        dealt = shards(args.rehearsal, n)
        mine = [(e, noise) for e, noise in units(args.rehearsal) if dealt[e["name"]] == k]
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for entry, noise in mine:
                for row in run_unit(entry, noise, budget=budget):
                    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                    print(_printable(row), flush=True)
                handle.flush()
        (args.out / f"loaded-shard-{k:03d}.json").write_text(json.dumps(loaded_digests()) + "\n", encoding="utf-8")
        return 0
    if not args.rehearsal:
        check_frozen()
    rows = [json.loads(line) for p in sorted(args.out.rglob("runs-shard-*.jsonl")) for line in p.read_text().splitlines()]
    loaded = [json.loads(p.read_text()) for p in sorted(args.out.rglob("loaded-shard-*.json"))]
    changed = sorted({f for d in loaded for f, h in d.items()
                      if not (ROOT / f).is_file() or hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h})
    expected = len(units(args.rehearsal))
    record = {"preregistration": PREREGISTRATION_ID, "rehearsal": args.rehearsal, "rows": len(rows),
              "loaded_files_changed": changed, "summary": summarise(rows, expected),
              "frozen": frozen_digests()}
    if changed:
        record["summary"]["expectations"]["read"] = False
        record["summary"]["expectations"]["why_not_read"] = f"loaded files differ from the checkout: {changed[:5]}"
        record["summary"]["outcome"] = "not read: a rerun is put to the owner"
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (args.out / "results.json").write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n",
                                           encoding="utf-8")
    print(json.dumps(record["summary"]["expectations"], indent=1, default=str), "\n", record["summary"]["outcome"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
