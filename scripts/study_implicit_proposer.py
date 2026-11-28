#!/usr/bin/env python3
"""Development: the proposer of polynomial relations as the search stage of the route in force.

Status: audit-only

**Development, not a result.** ``apedp_dyn/proposers/implicit_form.py`` proposes formulas solved from polynomial
relations between the inputs and the target. This puts it through the referee as a search stage and counts what
is reported, beside the same tables read with no search stage at all (the route's first pass alone; the product's
own stage needs PySR, which cannot be installed where this was run).

* ``stream``: the 122 tables the second family run left without a law. Each is marked by whether its law is
  within the proposer's reach: on the law's own rows without noise, the proposer's first relation is exact. The
  tables outside its reach are the trap, since everything it proposes there is an approximation.
* ``laws``: six laws with a short relation and three it cannot write (an exponential, a logarithm, a sine), as
  tables of the stream's size at its three noise levels.
* ``real``: the three real development tables by the small-table reading, and where the known form stands among
  the ten proposals.

Counted for a reading: a law reported; the law itself by the benchmark's test of form; accurate (R squared 0.999
on rows without noise, the line that defines a wrong law); and, beyond that line, accurate on rows drawn half as
wide again. No law of the stream is written here.

    python scripts/study_implicit_proposer.py --out benchmarks/outside_proposals/dev_implicit_proposer.json --jobs 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_real_data as real  # noqa: E402 - after the path is set
import study_family_stream as study  # noqa: E402
import study_handed_truth as handed  # noqa: E402

from apedp_dyn.discovery_l2 import family_route, small_table  # noqa: E402
from apedp_dyn.discovery_l2.across_tables import Table  # noqa: E402
from apedp_dyn.proposers import implicit_form  # noqa: E402
from apedp_dyn.proposers.fit_guard import FitGuardProposer  # noqa: E402
from apedp_dyn.proposers.implicit_form import ImplicitFormProposer  # noqa: E402
from apedp_dyn.recursion import curriculum  # noqa: E402

TAG = "DEV-IMPLICIT-PROPOSER-2026-11"
WAYS = ("first_pass_alone", "implicit_stage")
WIDER = 1.5                    #: the rows beyond the accuracy line are drawn over a range this many times as wide
WITHIN_REACH = 1e-5            #: the first relation's error on rows without noise, in units of the target's median
NOISES = (0.01, 0.05, 0.1)     #: the stream's noise levels
SEEDS = (0, 1)

#: name: (the law, its inputs' ranges); the last three have no relation the proposer can write
LAWS: dict[str, tuple[str, dict[str, tuple[float, float]]]] = {
    "time dilation": ("3.0*(sqrt(1 - v**2) - 1)", {"v": (0.1, 0.9)}),
    "saturation (Langmuir)": ("2.0*p/(1 + 0.8*p)", {"p": (0.1, 10.0)}),
    "Kepler with two masses": ("d**1.5/sqrt(m1 + m2)", {"m1": (0.5, 2.0), "m2": (0.05, 1.0), "d": (0.1, 5.0)}),
    "line (Hubble)": ("0.5*D", {"D": (0.03, 2.0)}),
    "lens": ("u*v/(u + v)", {"u": (0.5, 5.0), "v": (0.5, 5.0)}),
    "energy": ("sqrt(p**2 + m**2)", {"p": (0.2, 3.0), "m": (0.2, 3.0)}),
    "trap: exponential": ("2.0*exp(-0.7*x)", {"x": (0.2, 4.0)}),
    "trap: logarithm": ("-7.0*log(x) - 2.0", {"x": (1.0, 6.0)}),
    "trap: sine": ("sin(1.3*x) + 2.0", {"x": (0.2, 4.0)}),
}

#: the known law of each real development table as a relation: exponents of (input, target)
REAL_RELATIONS = {"kepler_solar_system": {(0, 2), (3, 0)}, "langmuir_table_ix": {(0, 1), (1, 0), (1, 1)},
                  "leavitt": {(0, 1), (1, 0), (0, 0)}}


def stage() -> FitGuardProposer:
    """The proposer as the route is given a search stage: behind the fit guard."""
    return FitGuardProposer(ImplicitFormProposer(), target_col="__target__")


def within_reach(names: Sequence[str], X: np.ndarray, clean: np.ndarray) -> bool:
    """Whether the proposer's first relation is exact on the law's own rows without noise."""
    proposer = ImplicitFormProposer()
    proposer.fit_and_propose(pd.DataFrame({**{n: X[:, i] for i, n in enumerate(names)}, "__target__": clean}), target_col="__target__")
    behind = proposer.last_run.get("behind") or []
    return bool(behind and behind[0]["error"] <= WITHIN_REACH)


def readings(table: Table, truth: sp.Expr, test_X: np.ndarray, test_clean: np.ndarray, wide_X: np.ndarray, wide_clean: np.ndarray,
             seed: int) -> dict[str, Any]:
    """The table read with no search stage and with the proposer as the stage, each scored."""
    out: dict[str, Any] = {}
    fine = np.isfinite(wide_clean)
    for way, proposer in (("first_pass_alone", None), ("implicit_stage", stage())):
        reading = family_route.read(table, proposer, noise="absolute", seed=seed)
        law = reading["law"] if reading["reported"] else None
        scored = study.score(law, truth, table.names, test_X, test_clean)
        wider = study.score(law, truth, table.names, wide_X[fine], wide_clean[fine])
        path = reading["record"].get("main_search")
        out[way] = {"action": reading["action"], "reported": bool(reading["reported"]), "the_law_itself": scored["recovered"],
                    "accurate": scored["accurate"], "accurate_on_wider_rows": wider["accurate"],
                    "found_by": getattr(path, "found_by", None), "proposed": len(reading["formulas"]),
                    "held_back": bool(reading["tentative_law"]), "seconds": reading["seconds"]}
        held = study.score(reading["tentative_law"], truth, table.names, test_X, test_clean)
        out[way]["held_back_was_the_law"] = held["recovered"]
    return out


def stream_unit(job: tuple[int, float, tuple[int, ...]]) -> list[dict[str, Any]]:
    index, noise, wanted = job
    spec = study.draw_families(handed.FAMILIES_DRAWN)[index]
    family, scoring = study.tables(spec, noise)
    symbols = [sp.Symbol(f"x{i}") for i in range(spec["inputs"])]
    rows = []
    for table, how in zip(family.tables, scoring, strict=True):
        if how["index"] not in wanted:
            continue
        rng = np.random.default_rng([spec["family"], how["index"], 7])
        wide_X = rng.uniform(curriculum.LOW, curriculum.LOW + WIDER * (curriculum.HIGH - curriculum.LOW), how["test_X"].shape)
        with np.errstate(all="ignore"):
            wide_clean = np.asarray(sp.lambdify(symbols, how["truth"], "numpy")(*wide_X.T), dtype=float) * np.ones(len(wide_X))
        rows.append({"family": spec["family"], "member": how["index"], "noise": noise, "inputs": spec["inputs"],
                     "within_reach": within_reach(table.names, how["test_X"], how["test_clean"]),
                     **readings(table, how["truth"], how["test_X"], how["test_clean"], wide_X, wide_clean, spec["family"])})
    return rows


def law_table(name: str, noise: float, seed: int) -> tuple[Table, sp.Expr, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A table of one law in the stream's sizes: rows to fit, a pool, rows to score on, and rows over a wider range."""
    law, ranges = LAWS[name]
    names = list(ranges)
    symbols = [sp.Symbol(n) for n in names]
    truth = sp.sympify(law, locals=dict(zip(names, symbols, strict=True)))
    fn: Callable[..., Any] = sp.lambdify(symbols, truth, "numpy")
    rng = np.random.default_rng([list(LAWS).index(name), seed, round(noise * 1e4)])
    total = curriculum.ROWS + curriculum.POOL + curriculum.TEST_ROWS
    X = np.column_stack([rng.uniform(lo, hi, total) for lo, hi in ranges.values()])
    clean = np.asarray(fn(*X.T), dtype=float) * np.ones(total)
    y = clean + noise * float(np.sqrt(np.mean(clean ** 2))) * rng.standard_normal(total)
    wide_X = np.column_stack([rng.uniform(lo, lo + WIDER * (hi - lo), curriculum.TEST_ROWS) for lo, hi in ranges.values()])
    with np.errstate(all="ignore"):
        wide_clean = np.asarray(fn(*wide_X.T), dtype=float) * np.ones(len(wide_X))
    dev, pool = curriculum.ROWS, curriculum.ROWS + curriculum.POOL
    pool_y = y[dev:pool]
    table = Table(name, names, X[:dev], y[:dev], X[dev:pool], lambda asked: pool_y[np.asarray(asked, dtype=int)], {})
    return table, truth, X[pool:], clean[pool:], wide_X, wide_clean


def law_unit(job: tuple[str, float, int]) -> dict[str, Any]:
    name, noise, seed = job
    table, truth, test_X, test_clean, wide_X, wide_clean = law_table(name, noise, seed)
    return {"law": name, "noise": noise, "seed": seed, "inputs": len(table.names),
            "within_reach": within_reach(table.names, test_X, test_clean),
            **readings(table, truth, test_X, test_clean, wide_X, wide_clean, seed)}


def real_unit(key: str) -> dict[str, Any]:
    """One real development table: the small-table reading with and without the stage, and the known form's place."""
    loaded = real.real_data.load(key)
    table, X, y = loaded["table"], loaded["X"], loaded["y"]
    stated = json.loads(real.KNOWN.read_text(encoding="utf-8"))["tables"][key]
    truth, _ = real.fitted_law(table, X, y)
    proposer = ImplicitFormProposer()
    proposals = proposer.fit_and_propose(pd.DataFrame({**X, "__target__": y}), target_col="__target__")
    forms = [bool(real.judged(str(p.formula), truth, X, y).get("same_form_as_the_known_law")) for p in proposals]
    # the benchmark's test of form refits constants, so it also passes the law with a term added (audit M-23, M-26);
    # the place of the law's own relation, term for term, is counted beside it
    own = REAL_RELATIONS[key]
    exact = [{tuple(e) for e in p.metadata["support"]} == own for p in proposals]
    columns = np.column_stack([np.asarray(X[n], dtype=float) for n in X])
    ranked = implicit_form.relations(columns / np.median(np.abs(columns), axis=0), np.asarray(y, dtype=float) / float(np.median(np.abs(y))),
                                     int(proposer.last_run["degree"]), int(proposer.last_run["max_terms"]))
    place = next((i for i, r in enumerate(ranked) if set(r.support) == own), None)
    of_its_size = [r for r in ranked if len(r.support) == len(own)]
    out: dict[str, Any] = {"key": key, "rows": len(y), "known_law": table["law"], "proposed": len(proposals),
                           "known_form_place_among_the_proposals": forms.index(True) + 1 if any(forms) else None,
                           "own_relation_place_among_the_proposals": exact.index(True) + 1 if any(exact) else None,
                           "own_relation_rank": None if place is None else place + 1, "relations_fitted": len(ranked),
                           "own_relation_rank_among_its_size": None if place is None else of_its_size.index(ranked[place]) + 1,
                           "relations_of_its_size": len(of_its_size), "first_proposal": str(proposals[0].formula) if proposals else None,
                           "setting": [proposer.last_run.get("degree"), proposer.last_run.get("max_terms")], "readings": {}}
    for way, search in (("first_pass_alone", None), ("implicit_stage", stage())):
        started = time.perf_counter()
        reading = small_table.read(X, y, noise=stated["noise"], proposer=search, seed=0, known=(), units=None)
        judged = real.judged(reading.get("law"), truth, X, y)
        out["readings"][way] = {"action": reading.get("action"), "reported": bool(reading.get("reported")),
                                "tentative": bool(reading.get("tentative")), "law": reading.get("law"),
                                "same_form_as_the_known_law": bool(judged.get("same_form_as_the_known_law")),
                                "r2_all_rows": judged.get("r2_all_rows"), "class_size": reading.get("class_size"),
                                "seconds": round(time.perf_counter() - started, 1)}
    return json.loads(json.dumps(out, default=str))


def counted(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The counts of one group of readings, for each way of reading."""
    out: dict[str, Any] = {"tables": len(rows)}
    for way in WAYS:
        got = [r[way] for r in rows]
        reported = [g for g in got if g["reported"]]
        out[way] = {"reported": len(reported), "the_law_itself": sum(g["the_law_itself"] for g in reported),
                    "accurate": sum(g["accurate"] for g in reported),
                    "reported_and_not_accurate": sum(not g["accurate"] for g in reported),
                    "accurate_on_wider_rows": sum(g["accurate_on_wider_rows"] for g in reported),
                    "found_by_the_stage": sum(g["found_by"] == "search" for g in reported),
                    "held_back": sum(g["held_back"] for g in got), "held_back_was_the_law": sum(g["held_back_was_the_law"] for g in got),
                    "not_run": sum(g["action"] == "not_run" for g in got)}
    return out


def summarise(rows: Sequence[dict[str, Any]], by: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"all": counted(rows)}
    for field in by:
        out[f"by_{field}"] = {str(value): counted([r for r in rows if r[field] == value])
                              for value in sorted({r[field] for r in rows}, key=str)}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--arms", nargs="+", choices=("stream", "laws", "real"), default=["stream", "laws", "real"])
    args = parser.parse_args(argv)
    record: dict[str, Any] = {"tag": TAG, "development_not_a_result": True,
                              "what": "the proposer of polynomial relations as the search stage of the route in force, beside the "
                                      "route's first pass alone; no law of the stream is written here",
                              "settings": {"wider": WIDER, "within_reach": WITHIN_REACH, "noises": list(NOISES), "seeds": list(SEEDS),
                                           "seconds": {}}}
    if args.out.exists():                      # an arm that is not run again is kept as it was recorded
        before = json.loads(args.out.read_text(encoding="utf-8"))
        for arm in ("stream", "laws", "real"):
            if arm not in args.arms and arm in before:
                record[arm] = before[arm]
                seconds = before.get("settings", {}).get("seconds")
                record["settings"]["seconds"][arm] = seconds.get(arm) if isinstance(seconds, dict) else None
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        if "stream" in args.arms:
            started = time.perf_counter()
            wanted: dict[tuple[int, float], list[int]] = {}
            for (family, member, noise), row in handed.second_run().items():
                if not row["reported"]:
                    wanted.setdefault((family, noise), []).append(member)
            jobs = [(family, noise, tuple(sorted(members))) for (family, noise), members in sorted(wanted.items())]
            rows = [row for unit in pool.map(stream_unit, jobs) for row in unit]
            record["stream"] = {"summary": summarise(rows, ("within_reach", "noise")), "rows": rows}
            record["settings"]["seconds"]["stream"] = round(time.perf_counter() - started)
            print(json.dumps(record["stream"]["summary"], indent=1), flush=True)
        if "laws" in args.arms:
            started = time.perf_counter()
            rows = list(pool.map(law_unit, [(name, noise, seed) for name in LAWS for noise in NOISES for seed in SEEDS]))
            record["laws"] = {"summary": summarise(rows, ("law", "noise")), "rows": rows}
            record["settings"]["seconds"]["laws"] = round(time.perf_counter() - started)
            print(json.dumps(record["laws"]["summary"]["by_law"], indent=1), flush=True)
        if "real" in args.arms:
            started = time.perf_counter()
            record["real"] = list(pool.map(real_unit, real.DEVELOPMENT))
            record["settings"]["seconds"]["real"] = round(time.perf_counter() - started)
            print(json.dumps(record["real"], indent=1), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
