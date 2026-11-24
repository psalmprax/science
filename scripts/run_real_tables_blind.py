#!/usr/bin/env python3
"""PREREG-REAL-TABLES-2026-11: the small-table reading on real tables it was not developed on, once.

Frozen with its plan by ``preregistration/DIGEST-REAL-TABLES.json``; this script refuses to run if a frozen file
has changed, and reads a held-back table only through ``real_data.fetch``, which needs the preregistration to
exist. Four arms (the plan says what each is for and what is expected):

* **A, known laws:** six held-back tables with a known law. The reading with what is known stated and the search
  stage on (the arm under test), without the searches, with nothing stated, and beside them the route in force
  called directly and the product's own reading. Each law given is compared with the known form.
* **B, pipe friction:** two whole pipes are left out by a rule fixed before the file was read; a law is read from
  the other rows and scored on the pipes left out, beside Haaland's and Colebrook's published formulas.
* **C, the expansion rate:** every third measurement is left out; a law is read from the rest, in units of each
  measurement's stated error, and scored on those left out beside the standard model's form.
* **D, generated small tables:** 120 tables drawn after freezing from a seed used nowhere else, the reading against
  the route in force called directly, no search stage in either.

    python scripts/run_real_tables_blind.py real --out DIR --searches 90
    python scripts/run_real_tables_blind.py generated --out DIR [--shard 0 --shards 2]
    python scripts/run_real_tables_blind.py collect --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import real_data  # noqa: E402
import run_real_data as dev  # noqa: E402
import run_small_tables as generated  # noqa: E402

from apedp_dyn.discovery_l2 import small_table  # noqa: E402
from apedp_dyn.ui import route_endpoint  # noqa: E402

PREREGISTRATION = "PREREG-REAL-TABLES-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-REAL-TABLES.json"
KNOWN = ROOT / "benchmarks" / "real_data" / "known_beforehand.json"
KNOWN_LAW = ("kepler_exoplanets", "kepler_binary_stars", "langmuir_sun_et_al", "time_dilation", "hubble_1929", "tully_fisher")
BLIND_SEED = 2026112477
BLIND_TABLES = 120
SEED = 0
MIN_PIPE_ROWS = 10           # a roughness with fewer rows than this is not a pipe for the split
READ_SECONDS = 900.0


def frozen_files_changed() -> list[str]:
    recorded = json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"]
    return [name for name, digest in recorded.items()
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]


def held_back(key: str, cache: Path) -> dict[str, Any]:
    table = real_data.registry()[key]
    path = real_data.fetch(key, preregistration=PREREGISTRATION, into=cache)
    return {"table": table, "columns": real_data.parse(table, path.read_text(encoding="utf-8"))}


def stage(searches: float) -> Any:
    return route_endpoint._proposer(searches)[0] if searches > 0 else None


def short(reading: dict[str, Any]) -> dict[str, Any]:
    keep = ("action", "law", "reported", "tentative", "tentative_reasons", "rows", "errors", "chosen", "class_size", "class",
            "best_by_left_out_error", "stability", "accuracy", "known_checks_on_the_law", "candidates", "error", "seconds")
    out = {k: reading[k] for k in keep if k in reading}
    out["set_aside_by_what_is_known"] = (reading.get("set_aside_by_what_is_known") or {}).get("forms")
    return json.loads(json.dumps(out, default=str))


# ------------------------------------------------------------------ arm A
def known_law_table(key: str, searches: float, cache: Path) -> dict[str, Any]:
    loaded = held_back(key, cache)
    table, columns = loaded["table"], loaded["columns"]
    stated = json.loads(KNOWN.read_text(encoding="utf-8"))["tables"][key]
    X = {name: columns[name] for name in table["inputs"]}
    y = columns[table["target"]]
    statements = [{k: v for k, v in s.items() if k != "why"} for s in stated["known"]]
    out: dict[str, Any] = {"key": key, "rows": len(y), "inputs": table["inputs"], "known_law": table["law"]}
    try:
        truth, truth_r2 = dev.fitted_law(table, X, y)
        out.update({"known_law_fitted": str(truth), "known_law_r2_all_rows": None if truth_r2 is None else round(truth_r2, 6)})
    except Exception as exc:  # noqa: BLE001 - a table on which the known form cannot be fitted is recorded as such and not scored
        out["known_law_error"] = f"{type(exc).__name__}: {exc}"[:200]
        truth = None
    arms = {
        "informed_with_searches": {"units": stated["units"], "known": statements, "proposer": stage(searches)},
        "informed": {"units": stated["units"], "known": statements, "proposer": None},
        "plain": {"units": None, "known": (), "proposer": None},
    }
    out["readings"] = {}
    for name, options in arms.items():
        reading = small_table.read(X, y, noise=stated["noise"], seed=SEED, seconds=READ_SECONDS, **options)
        entry = short(reading)
        if truth is not None:
            entry["judged"] = dev.judged(reading.get("law"), truth, X, y)
        out["readings"][name] = entry
    direct = dev.direct(X, y, SEED, searches) if len(y) >= dev.MIN_DIRECT_ROWS else {"action": "not_run", "error": "too few rows to split"}
    out["direct"] = {**direct, **({"judged": dev.judged(direct.get("law_as_confirmed"), truth, X, y)} if truth is not None else {})}
    product = dev.keep(route_endpoint.route_reading(X, y, search_seconds=searches, seed=SEED))
    out["product"] = {**product, **({"judged": dev.judged(product.get("law_as_confirmed"), truth, X, y)} if truth is not None else {})}
    return out


# ------------------------------------------------------------------ arm B
def haaland(reynolds: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    """Haaland (1983): 1/sqrt(f) = -1.8 log10((e/D / 3.7)**1.11 + 6.9/Re), the Darcy friction factor."""
    out: np.ndarray = (-1.8 * np.log10((roughness / 3.7) ** 1.11 + 6.9 / reynolds)) ** -2
    return out


def colebrook(reynolds: np.ndarray, roughness: np.ndarray, steps: int = 60) -> np.ndarray:
    """Colebrook and White (1939): 1/sqrt(f) = -2 log10(e/D / 3.7 + 2.51 / (Re sqrt(f))), solved by iteration."""
    inverse_root = 1.0 / np.sqrt(haaland(reynolds, roughness))
    for _ in range(steps):
        inverse_root = -2.0 * np.log10(roughness / 3.7 + 2.51 * inverse_root / reynolds)
    out: np.ndarray = inverse_root ** -2
    return out


def pipes_left_out(roughness: np.ndarray, reynolds: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """The rows of the pipes left out, by the rule fixed before the file was read."""
    with np.errstate(all="ignore"):
        rounded = np.array([float(f"{value:.3g}") for value in roughness])
    values, counts = np.unique(rounded, return_counts=True)
    pipes = [float(v) for v, c in zip(values, counts, strict=True) if c >= MIN_PIPE_ROWS]
    if len(pipes) >= 5:
        chosen = [pipes[1], pipes[4]]                       # the second and the fifth smoothest
    elif len(pipes) >= 3:
        chosen = [pipes[len(pipes) // 2]]
    else:                                                    # no pipes to speak of: every third row by Reynolds number
        test = np.zeros(len(roughness), dtype=bool)
        test[np.argsort(reynolds)[1::3]] = True
        return test, {"rule": "every third row by Reynolds number", "pipes": pipes}
    return np.isin(rounded, chosen), {"rule": "whole pipes", "pipes": pipes, "left_out": chosen}


def rmse(a: np.ndarray, b: np.ndarray) -> float | None:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if np.all(ok) else None


def law_on(law: str | None, names: list[str], X: np.ndarray) -> np.ndarray | None:
    if not law:
        return None
    symbols = [sp.Symbol(n) for n in names]
    try:
        model = sp.sympify(str(law), locals=dict(zip(names, symbols, strict=True)))
        with np.errstate(all="ignore"):
            out: np.ndarray = np.asarray(sp.lambdify(symbols, model, "numpy")(*X.T), dtype=float) * np.ones(len(X))
    except Exception:  # noqa: BLE001 - a law that cannot be evaluated scores nothing
        return None
    return out


def pipe_friction(searches: float, cache: Path) -> dict[str, Any]:
    columns = held_back("nikuradse_superpipe", cache)["columns"]
    reynolds, roughness, friction = columns["Re"], columns["eD"], columns["f"]
    usable = (reynolds > 0) & (roughness > 0) & (friction > 0)
    reynolds, roughness, friction = reynolds[usable], roughness[usable], friction[usable]
    test, split = pipes_left_out(roughness, reynolds)
    work = ~test
    ratio = float(np.median(friction[work] / haaland(reynolds[work], roughness[work])))
    factor = 4.0 if 0.125 <= ratio < 0.5 else 1.0             # a Fanning factor is a quarter of the Darcy one
    friction = friction * factor
    names = ["lRe", "leD"]
    logs = np.column_stack([np.log10(reynolds), np.log10(roughness)])
    target = np.log10(friction)
    out: dict[str, Any] = {"rows": len(target), "rows_work": int(work.sum()), "rows_left_out": int(test.sum()), "split": split,
                           "median_ratio_to_haaland_on_work_rows": round(ratio, 4), "factor_applied_to_f": factor}
    reading = small_table.read({n: logs[work, i] for i, n in enumerate(names)}, target[work], proposer=stage(searches),
                               noise="absolute", seed=SEED, seconds=READ_SECONDS)
    product = dev.keep(route_endpoint.route_reading({n: logs[work, i] for i, n in enumerate(names)}, target[work],
                                                    search_seconds=searches, seed=SEED))
    out["reading"], out["product"] = short(reading), product

    def refit_resid(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            inside = (roughness[work] / theta[1]) ** 1.11 + theta[2] / reynolds[work]
            value = np.log10((-theta[0] * np.log10(inside)) ** -2)
        res: np.ndarray = np.where(np.isfinite(value), value - target[work], 1e3)
        return res

    refit = least_squares(refit_resid, np.array([1.8, 3.7, 6.9]), max_nfev=2000)
    with np.errstate(all="ignore"):
        refitted = np.log10((-refit.x[0] * np.log10((roughness / refit.x[1]) ** 1.11 + refit.x[2] / reynolds)) ** -2)
    predictions = {
        "the reading's law": law_on(reading.get("law"), names, logs),
        "the route in force's law": law_on(product.get("law_as_confirmed"), names, logs),
        "Haaland, published constants": np.log10(haaland(reynolds, roughness)),
        "Colebrook and White, published constants": np.log10(colebrook(reynolds, roughness)),
        "Haaland's form, constants refitted on the work rows": refitted,
    }
    out["error_on_pipes_left_out"] = {name: None if pred is None else rmse(pred[test], target[test]) for name, pred in predictions.items()}
    out["error_on_work_rows"] = {name: None if pred is None else rmse(pred[work], target[work]) for name, pred in predictions.items()}
    out["haaland_refitted_constants"] = [float(v) for v in refit.x]
    return out


# ------------------------------------------------------------------ arm C
def expansion_rate(searches: float, cache: Path) -> dict[str, Any]:
    columns = held_back("cosmic_chronometers", cache)["columns"]
    z, rate, sigma = columns["z"], columns["H"], columns["sigma_H"]
    order = np.argsort(z, kind="stable")
    test = np.zeros(len(z), dtype=bool)
    test[order[1::3]] = True                                  # the second, fifth, eighth ... by redshift
    work = ~test
    stated = json.loads(KNOWN.read_text(encoding="utf-8"))["tables"]["cosmic_chronometers"]
    statements = [{k: v for k, v in s.items() if k != "why"} for s in stated["known"]]
    reading = small_table.read({"z": z[work]}, rate[work], sigma=sigma[work], known=statements, proposer=stage(searches),
                               seed=SEED, seconds=READ_SECONDS)

    def fitted(model: Any, start: list[float]) -> np.ndarray:
        got = least_squares(lambda theta: (model(theta, z[work]) - rate[work]) / sigma[work], np.array(start), max_nfev=2000)
        out: np.ndarray = model(got.x, z)
        return out

    with np.errstate(all="ignore"):
        predictions = {
            "the reading's law": law_on(reading.get("law"), ["z"], z.reshape(-1, 1)),
            "the standard model's form (flat, matter and a constant), fitted on the work rows":
                fitted(lambda t, x: t[0] * np.sqrt(np.abs(t[1]) * (1 + x) ** 3 + 1 - np.abs(t[1])), [70.0, 0.3]),
            "a straight line, fitted on the work rows": fitted(lambda t, x: t[0] * (1 + t[1] * x), [70.0, 1.0]),
        }

    def chi2(pred: np.ndarray | None, rows: np.ndarray) -> float | None:
        if pred is None or not np.all(np.isfinite(pred[rows])):
            return None
        return float(np.mean(((pred[rows] - rate[rows]) / sigma[rows]) ** 2))

    return {"rows": len(z), "rows_work": int(work.sum()), "rows_left_out": int(test.sum()), "reading": short(reading),
            "chi2_per_row_left_out": {name: chi2(pred, test) for name, pred in predictions.items()},
            "chi2_per_row_work": {name: chi2(pred, work) for name, pred in predictions.items()}}


# ------------------------------------------------------------------ arm D
def blind_laws() -> list[dict[str, Any]]:
    """120 laws drawn from a seed used nowhere else, leaving out every structure the development tables used."""
    used = frozenset(generated.curriculum.structure(sp.sympify(law["text"])) for law in generated.draw_laws(120))
    laws = generated.draw_laws(BLIND_TABLES, seed=BLIND_SEED, leave_out=used)
    return [{**law, "tag": "small-tables-blind"} for law in laws]


def summarise_known(tables: list[dict[str, Any]]) -> dict[str, Any]:
    def form(entry: dict[str, Any]) -> bool:
        return bool((entry.get("judged") or {}).get("same_form_as_the_known_law"))

    out: dict[str, Any] = {"tables": len(tables)}
    for arm in ("informed_with_searches", "informed", "plain"):
        got = [t["readings"][arm] for t in tables]
        out[arm] = {"read": sum(g.get("action") == "read" for g in got), "known_form": sum(form(g) for g in got),
                    "reported": sum(bool(g.get("reported")) for g in got),
                    "reported_without_the_known_form": sum(bool(g.get("reported")) and not form(g) for g in got),
                    "known_form_on": [t["key"] for t, g in zip(tables, got, strict=True) if form(g)]}
    for arm in ("direct", "product"):
        got = [t[arm] for t in tables]
        out[arm] = {"ran": sum(g.get("action") != "not_run" for g in got), "known_form": sum(form(g) for g in got),
                    "reported": sum(bool(g.get("reported")) for g in got),
                    "reported_without_the_known_form": sum(bool(g.get("reported")) and not form(g) for g in got),
                    "known_form_on": [t["key"] for t, g in zip(tables, got, strict=True) if form(g)]}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("real", "generated", "collect", "check"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--searches", type=float, default=90.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)
    changed = frozen_files_changed()
    if changed:
        raise SystemExit(f"frozen files changed since the digest was written: {changed}. Nothing was run.")
    if args.command == "check":
        print(f"{PREREGISTRATION}: every frozen file matches its digest")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "collect":
        record: dict[str, Any] = {"preregistration": PREREGISTRATION}
        real = args.out / "real.json"
        if real.exists():
            record["real"] = json.loads(real.read_text(encoding="utf-8"))
        rows = [json.loads(line) for path in sorted(args.out.glob("rows_*.jsonl")) for line in path.read_text().splitlines()]
        if rows:
            rows.sort(key=lambda r: r["index"])
            record["generated"] = {"tables": len(rows), "summary": generated.summarise(rows), "rows": rows}
        (args.out / "results.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"known": (record.get("real") or {}).get("summary"),
                          "generated": (record.get("generated") or {}).get("summary", {}).get("paired")}, indent=1))
        return 0
    if args.command == "generated":
        laws = blind_laws()[args.shard::args.shards]
        path = args.out / f"rows_{args.shard}.jsonl"
        with path.open("w", encoding="utf-8") as handle, mp.get_context("fork").Pool(max(args.workers, 1), maxtasksperchild=8) as pool:
            for row in pool.imap(generated.one, [(law, 0.0) for law in laws], chunksize=1):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
        return 0
    if args.searches > 0:
        try:
            import pysr  # noqa: F401 - the arm under test has the search stage; without PySR it would silently be another arm
        except Exception as exc:  # noqa: BLE001 - whatever stops PySR loading ends the run with its reason, before any table is read
            raise SystemExit(f"PySR cannot be imported ({type(exc).__name__}: {exc}). Nothing was run.") from exc
    started = time.perf_counter()
    cache = args.out / "held_back"
    tables = [known_law_table(key, args.searches, cache) for key in KNOWN_LAW]
    record = {"preregistration": PREREGISTRATION, "search_seconds": args.searches, "known_law": tables,
              "summary": summarise_known(tables), "pipe_friction": pipe_friction(args.searches, cache),
              "expansion_rate": expansion_rate(args.searches, cache), "seconds": round(time.perf_counter() - started, 1)}
    for path in cache.glob("*"):                              # the held-back files are not kept beside the results
        path.unlink()
    (args.out / "real.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
