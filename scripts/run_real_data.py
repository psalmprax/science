#!/usr/bin/env python3
"""Development: what the table route in force says on real measurements with a known law.

**Development, not a result.** It reads the three development tables of ``benchmarks/real_data/registry.json``
and nothing else; ``real_data.load`` refuses a held-back table.

For each table and each of ``SEEDS`` splits:

* ``product``: ``apedp_dyn.ui.route_endpoint.route_reading``, the reading the product gives a table handed to
  it: 70 % of the rows to work on, 30 % as the pool the referee confirms on. It refuses a table of fewer than
  24 rows;
* ``direct``: where the product refused for want of rows, the route itself
  (``vouched_route_v2.discover_law``) on the same split, to see what it does with a table that small.

Beside each law the route gave (reported, or tentative): whether it has the known law's form
(``run_srbench_baseline.symbolic_recovery`` against the known form fitted to the table), and its R² on all
rows beside the known form's. Nothing here is a threshold; the numbers are for reading.

    python scripts/run_real_data.py dev --out DIR [--searches 90]
    python scripts/run_real_data.py small --out DIR [--searches 90]

``small`` is the small-table reading (``apedp_dyn/discovery_l2/small_table.py``) on the same three tables: with
nothing stated, and with what is known about each law beforehand (``benchmarks/real_data/known_beforehand.json``);
with ``--searches`` the search stage's candidates take part in both.
"""
from __future__ import annotations

import argparse
import json
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
import run_srbench_baseline as base  # noqa: E402

from apedp_dyn.discovery_l2 import small_table, vouched_route_v2  # noqa: E402
from apedp_dyn.ui import route_endpoint  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
DEVELOPMENT = ("kepler_solar_system", "langmuir_table_ix", "leavitt")
MIN_DIRECT_ROWS = 6          # below this a 70/30 split leaves under two rows to confirm on


def r2(y: np.ndarray, pred: np.ndarray) -> float | None:
    if pred.shape != y.shape or not np.all(np.isfinite(pred)):
        return None
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - pred) ** 2)) / total if total > 0 else None


def fitted_law(table: dict[str, Any], X: dict[str, np.ndarray], y: np.ndarray) -> tuple[sp.Expr, float | None]:
    """The known law's form with its constants fitted to the table by least squares, and its R² there."""
    names = list(X)
    symbols = [sp.Symbol(n) for n in names]
    params = [sp.Symbol(p) for p in table["law_parameters"]]
    form = sp.sympify(table["law"], locals={**{n: sp.Symbol(n) for n in names}, **{p: sp.Symbol(p) for p in table["law_parameters"]}})
    fn = sp.lambdify([*params, *symbols], form, "numpy")
    columns = [X[n] for n in names]
    spread = float(np.std(y)) or 1.0

    def resid(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            value = np.asarray(fn(*theta, *columns), dtype=float) * np.ones(len(y))
        out: np.ndarray = np.where(np.isfinite(value), (value - y) / spread, 1e3)
        return out

    best = None
    rng = np.random.default_rng(0)
    for attempt in range(40):
        start = np.ones(len(params)) if attempt == 0 else rng.choice([-1.0, 1.0], len(params)) * 10 ** rng.uniform(-3, 3, len(params))
        try:
            fit = least_squares(resid, start, max_nfev=4000)
        except (ValueError, FloatingPointError):
            continue
        if best is None or fit.cost < best.cost:
            best = fit
    if best is None:
        raise RuntimeError(f"{table['key']}: the known form could not be fitted")
    fitted = form.xreplace({p: sp.Float(float(v)) for p, v in zip(params, best.x, strict=True)})
    with np.errstate(all="ignore"):
        pred = np.asarray(fn(*best.x, *columns), dtype=float) * np.ones(len(y))
    return fitted, r2(y, pred)


def judged(law: str | None, truth: sp.Expr, X: dict[str, np.ndarray], y: np.ndarray) -> dict[str, Any]:
    """A law the route gave, beside the known one: the same form or not, and its R² on every row."""
    if not law:
        return {"law": None}
    names = list(X)
    symbols = [sp.Symbol(n) for n in names]
    model = sp.sympify(str(law), locals={n: sp.Symbol(n) for n in names})
    columns = np.column_stack([X[n] for n in names])
    with np.errstate(all="ignore"):
        pred = np.asarray(sp.lambdify(symbols, model, "numpy")(*columns.T), dtype=float) * np.ones(len(y))
    form = base.symbolic_recovery(model, truth, names, columns)
    score = r2(y, pred)
    return {"law": str(law), "same_form_as_the_known_law": bool(form["recovered"]), "refit_error": form.get("rel_rmse"),
            "r2_all_rows": None if score is None else round(score, 6), "constants": len(model.atoms(sp.Float))}


def keep(record: dict[str, Any]) -> dict[str, Any]:
    names = ("route", "action", "law", "law_as_confirmed", "reported", "tentative", "tentative_reason", "vouching",
             "measured_range", "found_by", "search_stage", "candidates", "complete", "rows_development", "rows_pool",
             "rows_read_from_pool", "rejected_law", "error", "seconds")
    return json.loads(json.dumps({k: record[k] for k in names if k in record}, default=str))


def direct(X: dict[str, np.ndarray], y: np.ndarray, seed: int, searches: float) -> dict[str, Any]:
    """The route itself on a table the product refuses for want of rows, with the product's own split."""
    started = time.perf_counter()
    names = list(X)
    columns = np.column_stack([X[n] for n in names])
    order = np.random.default_rng(seed).permutation(len(y))
    cut = round((1.0 - route_endpoint.POOL_SHARE) * len(y))
    dev, pool = order[:cut], order[cut:]
    pool_y = y[pool]
    out: dict[str, Any] = {"rows_development": len(dev), "rows_pool": len(pool)}
    try:
        proposer, stage = route_endpoint._proposer(searches)
        got = vouched_route_v2.discover_law({n: columns[dev, i] for i, n in enumerate(names)}, y[dev], columns[pool],
                                            lambda rows: pool_y[rows], ledger=None, search_id="real-data-dev",
                                            proposer=proposer, noise="absolute", seed=seed)
        out.update({"route": got["route"], "action": got["action"], "law": got["law"], "law_as_confirmed": got["law_as_confirmed"],
                    "reported": bool(got["reported"]), "tentative": got.get("tentative_law") is not None,
                    "tentative_reason": got.get("tentative_reason"), "vouching": dict(got.get("vouching") or {}),
                    "measured_range": dict(got.get("measured_range") or {}), "search_stage": stage})
    except Exception as exc:  # noqa: BLE001 - what the route does on a table this small is the finding, an error included
        out.update({"action": "not_run", "error": f"{type(exc).__name__}: {exc}"[:300]})
    out["seconds"] = round(time.perf_counter() - started, 2)
    return json.loads(json.dumps(out, default=str))


def run_table(key: str, searches: float) -> dict[str, Any]:
    loaded = real_data.load(key)
    table, X, y = loaded["table"], loaded["X"], loaded["y"]
    truth, truth_r2 = fitted_law(table, X, y)
    readings = []
    for seed in SEEDS:
        product = keep(route_endpoint.route_reading(X, y, search_seconds=searches, seed=seed))
        entry: dict[str, Any] = {"seed": seed, "product": product,
                                 "product_law": judged(product.get("law_as_confirmed"), truth, X, y)}
        if product.get("action") == "not_run" and len(y) >= MIN_DIRECT_ROWS:
            entry["direct"] = direct(X, y, seed, searches)
            entry["direct_law"] = judged(entry["direct"].get("law_as_confirmed"), truth, X, y)
        readings.append(entry)

    def count(where: str, what: str) -> int:
        return sum(bool((r.get(where) or {}).get(what)) for r in readings)

    return {"key": key, "rows": len(y), "inputs": table["inputs"], "target": table["target"], "known_law": table["law"],
            "known_law_fitted": str(truth), "known_law_r2_all_rows": None if truth_r2 is None else round(truth_r2, 6),
            "readings": readings,
            "summary": {"splits": len(readings),
                        "product_refused": sum(r["product"].get("action") == "not_run" for r in readings),
                        "product_reported": count("product", "reported"),
                        "product_reported_the_law": sum(bool(r["product"].get("reported")) and
                                                        bool(r["product_law"].get("same_form_as_the_known_law")) for r in readings),
                        "product_gave_the_law_tentatively": sum(not r["product"].get("reported") and
                                                                bool(r["product_law"].get("same_form_as_the_known_law"))
                                                                for r in readings),
                        "direct_run": sum("direct" in r for r in readings),
                        "direct_reported": count("direct", "reported"),
                        "direct_reported_the_law": sum(bool((r.get("direct") or {}).get("reported")) and
                                                       bool((r.get("direct_law") or {}).get("same_form_as_the_known_law"))
                                                       for r in readings),
                        "direct_gave_the_law_tentatively": sum("direct" in r and not r["direct"].get("reported") and
                                                               bool((r.get("direct_law") or {}).get("same_form_as_the_known_law"))
                                                               for r in readings),
                        "direct_errors": sum("error" in (r.get("direct") or {}) for r in readings)}}


KNOWN = ROOT / "benchmarks" / "real_data" / "known_beforehand.json"


def small_readings(key: str, searches: float) -> dict[str, Any]:
    """The small-table reading on one development table: nothing stated, and what is known stated."""
    loaded = real_data.load(key)
    table, X, y = loaded["table"], loaded["X"], loaded["y"]
    stated = json.loads(KNOWN.read_text(encoding="utf-8"))["tables"][key]
    statements = [{k: v for k, v in s.items() if k != "why"} for s in stated["known"]]
    truth, truth_r2 = fitted_law(table, X, y)
    out: dict[str, Any] = {"key": key, "rows": len(y), "known_law": table["law"], "known_law_fitted": str(truth),
                           "known_law_r2_all_rows": None if truth_r2 is None else round(truth_r2, 6), "readings": {}}
    for name, options in (("plain", {"known": (), "units": None}), ("informed", {"known": statements, "units": stated["units"]})):
        proposer = route_endpoint._proposer(searches)[0] if searches > 0 else None
        reading = small_table.read(X, y, noise=stated["noise"], proposer=proposer, seed=0, **options)
        keep_names = ("action", "law", "reported", "tentative", "tentative_reasons", "errors", "chosen", "class_size", "class",
                      "best_by_left_out_error", "stability", "accuracy", "known_checks_on_the_law", "candidates", "error", "seconds")
        entry = json.loads(json.dumps({k: reading[k] for k in keep_names if k in reading}, default=str))
        entry["set_aside_by_what_is_known"] = (reading.get("set_aside_by_what_is_known") or {}).get("forms")
        entry["judged"] = judged(reading.get("law"), truth, X, y)
        out["readings"][name] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("dev", "small"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--searches", type=float, default=0.0, help="seconds for the search stage in force (0: first pass alone)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "small":
        readings = [small_readings(key, args.searches) for key in DEVELOPMENT]
        record = {"what": "development: the small-table reading on three tables of real measurements; not a result",
                  "search_seconds": args.searches, "tables": readings}
        (args.out / "real_data_small.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        for table in readings:
            for name, entry in table["readings"].items():
                print(table["key"], name, entry.get("action"), "reported" if entry.get("reported") else "tentative",
                      "| known form:", entry["judged"].get("same_form_as_the_known_law"), "|", str(entry.get("law"))[:80],
                      "|", (entry.get("chosen") or {}).get("source"))
        return 0
    tables = [run_table(key, args.searches) for key in DEVELOPMENT]
    record = {"what": "development: the table route in force on three tables of real measurements; not a result",
              "search_seconds": args.searches, "min_rows_of_the_product": route_endpoint.MIN_ROWS,
              "pool_share": route_endpoint.POOL_SHARE, "last_checks": vouched_route_v2.version(), "tables": tables}
    (args.out / "real_data_dev.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for table in tables:
        print(table["key"], table["rows"], "rows; known law R2", table["known_law_r2_all_rows"], json.dumps(table["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
