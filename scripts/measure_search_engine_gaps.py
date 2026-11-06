#!/usr/bin/env python3
"""Measure what the symbolic search engines can reach, before and after the fix.

`tests/test_search_engines_real.py` pins behaviour; it does not say which laws a
search *cannot* express. Two reachability gaps were measured and are recorded
here:

* **`HierarchicalSearch`**: the classic template set is six shapes -- `x`, `x**2`,
  `sin(x)`, `exp(x)`, `x0+x1`, `x0*x1` -- expanded by `x**2`, `x*2`, `2*x`,
  `sin(x)`, `x+1`. A decay (`exp(-x)`), a reciprocal, a square root, a cubic, a
  logarithm, a cosine, a rational saturation and a ratio are therefore
  *unreachable at any budget*, and a law found up to a constant could not be
  scored as such because no coefficient was fitted.
* **`beam_search_expressions`**: the atom set is the variables plus `1` and `2`,
  composed with `+ - * /`. Every transcendental form is again unreachable.

The extended sets close both: more structural families, plus an exact
least-squares scale and offset fitted to each candidate before it is scored.

The measurement is a train/holdout split of planted laws. The search sees the
training rows only; the holdout rows score the candidate the search returned,
refitted on the training rows. "Recovered" means holdout R^2 >= 0.99 measured on
rows no part of the search has seen.

Example::

    .venv/bin/python scripts/measure_search_engine_gaps.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apedp.advanced_search import (  # noqa: E402 - after the path fix
    _TRIG_FREQUENCY_SCALES,
    HierarchicalSearch,
    _evaluate_expression,
    _fit_scale_offset,
)
from apedp.feature_selection import beam_search_expressions  # noqa: E402

DEFAULT_RECORD = ROOT / "benchmarks" / "search_engine_gaps.json"

#: Planted laws, one per structural family. Coefficients are non-unit on purpose:
#: a search that only matches the shape exactly would fail every one of them.
LAWS = (
    ("linear", "3.0*x0", 1),
    ("quadratic", "2.0*x0**2 + 1.0", 1),
    ("cubic", "0.5*x0**3", 1),
    ("square_root", "4.0*sqrt(x0)", 1),
    ("inverse", "1.2/x0", 1),
    ("power_2_5", "x0**2.5", 1),
    ("exponential_decay", "2.0*exp(-0.7*x0)", 1),
    ("logarithm", "3.0*log(x0)", 1),
    ("sine", "sin(2.0*x0)", 1),
    ("rational_saturation", "3.0*x0/(1.0+0.5*x0)", 1),
    ("product", "x0*x1", 2),
    ("ratio", "x0/x1", 2),
)

RECOVERY_R2 = 0.99


def sample_law(expr: str, seed: int, n: int, n_vars: int) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    columns = [rng.uniform(0.5, 3.0, n) for _ in range(n_vars)]
    scope = {f"x{i}": column for i, column in enumerate(columns)}
    scope.update({"sqrt": np.sqrt, "exp": np.exp, "log": np.log, "sin": np.sin})
    y = np.asarray(eval(expr, {"__builtins__": {}}, scope), float)  # noqa: S307 - fixed law strings
    noise = 0.01 * float(np.std(y))
    return columns, y + rng.normal(0.0, noise, n)


def holdout_r2(expr: str, names: list[str], x_train: list[np.ndarray], y_train: np.ndarray,
               x_test: list[np.ndarray], y_test: np.ndarray) -> dict | None:
    """Refit the candidate's scale and offset on train, score it on holdout."""
    fit = _fit_scale_offset(expr, names, x_train, y_train)
    if fit is None:
        return None
    _, scale, offset = fit
    prediction = _evaluate_expression(expr, names, x_test)
    if prediction is None:
        return None
    prediction = np.broadcast_to(prediction, y_test.shape).ravel()
    if not np.all(np.isfinite(prediction)):
        return None
    residual = y_test - (scale * prediction + offset)
    sst = float(np.sum((y_test - y_test.mean()) ** 2))
    if sst <= 0.0:
        return None
    return {"holdout_r2": 1.0 - float(np.sum(residual ** 2)) / sst,
            "scale": scale, "offset": offset}


def run_hierarchical(template_set: str, names, x_train, y_train, x_test, y_test,
                     budget: int) -> dict:
    engine = HierarchicalSearch(template_set=template_set)
    candidates = engine.search(np.column_stack(x_train), y_train, budget=budget,
                               variable_names=names)
    if not candidates:
        return {"expression": None, "train_r2": None, "holdout_r2": None,
                "recovered": False, "n_candidates": 0}
    best = candidates[0]
    scored = holdout_r2(best["expression"], names, x_train, y_train, x_test, y_test)
    holdout = scored["holdout_r2"] if scored else None
    return {
        "expression": best["expression"],
        "train_r2": float(best["r2"]),
        "holdout_r2": holdout,
        "recovered": holdout is not None and holdout >= RECOVERY_R2,
        "n_candidates": len(candidates),
    }


def run_beam(include_functions: bool, names, x_train, y_train, x_test, y_test,
             depth: int, width: int) -> dict:
    expressions = beam_search_expressions(names, depth=depth, width=width,
                                          include_functions=include_functions)
    best_expr, best_train = None, None
    for expression in expressions:
        fit = _fit_scale_offset(expression, names, x_train, y_train)
        if fit is None:
            continue
        if best_train is None or fit[0] > best_train:
            best_expr, best_train = expression, fit[0]
    holdout = None
    if best_expr is not None:
        scored = holdout_r2(best_expr, names, x_train, y_train, x_test, y_test)
        holdout = scored["holdout_r2"] if scored else None
    return {
        "expression": best_expr,
        "train_r2": best_train,
        "holdout_r2": holdout,
        "recovered": holdout is not None and holdout >= RECOVERY_R2,
        "n_candidates": len(expressions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--holdout", type=int, default=100)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--beam-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()

    rows = []
    for index, (name, expression, n_vars) in enumerate(LAWS):
        names = [f"x{i}" for i in range(n_vars)]
        columns, y = sample_law(expression, args.seed + index,
                                args.train + args.holdout, n_vars)
        x_train = [column[:args.train] for column in columns]
        x_test = [column[args.train:] for column in columns]
        y_train, y_test = y[:args.train], y[args.train:]
        rows.append({
            "law": name,
            "planted_expression": expression,
            "variables": names,
            "classic": run_hierarchical("classic", names, x_train, y_train,
                                        x_test, y_test, args.budget),
            "extended": run_hierarchical("extended", names, x_train, y_train,
                                         x_test, y_test, args.budget),
            "beam_classic": run_beam(False, names, x_train, y_train, x_test, y_test,
                                     args.beam_depth, args.beam_width),
            "beam_extended": run_beam(True, names, x_train, y_train, x_test, y_test,
                                      args.beam_depth, args.beam_width),
        })

    engines = ("classic", "extended", "beam_classic", "beam_extended")
    recovered = {engine: [row["law"] for row in rows if row[engine]["recovered"]]
                 for engine in engines}
    closed = [row["law"] for row in rows
              if row["extended"]["recovered"] and not row["classic"]["recovered"]]
    still = [row["law"] for row in rows
             if not row["extended"]["recovered"] and not row["beam_extended"]["recovered"]]

    record = {
        "experiment": "reachability of planted law families, classic versus extended templates",
        "protocol": {
            "train_rows": args.train,
            "holdout_rows": args.holdout,
            "noise": "1% of the target's standard deviation",
            "budget": args.budget,
            "beam_depth": args.beam_depth,
            "beam_width": args.beam_width,
            "seed": args.seed,
            "extended_trig_frequency_scales": list(_TRIG_FREQUENCY_SCALES),
            "recovery_criterion": f"holdout R^2 >= {RECOVERY_R2}, coefficients refitted on train",
            "leakage": "search sees training rows only; the holdout rows score the returned "
                       "candidate and are never read during the search",
        },
        "laws": [
            {
                "law": row["law"],
                "planted_expression": row["planted_expression"],
                "variables": row["variables"],
                "engines": {engine: row[engine] for engine in engines},
            }
            for row in rows
        ],
        "summary": {
            "n_laws": len(rows),
            "recovered_by_engine": {engine: len(recovered[engine]) for engine in engines},
            "recovered_laws": recovered,
            "closed_by_extended_hierarchical": closed,
            "still_unreachable_by_both_extended_engines": still,
        },
        "checks": {
            "every_recovered_candidate_clears_the_criterion": all(
                row[engine]["holdout_r2"] is None
                or row[engine]["holdout_r2"] < RECOVERY_R2
                or row[engine]["recovered"]
                for row in rows for engine in engines),
            "extended_covers_every_classic_recovery": set(recovered["classic"]).issubset(
                recovered["extended"]),
            "extended_covers_every_beam_classic_recovery": set(
                recovered["beam_classic"]).issubset(recovered["beam_extended"]),
        },
    }

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")

    print(f"{'law':22s} " + " ".join(f"{engine:>13s}" for engine in engines))
    for row in rows:
        cells = []
        for engine in engines:
            value = row[engine]["holdout_r2"]
            cells.append("      none   " if value is None else f"{value:13.4f}")
        print(f"{row['law']:22s} " + " ".join(cells))
    print()
    for engine in engines:
        print(f"{engine:14s} recovered {len(recovered[engine])}/{len(rows)}: "
              f"{', '.join(recovered[engine]) or '-'}")
    print(f"closed by the extended hierarchical set: {', '.join(closed) or 'none'}")
    print(f"still unreachable: {', '.join(still) or 'none'}")
    print(f"record: {args.record}")
    return 0 if all(record["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
