#!/usr/bin/env python3
"""Does counting independent trials keep the deflated Sharpe ratio honest and make it see more? (development)

Status: audit-only

**Development, not a result, and not a test of anything in force.** The owner asked on 26 November to
build and test the idea found in US patent application US20190294990A1 (the method of López de Prado and
Lewis, 2019): deflate the selected strategy against the number of *independent* trials, estimated by
clustering the trials' returns, rather than against every trial. ``apedp_dyn.epistemic.independent_trials``
does it three ways; this script measures, on simulated searches whose structure is known, how often each
calls a strategy significant at the canonical DSR threshold:

* when no strategy has an edge (the false-positive rate; the line is 5 %),
* when one family of strategies has a true edge of annualised Sharpe ratio 2.5 or 3.5 over two years of
  daily bars (how often the edge is found, and how often something else is called significant instead).

Families: 50 independent strategies; 10 families of 10 near copies at within-family correlations 0.5,
0.8 and 0.95; five families of 2, 5, 10, 20 and 40 at 0.8; and, where counting by clusters could count too
few, 10 families of 10 at correlations 0.1, 0.2 and 0.35, and 4 families of 10 at 0.8 beside 60 independent
strategies. Every strategy has daily volatility 1 %.
The seed and the repetitions are fixed; nothing here is a held-out measurement.

    python scripts/study_independent_trials.py --out DIR [--null 300] [--alt 150]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.epistemic import independent_trials as it  # noqa: E402

TAG = "DEV-INDEPENDENT-TRIALS-2026-11"
SEED = 2026112611
BARS = 500
BARS_PER_YEAR = 252
VOL = 0.01
STRUCTURES = {
    "independent 50": {"sizes": [1] * 50, "rho": 0.0},
    "10 x 10, rho 0.5": {"sizes": [10] * 10, "rho": 0.5},
    "10 x 10, rho 0.8": {"sizes": [10] * 10, "rho": 0.8},
    "10 x 10, rho 0.95": {"sizes": [10] * 10, "rho": 0.95},
    "2/5/10/20/40, rho 0.8": {"sizes": [2, 5, 10, 20, 40], "rho": 0.8},
    # where counting by clusters could undercount: families whose members barely move together
    "10 x 10, rho 0.1": {"sizes": [10] * 10, "rho": 0.1},
    "10 x 10, rho 0.2": {"sizes": [10] * 10, "rho": 0.2},
    "10 x 10, rho 0.35": {"sizes": [10] * 10, "rho": 0.35},
    "4 x 10 at rho 0.8 and 60 alone": {"sizes": [10] * 4 + [1] * 60, "rho": 0.8},
}
EDGES = (2.5, 3.5)
METHODS = ("as_referee", "by_cluster", "by_eigenvalues")


def world(sizes: list[int], rho: float, edge_family: int | None, annual_sharpe: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns (bars x strategies) and each strategy's family. Members of a family are sqrt(rho) f + sqrt(1 - rho) e
    with a shared factor f (a family of one is f itself); the edge family's factor has the drift that gives
    each member the annualised Sharpe ratio asked for."""
    cols, fam = [], []
    per_bar = annual_sharpe / math.sqrt(BARS_PER_YEAR)
    for g, size in enumerate(sizes):
        alone = size == 1
        f = rng.standard_normal(BARS)
        if g == edge_family:
            f = f + (per_bar if alone else per_bar / math.sqrt(rho))
        for _ in range(size):
            x = f if alone else math.sqrt(rho) * f + math.sqrt(1.0 - rho) * rng.standard_normal(BARS)
            cols.append(VOL * x)
            fam.append(g)
    return np.column_stack(cols), np.array(fam)


def run(null_reps: int, alt_reps: int) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    cells: dict[str, Any] = {}
    for name, spec in STRUCTURES.items():
        sizes, rho = spec["sizes"], spec["rho"]
        for edge in (0.0, *EDGES):
            reps = null_reps if edge == 0 else alt_reps
            family = None if edge == 0 else int(np.argmax(sizes) if len(set(sizes)) > 1 else 0)
            counts = {m: {"significant": 0, "edge_found": 0, "other_called": 0} for m in METHODS}
            clusters, eig = [], []
            for _ in range(reps):
                r, fam = world(sizes, rho, family, edge, rng)
                d = it.deflated(r)
                clusters.append(d["by_cluster"]["n_trials"])
                eig.append(d["by_eigenvalues"]["n_trials"])
                on_edge = family is not None and int(fam[d["selected"]]) == family
                for m in METHODS:
                    if d[m]["significant"]:
                        counts[m]["significant"] += 1
                        if family is not None:
                            counts[m]["edge_found" if on_edge else "other_called"] += 1
            key = f"{name} | {'no edge' if edge == 0 else f'edge {edge}'}"
            row: dict[str, Any] = {"strategies": sum(sizes), "families": len(sizes), "rho": rho, "annual_sharpe_of_edge": edge,
                                   "repetitions": reps, "clusters_median": float(np.median(clusters)),
                                   "clusters_range": [int(min(clusters)), int(max(clusters))],
                                   "eigen_effective_median": float(np.median(eig))}
            for m in METHODS:
                c = counts[m]
                row[m] = {"rate_significant": round(c["significant"] / reps, 4),
                          "upper_95": round(_upper(c["significant"], reps), 4)}
                if family is not None:
                    row[m]["rate_edge_found"] = round(c["edge_found"] / reps, 4)
                    row[m]["rate_other_called"] = round(c["other_called"] / reps, 4)
            cells[key] = row
            print(key, {m: row[m] for m in METHODS}, "clusters", row["clusters_median"], flush=True)
    return cells


def _upper(k: int, n: int) -> float:
    """Clopper-Pearson 95 % upper bound on a rate."""
    from scipy import stats
    return 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--null", type=int, default=300)
    parser.add_argument("--alt", type=int, default=150)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    cells = run(args.null, args.alt)
    nulls = {k: v for k, v in cells.items() if v["annual_sharpe_of_edge"] == 0}
    record = {"tag": TAG, "development_not_a_result": True, "seed": SEED, "bars": BARS, "daily_volatility": VOL,
              "threshold": it.DSR_THRESHOLD, "methods": list(METHODS),
              "worst_false_positive_rate": {m: max(v[m]["rate_significant"] for v in nulls.values()) for m in METHODS},
              "cells": cells, "seconds": round(time.time() - began, 1)}
    (args.out / "independent_trials.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["worst_false_positive_rate"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
