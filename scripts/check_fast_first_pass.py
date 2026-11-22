#!/usr/bin/env python3
"""Does the native first pass return what the Python reference returns? Many datasets, both run, compared.

    python scripts/check_fast_first_pass.py --datasets 200 [--seed 0] [--out FILE]

Each dataset is a law of the invented-law grammar (or a plain polynomial, or pure noise) on one to
three inputs, with 40 to 700 rows and noise between none and 10 %, the kinds of rows the main
search is given. ``confirmed_stop.search_confirmed`` (the reference, frozen) and
``fast_first_pass.search_confirmed`` (the native kernel) are run on each and their results compared
field by field: the same best law with the same floats, the same counts, the same early stop.

This is the evidence the fast path rests on (its error bounds are an analysis, not a proof), so a
single difference is a failure and is printed with the dataset that produced it. Timings of both
are reported; they are measurements on this machine, with whatever else it was doing.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_invented_laws as grammar  # noqa: E402

from apedp_dyn.discovery_l2 import confirmed_stop, fast_first_pass, main_search  # noqa: E402
from apedp_dyn.discovery_l2.guided_search import GuidedSearch  # noqa: E402

NOISE = (0.0, 0.0, 1e-4, 1e-3, 1e-2, 1e-2, 0.03, 0.1)
ROWS = (40, 70, 100, 210, 300, 500, 700)
KINDS = ("law", "law", "law", "polynomial", "noise", "law in other units")


def dataset(rng: np.random.Generator) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """One dataset and what it is."""
    inputs = int(rng.integers(1, 4))
    rows = int(rng.choice(ROWS))
    noise = float(rng.choice(NOISE))
    kind = str(rng.choice(KINDS))
    symbols = [sp.Symbol(f"x{i}") for i in range(inputs)]
    X = rng.uniform(grammar.LOW, grammar.HIGH, (rows, inputs))
    text = "noise"
    if kind == "noise":
        y = rng.standard_normal(rows)
    elif kind == "polynomial":
        powers = rng.integers(0, 3, inputs)
        y = float(rng.uniform(0.5, 3.0)) * np.prod(X ** powers, axis=1) + float(rng.uniform(-2.0, 2.0))
        text = f"polynomial {powers.tolist()}"
    else:
        for _ in range(200):
            expr = grammar.draw(rng, inputs)
            with np.errstate(all="ignore"):
                y = np.asarray(sp.lambdify(symbols, expr, "numpy")(*X.T), dtype=float) * np.ones(rows)
            if np.all(np.isfinite(y)) and float(np.std(y)) > 0:
                break
        text = str(expr)
        if kind == "law in other units":            # inputs and target far from order one
            X = X * 10.0 ** rng.integers(-3, 4, inputs)
            with np.errstate(all="ignore"):
                y = np.asarray(sp.lambdify(symbols, expr, "numpy")(*X.T), dtype=float) * np.ones(rows)
            if not np.all(np.isfinite(y)) or float(np.std(y)) == 0:
                y = X[:, 0] * 1e3
            y = y * 10.0 ** int(rng.integers(-6, 7))
    y = y + noise * float(np.sqrt(np.mean(y ** 2))) * rng.standard_normal(rows)
    return {"kind": kind, "inputs": inputs, "rows": rows, "noise": noise, "law": text}, X, y


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if fast_first_pass.kernel() is None:
        print("the native kernel is not installed or has no first-pass functions: nothing to compare")
        return 2
    guided = GuidedSearch(main_search.size_prior)
    rng = np.random.default_rng(args.seed)
    different: list[dict[str, Any]] = []
    seconds = {"reference": 0.0, "native": 0.0}
    paths = {"stopped early": 0, "full search": 0, "exhausted": 0}
    slowest = 0.0
    for number in range(args.datasets):
        what, X, y = dataset(rng)
        names = [f"x{i}" for i in range(what["inputs"])]
        size = 6 if len(names) <= 5 else 5
        guided.order(X.shape[1], size)               # the enumeration is shared by both and not what is timed
        started = time.perf_counter()
        reference = confirmed_stop.search_confirmed(guided, X, y, names, size)
        middle = time.perf_counter()
        native = fast_first_pass.search_confirmed(guided, X, y, names, size)
        done = time.perf_counter()
        seconds["reference"] += middle - started
        seconds["native"] += done - middle
        slowest = max(slowest, done - middle)
        paths["stopped early" if reference.stopped_early else "full search" if reference.residual_structure == 1.0 else "exhausted"] += 1
        if native != reference:
            different.append({**what, "number": number, "reference": repr(reference), "native": repr(native)})
            print(f"DIFFERENT on dataset {number}: {what}\n  reference {reference!r}\n  native    {native!r}", flush=True)
    record = {"datasets": args.datasets, "seed": args.seed, "different": len(different), "differences": different,
              "paths": paths, "seconds": {k: round(v, 2) for k, v in seconds.items()},
              "times_faster": round(seconds["reference"] / max(seconds["native"], 1e-9), 1),
              "slowest_native_seconds": round(slowest, 2)}
    print(json.dumps({k: v for k, v in record.items() if k != "differences"}, indent=1))
    if args.out:
        args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if different else 0


if __name__ == "__main__":
    raise SystemExit(main())
