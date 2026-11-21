#!/usr/bin/env python3
"""Sixty laws that nobody has published, for testing a search that might remember the published ones.

A language model may have read every textbook law, so a benchmark of textbook laws cannot tell
what it finds from what it recalls. These laws are drawn by a seeded generator from a small
grammar: one to three inputs on (1, 5), sums, differences, products and quotients of a few
building blocks (an input, a power of one, an exponential decay, a logarithm, a sine, a saturating
ratio, a square root), with constants taken from a list of unround numbers. A draw is kept only if
it uses every input, is finite and varies on the domain, is not a polynomial of degree two or
less, costs between ``MIN_COST`` and ``MAX_COST`` operations and constants, has a structure not
drawn before, and needs every one of its parts: replacing any function in it by that function's
average over the domain must move the law by at least ``PART_MATTERS`` of its spread. Without
that last rule a law can carry a part that no data could reveal (a ``tanh`` that is flat on the
whole domain), and a search that drops it would be marked wrong (audit M-03).

    python scripts/build_invented_laws.py           # write benchmarks/invented_laws.json
    python scripts/build_invented_laws.py --check   # the file is what the generator gives

The first ``DEVELOPMENT`` laws are for development; the rest are held back for a preregistered
run. Nothing claims these resemble natural laws: they are a test of search, not of physics.

``benchmarks/invented_laws_fingerprints.json`` holds the SHA-256 of every law's structure and
nothing else. Anything that draws further laws from this grammar for training (the loop of
``apedp_dyn.evolution``) uses it to leave out structures that are in the benchmark, without
reading a held-back law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "invented_laws.json"
FINGERPRINTS = ROOT / "benchmarks" / "invented_laws_fingerprints.json"

SEED = 20261121
COUNT = 60
DEVELOPMENT = 20
LOW, HIGH = 1.0, 5.0
MIN_COST, MAX_COST = 5, 14
CONSTANTS = (0.3, 0.45, 0.7, 1.3, 1.7, 2.3, 2.9, 3.7)
PART_MATTERS = 0.05     # of the law's spread: what replacing a part by its average has to change
MAX_DRAWS = 200_000


def block(rng: np.random.Generator, x: sp.Symbol) -> sp.Expr:
    """One building block in one input."""
    a = sp.Float(float(rng.choice(CONSTANTS)))
    kind = int(rng.integers(0, 9))
    if kind == 0:
        return x
    if kind == 1:
        return x ** int(rng.choice([2, 3]))
    if kind == 2:
        return 1 / x
    if kind == 3:
        return sp.exp(-a * x)
    if kind == 4:
        return sp.log(1 + a * x)
    if kind == 5:
        return sp.sin(a * x)
    if kind == 6:
        return x / (a + x)
    if kind == 7:
        return sp.sqrt(a + x)
    return sp.tanh(a * x)


def draw(rng: np.random.Generator, inputs: int) -> sp.Expr:
    """Blocks in the inputs (each input at least once), joined left to right by random operations."""
    symbols = [sp.Symbol(f"x{i}") for i in range(inputs)]
    extra = int(rng.integers(0, 2))
    order = [*symbols, *(symbols[int(rng.integers(0, inputs))] for _ in range(extra))]
    rng.shuffle(order)
    expr = sp.Float(float(rng.choice(CONSTANTS))) * block(rng, order[0])
    for symbol in order[1:]:
        term, scale = block(rng, symbol), sp.Float(float(rng.choice(CONSTANTS)))
        kind = int(rng.integers(0, 4))
        if kind == 0:
            expr = expr + scale * term
        elif kind == 1:
            expr = expr - scale * term
        elif kind == 2:
            expr = expr * term
        else:
            expr = expr / (scale + term)
    return expr


def cost(expr: sp.Expr) -> int:
    return int(sp.count_ops(expr)) + len(expr.atoms(sp.Float))


def structure(expr: sp.Expr) -> str:
    return str(expr.xreplace(dict.fromkeys(expr.atoms(sp.Float), sp.Symbol("C"))))


def acceptable(expr: sp.Expr, inputs: int, rng: np.random.Generator) -> bool:
    symbols = [sp.Symbol(f"x{i}") for i in range(inputs)]
    if expr.free_symbols != set(symbols) or not MIN_COST <= cost(expr) <= MAX_COST:
        return False
    if expr.is_polynomial(*symbols) and sp.Poly(expr, *symbols).total_degree() <= 2:
        return False
    X = rng.uniform(LOW, HIGH, (400, inputs))
    with np.errstate(all="ignore"):
        y = np.asarray(sp.lambdify(symbols, expr, "numpy")(*X.T), dtype=float) * np.ones(len(X))
    if not np.all(np.isfinite(y)) or np.iscomplexobj(y):
        return False
    spread, size = float(np.std(y)), float(np.sqrt(np.mean(y ** 2)))
    if size <= 0 or spread < 0.05 * size or float(np.max(np.abs(y))) > 1e3 * float(np.median(np.abs(y)) + 1e-12):
        return False
    for part in sorted(set(sp.preorder_traversal(expr)), key=str):
        if part == expr or not isinstance(part, (sp.exp, sp.log, sp.sin, sp.tanh, sp.Pow, sp.Add)) or not part.free_symbols:
            continue                          # a function, a power or a sum inside the law: does its variation matter?
        with np.errstate(all="ignore"):
            values = np.asarray(sp.lambdify(symbols, part, "numpy")(*X.T), dtype=float) * np.ones(len(X))
            flat = expr.xreplace({part: sp.Float(float(np.mean(values)))})
            other = np.asarray(sp.lambdify(symbols, flat, "numpy")(*X.T), dtype=float) * np.ones(len(X))
        if not np.all(np.isfinite(other)) or float(np.sqrt(np.mean((other - y) ** 2))) < PART_MATTERS * spread:
            return False
    return True


def build() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    laws: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(MAX_DRAWS):
        if len(laws) == COUNT:
            break
        inputs = int(rng.integers(1, 4))
        expr = draw(rng, inputs)
        key = structure(expr)
        if key in seen or not acceptable(expr, inputs, rng):
            continue
        seen.add(key)
        laws.append({"id": 9000 + len(laws), "inputs": inputs, "text": str(expr), "cost": cost(expr),
                     "split": "development" if len(laws) < DEVELOPMENT else "held_out"})
    if len(laws) != COUNT:
        raise SystemExit(f"only {len(laws)} laws after {MAX_DRAWS} draws")
    return {"note": "Invented by scripts/build_invented_laws.py; not published laws and not meant to resemble any.",
            "seed": SEED, "domain": [LOW, HIGH], "constants": list(CONSTANTS), "laws": laws}


def fingerprint(text: str) -> str:
    """The SHA-256 of the structure of a law given as text: what two laws share if they differ only in their constants."""
    expr = sp.sympify(text, locals={f"x{i}": sp.Symbol(f"x{i}") for i in range(3)})
    return hashlib.sha256(structure(expr).encode()).hexdigest()


def fingerprints(record: dict[str, Any]) -> dict[str, Any]:
    """The structure fingerprints of every law of the benchmark, sorted, with no law's text."""
    return {"note": "SHA-256 of the structure of each law in invented_laws.json (constants replaced by C), sorted. "
                    "For leaving these structures out of training sets; no law can be read back from it.",
            "fingerprints": sorted(fingerprint(law["text"]) for law in record["laws"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    record = build()
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    prints = json.dumps(fingerprints(record), indent=1, sort_keys=True) + "\n"
    if args.check:
        for path, expected in ((OUT, text), (FINGERPRINTS, prints)):
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                print(f"{path.name} is not what the generator gives")
                return 1
        print(f"{OUT.name}: {len(record['laws'])} laws, as generated; {FINGERPRINTS.name} matches")
        return 0
    OUT.write_text(text, encoding="utf-8")
    FINGERPRINTS.write_text(prints, encoding="utf-8")
    by_inputs = {k: sum(law["inputs"] == k for law in record["laws"]) for k in (1, 2, 3)}
    print(f"wrote {OUT.relative_to(ROOT)}: {len(record['laws'])} laws, by number of inputs {by_inputs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
