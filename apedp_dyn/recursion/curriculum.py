"""A curriculum the system makes for itself: invented laws at the difficulty it solves about half the time.

Status: experimental

Laws are drawn from the grammar of the invented-laws benchmark (blocks in one
input joined by sums, differences, products and quotients, with unround
constants), at a **level** from 0 to 6 that fixes how many inputs and blocks a
law has. Because the laws are invented here, the true law of every problem is
known, which is what lets a solve be scored without a human.

* ``Curriculum.batch`` draws laws around the current level; ``update`` moves the
  level up when more than ``UP`` of a batch was solved and down when fewer than
  ``DOWN`` was, so study stays where there is something to learn;
* ``measurement`` draws from every level of ``MEASURED_LEVELS`` equally. It does
  not move with the curriculum: a rate measured at each generation is only
  comparable if the problems come from one fixed distribution.

Every draw is seeded by its purpose and generation (``"study"``,
``"selection"``, ``"measurement"``, ...), so the three kinds of problem never
share a law, and no law is drawn twice in a run.

**The benchmark is left out without being read.** A drawn law whose structure
has the fingerprint of a law of ``benchmarks/invented_laws.json`` is skipped,
using ``benchmarks/invented_laws_fingerprints.json`` (hashes only). This module
never opens the benchmark itself, so the forty held-back laws stay unread.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
FINGERPRINTS = ROOT / "benchmarks" / "invented_laws_fingerprints.json"

LOW, HIGH = 1.0, 5.0
CONSTANTS = (0.3, 0.45, 0.7, 1.3, 1.7, 2.3, 2.9, 3.7)
PART_MATTERS = 0.05
#: level -> (inputs, blocks)
LEVELS = ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (3, 5))
MEASURED_LEVELS = (1, 2, 3, 4, 5)
UP, DOWN = 0.6, 0.4
ROWS, POOL, TEST_ROWS = 700, 300, 400
NOISES = (0.0, 0.01)
MAX_DRAWS = 20_000


def _block(rng: np.random.Generator, x: sp.Symbol) -> sp.Expr:
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


def draw(rng: np.random.Generator, level: int) -> tuple[sp.Expr, int]:
    """One law at a level: its blocks in its inputs (each input at least once), joined left to right."""
    inputs, blocks = LEVELS[int(np.clip(level, 0, len(LEVELS) - 1))]
    symbols = [sp.Symbol(f"x{i}") for i in range(inputs)]
    order = [*symbols, *(symbols[int(rng.integers(0, inputs))] for _ in range(blocks - inputs))]
    rng.shuffle(order)
    expr = sp.Float(float(rng.choice(CONSTANTS))) * _block(rng, order[0])
    for symbol in order[1:]:
        term, scale = _block(rng, symbol), sp.Float(float(rng.choice(CONSTANTS)))
        kind = int(rng.integers(0, 4))
        if kind == 0:
            expr = expr + scale * term
        elif kind == 1:
            expr = expr - scale * term
        elif kind == 2:
            expr = expr * term
        else:
            expr = expr / (scale + term)
    return expr, inputs


def structure(expr: sp.Expr) -> str:
    """A law with every constant replaced by ``C``: what the benchmark's fingerprints are taken of."""
    return str(expr.xreplace(dict.fromkeys(expr.atoms(sp.Float), sp.Symbol("C"))))


def fingerprint(expr: sp.Expr) -> str:
    return hashlib.sha256(structure(expr).encode()).hexdigest()


def benchmark_fingerprints() -> frozenset[str]:
    return frozenset(json.loads(FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])


def acceptable(expr: sp.Expr, inputs: int, rng: np.random.Generator) -> bool:
    """Uses every input, finite and varying on the domain, not a plain line, and every part of it matters."""
    symbols = [sp.Symbol(f"x{i}") for i in range(inputs)]
    if expr.free_symbols != set(symbols):
        return False
    if expr.is_polynomial(*symbols) and sp.Poly(expr, *symbols).total_degree() <= 1:
        return False
    X = rng.uniform(LOW, HIGH, (400, inputs))
    with np.errstate(all="ignore"):
        y = np.asarray(sp.lambdify(symbols, expr, "numpy")(*X.T), dtype=float) * np.ones(len(X))
    if not np.all(np.isfinite(y)):
        return False
    spread, size = float(np.std(y)), float(np.sqrt(np.mean(y ** 2)))
    if size <= 0 or spread < 0.05 * size or float(np.max(np.abs(y))) > 1e3 * float(np.median(np.abs(y)) + 1e-12):
        return False
    for part in sorted(set(sp.preorder_traversal(expr)), key=str):
        if part == expr or not isinstance(part, (sp.exp, sp.log, sp.sin, sp.tanh, sp.Pow, sp.Add)) or not part.free_symbols:
            continue
        with np.errstate(all="ignore"):
            values = np.asarray(sp.lambdify(symbols, part, "numpy")(*X.T), dtype=float) * np.ones(len(X))
            flat = expr.xreplace({part: sp.Float(float(np.mean(values)))})
            other = np.asarray(sp.lambdify(symbols, flat, "numpy")(*X.T), dtype=float) * np.ones(len(X))
        if not np.all(np.isfinite(other)) or float(np.sqrt(np.mean((other - y) ** 2))) < PART_MATTERS * spread:
            return False
    return True


@dataclass(frozen=True)
class Problem:
    """One invented law and how its rows are drawn. The rows are a function of ``key`` alone."""

    key: str
    text: str
    inputs: int
    level: int
    noise: float

    @property
    def names(self) -> list[str]:
        return [f"x{i}" for i in range(self.inputs)]

    @property
    def truth(self) -> sp.Expr:
        return sp.sympify(self.text, locals={n: sp.Symbol(n) for n in self.names})

    def rows(self) -> dict[str, np.ndarray]:
        """Rows to fit on (``X``, ``y``), the pool the route may ask to measure, and clean rows to score on."""
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"rows|{self.key}".encode()).digest()[:8], "big"))
        X = rng.uniform(LOW, HIGH, (ROWS + POOL + TEST_ROWS, self.inputs))
        symbols = [sp.Symbol(n) for n in self.names]
        clean = np.asarray(sp.lambdify(symbols, self.truth, "numpy")(*X.T), dtype=float) * np.ones(len(X))
        size = float(np.sqrt(np.mean(clean[: ROWS + POOL] ** 2)))
        y = clean + self.noise * size * rng.standard_normal(len(X))
        return {"X": X[:ROWS], "y": y[:ROWS], "pool_X": X[ROWS: ROWS + POOL], "pool_y": y[ROWS: ROWS + POOL],
                "test_X": X[ROWS + POOL:], "test_clean": clean[ROWS + POOL:]}

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "text": self.text, "inputs": self.inputs, "level": self.level, "noise": self.noise}


@dataclass
class Curriculum:
    """Where the system is studying, and every structure drawn so far."""

    seed: int
    level: float = 2.0
    seen: set[str] = field(default_factory=set)
    history: list[dict[str, Any]] = field(default_factory=list)
    _excluded: frozenset[str] = field(default_factory=benchmark_fingerprints)

    def _draw(self, count: int, purpose: str, levels: list[int]) -> list[Problem]:
        key = hashlib.sha256(f"curriculum|{self.seed}|{purpose}".encode()).digest()[:8]
        rng = np.random.default_rng(int.from_bytes(key, "big"))
        out: list[Problem] = []
        for attempt in range(MAX_DRAWS):
            if len(out) == count:
                break
            level = levels[len(out) % len(levels)]
            expr, inputs = draw(rng, level)
            shape = structure(expr)
            if shape in self.seen or fingerprint(expr) in self._excluded or not acceptable(expr, inputs, rng):
                continue
            self.seen.add(shape)
            out.append(Problem(f"{self.seed}|{purpose}|{len(out)}|{attempt}", str(expr), inputs, level,
                               NOISES[len(out) % len(NOISES)]))
        if len(out) != count:
            raise RuntimeError(f"only {len(out)} of {count} laws could be drawn for {purpose!r}")
        return out

    def batch(self, count: int, purpose: str) -> list[Problem]:
        """Laws around the current level: half at it, a quarter one below, a quarter one above."""
        at = round(self.level)
        top = len(LEVELS) - 1
        return self._draw(count, purpose, [at, max(at - 1, 0), at, min(at + 1, top)])

    def measurement(self, count: int, purpose: str) -> list[Problem]:
        """Laws from the fixed measured levels, equally: the same distribution at every generation."""
        return self._draw(count, purpose, list(MEASURED_LEVELS))

    def update(self, solved: int, of: int, purpose: str = "") -> float:
        """Move the level by what was solved; returns the new level."""
        share = solved / of if of else 0.0
        before = self.level
        if share > UP:
            self.level = min(self.level + 1.0, float(len(LEVELS) - 1))
        elif share < DOWN:
            self.level = max(self.level - 1.0, 0.0)
        self.history.append({"purpose": purpose, "solved": solved, "of": of, "level_before": before, "level_after": self.level})
        return self.level
