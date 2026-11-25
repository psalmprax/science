"""What the system has solved before, found again by what a table looks like.

Status: experimental

**Why.** Every reading starts from nothing. The memory of confirmed laws keeps their parts, and
offers the same parts to every table whatever the table looks like. A store of *cases* keeps each
solved table's law beside a short description of the table, so that a new table can be met with
the laws of the tables most like it: as examples for a language model, as shapes to refit, or as
a way of ordering what another proposer offers.

**What is kept.** For each case: the law as text, the number of inputs, and a signature of the
table made of rank statistics only (how each input moves with the target, how the target bends
over each input, how the target is spread). Nothing of the rows themselves is kept. The signature
does not depend on the order of the inputs or on the units of anything.

**What it is not.** Not a search and not a judge: it hands back cases, and whatever is done with
them goes to the referee like any proposal (AGENTS.md 1.1). A case must come from a law confirmed
on *another* table; a store that held a table's own answer would be reading the answer back.
**Not in force**, and nothing builds it but its study script and tests.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp
from scipy.stats import rankdata

MAX_INPUTS = 5
PER_INPUT = 2
SPREAD_FEATURES = 3
#: what one unit of each number is worth: rank correlations and a share run over about 1, the skew over about 2.5, the
#: spread (decades between the 5th and 95th percentile of the target's size) over about 3. Fixed, so that two tables are
#: as far apart whatever else the store holds.
SCALE = np.array([1.0] * (PER_INPUT * MAX_INPUTS) + [1.0, 2.5, 3.0])


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    sa, sb = float(np.std(ra)), float(np.std(rb))
    return 0.0 if sa == 0.0 or sb == 0.0 else float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def signature(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A table as a few rank statistics: per input, how the target moves and bends over it; then how the target is spread."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(y)), X])
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    per_input = sorted(((_rank_correlation(X[:, j], y), _rank_correlation(np.abs(X[:, j] - np.median(X[:, j])), residual))
                        for j in range(X.shape[1])), reverse=True)[:MAX_INPUTS]
    flat = [value for pair in per_input for value in pair] + [0.0] * (PER_INPUT * (MAX_INPUTS - len(per_input)))
    spread = float(np.std(y)) or 1.0
    centred = (y - float(np.mean(y))) / spread
    low, high = np.quantile(np.abs(y), [0.05, 0.95])
    out: np.ndarray = np.array([*flat, float(np.mean(y < 0)), float(np.clip(np.mean(centred ** 3), -5, 5)),
                                float(np.clip(np.log10((high + 1e-12) / (low + 1e-12)), 0, 6))], dtype=float)
    return out


def skeleton(law: str | sp.Expr) -> str:
    """A law with each fitted constant replaced by ``c0, c1, ...`` in the order they are met; small whole numbers stay."""
    counter = [0]

    def walk(expr: sp.Expr) -> sp.Expr:
        if expr.is_Float:
            counter[0] += 1
            return sp.Symbol(f"c{counter[0] - 1}")
        if not expr.args:
            return expr
        return expr.func(*[walk(arg) for arg in expr.args])

    return str(walk(sp.sympify(law)))


@dataclass(frozen=True)
class Case:
    """One solved table: its law, and what the table looked like."""

    key: str
    law: str
    inputs: int
    signature: tuple[float, ...]


class Experience:
    """Solved cases, and the ones most like a new table."""

    def __init__(self, cases: Sequence[Case] = ()) -> None:
        self.cases: list[Case] = list(cases)

    def add(self, key: str, law: str, X: np.ndarray, y: np.ndarray) -> Case:
        case = Case(str(key), str(law), int(np.asarray(X).shape[1]), tuple(float(v) for v in signature(X, y)))
        self.cases.append(case)
        return case

    def nearest(self, X: np.ndarray, y: np.ndarray, k: int = 8, exclude: Sequence[str] = ()) -> list[Case]:
        """The ``k`` cases with as many inputs whose tables looked most like this one, nearest first."""
        inputs = int(np.asarray(X).shape[1])
        skipped = set(exclude)
        same = [case for case in self.cases if case.inputs == inputs and case.key not in skipped]
        if not same:
            return []
        table = np.array([case.signature for case in same], dtype=float)
        distance = np.sqrt((((table - signature(X, y)) / SCALE) ** 2).sum(axis=1))
        return [same[int(index)] for index in np.argsort(distance, kind="stable")[: int(k)]]

    def as_dict(self) -> dict[str, Any]:
        return {"cases": [{"key": c.key, "law": c.law, "inputs": c.inputs, "signature": list(c.signature)} for c in self.cases]}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Experience:
        return cls([Case(str(c["key"]), str(c["law"]), int(c["inputs"]), tuple(float(v) for v in c["signature"]))
                    for c in record.get("cases", [])])
