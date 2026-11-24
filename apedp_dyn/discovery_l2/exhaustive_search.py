"""Exhaustive search over small expressions, scored by description length.

Status: experimental

Every expression up to ``max_complexity`` nodes over the given variables and
operators is generated bottom-up, so within that limit nothing is missed by a
stochastic search (the idea of Exhaustive Symbolic Regression, Bartlett, Desmond
& Ferreira 2023). Differences from that method, stated so the name is not
borrowed for more than is done here:

- expressions carry no free constants inside them. Each is fitted as
  ``a * f(x) + b``, and the best single terms are also tried in sums
  ``a * f + b * g + c`` with every simple term. Exhaustive over terms, greedy
  over sums.
- two expressions that are the same function up to scale and offset are one
  candidate: duplicates are removed by a numeric fingerprint (values on fixed
  probe points, centred and scaled), which is what makes the count tractable.
- the score is a two-part description length with a BIC parameter cost:
  ``n/2 ln(RSS/n) + nodes * ln(alphabet) + params/2 * ln(n)``; not the
  Fisher-information code of the ESR papers.

The candidate list is deterministic, so any number of workers can each take a
slice of it (``score_slice``) and the coordinator merges their best
(``merge``); ``work_queue.py`` carries the jobs.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

UNARY = {
    "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos,
    "inv": lambda a: 1.0 / a, "sq": lambda a: a * a,
}
BINARY = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}
COMMUTATIVE = {"+", "*"}
Node = tuple[Any, ...]
N_PROBE = 24


@dataclass(frozen=True)
class Scored:
    expression: str
    terms: tuple[str, ...]
    complexity: int
    n_params: int
    rss: float
    description_length: float
    coef: tuple[float, ...]


def _probes(n_vars: int) -> np.ndarray:
    probes: np.ndarray = np.random.default_rng(20261119).uniform(0.6, 3.1, (N_PROBE, n_vars))
    return probes


def evaluate(node: Node, X: np.ndarray) -> np.ndarray:
    """Values of an expression tree on the rows of ``X`` (non-finite where undefined)."""
    with np.errstate(all="ignore"):
        if node[0] == "x":
            out: np.ndarray = X[:, node[1]]
        elif len(node) == 2:
            out = UNARY[node[0]](evaluate(node[1], X))
        else:
            out = BINARY[node[0]](evaluate(node[1], X), evaluate(node[2], X))
    return out


def to_string(node: Node, names: Sequence[str]) -> str:
    """The expression in a form sympy parses."""
    if node[0] == "x":
        return str(names[node[1]])
    if len(node) == 2:
        inner = to_string(node[1], names)
        return {"inv": f"(1/({inner}))", "sq": f"(({inner})**2)"}.get(node[0], f"{node[0]}({inner})")
    return f"(({to_string(node[1], names)}) {node[0]} ({to_string(node[2], names)}))"


def _fingerprint(values: np.ndarray) -> bytes | None:
    if not np.all(np.isfinite(values)):
        return None
    centred = values - values.mean()
    scale = float(np.sqrt(np.mean(centred ** 2)))
    if scale < 1e-9 * max(1.0, float(np.max(np.abs(values)))):
        return None  # constant: the fitted offset already covers it
    unit = centred / scale
    if unit[np.argmax(np.abs(unit))] < 0:
        unit = -unit
    key: bytes = np.round(unit, 7).tobytes()
    return key


def enumerate_expressions(
    n_vars: int, max_complexity: int, unary: Sequence[str] = tuple(UNARY), binary: Sequence[str] = tuple(BINARY),
) -> list[tuple[Node, int]]:
    """Every distinct function (up to scale and offset) of at most ``max_complexity`` nodes, in a fixed order."""
    probes = _probes(n_vars)
    seen: set[bytes] = set()
    levels: dict[int, list[tuple[Node, np.ndarray]]] = {}
    out: list[tuple[Node, int]] = []

    def admit(node: Node, values: np.ndarray, c: int) -> None:
        key = _fingerprint(values)
        if key is None or key in seen:
            return
        seen.add(key)
        levels.setdefault(c, []).append((node, values))
        out.append((node, c))

    for i in range(n_vars):
        admit(("x", i), probes[:, i], 1)
    for c in range(2, max_complexity + 1):
        with np.errstate(all="ignore"):
            for name in unary:
                for child, values in levels.get(c - 1, []):
                    admit((name, child), UNARY[name](values), c)
            for left_c in range(1, c - 1):
                right_c = c - 1 - left_c
                for name in binary:
                    if name in COMMUTATIVE and left_c > right_c:
                        continue
                    for (ln_, lv), (rn, rv) in itertools.product(levels.get(left_c, []), levels.get(right_c, [])):
                        admit((name, ln_, rn), BINARY[name](lv, rv), c)
    return out


def _fit(columns: list[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, float] | None:
    A = np.column_stack([*columns, np.ones(len(y))])
    if not np.all(np.isfinite(A)):
        return None
    coef, _res, rank, _sv = np.linalg.lstsq(A, y, rcond=None)
    if rank < A.shape[1]:
        return None
    # Below ten digits of the target's size a residual is rounding, not misfit;
    # without this floor an exact law loses to itself plus a term fitted to rounding.
    floor = len(y) * (1e-10 * float(np.sqrt(np.mean(y ** 2)))) ** 2
    return coef, max(float(np.sum((y - A @ coef) ** 2)), floor)


def description_length(rss: float, n: int, complexity: int, n_params: int, alphabet: int,
                       structure_nats: float | None = None) -> float:
    """``structure_nats`` replaces the flat ``complexity * ln(alphabet)`` cost when a learned prior supplies one."""
    structure = complexity * math.log(alphabet) if structure_nats is None else structure_nats
    return 0.5 * n * math.log(max(rss, 1e-300) / n) + structure + 0.5 * n_params * math.log(n)


def score_slice(
    X: np.ndarray, y: np.ndarray, names: Sequence[str], max_complexity: int, *, part: int = 0, parts: int = 1,
    keep: int = 40, unary: Sequence[str] = tuple(UNARY), binary: Sequence[str] = tuple(BINARY),
    structure_bits: Callable[[Node], float] | None = None,
) -> list[dict[str, Any]]:
    """The ``keep`` best single-term fits among every ``parts``-th candidate starting at ``part``."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    alphabet = X.shape[1] + len(unary) + len(binary)
    rows: list[dict[str, Any]] = []
    for index, (node, c) in enumerate(enumerate_expressions(X.shape[1], max_complexity, unary, binary)):
        if index % parts != part:
            continue
        fit = _fit([evaluate(node, X)], y)
        if fit is None:
            continue
        coef, rss = fit
        nats = None if structure_bits is None else structure_bits(node) * math.log(2.0)
        rows.append({"index": index, "expression": to_string(node, names), "complexity": c, "rss": rss,
                     "coef": [float(v) for v in coef], "structure_nats": nats,
                     "description_length": description_length(rss, len(y), c, 2, alphabet, nats)})
    rows.sort(key=lambda r: (r["description_length"], r["index"]))
    return rows[:keep]


def merge(
    slices: Sequence[Sequence[dict[str, Any]]], X: np.ndarray, y: np.ndarray, names: Sequence[str],
    max_complexity: int, *, pair_pool: int = 40, partner_complexity: int = 3,
    structure_bits: Callable[[Node], float] | None = None, unary: Sequence[str] = tuple(UNARY), binary: Sequence[str] = tuple(BINARY),
) -> list[Scored]:
    """Combine the slices' best terms, pair each of the best ``pair_pool`` with every simple term, rank everything.

    A sum's second term need not fit well alone (in 3*sqrt(x0) + 2*x1 neither
    does), so partners are all terms of at most ``partner_complexity`` nodes,
    not only other leaders.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    alphabet = X.shape[1] + len(unary) + len(binary)
    singles = sorted((r for part in slices for r in part), key=lambda r: (r["description_length"], r["index"]))
    ranked = [Scored(r["expression"], (r["expression"],), r["complexity"], 2, r["rss"], r["description_length"], tuple(r["coef"]))
              for r in singles]
    nodes = enumerate_expressions(X.shape[1], max_complexity, unary, binary)
    partners = [(to_string(node, names), c, evaluate(node, X),
                 None if structure_bits is None else structure_bits(node) * math.log(2.0))
                for node, c in nodes if c <= partner_complexity]
    for first in singles[:pair_pool]:
        first_values = evaluate(nodes[first["index"]][0], X)
        for expr, c_b, partner_values, partner_nats in partners:
            if expr == first["expression"]:
                continue
            fit = _fit([first_values, partner_values], y)
            if fit is None:
                continue
            coef, rss = fit
            c = first["complexity"] + c_b + 1
            pair_nats = (None if partner_nats is None or first.get("structure_nats") is None
                         else first["structure_nats"] + partner_nats + math.log(alphabet))
            ranked.append(Scored(f"({first['expression']}) + ({expr})", (first["expression"], expr), c, 3, rss,
                                 description_length(rss, len(y), c, 3, alphabet, pair_nats),
                                 tuple(float(v) for v in coef)))
    ranked.sort(key=lambda s: s.description_length)
    return ranked


def search(
    X: np.ndarray, y: np.ndarray, names: Sequence[str], max_complexity: int = 5,
    unary: Sequence[str] = tuple(UNARY), binary: Sequence[str] = tuple(BINARY),
    structure_bits: Callable[[Node], float] | None = None,
) -> list[Scored]:
    """Single-process convenience: one slice, then merge.

    ``structure_bits(node)`` is an optional learned code length for a
    candidate's structure (``apedp_mmlm.equation_language.search_prior``); without
    it every node costs ``ln(alphabet)``.
    """
    one = score_slice(X, y, names, max_complexity, unary=unary, binary=binary, structure_bits=structure_bits)
    return merge([one], X, y, names, max_complexity, unary=unary, binary=binary, structure_bits=structure_bits)


def model_string(scored: Scored) -> str:
    """The fitted model as a sympy-parsable string, coefficients included."""
    parts = [f"{c!r}*({term})" for c, term in zip(scored.coef, scored.terms, strict=False)]
    return " + ".join([*parts, repr(scored.coef[-1])])


def slice_job(job: dict[str, Any]) -> dict[str, Any]:
    """A :mod:`work_queue` job: score one slice. ``job``: ``job_id``, ``data`` (``.npz`` with ``X`` and ``y``),
    ``names``, ``max_complexity``, ``part``, ``parts``."""
    arrays = np.load(job["data"])
    best = score_slice(arrays["X"], arrays["y"], job["names"], int(job["max_complexity"]),
                       part=int(job["part"]), parts=int(job["parts"]))
    return {"job_id": job["job_id"], "best": best}
