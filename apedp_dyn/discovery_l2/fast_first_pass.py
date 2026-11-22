"""The first pass of the law search on a native kernel, returning what the Python reference returns.

Status: experimental

**Why.** A decision of the main search takes about eight seconds, and about
seven of them are the first pass: ``confirmed_stop.search_confirmed`` fits
every enumerated expression to the rows, one NumPy call at a time, and when
none settles the matter ``exhaustive_search.search`` does it again and adds
sums of two terms. Anything that calls the referee in a loop pays that each
time.

**What this is.** ``search_confirmed`` here has the reference's signature and
returns the reference's ``GuidedResult``. The kernel
(``rust/apedp-kernel/src/first_pass.rs``) fits the whole expression table in
one pass, sharing every sub-expression's column. Its numbers are close to the
reference's and not identical (other transcendental functions, least squares
by orthogonalisation instead of LAPACK), so they are used for one thing only:
to rule candidates *out*. Each comes with a bound on how far the reference's
value can be from it (``_doubt``); every candidate a decision could still
turn on, given that bound, is computed again with the reference's own
functions, and the decision is taken on those values with the reference's own
statements, in the reference's order. What is returned therefore carries the
reference's floats.

**What it rests on.** The bound is an error analysis with a safety factor, not
a proof: if the reference and the kernel disagreed by more than the bound on a
candidate that mattered, the two would return different results. That is what
``tests/test_fast_first_pass.py`` and ``scripts/check_fast_first_pass.py``
look for (the same answer on many datasets), and ``check=True`` runs both on
every call and raises on any difference.

**Nothing frozen is edited.** The reference modules are imported and called.
``enabled()`` makes the main search use this first pass inside a ``with``
block by replacing one name in the ``main_search`` module at run time; the
file on disk, which the preregistration digests cover, is the same. Outside
such a block, and whenever the kernel is not installed or lacks these
functions, everything runs as before.
"""
from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from apedp_dyn.discovery_l2 import confirmed_stop, main_search
from apedp_dyn.discovery_l2 import exhaustive_search as es
from apedp_dyn.discovery_l2.guided_search import (
    GuidedResult,
    GuidedSearch,
    neighbours,
    residual_structure,
)

OPS = {"x": 0, "sqrt": 1, "exp": 2, "log": 3, "sin": 4, "cos": 5, "inv": 6, "sq": 7, "+": 8, "-": 9, "*": 10, "/": 11}
FITTED, NOT_FINITE, RANK_DEFICIENT, NEAR_RANK_THRESHOLD = 0, 1, 2, 3
#: How many times the first-order error estimate a value of the kernel is trusted to be from the reference's.
SAFETY = 1e3
#: Below this many times the floor a residual sum of squares is rounding in both implementations: no bound is claimed.
NEAR_FLOOR = 1e4
KEEP, PAIR_POOL, PARTNER_COMPLEXITY = 40, 40, 3       # the reference's defaults in score_slice and merge
REFERENCE_ATTRIBUTE = "search_confirmed"                  # the name the frozen main search calls its first pass by


class FirstPassMismatchError(AssertionError):
    """``check=True`` found the native first pass and the reference returning different results."""


def kernel() -> Any:
    """The native module if it is installed and has the first-pass functions, else ``None``."""
    try:
        import apedp_kernel
    except ImportError:
        return None
    return apedp_kernel if hasattr(apedp_kernel, "first_pass_singles") and hasattr(apedp_kernel, "first_pass_pairs") else None


@dataclass(frozen=True)
class Table:
    """An enumeration as arrays: operation, children, complexity."""

    ops: np.ndarray
    left: np.ndarray
    right: np.ndarray
    complexity: np.ndarray


_TABLES: dict[int, tuple[Any, Table]] = {}


def table_of(nodes: Sequence[tuple[es.Node, int]]) -> Table:
    """The expression table of an enumeration (cached per enumeration object)."""
    held = _TABLES.get(id(nodes))
    if held is not None and held[0] is nodes:
        return held[1]
    index: dict[es.Node, int] = {}
    ops: np.ndarray = np.empty(len(nodes), dtype=np.uint8)
    left: np.ndarray = np.full(len(nodes), -1, dtype=np.int32)
    right: np.ndarray = np.full(len(nodes), -1, dtype=np.int32)
    for i, (node, _complexity) in enumerate(nodes):
        ops[i] = OPS[node[0]]
        if node[0] == "x":
            left[i] = node[1]
        else:
            left[i] = index[node[1]]
            if len(node) == 3:
                right[i] = index[node[2]]
        index[node] = i
    table = Table(ops, left, right, np.array([c for _node, c in nodes], dtype=np.int64))
    _TABLES[id(nodes)] = (nodes, table)
    return table


def _doubt(rss: np.ndarray, condition: np.ndarray, total: float, floor: float) -> np.ndarray:
    """A bound on the relative difference between the reference's residual sum of squares and the kernel's.

    To first order both differ from the exact least-squares value by rounding times the
    conditioning of the fit times how small the residual is against what was fitted. Near the
    floor, or where that estimate is not small, no bound is claimed (infinity).
    """
    with np.errstate(all="ignore"):
        estimate = SAFETY * np.finfo(float).eps * condition * (1.0 + np.sqrt(total / rss))
    out: np.ndarray = np.where((rss >= NEAR_FLOOR * floor) & (estimate < 0.25) & np.isfinite(estimate), estimate, np.inf)
    return out


def _lengths(rss: np.ndarray, n: int, structure_nats: np.ndarray, params: int) -> np.ndarray:
    with np.errstate(all="ignore"):
        out: np.ndarray = 0.5 * n * np.log(np.maximum(rss, 1e-300) / n) + structure_nats + 0.5 * params * math.log(n)
    return out


def _full_search(native: Any, nodes: Sequence[tuple[es.Node, int]], nats: np.ndarray, table: Table, X: np.ndarray, y: np.ndarray,
                 names: Sequence[str], alphabet: int, rss: np.ndarray, condition: np.ndarray, status: np.ndarray,
                 floor: float, total: float) -> es.Scored:
    """``exhaustive_search.search(...)[0]``: the best single term or sum of two, by the reference's ordering."""
    n = len(y)
    fitted = status == FITTED
    doubt = _doubt(rss, condition, total, floor)
    length = _lengths(rss, n, nats, 2)
    slack = np.where(np.isfinite(doubt), 0.5 * n * doubt, np.inf)
    # The reference keeps the KEEP best by (description length, index). A candidate whose best case is
    # worse than the KEEP-th best worst case cannot be among them.
    upper = np.sort((length + slack)[fitted])
    cut = upper[min(KEEP, len(upper)) - 1] if len(upper) else np.inf
    wanted = np.nonzero((fitted & (length - slack <= cut)) | (status == NEAR_RANK_THRESHOLD))[0]
    rows: list[dict[str, Any]] = []
    for i in wanted:
        node, c = nodes[int(i)]
        fit = es._fit([es.evaluate(node, X)], y)
        if fit is None:
            continue
        coef, exact = fit
        rows.append({"index": int(i), "expression": es.to_string(node, names), "complexity": c, "rss": exact,
                     "coef": [float(v) for v in coef], "structure_nats": float(nats[int(i)]),
                     "description_length": es.description_length(exact, n, c, 2, alphabet, float(nats[int(i)]))})
    rows.sort(key=lambda r: (r["description_length"], r["index"]))
    singles = rows[:KEEP]
    ranked = [es.Scored(r["expression"], (r["expression"],), r["complexity"], 2, r["rss"], r["description_length"],
                        tuple(r["coef"])) for r in singles]
    best: es.Scored | None = ranked[0] if ranked else None
    partners = [i for i, (_node, c) in enumerate(nodes) if c <= PARTNER_COMPLEXITY]
    firsts = singles[:PAIR_POOL]
    if not firsts or not partners:
        if best is None:
            raise ValueError("no candidate could be fitted")
        return best
    with np.errstate(all="ignore"):
        first_values = [es.evaluate(nodes[r["index"]][0], X) for r in firsts]
        partner_values = [es.evaluate(nodes[i][0], X) for i in partners]
        pair_rss, pair_condition, pair_status = native.first_pass_pairs(
            np.ascontiguousarray(np.array(first_values, dtype=float)), np.ascontiguousarray(np.array(partner_values, dtype=float)),
            np.ascontiguousarray(y), floor)
    first_nats = np.array([r["structure_nats"] for r in firsts])
    first_c = np.array([r["complexity"] for r in firsts])
    partner_nats, partner_c = nats[partners], table.complexity[partners]
    pair_nats = first_nats[:, None] + partner_nats[None, :] + math.log(alphabet)
    pair_length = _lengths(pair_rss, n, pair_nats, 3)
    pair_doubt = _doubt(pair_rss, pair_condition, total, floor)
    pair_slack = np.where(np.isfinite(pair_doubt), 0.5 * n * pair_doubt, np.inf)
    same = np.array([[r["index"] == i for i in partners] for r in firsts])       # the reference skips a term paired with itself
    pair_fitted = (pair_status == FITTED) & ~same
    limit = float(np.min((pair_length + pair_slack)[pair_fitted])) if np.any(pair_fitted) else np.inf
    if best is not None:
        limit = min(limit, best.description_length)
    contenders = np.argwhere((pair_fitted & (pair_length - pair_slack <= limit)) | ((pair_status == NEAR_RANK_THRESHOLD) & ~same))
    for f, p in contenders:                                # row-major: the order the reference appends its pairs in
        first, other = firsts[int(f)], partners[int(p)]
        fit = es._fit([first_values[int(f)], partner_values[int(p)]], y)
        if fit is None:
            continue
        coef, exact = fit
        c = int(first_c[int(f)] + partner_c[int(p)] + 1)
        nats_here = first["structure_nats"] + float(nats[other]) + math.log(alphabet)
        scored = es.Scored(f"({first['expression']}) + ({es.to_string(nodes[other][0], names)})",
                           (first["expression"], es.to_string(nodes[other][0], names)), c, 3, exact,
                           es.description_length(exact, n, c, 3, alphabet, nats_here), tuple(float(v) for v in coef))
        if best is None or scored.description_length < best.description_length:    # a stable sort keeps the earlier of equals
            best = scored
    if best is None:
        raise ValueError("no candidate could be fitted")
    return best


def search_confirmed(guided: GuidedSearch, X: np.ndarray, y: np.ndarray, names: Sequence[str], max_complexity: int, *,
                     patience: float = 3.0, floor: int = 300, check: bool = False) -> GuidedResult:
    """``confirmed_stop.search_confirmed`` on the native kernel; the reference itself where the kernel cannot be used.

    ``check=True`` also runs the reference and raises ``FirstPassMismatchError`` if the two differ.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    native = kernel()
    usable = native is not None and X.ndim == 2 and len(y) >= 4 and bool(np.all(np.isfinite(X)) and np.all(np.isfinite(y)))
    if not usable:
        return confirmed_stop.search_confirmed(guided, X, y, names, max_complexity, patience=patience, floor=floor)
    result = _search_confirmed(native, guided, X, y, names, max_complexity, patience, floor)
    if check:
        reference = confirmed_stop.search_confirmed(guided, X, y, names, max_complexity, patience=patience, floor=floor)
        if result != reference:
            raise FirstPassMismatchError(f"native {result!r}\nreference {reference!r}")
    return result


def _search_confirmed(native: Any, guided: GuidedSearch, X: np.ndarray, y: np.ndarray, names: Sequence[str],
                      max_complexity: int, patience: float, floor: int) -> GuidedResult:
    nodes, nats, order = guided.order(X.shape[1], max_complexity)
    table = table_of(nodes)
    alphabet = X.shape[1] + len(es.UNARY) + len(es.BINARY)
    n = len(y)
    nn = neighbours(X, min(guided.k, n - 1))
    rss_floor = n * (1e-10 * float(np.sqrt(np.mean(y ** 2)))) ** 2
    total = float(np.sum((y - y.mean()) ** 2))
    rss, _slope, _intercept, structure, condition, status = native.first_pass_singles(
        table.ops, table.left, table.right, np.ascontiguousarray(X), np.ascontiguousarray(y),
        np.ascontiguousarray(nn, dtype=np.uint32), rss_floor)
    doubt = _doubt(rss, condition, total, rss_floor)
    length = _lengths(rss, n, nats, 2)
    slack = np.where(np.isfinite(doubt), 0.5 * n * doubt, np.inf)
    # A candidate can only become the best if its residuals show no structure. Those that might:
    with np.errstate(all="ignore"):
        might = ((status == FITTED) & ~(structure > 10.0 * doubt)) | (status == NEAR_RANK_THRESHOLD)
    best: es.Scored | None = None
    deadline: float | None = None
    for position in np.nonzero(might[order])[0]:
        count = int(position) + 1
        if deadline is not None and count > deadline:
            break
        index = int(order[position])
        if best is not None and length[index] - slack[index] >= best.description_length:
            continue                                        # the reference's fit could not beat the best either
        node, c = nodes[index]
        column = es.evaluate(node, X)
        fit = es._fit([column], y)
        if fit is None:
            continue
        coef, exact = fit
        exact_length = es.description_length(exact, n, c, 2, alphabet, float(nats[index]))
        if best is not None and exact_length >= best.description_length:
            continue
        if residual_structure(y - (coef[0] * column + coef[1]), nn) > 0.0:
            continue
        expression = es.to_string(node, names)
        best = es.Scored(expression, (expression,), c, 2, exact, exact_length, tuple(float(v) for v in coef))
        deadline = max(deadline or 0.0, patience * count, count + floor)
    if deadline is not None and len(order) > deadline:      # the reference stops at the first count past the deadline
        assert best is not None
        return GuidedResult(best, math.floor(deadline), len(nodes), True, 0.0)
    if best is not None:
        return GuidedResult(best, len(nodes), len(nodes), False, 0.0)
    full = _full_search(native, nodes, nats, table, X, y, names, alphabet, rss, condition, status, rss_floor, total)
    return GuidedResult(full, len(nodes), len(nodes), False, 1.0)


@contextmanager
def enabled(check: bool = False) -> Iterator[None]:
    """Inside the block the main search's first pass is this one (the reference if the kernel is absent)."""

    def first_pass(guided: GuidedSearch, X: np.ndarray, y: np.ndarray, names: Sequence[str], max_complexity: int,
                   **options: Any) -> GuidedResult:
        return search_confirmed(guided, X, y, names, max_complexity, check=check, **options)

    original = getattr(main_search, REFERENCE_ATTRIBUTE)
    setattr(main_search, REFERENCE_ATTRIBUTE, first_pass)
    try:
        yield
    finally:
        setattr(main_search, REFERENCE_ATTRIBUTE, original)
