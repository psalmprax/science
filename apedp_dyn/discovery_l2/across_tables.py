"""Learning across tables: what the tables of one family can tell each other.

Status: experimental

**Not in force.** Nothing calls it but ``family_route.py``, its study script and its tests. Putting
any of it in force needs a blind test and the owner's word (AGENTS.md 1.5). It is the redesign of
the learning loop proposed in ``docs/SOLUTIONS_FOR_NO_GAIN_ITEMS_2026-11-27.md`` (section 2) and
built on the owner's word of 27 November 2026.

**Why.** The learning loop (``learning_loop.py``) learns the parts and the text of each confirmed
law and offers them to every later table. Measured, that added nothing on laws drawn one by one
from a grammar, and its one near-gain was on ODE systems, whose components share parts
(``benchmarks/learning_stream/dev_families.json``). What it never used is the knowledge that
transfers exactly: tables that describe *one* phenomenon under different conditions share one
shape, and differ only in its constants.

**A family** (``Family``) is a set of tables with the same inputs, said to belong together: one
experiment at several temperatures, the runs of one rig, or one wide table in which an input takes
only a few values, whose rows are then grouped by that value (``by_levels``). Each member may carry
*conditions*: numbers that are fixed inside the member and differ between members. A family is
declared, or read from repeated values. None is guessed from likeness.

**What a family gives.**

1. *Shapes for its members* (``sibling_laws``). A law has a shape: the law with each fitted constant
   made a free parameter (``shape_of``). Every law the route listed for any member is refitted on
   every member, with constants of its own for each, and the shapes are ordered by what all the rows
   of the family say (the sum of the members' BIC), shapes the referee reported on a member first. A
   member is then handed the best shapes its own list did not hold, refitted on its own rows. The
   shape is decided by all rows while each table keeps its constants.
2. *A wider law* (``wider_laws``). For a shape fitted on every member, each constant set against
   the members' condition is a small table. A constant that does not move beyond its own
   uncertainty is one number; one that does is given a few short laws of the condition
   (``laws_of_a_constant``). Put into the shape, they make one law over the inputs *and* the
   condition. That law is **not** judged on the handful of members: it is handed to the route on the
   pooled table (``pooled``), all rows of all members with the condition as one more input, and the
   referee decides there as on any table. A law of three inputs is found as a law of two and a law
   of one, and never searched as three.

**What it never does.** It proposes (AGENTS.md 1.1). It reads development rows only: no pool row and
no measurement. Every law it hands over goes through the route in force, a member's law on that
member's rows and a wider law on the family's rows.

**Limits, known before any test.** One condition only for the wider law. A shape whose constants
cannot be told apart on a member (a redundant pair) gives no wider law. The short laws of a
constant are a fixed list of eight forms; a constant that follows another form is at best
approximated, and the referee sees the approximation on all rows. Fitting is local least squares
from a few starts, so a shape can be scored worse than it deserves. A family that is declared
wrongly (tables that share no shape) gets nothing from its siblings but costs their refits.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

MAX_LEVELS = 12        #: an input with at most this many values in a table can be read as a condition
MIN_ROWS = 30          #: rows a value needs, among the rows to fit on and among the pool, to be a member of its own
MAX_SHAPES = 30        #: shapes scored across a family
HANDED = 6             #: laws a member is handed from its siblings
STARTS = 3             #: starts of a refit beside the ones it is given
MAX_CONSTANTS = 10     #: a law with more fitted constants than this is not read as a shape
MIN_MEMBERS = 4        #: members a family needs before its constants are read as a table
VARIES_SIGMAS = 3.0    #: a constant moves if it leaves its weighted mean by more than this many of its standard errors
FORMS_KEPT = 2         #: short laws kept for a constant that moves
WIDER_LAWS = 8         #: wider laws handed to the route for one family
ERROR_FLOOR = 1.0e-8   #: a standard error is not taken smaller than this share of the constant's size
_C = sp.Symbol("C")
_BAD = 1.0e6


@dataclass
class Table:
    """One table as the route reads it: rows to fit on, a pool the referee may ask to have measured, and its conditions."""

    key: str
    names: list[str]
    X: np.ndarray
    y: np.ndarray
    pool_X: np.ndarray
    measure: Callable[[np.ndarray], np.ndarray]
    conditions: dict[str, float] = field(default_factory=dict)

    def columns(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(self.X[:, i], dtype=float) for i, name in enumerate(self.names)}


@dataclass
class Family:
    """Tables with the same inputs that are said to share one shape."""

    key: str
    tables: list[Table]

    def __post_init__(self) -> None:
        if len(self.tables) < 2:
            raise ValueError("a family has at least two tables")
        first = self.tables[0]
        for table in self.tables[1:]:
            if table.names != first.names:
                raise ValueError(f"the tables of a family have the same inputs in the same order: {table.key} differs")
            if set(table.conditions) != set(first.conditions):
                raise ValueError(f"the tables of a family carry the same conditions: {table.key} differs")
        if set(first.conditions) & set(first.names):
            raise ValueError("a condition cannot have the name of an input")

    @property
    def names(self) -> list[str]:
        return list(self.tables[0].names)

    @property
    def conditions(self) -> list[str]:
        return sorted(self.tables[0].conditions)


def by_levels(table: Table, name: str | None = None, *, max_levels: int = MAX_LEVELS, min_rows: int = MIN_ROWS) -> Family | None:
    """A table in which input ``name`` takes few values, as a family: one member a value, that input its condition.

    Without ``name``, the input with the fewest values that qualifies. ``None`` if no input does: every value needs
    ``min_rows`` rows to fit on and as many in the pool, and there must be at least two values and two inputs.
    """
    if len(table.names) < 2:
        return None
    best: tuple[int, int] | None = None
    for j, candidate in enumerate(table.names):
        if name is not None and candidate != name:
            continue
        values = np.unique(table.X[:, j])
        if not 2 <= len(values) <= max_levels:
            continue
        if all(int(np.sum(table.X[:, j] == v)) >= min_rows and int(np.sum(table.pool_X[:, j] == v)) >= min_rows for v in values):
            if best is None or len(values) < best[0]:
                best = (len(values), j)
    if best is None:
        return None
    j = best[1]
    rest = [i for i in range(len(table.names)) if i != j]
    members = []
    for value in np.unique(table.X[:, j]):
        rows = table.X[:, j] == value
        pool_rows = np.flatnonzero(table.pool_X[:, j] == value)

        def measure(asked: np.ndarray, pool_rows: np.ndarray = pool_rows) -> np.ndarray:
            measured: np.ndarray = np.asarray(table.measure(pool_rows[np.asarray(asked, dtype=int)]), dtype=float)
            return measured

        members.append(Table(f"{table.key}|{table.names[j]}={float(value):g}", [table.names[i] for i in rest],
                             table.X[rows][:, rest], table.y[rows], table.pool_X[pool_rows][:, rest], measure,
                             {table.names[j]: float(value)}))
    return Family(table.key, members)


# ----------------------------------------------------------------------------------------------- shapes

@dataclass(frozen=True)
class Shape:
    """A law with each of its fitted constants made a parameter."""

    text: str                      #: in the inputs and the parameters ``_k0, _k1, ...``
    constants: int
    start: tuple[float, ...]       #: the constants of the law it was read from
    key: str                       #: the law with every constant replaced by ``C``: two laws of one shape have one key

    @property
    def params(self) -> list[sp.Symbol]:
        return [sp.Symbol(f"_k{i}") for i in range(self.constants)]

    def expr(self, names: Sequence[str]) -> sp.Expr:
        return sp.sympify(self.text, locals={**{n: sp.Symbol(n) for n in names}, **{str(p): p for p in self.params}})

    def law(self, names: Sequence[str], theta: Sequence[float]) -> str:
        """The shape with ``theta`` for its constants, as a law the route can read."""
        values = {p: sp.Float(float(f"{float(v):.10g}")) for p, v in zip(self.params, theta, strict=True)}
        return str(self.expr(names).xreplace(values))


def shape_of(law: str | sp.Expr, names: Sequence[str], max_constants: int = MAX_CONSTANTS) -> Shape | None:
    """The shape of a law: every fitted constant (a float; whole numbers and fractions stay) a parameter of its own.

    ``None`` for a text that is not a law in ``names``, has no input in it, or has more than ``max_constants`` constants.
    """
    symbols = {n: sp.Symbol(n) for n in names}
    try:
        expr = sp.sympify(str(law), locals=symbols)
    except (sp.SympifyError, TypeError, SyntaxError, AttributeError):
        return None
    if not isinstance(expr, sp.Expr) or not expr.free_symbols or not expr.free_symbols <= set(symbols.values()):
        return None
    start: list[float] = []

    def walk(node: sp.Expr) -> sp.Expr:
        if node.is_Float:
            start.append(float(node))
            return sp.Symbol(f"_k{len(start) - 1}")
        if not node.args:
            return node
        return node.func(*[walk(arg) for arg in node.args])

    parametrised = walk(expr)
    if len(start) > max_constants or not all(math.isfinite(v) for v in start):
        return None
    key = str(parametrised.xreplace({sp.Symbol(f"_k{i}"): _C for i in range(len(start))}))
    return Shape(str(parametrised), len(start), tuple(start), key)


@lru_cache(maxsize=512)
def _compiled(text: str, names: tuple[str, ...], constants: int) -> Any:
    params = [sp.Symbol(f"_k{i}") for i in range(constants)]
    expr = sp.sympify(text, locals={**{n: sp.Symbol(n) for n in names}, **{str(p): p for p in params}})
    return sp.lambdify([*(sp.Symbol(n) for n in names), *params], expr, "numpy")


@dataclass(frozen=True)
class Fit:
    """A shape's constants on one table and what they leave."""

    theta: tuple[float, ...]
    rss: float
    rows: int
    relative: float                #: RMS of what is left over the RMS of the target

    def bic(self) -> float:
        return self.rows * math.log(max(self.rss / self.rows, 1.0e-300)) + len(self.theta) * math.log(self.rows)


def _residual(shape: Shape, names: Sequence[str], X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    fn = _compiled(shape.text, tuple(names), shape.constants)
    size = float(np.sqrt(np.mean(y ** 2))) or 1.0

    def residual(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            values = np.asarray(fn(*X.T, *theta), dtype=float) * np.ones(len(y))
        out: np.ndarray = np.where(np.isfinite(values), (values - y) / size, _BAD)
        return out

    return residual


def refit(shape: Shape, names: Sequence[str], X: np.ndarray, y: np.ndarray, starts: Sequence[Sequence[float]] = (),
          *, seed: int = 0, extra: int = STARTS) -> Fit | None:
    """The shape's constants fitted to a table by least squares: from ``starts``, from the law it was read from, and from
    ``extra`` scattered copies of the best of those. ``None`` if no start gives a finite fit."""
    y = np.asarray(y, dtype=float)
    residual = _residual(shape, names, np.asarray(X, dtype=float), y)
    size = float(np.sqrt(np.mean(y ** 2))) or 1.0
    if shape.constants == 0:
        left = residual(np.array([]))
        return None if float(np.max(np.abs(left))) >= _BAD else Fit((), float(np.sum((left * size) ** 2)), len(y),
                                                                   float(np.sqrt(np.mean(left ** 2))))
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray] | None = None

    def attempt(start: np.ndarray) -> None:
        nonlocal best
        try:
            fit = least_squares(residual, start, max_nfev=200)
        except (ValueError, TypeError, FloatingPointError):
            return
        cost = float(np.sum(fit.fun ** 2))
        if np.all(np.isfinite(fit.x)) and math.isfinite(cost) and float(np.max(np.abs(fit.fun))) < _BAD and (
                best is None or cost < best[0]):
            best = (cost, np.asarray(fit.x, dtype=float))

    given: list[np.ndarray] = [np.asarray(s, dtype=float) for s in (*starts, shape.start) if len(s) == shape.constants]
    for start in given:
        attempt(start)
    centre = best[1] if best is not None else np.asarray(shape.start, dtype=float)
    for _ in range(extra):
        attempt(centre * np.exp(rng.normal(0.0, 0.5, shape.constants)))
    if best is None:
        return None
    cost, theta = best
    return Fit(tuple(float(v) for v in theta), cost * size ** 2, len(y), float(np.sqrt(cost / len(y))))


def _jacobian(shape: Shape, names: Sequence[str], X: np.ndarray, y: np.ndarray, theta: Sequence[float]) -> np.ndarray:
    residual = _residual(shape, names, np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    point: np.ndarray = np.asarray(theta, dtype=float)
    base = residual(point)
    out = np.empty((len(base), len(point)))
    for j in range(len(point)):
        step = 1.0e-6 * max(1.0, abs(float(point[j])))
        moved = point.copy()
        moved[j] += step
        out[:, j] = (residual(moved) - base) / step
    return out


def standard_errors(shape: Shape, names: Sequence[str], X: np.ndarray, y: np.ndarray, theta: Sequence[float]) -> np.ndarray | None:
    """One standard error for each constant at ``theta``, from the fit's own curvature; ``None`` where the constants
    cannot be told apart on these rows (the curvature has no inverse worth the name) or there are too few rows."""
    m = len(theta)
    if m == 0 or len(y) <= m:
        return None
    base = _residual(shape, names, np.asarray(X, dtype=float), np.asarray(y, dtype=float))(np.asarray(theta, dtype=float))
    jacobian = _jacobian(shape, names, X, y, theta)
    if not np.all(np.isfinite(jacobian)):
        return None
    scale = np.sqrt(np.sum(jacobian ** 2, axis=0))
    if np.any(scale == 0.0):
        return None
    singular = np.linalg.svd(jacobian / scale, compute_uv=False)
    if singular[-1] < 1.0e-6 * singular[0]:
        return None
    covariance = np.linalg.inv(jacobian.T @ jacobian) * float(np.sum(base ** 2)) / (len(base) - m)
    out: np.ndarray = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return out


# ----------------------------------------------------------------------------------- shapes for the members

@dataclass
class Scored:
    """A shape fitted on every member of a family."""

    shape: Shape
    fits: list[Fit | None]
    from_members: set[int]         #: the members whose own lists held it
    reported_on: set[int]          #: the members on which the referee reported it

    @property
    def score(self) -> float:
        """The sum of the members' BIC: what all rows of the family say. Infinite if it cannot be fitted on one."""
        return math.inf if any(f is None for f in self.fits) else float(sum(f.bic() for f in self.fits if f is not None))


def score_shapes(family: Family, listed: Sequence[Sequence[str]], reported: Sequence[str | None], *, seed: int = 0,
                 max_shapes: int = MAX_SHAPES) -> list[Scored]:
    """Every shape the route listed for any member, fitted on every member, best first.

    ``listed[i]`` are member ``i``'s laws in the route's order and ``reported[i]`` the law the referee reported on it,
    if any. Shapes reported on a member come first (more members first), then the rest; inside each group by the sum
    of the members' BIC. At most ``max_shapes`` are scored: every reported shape, then each member's first ones in turn.
    """
    names = family.names
    wanted: dict[str, Scored] = {}

    def take(law: str, member: int, is_reported: bool) -> None:
        shape = shape_of(law, names)
        if shape is None:
            return
        if shape.key not in wanted:
            if len(wanted) >= max_shapes and not is_reported:
                return
            wanted[shape.key] = Scored(shape, [], set(), set())
        wanted[shape.key].from_members.add(member)
        if is_reported:
            wanted[shape.key].reported_on.add(member)

    for member, law in enumerate(reported):
        if law:
            take(law, member, True)
    longest = max((len(laws) for laws in listed), default=0)
    for place in range(longest):
        for member, laws in enumerate(listed):
            if place < len(laws):
                take(laws[place], member, False)
    for item in wanted.values():
        previous: tuple[float, ...] | None = None
        for index, table in enumerate(family.tables):
            fit = refit(item.shape, names, table.X, table.y, [previous] if previous is not None else [], seed=seed + index)
            item.fits.append(fit)
            previous = fit.theta if fit is not None else previous
    return sorted(wanted.values(), key=lambda s: (0 if s.reported_on else 1, -len(s.reported_on), s.score, s.shape.key))


def sibling_laws(family: Family, scored: Sequence[Scored], member: int, handed: int = HANDED) -> list[str]:
    """What member ``member`` is handed: the best shapes its own list did not hold, refitted on its own rows."""
    out: list[str] = []
    for item in scored:
        if len(out) >= handed:
            break
        fit = item.fits[member] if member < len(item.fits) else None
        if fit is None or member in item.from_members:
            continue                      # it cannot be fitted here, or the member's own list already holds the shape
        out.append(item.shape.law(family.names, fit.theta))
    return out


# ------------------------------------------------------------------------------------------- a wider law

#: Short laws of one condition ``c`` for a constant that moves: name, the law in ``c`` and two numbers ``a``, ``b``,
#: and whether it needs ``c > 0``. Fitted by least squares weighted by the constant's standard errors.
_FORMS: tuple[tuple[str, Callable[[Any, Any, Any], Any], bool], ...] = (
    ("a + b*c", lambda c, a, b: a + b * c, False),
    ("a + b/c", lambda c, a, b: a + b / c, True),
    ("a + b*c**2", lambda c, a, b: a + b * c ** 2, False),
    ("a + b*sqrt(c)", lambda c, a, b: a + b * sp.sqrt(c) if isinstance(c, sp.Basic) else a + b * np.sqrt(c), True),
    ("a + b*log(c)", lambda c, a, b: a + b * sp.log(c) if isinstance(c, sp.Basic) else a + b * np.log(c), True),
    ("a*c**b", lambda c, a, b: a * c ** b, True),
    ("a*exp(b*c)", lambda c, a, b: a * sp.exp(b * c) if isinstance(c, sp.Basic) else a * np.exp(b * c), False),
)


def _floored(values: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """Standard errors no smaller than rounding: rows without noise fit exactly and leave an error of zero."""
    out: np.ndarray = np.maximum(np.asarray(errors, dtype=float), ERROR_FLOOR * max(1.0, float(np.max(np.abs(values)))))
    return out


def moves(values: np.ndarray, errors: np.ndarray, sigmas: float = VARIES_SIGMAS) -> bool:
    """Whether a constant differs between members by more than its own uncertainty allows."""
    values = np.asarray(values, dtype=float)
    errors = _floored(values, errors)
    weights = 1.0 / errors ** 2
    mean = float(np.sum(weights * values) / np.sum(weights))
    return bool(np.any(np.abs(values - mean) > sigmas * errors))


def laws_of_a_constant(values: np.ndarray, errors: np.ndarray, condition: np.ndarray, symbol: sp.Symbol,
                       kept: int = FORMS_KEPT) -> list[tuple[sp.Expr, float]]:
    """Short laws of the condition for one constant across the members, best first, with what each leaves (chi-squared).

    One number if the constant does not move. Otherwise the ``kept`` best of the eight forms. A handful of members
    cannot decide between forms, and this does not try to: what it returns is put into the shape and judged on all rows.
    """
    values, c = np.asarray(values, dtype=float), np.asarray(condition, dtype=float)
    errors = _floored(values, errors)
    weights = 1.0 / errors
    mean = float(np.sum(weights ** 2 * values) / np.sum(weights ** 2))
    if not moves(values, errors):
        return [(sp.Float(mean), float(np.sum((weights * (values - mean)) ** 2)))]
    out: list[tuple[sp.Expr, float]] = []
    positive = bool(np.all(c > 0.0))
    for _name, form, needs_positive in _FORMS:
        if needs_positive and not positive:
            continue

        def left(theta: np.ndarray, form: Callable[[Any, Any, Any], Any] = form) -> np.ndarray:
            with np.errstate(all="ignore"):
                predicted = np.asarray(form(c, theta[0], theta[1]), dtype=float) * np.ones(len(c))
            out_: np.ndarray = np.where(np.isfinite(predicted), weights * (predicted - values), _BAD)
            return out_

        best: tuple[float, np.ndarray] | None = None
        for start in ((mean, 0.0), (mean, 1.0), (mean, -1.0), (1.0, 1.0), (float(values[0]), 0.1), (float(values[0]), -0.1)):
            try:
                fit = least_squares(left, np.array(start, dtype=float), max_nfev=400)
            except (ValueError, TypeError, FloatingPointError):
                continue
            cost = float(np.sum(fit.fun ** 2))
            if np.all(np.isfinite(fit.x)) and math.isfinite(cost) and (best is None or cost < best[0]):
                best = (cost, fit.x)
        if best is None or float(np.max(np.abs(left(best[1])))) >= _BAD:
            continue
        a, b = (sp.Float(float(f"{float(v):.10g}")) for v in best[1])
        out.append((form(symbol, a, b), best[0]))
    out.sort(key=lambda pair: pair[1])
    return out[:kept]


@dataclass
class Constants:
    """A shape's constants on every member, with their standard errors, against one condition."""

    shape: Shape
    condition: str
    at: np.ndarray                 #: the condition's value for each member
    values: np.ndarray             #: members x constants
    errors: np.ndarray
    moving: list[bool]

    def as_record(self) -> dict[str, Any]:
        return {"shape": self.shape.key, "condition": self.condition, "at": [float(v) for v in self.at],
                "values": [[float(v) for v in row] for row in self.values],
                "standard_errors": [[float(v) for v in row] for row in self.errors], "moving": list(self.moving)}


def told_apart(shape: Shape, names: Sequence[str], X: np.ndarray, y: np.ndarray, theta: Sequence[float]) -> Shape:
    """The shape with the constants these rows cannot tell apart held at ``theta``: what is left can be given a table.

    A search writes ``a*(b*x + c)``: three constants for two numbers. Which ones to hold is read from the fit's
    curvature (a QR decomposition with pivoting: the columns it takes last add nothing the others do not). Holding
    them loses nothing, since the same law is still in the shape. The shape itself if every constant counts.
    """
    m = shape.constants
    if m == 0:
        return shape
    jacobian = _jacobian(shape, names, X, y, theta)
    if not np.all(np.isfinite(jacobian)):
        return shape
    scale = np.sqrt(np.sum(jacobian ** 2, axis=0))
    scale[scale == 0.0] = 1.0
    from scipy.linalg import qr

    _, r, pivots = qr(jacobian / scale, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    kept = sorted(int(p) for p, size in zip(pivots, diagonal, strict=True) if size > 1.0e-6 * diagonal[0])
    if len(kept) == m:
        return shape
    params = shape.params
    held = {params[j]: sp.Float(float(f"{float(theta[j]):.10g}")) for j in range(m) if j not in kept}
    renamed = {params[j]: sp.Symbol(f"_k{new}") for new, j in enumerate(kept)}
    expr = shape.expr(names).xreplace(held).xreplace(renamed)
    return Shape(str(expr), len(kept), tuple(float(theta[j]) for j in kept), shape.key)


def constants_of(family: Family, shape: Shape, condition: str, *, seed: int = 0) -> Constants | None:
    """The shape fitted along the condition, each member started from its neighbour so the constants stay on one branch.

    Constants the first member cannot tell apart are held at its values for every member (``told_apart``). ``None`` if
    a member cannot be fitted, or still cannot tell the remaining constants apart.
    """
    names = family.names
    order = sorted(range(len(family.tables)), key=lambda i: family.tables[i].conditions[condition])
    first = family.tables[order[0]]
    opening = refit(shape, names, first.X, first.y, seed=seed)
    if opening is None:
        return None
    shape = told_apart(shape, names, first.X, first.y, opening.theta)
    if shape.constants == 0:
        return None
    values = np.empty((len(order), shape.constants))
    errors = np.empty_like(values)
    previous: tuple[float, ...] | None = None
    for row, index in enumerate(order):
        table = family.tables[index]
        fit = refit(shape, names, table.X, table.y, [previous] if previous is not None else [], seed=seed + index)
        if fit is None:
            return None
        spread = standard_errors(shape, names, table.X, table.y, fit.theta)
        if spread is None:
            return None
        values[row], errors[row], previous = fit.theta, spread, fit.theta
    at = np.array([family.tables[i].conditions[condition] for i in order], dtype=float)
    return Constants(shape, condition, at, values, errors, [moves(values[:, j], errors[:, j]) for j in range(shape.constants)])


def pooled(family: Family, condition: str) -> Table:
    """All rows of all members as one table, the condition one more input; the pool likewise, each row measured by its member."""
    if condition in family.names:
        raise ValueError("the condition already is an input")
    blocks, pools, targets, sizes = [], [], [], []
    for table in family.tables:
        value = float(table.conditions[condition])
        blocks.append(np.column_stack([table.X, np.full(len(table.X), value)]))
        pools.append(np.column_stack([table.pool_X, np.full(len(table.pool_X), value)]))
        targets.append(np.asarray(table.y, dtype=float))
        sizes.append(len(table.pool_X))
    offsets = np.concatenate([[0], np.cumsum(sizes)])

    def measure(asked: np.ndarray) -> np.ndarray:
        asked = np.asarray(asked, dtype=int)
        out = np.empty(len(asked))
        member = np.searchsorted(offsets, asked, side="right") - 1
        for index in np.unique(member):
            mine = member == index
            out[mine] = np.asarray(family.tables[int(index)].measure(asked[mine] - offsets[int(index)]), dtype=float)
        return out

    return Table(f"{family.key}|pooled over {condition}", [*family.names, condition], np.vstack(blocks),
                 np.concatenate(targets), np.vstack(pools), measure)


def wider_laws(family: Family, scored: Sequence[Scored], condition: str, *, seed: int = 0, shapes: int = 3,
               limit: int = WIDER_LAWS) -> tuple[list[str], list[dict[str, Any]]]:
    """Laws over the inputs and the condition, for the route to judge on the pooled table; and how each was made.

    For each of the first ``shapes`` shapes that fit every member: its constants along the condition, a short law (or
    one number) for each, the shape with those put in, and every constant of the result refitted on all rows of the
    family. Best fit on those rows first.
    """
    if len(family.tables) < MIN_MEMBERS or len({t.conditions[condition] for t in family.tables}) < MIN_MEMBERS:
        return [], [{"why_not": f"fewer than {MIN_MEMBERS} members with different values of {condition}"}]
    whole = pooled(family, condition)
    symbol = sp.Symbol(condition)
    found: list[tuple[float, str, dict[str, Any]]] = []
    seen: set[str] = set()
    tried = 0
    for item in scored:
        if tried >= shapes:
            break
        if not math.isfinite(item.score) or item.shape.constants == 0:
            continue
        tried += 1
        table = constants_of(family, item.shape, condition, seed=seed)
        if table is None:
            continue
        fitted = table.shape                       # the shape with what cannot be told apart held
        choices = [laws_of_a_constant(table.values[:, j], table.errors[:, j], table.at, symbol) for j in range(fitted.constants)]
        if any(not options for options in choices):
            continue
        combos: list[tuple[float, list[sp.Expr]]] = [(0.0, [])]
        for options in choices:
            combos = sorted(((cost + extra, [*laws, law]) for cost, laws in combos for law, extra in options),
                            key=lambda pair: pair[0])[:limit]
        base = fitted.expr(family.names)
        for _cost, laws in combos:
            wide = base.xreplace(dict(zip(fitted.params, laws, strict=True)))
            wide_shape = shape_of(str(wide), whole.names)
            if wide_shape is None or wide_shape.key in seen:
                continue
            seen.add(wide_shape.key)
            fit = refit(wide_shape, whole.names, whole.X, whole.y, seed=seed, extra=1)
            if fit is None:
                continue
            law = wide_shape.law(whole.names, fit.theta)
            found.append((fit.bic(), law, {"law": law, "members_shape": item.shape.key, "relative_error": round(fit.relative, 6),
                                           "constants": table.as_record()}))
    found.sort(key=lambda entry: entry[0])
    return [law for _, law, _ in found[:limit]], [how for _, _, how in found[:limit]]


def conditions_table(family: Family) -> Mapping[str, list[float]]:
    """The members' conditions, one list a condition, in the members' order."""
    return {name: [float(t.conditions[name]) for t in family.tables] for name in family.conditions}
