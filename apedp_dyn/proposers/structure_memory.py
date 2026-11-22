"""Memory of confirmed laws as a source of candidates.

Status: experimental

**Why.** Every search here starts each problem knowing nothing of the last one.
The learned prior (``apedp_mmlm``) changes the *order* in which an enumeration
is tried; it does not put a law's parts on the table. A law the referee has
confirmed is knowledge of another kind: its parts are forms that occur in this
domain, and the next law of the domain is likely built from some of them.

**What is remembered.** From each confirmed law, its *motifs*: the parts that
depend on one variable only, with the variable and every constant taken out
(``2.3*x0/(x0 + 1.7) - 0.7*exp(-0.45*x1)`` gives ``v/(C + v)`` and
``exp(C*v)``). A motif is a shape, not a law: no constant value is kept, so
nothing is carried over that the new data would not have to fit again. The
whole law's shape is kept too, for a problem that is the same law in other
variables.

**What is proposed.** For a new problem, laws grown one motif at a time over
its variables: start from a scaled motif, then add a scaled motif, multiply by
one (or by a constant plus one), or divide by a constant plus one. After each step every constant is
fitted to the rows by least squares, and the best few by BIC are grown
further (a beam). The best laws found, and every remembered whole shape that
fits the number of variables, are returned as proposals.

**It decides nothing.** Its laws are candidates for the path's confirmation
like any search's (AGENTS.md 1.1), and what it remembers must have been
confirmed by the referee on *other* problems: a memory that had seen a
problem's own answer would be reading the answer back.

**Limits.** It can only build what its motifs and three ways of joining them
reach, and a law whose parts were never confirmed before is invisible to it.
A periodic motif's constant is started at a scan of frequencies, because a
frequency's basin is narrow; other constants at three values (``STARTS``).
Not measured in a preregistered run.
"""
from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import least_squares

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.proposers.composed import formula_proposal

V, C = sp.Symbol("v"), sp.Symbol("C")
JOINS = ("add", "multiply", "times", "divide")     # times: by a constant plus the motif; divide: by a constant plus the motif
STARTS = (0.3, 1.0, 3.0)        # where a motif's first constant starts its fit
PERIODIC = ("sin(", "cos(", "tan(")
HALF_PERIODS = 12               # a periodic motif's constant starts at 1 to this many quarter turns over the variable's range
MAX_MOTIF_OPS = 10              # a one-variable part with more operations than this is not kept as a motif
EXACT = 1e-6                    # relative error below which a fit is taken as exact and the search stops
BAD = 1e3                       # a residual where a candidate is not finite


def one_variable_parts(expr: sp.Expr) -> list[sp.Expr]:
    """The parts of ``expr`` that depend on exactly one variable, as large as they come."""
    free = expr.free_symbols
    if not free:
        return []
    if len(free) == 1:
        return [expr]
    if expr.is_Add or expr.is_Mul:
        grouped: dict[sp.Symbol, list[sp.Expr]] = {}
        out: list[sp.Expr] = []
        for arg in expr.args:
            symbols = arg.free_symbols
            if len(symbols) == 1:
                grouped.setdefault(next(iter(symbols)), []).append(arg)
            elif len(symbols) > 1:
                out.extend(one_variable_parts(arg))
        return [*out, *(expr.func(*args) for args in grouped.values())]
    return [part for arg in expr.args for part in one_variable_parts(arg)]


def _core(part: sp.Expr) -> sp.Expr:
    """A one-variable part without its scale and offset."""
    return part.as_coeff_Mul()[1].as_coeff_Add()[1].as_coeff_Mul()[1]


def shape_of(part: sp.Expr) -> str | None:
    """A one-variable part as a motif: scale and offset stripped, the variable ``v``, every constant ``C``."""
    (variable,) = part.free_symbols
    rest = _core(part)
    if sp.count_ops(rest) > MAX_MOTIF_OPS:
        return None
    rest = rest.xreplace({variable: V})
    return str(rest.xreplace(dict.fromkeys(rest.atoms(sp.Float), C)))


def shapes_in(part: sp.Expr) -> list[str | None]:
    """The motif of a one-variable part and of the pieces it is plainly made of: its factors and its denominators."""
    out = [shape_of(part)]
    core = _core(part)
    for factor in (core.args if core.is_Mul else (core,)):
        if not factor.free_symbols or factor.is_Symbol:
            continue                                           # a number, or the bare variable: nothing to remember
        if factor.is_Pow and factor.exp.is_negative:
            if not _core(factor.base).is_Symbol:
                out.extend(shapes_in(factor.base))             # what a quotient divides by
        elif core.is_Mul:
            out.append(shape_of(factor))
    return out


def motifs_of(law: str | sp.Expr) -> list[str]:
    """The motifs of a law, each once, in the order met."""
    expr = sp.sympify(law) if isinstance(law, str) else law
    found = [shape for part in one_variable_parts(expr) for shape in shapes_in(part)]
    return list(dict.fromkeys(shape for shape in found if shape is not None))


def skeleton_of(law: str | sp.Expr) -> tuple[str, int]:
    """The whole law with its variables numbered by first appearance and every constant ``C``; and how many variables."""
    expr = sp.sympify(law) if isinstance(law, str) else law
    symbols = sorted(expr.free_symbols, key=lambda s: min(str(expr).find(str(s)), 10 ** 9))
    renamed = expr.xreplace({s: sp.Symbol(f"u{i}") for i, s in enumerate(symbols)})
    return str(renamed.xreplace(dict.fromkeys(renamed.atoms(sp.Float), C))), len(symbols)


@dataclass
class Memory:
    """Shapes of confirmed laws: how often each motif was met, and each whole law's shape."""

    motifs: dict[str, int] = field(default_factory=dict)
    skeletons: dict[str, int] = field(default_factory=dict)     #: shape -> number of variables
    laws: int = 0

    def learn(self, law: str) -> list[str]:
        """Add one confirmed law; returns its motifs."""
        expr = sp.sympify(law)
        if not expr.free_symbols:
            return []
        found = motifs_of(expr)
        for shape in found:
            self.motifs[shape] = self.motifs.get(shape, 0) + 1
        shape, variables = skeleton_of(expr)
        self.skeletons[shape] = variables
        self.laws += 1
        return found

    def as_dict(self) -> dict[str, Any]:
        return {"laws": self.laws, "motifs": dict(sorted(self.motifs.items())), "skeletons": dict(sorted(self.skeletons.items()))}

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Memory:
        return cls(dict(record.get("motifs", {})), dict(record.get("skeletons", {})), int(record.get("laws", 0)))

    def save(self, path: Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(self.as_dict(), indent=1, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Any) -> Memory:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def with_parameters(shape: str, variables: dict[str, sp.Symbol], prefix: str) -> tuple[sp.Expr, list[sp.Symbol]]:
    """A shape with each ``C`` a parameter of its own, in the given variables."""
    params: list[sp.Symbol] = []

    def fresh(expr: sp.Expr) -> sp.Expr:
        if expr == C:
            params.append(sp.Symbol(f"{prefix}{len(params)}"))
            return params[-1]
        if not expr.args:
            return expr
        return expr.func(*[fresh(arg) for arg in expr.args])

    return fresh(sp.sympify(shape, locals={"C": C, **variables})), params


@dataclass(frozen=True)
class _Motif:
    shape: str
    params: int
    fn: Callable[..., np.ndarray]
    periodic: bool = False          #: whether its constant is a frequency

    def scan(self, column: np.ndarray) -> tuple[float, ...]:
        """Frequencies to start a periodic motif's constant at: 1 to ``HALF_PERIODS`` quarter turns over the column's range."""
        width = max(float(np.ptp(column)), 1e-12)
        return tuple(k * np.pi / (2.0 * width) for k in range(1, HALF_PERIODS + 1))


@dataclass(frozen=True)
class Grown:
    """A law grown from motifs: the steps, its fitted constants, and how it fits."""

    steps: tuple[tuple[str, str, int], ...]       #: (join, motif shape, variable index); the first join is ``start``
    theta: tuple[float, ...]
    bic: float
    relative_error: float


CONSTANT = "1"                  # the constant, always available, and only ever added


def _compiled(shapes: Sequence[str]) -> dict[str, _Motif]:
    out: dict[str, _Motif] = {CONSTANT: _Motif(CONSTANT, 0, lambda v: np.ones(len(v)))}
    for shape in shapes:
        if shape == CONSTANT:
            continue
        try:
            expr, params = with_parameters(shape, {"v": V}, "_m")
            out[shape] = _Motif(shape, len(params), sp.lambdify([V, *params], expr, "numpy"),
                                any(name in shape for name in PERIODIC))
        except (sp.SympifyError, TypeError, SyntaxError, ValueError):
            continue
    return out


def _layout(steps: Sequence[tuple[str, str, int]], motifs: dict[str, _Motif]) -> list[tuple[int, int]]:
    """Per step: where its own constant (scale or offset) sits in theta, or -1, and where the motif's constants start."""
    out, at = [], 0
    for join, shape, _variable in steps:
        own = -1 if join == "multiply" else at
        at += 0 if join == "multiply" else 1
        out.append((own, at))
        at += motifs[shape].params
    return out


def _count(steps: Sequence[tuple[str, str, int]], motifs: dict[str, _Motif]) -> int:
    return sum((0 if join == "multiply" else 1) + motifs[shape].params for join, shape, _v in steps)


def _values(steps: Sequence[tuple[str, str, int]], theta: np.ndarray, X: np.ndarray, motifs: dict[str, _Motif]) -> np.ndarray:
    value: np.ndarray = np.zeros(len(X))
    for (join, shape, variable), (own, start) in zip(steps, _layout(steps, motifs), strict=True):
        motif = motifs[shape]
        part = motif.fn(X[:, variable], *theta[start:start + motif.params])
        if join == "start":
            value = theta[own] * part
        elif join == "add":
            value = value + theta[own] * part
        elif join == "multiply":
            value = value * part
        elif join == "times":
            value = value * (theta[own] + part)
        else:
            value = value / (theta[own] + part)
    out: np.ndarray = np.asarray(value, dtype=float)
    return out


def _fit(steps: tuple[tuple[str, str, int], ...], start: np.ndarray, X: np.ndarray, y: np.ndarray,
         motifs: dict[str, _Motif], free: int | None = None) -> Grown | None:
    """Least squares for the constants of ``steps`` from ``start``; with ``free``, only the last ``free`` of them move."""
    spread = float(np.std(y)) or 1.0
    fixed = len(start) - (len(start) if free is None else free)

    def resid(moving: np.ndarray) -> np.ndarray:
        theta = np.concatenate([start[:fixed], moving])
        with np.errstate(all="ignore"):
            value = _values(steps, theta, X, motifs)
        out: np.ndarray = np.where(np.isfinite(value), (value - y) / spread, BAD)
        return out

    if fixed == len(start):
        moved, final = start[fixed:], resid(start[fixed:])
    else:
        try:
            fit = least_squares(resid, start[fixed:], method="lm", max_nfev=40 * (len(start) - fixed + 1))
        except (ValueError, TypeError, FloatingPointError):
            return None
        moved, final = fit.x, resid(fit.x)
    if not np.all(np.isfinite(moved)) or np.any(final >= BAD):
        return None
    n = len(y)
    mse = float(np.mean(final ** 2))
    relative = float(np.sqrt(mse)) * spread / max(float(np.sqrt(np.mean(y ** 2))), 1e-300)
    bic = n * float(np.log(max(mse, 1e-30))) + len(start) * float(np.log(n))
    return Grown(steps, (*(float(v) for v in start[:fixed]), *(float(v) for v in moved)), bic, relative)


def written(grown: Grown, names: Sequence[str], motifs: dict[str, _Motif], digits: int = 8) -> sp.Expr:
    """A grown law as an expression in ``names``, its constants printed to ``digits`` figures."""
    theta = [sp.Float(float(f"{value:.{digits}g}")) for value in grown.theta]
    expr: sp.Expr = sp.Integer(0)
    for (join, shape, variable), (own, start) in zip(grown.steps, _layout(grown.steps, motifs), strict=True):
        part, params = with_parameters(shape, {"v": sp.Symbol(names[variable])}, "_w")
        part = part.xreplace(dict(zip(params, theta[start:start + len(params)], strict=True)))
        if shape == CONSTANT:
            part = sp.Integer(1)
        if join == "start":
            expr = theta[own] * part
        elif join == "add":
            expr = expr + theta[own] * part
        elif join == "multiply":
            expr = expr * part
        elif join == "times":
            expr = expr * (theta[own] + part)
        else:
            expr = expr / (theta[own] + part)
    return expr


def grow(memory: Memory, X: np.ndarray, y: np.ndarray, *, beam: int = 6, max_motifs: int = 4, seconds: float = 60.0,
         keep: int = 12, refit: int = 10) -> tuple[list[Grown], dict[str, _Motif], dict[str, Any]]:
    """Laws grown from the memory's motifs on rows ``X, y``: the best ``keep`` by BIC, the motifs, and what was done.

    Each of the ``beam`` best laws of a step is extended by every motif on every variable in each
    of the three ways. An extension is first fitted in its new constants alone, from each of
    ``STARTS``; the ``refit`` most promising extensions of a law then have all constants fitted.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    motifs = _compiled(sorted(memory.motifs, key=lambda shape: (-memory.motifs[shape], shape)))
    deadline = time.monotonic() + seconds
    report: dict[str, Any] = {"motifs": len(motifs) - 1, "fits": 0, "steps": 0, "stopped": "grown to the limit"}
    found: dict[tuple[tuple[str, str, int], ...], Grown] = {}

    def extended(steps: tuple[tuple[str, str, int], ...], before: Sequence[float]) -> Grown | None:
        """The extension fitted in its own constants only, from the best of the starts."""
        motif = motifs[steps[-1][1]]
        own = [] if steps[-1][0] == "multiply" else [1.0]
        best: Grown | None = None
        # A frequency has a narrow basin, so a periodic motif's constant is started at a scan; others at three values.
        firsts = (1.0,) if not motif.params else motif.scan(X[:, steps[-1][2]]) if motif.periodic else STARTS
        for first in firsts:
            start = np.array([*before, *own, *([first] + [1.0] * (motif.params - 1) if motif.params else [])], dtype=float)
            report["fits"] += 1
            fitted = _fit(steps, start, X, y, motifs, free=len(own) + motif.params)
            if fitted is not None and (best is None or fitted.bic < best.bic):
                best = fitted
        return best

    def polished(rough: Grown) -> Grown:
        report["fits"] += 1
        fitted = _fit(rough.steps, np.array(rough.theta), X, y, motifs)
        return fitted if fitted is not None and fitted.bic < rough.bic else rough

    held: list[Grown] = []
    for shape, variable in itertools.product(motifs, range(X.shape[1])):
        if shape == CONSTANT:
            continue
        first = extended((("start", shape, variable),), [])
        if first is not None:
            found[first.steps] = first
            held.append(first)
    report["steps"] = 1
    for _ in range(max_motifs - 1):
        held = sorted(held, key=lambda g: g.bic)[:beam]
        if not held or held[0].relative_error < EXACT:
            report["stopped"] = "an exact fit" if held else "nothing fitted"
            break
        grown: list[Grown] = []
        for parent in held:
            if time.monotonic() > deadline:
                report["stopped"] = "out of time"
                break
            rough: list[Grown] = []
            for join, shape, variable in itertools.product(JOINS, motifs, range(X.shape[1])):
                if shape == CONSTANT and (join != "add" or variable != 0 or any(step[1] == CONSTANT for step in parent.steps)):
                    continue                    # one intercept at most, and a constant is only ever added
                steps = (*parent.steps, (join, shape, variable))
                if steps in found:
                    continue
                candidate = extended(steps, parent.theta)
                if candidate is not None:
                    rough.append(candidate)
            for candidate in sorted(rough, key=lambda g: g.bic)[:refit]:
                fitted = polished(candidate)
                found[fitted.steps] = fitted
                grown.append(fitted)
        report["steps"] += 1
        held = grown
        if report["stopped"] == "out of time":
            break
    return sorted(found.values(), key=lambda g: g.bic)[:keep], motifs, report


def remembered_whole(memory: Memory, names: Sequence[str], X: np.ndarray, y: np.ndarray, limit: int = 4,
                     seconds: float = 15.0, tries: int = 8) -> list[tuple[float, sp.Expr]]:
    """Every remembered whole-law shape in as many variables as the problem has, constants fitted; best BIC first.

    Each assignment of the shape's variables to the problem's is fitted from ``tries`` starting
    points (all ones, then seeded draws of sign and size): no constant's value was remembered.
    """
    out: list[tuple[float, sp.Expr]] = []
    spread = float(np.std(y)) or 1.0
    deadline = time.monotonic() + seconds
    rng = np.random.default_rng(0)
    for shape, variables in memory.skeletons.items():
        if variables != len(names):                     # a law of the problem uses every one of its variables
            continue
        for chosen in itertools.permutations(range(len(names))):
            if time.monotonic() > deadline:
                return sorted(out, key=lambda pair: pair[0])[:limit]
            try:
                expr, params = with_parameters(shape, {f"u{i}": sp.Symbol(names[j]) for i, j in enumerate(chosen)}, "_s")
                fn = sp.lambdify([*(sp.Symbol(n) for n in names), *params], expr, "numpy")
            except (sp.SympifyError, TypeError, SyntaxError, ValueError):
                continue
            if not params or len(params) > len(y):
                continue

            def resid(theta: np.ndarray, fn: Any = fn) -> np.ndarray:
                with np.errstate(all="ignore"):
                    value = np.asarray(fn(*X.T, *theta), dtype=complex) * np.ones(len(X))
                ok = np.isfinite(value.real) & (np.abs(value.imag) < 1e-12)
                res: np.ndarray = np.where(ok, (value.real - y) / spread, BAD)
                return res

            best: tuple[float, np.ndarray] | None = None
            for attempt in range(tries):
                start = np.ones(len(params)) if attempt == 0 else rng.choice([-1.0, 1.0], len(params)) * rng.choice(STARTS, len(params))
                try:
                    fit = least_squares(resid, start, max_nfev=60 * len(params))
                except (ValueError, TypeError, FloatingPointError):
                    continue
                final = resid(fit.x)
                if np.any(final >= BAD) or not np.all(np.isfinite(fit.x)):
                    continue
                mse = float(np.mean(final ** 2))
                if best is None or mse < best[0]:
                    best = (mse, fit.x)
                if mse < EXACT ** 2:
                    break
            if best is None:
                continue
            bic = len(y) * float(np.log(max(best[0], 1e-30))) + len(params) * float(np.log(len(y)))
            out.append((bic, expr.xreplace({p: sp.Float(float(f"{v:.8g}")) for p, v in zip(params, best[1], strict=True)})))
    return sorted(out, key=lambda pair: pair[0])[:limit]


class StructureMemoryProposer(ProposerBase):
    """Laws grown from the motifs of laws confirmed before, and those laws' own shapes, as proposals."""

    def __init__(self, memory: Memory, max_proposals: int = 3, beam: int = 6, max_motifs: int = 4, seconds: float = 60.0,
                 target_col: str = "target") -> None:
        self._memory, self._max = memory, int(max_proposals)
        self._beam, self._max_motifs, self._seconds = int(beam), int(max_motifs), float(seconds)
        self._target_col = target_col
        self.last_run: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return "structure_memory"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [c for c in frame.columns if c != target]
        X = frame[features].to_numpy(dtype=float)
        y = frame[target].to_numpy(dtype=float)
        best, motifs, report = grow(self._memory, X, y, beam=self._beam, max_motifs=self._max_motifs, seconds=self._seconds)
        texts: list[str] = []
        seen: set[str] = set()
        candidates = [*[(g.bic, written(g, features, motifs)) for g in best], *remembered_whole(self._memory, features, X, y)]
        for _bic, law in sorted(candidates, key=lambda pair: pair[0]):
            if law.free_symbols != {sp.Symbol(f) for f in features}:
                continue                        # a law of the problem uses every one of its variables
            key = str(law.xreplace(dict.fromkeys(law.atoms(sp.Float), C)))
            if key not in seen:
                seen.add(key)
                texts.append(str(law))
            if len(texts) == self._max:
                break
        self.last_run = {**report, "laws_remembered": self._memory.laws, "proposed": len(texts)}
        return [formula_proposal(self.proposer_id, text, features, {"rank": rank}) for rank, text in enumerate(texts)]
