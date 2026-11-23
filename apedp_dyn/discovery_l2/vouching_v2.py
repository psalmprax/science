"""The last checks, second version: an error is looked for in the shape it would take, after the law's constants have had their say.

Status: experimental

**Why this module exists.** ``vouching.py`` (the first version) is frozen by two
preregistrations and showed three faults on runs made after it was written:

* **M-33.** Its power rule planted an error in a fixed shape and asked whether its
  neighbour test would see it. But a law's constants are fitted to the rows, and
  fitting absorbs whatever part of an error the constants can follow. What is left
  is rougher and smaller than what was planted, so the rule was too confident.
* **M-34.** A counterfeit that puts a pole of the law a little off is wrong by a
  great deal on two or three rows and right elsewhere. The neighbour test asks
  whether a residual is predicted by its neighbours', and a spike is not. **This
  version does not repair it** (below).
* **M-35.** For the same reason, a rival that parts from the law on a few rows was
  judged undetectable, and the law was withheld. On the physics benchmark that cost
  43 of the 193 laws the route recovered, and stopped nothing.

**What is done here**, to a law the route is about to report and to the candidates
it could not tell from it. Two tests, each at half the canonical level, so that a
right law is refused at most at that level:

1. *Smooth structure* (``vouching.structure_test``, unchanged): the residual
   statistic at three neighbourhood sizes against random sign flips.
2. *Shapes* (``shape_test``): the residuals are regressed on polynomials of the
   inputs up to the third degree, each first made orthogonal to the member's own
   gradient in its constants, and an F test asks whether they explain anything. A
   shape the constants can follow almost entirely is dropped: refitting would turn
   that error into other constants, not into a wrong law.

**A third test was tried and dropped**: the largest standardized residual against
the largest of as many Gaussian draws, for an error on a few rows that no shape
points to (M-34). Replayed on the red-team campaigns' worlds (development), it
refused most right laws under noise of another shape than Gaussian (it kept 28 of
157 laws on the first campaign's randomly drawn worlds of that arena, where the
first version kept 157), and it did not reduce the wrong laws the second
campaign's climb got past the checks (259 against 262). On both benchmarks it
changed nothing. Without it M-34 stays open.

**The rivals are not used as shapes of error**, though the first version planted
them. They were proposed by searches run on these same rows, so their forms carry
the rows' noise, and a test in their direction takes that noise for an error.
Tried while writing this module (development, on the physics benchmark's recorded
run): testing the residuals on the rivals' differences refused 8 of 90 right laws
at 1 % noise; refitting each rival's constants on one half and testing on the
other still refused 7, while the same test on fresh noise refused none, because
it is the choice of form that carries the noise, and refitting does not undo it.
The rivals are judged by the class rule only, as before.

The class rule and the fit-error rule are the first version's. The **power rule**
asks whether an error in one of the shapes the second test tries, of the size that
makes a law wrong, would have been seen: for a known shape that chance follows from
the noncentral F distribution, so it is computed, not simulated, and it does not
depend on how smooth the shape is.
A law whose constants can follow every shape tried is withheld: it can bend to any
error it was offered, so the data cannot vouch for it.

**What this cannot do.** It finds no law. A right law is refused at the canonical
level, one time in a hundred. It does not see an error that sits on a few rows and
that no polynomial of low degree follows: a pole a little off (M-34). The F test
assumes noise of one spread; noise that grows with the target makes it refuse right
laws more often than the level says. The shapes tried are finite. And every choice here was made
after reading the runs that showed the first version's faults: what it does on
laws nobody has looked at is for a run fixed in advance to say.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp
from scipy import stats

from apedp_dyn.discovery_l2 import vouching as first

ACCURACY_R2 = first.ACCURACY_R2
LEVEL = first.LEVEL
POWER = first.POWER
TESTS = 2                         # smooth structure and shapes: each at LEVEL / TESTS
DEGREE = 4                        # polynomial shapes of the inputs up to this degree
MAX_SHAPES = 160                  # more inputs than this allows: the third degree is left out
KEEP = 0.1                        # a shape is dropped if less than this share of it is left after the constants
RANK_TOL = 1e-8
NUMERIC_FLOOR = first.NUMERIC_FLOOR
MAX_ROWS = first.MAX_ROWS


@dataclass
class Member:
    """One candidate after the three tests."""

    law: str
    usable: bool
    exact: bool = False
    refused: bool = False
    p_structure: float = 1.0
    p_shapes: float = 1.0
    shapes: int = 0                            #: shapes the second test kept
    size: tuple[int, int] = (0, 0)
    residuals: np.ndarray | None = field(default=None, repr=False)
    scales: list[dict[str, float]] = field(default_factory=list, repr=False)


@dataclass
class Vouched:
    """What is reported after the checks, and why."""

    law: str | None
    reason: str
    switched: bool = False
    members: list[Member] = field(default_factory=list)
    power: float | None = None
    record: dict[str, Any] = field(default_factory=dict)


def _size(text: str, names: Sequence[str]) -> tuple[int, int]:
    """Operations and constants, as the first version counts them; text that is not a formula is as large as can be."""
    try:
        ops, constants = first._size(text, names)
        return int(ops), int(constants)
    except (AttributeError, NameError, SyntaxError, TypeError, ValueError):
        return (10 ** 6, 10 ** 6)


def gradient(text: str, names: Sequence[str], X: np.ndarray) -> np.ndarray:
    """The law's derivative in each of its constants on the rows, and a column of ones: what refitting can follow."""
    symbols = [sp.Symbol(name) for name in names]
    columns = [np.ones(len(X))]
    try:
        expr = sp.sympify(text, locals={name: sp.Symbol(name) for name in names})
        constants = sorted(expr.atoms(sp.Float), key=str)
        holders = [sp.Symbol(f"_c{k}") for k in range(len(constants))]
        free = expr.xreplace(dict(zip(constants, holders, strict=True)))
        for holder in holders:
            derivative = sp.diff(free, holder).xreplace(dict(zip(holders, constants, strict=True)))
            fn = sp.lambdify(symbols, derivative, ["scipy", "numpy"])
            with np.errstate(all="ignore"):
                column = np.asarray(fn(*X.T), dtype=float) * np.ones(len(X))
            if np.all(np.isfinite(column)):
                columns.append(column)
    except (sp.SympifyError, TypeError, ValueError, ArithmeticError, NameError, AttributeError):
        pass
    stacked: np.ndarray = np.column_stack(columns)
    return stacked


def basis(columns: np.ndarray) -> np.ndarray:
    """An orthonormal basis of the columns' span (rank-revealing; near-dependent columns count once)."""
    if columns.size == 0:
        return np.zeros((columns.shape[0], 0))
    scale = np.maximum(np.linalg.norm(columns, axis=0), 1e-300)
    u, s, _ = np.linalg.svd(columns / scale, full_matrices=False)
    rank = int(np.sum(s > RANK_TOL * max(float(s[0]) if len(s) else 0.0, 1e-300)))
    out: np.ndarray = np.asarray(u[:, :rank])
    return out


def polynomial_shapes(X: np.ndarray, degree: int = DEGREE) -> list[np.ndarray]:
    """Monomials of the standardized inputs, from the first degree up; the third left out where it would be too many."""
    Z = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    inputs = Z.shape[1]
    out: list[np.ndarray] = []
    for d in range(1, degree + 1):
        combos = list(itertools.combinations_with_replacement(range(inputs), d))
        if d > 2 and len(out) + len(combos) > MAX_SHAPES:
            break
        for combo in combos:
            out.append(np.prod(Z[:, list(combo)], axis=1))
    return out


def kept_shapes(shapes: Sequence[np.ndarray], gradient_basis: np.ndarray) -> list[np.ndarray]:
    """Each shape with what the constants can follow taken out, scaled to unit root mean square; absorbed ones dropped."""
    out = []
    for shape in shapes:
        if not np.all(np.isfinite(shape)):
            continue
        centred = shape - shape.mean()
        spread = float(np.sqrt(np.mean(centred ** 2)))
        if spread <= 0.0:
            continue
        left = centred - gradient_basis @ (gradient_basis.T @ centred)
        rms = float(np.sqrt(np.mean(left ** 2)))
        if rms >= KEEP * spread:
            out.append(left / rms)
    return out


def shape_test(residuals: np.ndarray, gradient_basis: np.ndarray, shapes: Sequence[np.ndarray]
               ) -> tuple[float, int, int, float]:
    """F test of the residuals on the kept shapes: p-value, the shapes' rank, the residual degrees of freedom, sigma^2."""
    n = len(residuals)
    p = gradient_basis.shape[1]
    r = residuals - gradient_basis @ (gradient_basis.T @ residuals)
    if not shapes:
        return 1.0, 0, n - p, float(np.sum(r ** 2) / max(n - p, 1))
    U = basis(np.column_stack(shapes))
    U = basis(U - gradient_basis @ (gradient_basis.T @ U)) if U.shape[1] else U
    q = U.shape[1]
    df = n - p - q
    if q == 0 or df <= 0:
        return 1.0, q, max(df, 1), float(np.sum(r ** 2) / max(n - p, 1))
    explained = U @ (U.T @ r)
    left = r - explained
    sigma2 = float(np.sum(left ** 2) / df)
    if sigma2 <= 0.0:
        return 0.0, q, df, 0.0
    F = float(np.sum(explained ** 2) / q) / sigma2
    return float(stats.f.sf(F, q, df)), q, df, sigma2


def power_for(shapes_count: int, df: int, sigma2: float, amplitude: float, rows: int, level: float) -> float:
    """The chance the shape test sees an error of root mean square ``amplitude`` in one of its shapes."""
    if shapes_count == 0 or sigma2 <= 0.0:
        return 1.0 if sigma2 <= 0.0 else 0.0
    critical = float(stats.f.isf(level, shapes_count, df))
    noncentrality = amplitude ** 2 * rows / sigma2
    return float(stats.ncf.sf(critical, shapes_count, df, noncentrality))


@dataclass
class Tested:
    """What the power rule needs from a member's tests."""

    rank: int
    df: int
    sigma2: float
    rows: int


def _check(member: Member, values: np.ndarray, names: Sequence[str], X: np.ndarray, y: np.ndarray,
           operators: Sequence[tuple[int, np.ndarray]], *, flips: int, seed: int, degree: int) -> Tested:
    """The two tests on one member, and what the power rule will need."""
    residuals = y - values
    member.residuals = residuals
    G = basis(gradient(member.law, names, X))
    shapes = kept_shapes(polynomial_shapes(X, degree), G)
    member.shapes = len(shapes)
    member.p_structure, member.scales = first.structure_test(residuals, operators, flips=flips, seed=seed)
    member.p_shapes, q, df, sigma2 = shape_test(residuals, G, shapes)
    member.refused = min(member.p_structure, member.p_shapes) < LEVEL / TESTS
    return Tested(q, df, sigma2, len(y))


def vouch(law: str, others: Sequence[str], names: Sequence[str], X: np.ndarray, y: np.ndarray,
          X_more: np.ndarray | None = None, *, accuracy: float = ACCURACY_R2, power: float = POWER,
          flips: int = first.FLIPS, seed: int = 0, degree: int = DEGREE) -> Vouched:
    """The law to report in place of ``law``, or none, with the reason. Arguments as ``vouching.vouch``'s."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) > MAX_ROWS:
        keep = np.sort(np.random.default_rng(seed).choice(len(y), MAX_ROWS, replace=False))
        X, y = X[keep], y[keep]
    box = X if X_more is None or not len(X_more) else np.vstack([X, np.asarray(X_more, dtype=float)])
    operators = first.differences(X)
    scale = float(np.sqrt(np.mean(y ** 2))) or 1.0
    texts = list(dict.fromkeys([law, *others]))
    fitted: dict[str, np.ndarray] = {}
    members: list[Member] = []
    for text in texts:
        values = first._values(text, names, X)
        member = Member(text, values is not None, size=_size(text, names))
        members.append(member)
        if values is None:
            member.refused = True
        else:
            fitted[text] = values
    tested: dict[str, Tested] = {}
    for member in members:
        if not member.usable:
            continue
        values = fitted[member.law]
        if float(np.sqrt(np.mean((y - values) ** 2))) < NUMERIC_FLOOR * scale:
            member.exact = True
            continue
        tested[member.law] = _check(member, values, names, X, y, operators, flips=flips, seed=seed, degree=degree)
    record: dict[str, Any] = {"members": len(members), "refused": sum(m.refused for m in members), "level": LEVEL,
                              "tests": TESTS, "accuracy": accuracy, "rows": len(y)}
    kept = sorted((m for m in members if not m.refused), key=lambda m: (m.size, members.index(m)))
    if not kept:
        return Vouched(None, "every candidate failed a test of its residuals", members=members, record=record)
    chosen = members[0] if not members[0].refused else kept[0]
    on_box = {m.law: first._values(m.law, names, box) for m in kept}
    mine = on_box[chosen.law]
    if mine is None:
        return Vouched(None, "the law is not finite on the inputs at hand", members=members, record=record)
    spread = float(np.var(mine))
    apart = [m.law for m in kept if m is not chosen
             and (on_box[m.law] is None
                  or float(np.mean((on_box[m.law] - mine) ** 2)) > (1.0 - accuracy) * max(spread, 1e-300))]
    record["apart"] = len(apart)
    if apart:
        return Vouched(None, f"{len(apart) + 1} candidates passed every test and differ by more than the accuracy claimed",
                       members=members, record={**record, "class": [chosen.law, *apart]})
    switched = chosen is not members[0]
    note = "another member of the class is reported: the first failed a test" if switched else ""
    if chosen.exact:
        return Vouched(chosen.law, note, switched, members, 1.0, record)
    residuals = y - fitted[chosen.law]
    signal = float(np.var(fitted[chosen.law]))
    allowed = (1.0 - accuracy) * signal
    fit_error = chosen.size[1] * float(np.var(residuals)) / len(y)
    record.update({"allowed_error": allowed, "fit_error": fit_error})
    if fit_error >= allowed:
        return Vouched(None, "at this noise and with these rows its constants cannot be fitted to the accuracy claimed",
                       members=members, record=record)
    t = tested[chosen.law]
    record["shape_rank"] = t.rank
    if t.rank == 0:
        return Vouched(None, "its constants can follow every shape of error tried, so the rows cannot vouch for it",
                       members=members, power=0.0, record=record)
    chance = power_for(t.rank, t.df, t.sigma2, float(np.sqrt(allowed)), t.rows, LEVEL / TESTS)
    record["power"] = round(chance, 4)
    if chance < power:
        return Vouched(None, f"an error large enough to make it wrong would have been seen {chance:.0%} of the time, "
                             f"not {power:.0%}", members=members, power=chance, record=record)
    return Vouched(chosen.law, note, switched, members, chance, record)
