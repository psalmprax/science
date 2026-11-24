"""What is known about a law before the table is read, stated so that a candidate can be checked against it.

Status: experimental

**Not in force; development only.** ``small_table.py`` uses it; nothing else does.

A table of ten rows does not single out a law: on Langmuir's own Table IX a
cosine of the fourth root of the pressure fits better than Langmuir's isotherm.
The scientist who reads such a table knows things the rows do not say: the
surface holds nothing at zero pressure, more pressure never means less gas, and
the surface fills up. This module lets those be written down and checked on a
fitted candidate, so that the choice among forms a table cannot tell apart is
made by knowledge stated beforehand and not by the last decimal of a fit.

A statement is a mapping with a ``kind``:

* ``increasing`` / ``decreasing`` in one ``input`` over ``over`` (default: the
  table's range of that input). ``over_times: [a, b]`` gives the interval as
  multiples of the table's largest value of that input, for a statement written
  before the table is read (``[0, 4]``: from zero to four times as far as the
  table goes);
* ``concave`` / ``convex`` in one ``input`` over ``over``: the slope does not
  grow, or does not shrink (a quantity that levels off is increasing and concave);
* ``levels_off`` in one ``input``: far beyond the table the target stops
  growing, as a surface that fills up does. Checked as the local exponent
  ``x f'(x) / f(x)`` at ``FAR`` times the table's largest value of that input
  being below ``FLAT_EXPONENT`` in size: a power law keeps its exponent there
  and a saturating ratio's goes to zero;
* ``zero_at_zero``: the target goes to zero as every input does;
* ``positive``: the target stays above zero over the stated ranges;
* ``finite``: the law is defined over ``over`` (checked for every kind anyway).

**How a statement is checked.** Where the fitted law is a polynomial or a
ratio of polynomials in the input, the sign of its derivative is decided for
every point of the interval by an SMT solver (``apedp_dyn.recursion.smt``): a
proof, recorded as ``"proved"``. Otherwise the law is evaluated on ``GRID``
points of the interval, with the other inputs at five levels of their own
range, recorded as ``"grid"``: a check, not a proof. ``holds`` is ``True``,
``False`` (with the point where it fails) or ``None`` when the law could not be
evaluated.

The tolerances below are development constants of this module, not thresholds
of the referee. They say how much of the target's size counts as nothing.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import sympy as sp

from apedp_dyn.recursion import smt

KINDS = ("increasing", "decreasing", "concave", "convex", "levels_off", "zero_at_zero", "positive", "finite")
GRID = 200
LEVELS = 5                   # values of the other inputs a one-input statement is checked at
SLACK = 1e-6                 # of the target's size: a slope or bend smaller than this is counted as flat
ZERO_SHARE = 0.02            # of the target's largest size: what "goes to zero" allows at the origin
NEAR_ZERO = 1e-6             # of an input's smallest value: how close to zero the origin is approached
FAR = 1000.0                 # times the table's largest value of an input: where "levels off" is looked at
FLAT_EXPONENT = 0.05         # the size of the local exponent below which the target counts as no longer growing


def _values(expr: sp.Expr, names: Sequence[str], points: np.ndarray) -> np.ndarray:
    fn = sp.lambdify([sp.Symbol(n) for n in names], expr, "numpy")
    with np.errstate(all="ignore"):
        out = np.asarray(fn(*points.T), dtype=complex) * np.ones(len(points))
    real: np.ndarray = np.where(np.abs(out.imag) < 1e-12, out.real, np.nan)
    return real


def _levels(names: Sequence[str], ranges: Mapping[str, tuple[float, float]], leave: str | None) -> list[dict[str, float]]:
    """The other inputs at ``LEVELS`` values across their ranges (one setting if there are no other inputs)."""
    others = [n for n in names if n != leave]
    if not others:
        return [{}]
    shares = np.linspace(0.0, 1.0, LEVELS)
    return [{n: float(ranges[n][0] + share * (ranges[n][1] - ranges[n][0])) for n in others} for share in shares]


def _interval(statement: Mapping[str, Any], ranges: Mapping[str, tuple[float, float]], name: str) -> tuple[float, float]:
    over, times = statement.get("over"), statement.get("over_times")
    if times:
        reach = max(abs(ranges[name][0]), abs(ranges[name][1]))
        return float(times[0]) * reach, float(times[1]) * reach
    low, high = (float(over[0]), float(over[1])) if over else ranges[name]
    return low, high


def _proved_monotone(expr: sp.Expr, name: str, low: float, high: float, rising: bool) -> bool | None:
    """A proof that a one-input rational law rises (or falls) on the whole interval, or its refutation; else ``None``."""
    symbol = sp.Symbol(name)
    if expr.free_symbols != {symbol}:
        return None
    try:
        slope = sp.together(sp.diff(expr, symbol))
        top, bottom = sp.fraction(slope)
        box = {name: (low, high)}
        pole = smt.zero_in_box(bottom, box)
        turn = smt.zero_in_box(top, box)
    except (sp.polys.polyerrors.BasePolynomialError, ValueError, TypeError):
        return None
    if pole["decided"] is not False or turn["decided"] is None:
        return None
    try:
        middle = float(slope.subs(symbol, 0.5 * (low + high)))
    except (TypeError, ValueError):
        return None
    if turn["decided"] is False:                          # the slope never vanishes: its sign at one point is its sign
        return (middle > 0) == rising
    return None                                           # the slope touches zero somewhere: left to the grid


def check(expr: sp.Expr, names: Sequence[str], statement: Mapping[str, Any], ranges: Mapping[str, tuple[float, float]],
          size: float) -> dict[str, Any]:
    """Whether a fitted law agrees with one statement. ``size`` is the target's largest absolute value in the table."""
    kind = statement.get("kind")
    if kind not in KINDS:
        raise ValueError(f"a statement's kind is one of {KINDS}, not {kind!r}")
    out: dict[str, Any] = {"kind": kind, "input": statement.get("input")}
    scale = float(size) or 1.0
    if kind == "zero_at_zero":
        point = np.array([[NEAR_ZERO * min(abs(ranges[n][0]), abs(ranges[n][1]), 1.0) or NEAR_ZERO for n in names]])
        value = _values(expr, names, point)[0]
        ok = bool(np.isfinite(value) and abs(value) <= ZERO_SHARE * scale)
        return {**out, "holds": ok, "how": "grid", "value_near_zero": None if not np.isfinite(value) else float(value)}
    name = statement.get("input") or (names[0] if len(names) == 1 else None)
    if name is None or name not in names:
        raise ValueError(f"the statement {dict(statement)} needs an 'input' among {list(names)}")
    low, high = _interval(statement, ranges, name)
    if kind == "levels_off":
        far = FAR * max(abs(ranges[name][0]), abs(ranges[name][1]))
        worst_exponent = 0.0
        for setting in _levels(names, ranges, name):
            pair = np.array([[far * step if n == name else setting[n] for n in names] for step in (1.0, 1.01)])
            values = _values(expr, names, pair)
            if not np.all(np.isfinite(values)) or values[0] == 0:
                return {**out, "input": name, "holds": False, "how": "grid", "not_defined_at": far}
            exponent = float(np.log(abs(values[1] / values[0])) / np.log(1.01)) if values[1] * values[0] > 0 else float("inf")
            worst_exponent = max(worst_exponent, abs(exponent))
        return {**out, "input": name, "holds": bool(worst_exponent < FLAT_EXPONENT), "how": "grid", "looked_at": far,
                "local_exponent": None if not np.isfinite(worst_exponent) else round(worst_exponent, 4)}
    if kind in ("increasing", "decreasing") and len(names) == 1:
        proved = _proved_monotone(expr, name, low, high, rising=kind == "increasing")
        if proved is not None:
            return {**out, "input": name, "holds": proved, "how": "proved", "over": [low, high]}
    grid = np.linspace(low, high, GRID)
    worst: dict[str, Any] | None = None
    for setting in _levels(names, ranges, name):
        points = np.column_stack([grid if n == name else np.full(GRID, setting[n]) for n in names])
        values = _values(expr, names, points)
        if not np.all(np.isfinite(values)):
            at = float(grid[int(np.argmin(np.isfinite(values)))])
            return {**out, "input": name, "holds": False, "how": "grid", "over": [low, high], "not_defined_at": at}
        steps = np.diff(values)
        bends = np.diff(values, 2)
        slack = SLACK * scale
        failing = {"finite": np.zeros(0, dtype=bool), "positive": values <= 0.0,
                   "increasing": steps < -slack, "decreasing": steps > slack,
                   "concave": bends > slack, "convex": bends < -slack}[str(kind)]
        if np.any(failing):
            worst = {"at": float(grid[int(np.argmax(failing))]), "others": setting}
            break
    return {**out, "input": name, "holds": worst is None, "how": "grid", "over": [low, high],
            **({"fails": worst} if worst else {})}


def check_all(expr: sp.Expr, names: Sequence[str], statements: Sequence[Mapping[str, Any]],
              ranges: Mapping[str, tuple[float, float]], size: float) -> list[dict[str, Any]]:
    return [check(expr, names, statement, ranges, size) for statement in statements]


def agrees(checks: Sequence[Mapping[str, Any]]) -> bool:
    """A law agrees with what is known if no statement is shown false. A statement that could not be checked does not exclude it."""
    return not any(item.get("holds") is False for item in checks)
