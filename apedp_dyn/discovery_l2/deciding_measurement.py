"""Where rival laws disagree most: the measurement that would decide between them.

Status: experimental

When several laws fit the rows at hand equally well, reporting one of them
hides what is known. What is known is a set of rivals and, for any place
where an input could be measured, how far apart their predictions are there.
This module turns the set into that place.

* ``disagreement``: at each of some points, how far apart the rivals'
  predictions are, in units of the measurement's error.
* ``deciding_points``: where inputs can be set (an experiment), the points of
  a box at which the rivals disagree most, spread out so that they are not all
  the same point.
* ``deciding_rows``: where inputs cannot be set and only later observations
  can be chosen among (a market, a survey), the rows of a pool at which they
  disagree most.
* ``describe``: the result in a sentence a person can act on.

**What it does not do.** It says where a measurement would separate the
rivals it is given. It cannot know of a law that is not among them, and a
point far from the data is also where every one of them is least to be
trusted. It decides nothing: the measurement, once made, goes to the path's
confirmation like any other row.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

CLOUD = 4000                # trial points drawn in the box
SEPARATION = 0.15           # least distance between chosen points, as a share of the box's diagonal


@dataclass(frozen=True)
class DecidingPoint:
    point: tuple[float, ...]        #: where to measure, one value per input
    spread: float                   #: how far apart the rivals are there, in units of the measurement's error
    pair: tuple[int, int]           #: the two rivals furthest apart there (positions in the list given)


def predictions(rivals: Sequence[str | sp.Expr], names: Sequence[str], points: np.ndarray) -> np.ndarray:
    """Rivals by points: what each rival predicts at each point; not-a-number where it predicts nothing."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    symbols = [sp.Symbol(n) for n in names]
    out = np.full((len(rivals), len(points)), np.nan)
    for i, rival in enumerate(rivals):
        values: np.ndarray | None
        try:
            expr = sp.sympify(rival, locals={n: sp.Symbol(n) for n in names}) if isinstance(rival, str) else rival
            fn = sp.lambdify(symbols, expr, "numpy")
            with np.errstate(all="ignore"):
                values = np.asarray(fn(*points.T), dtype=complex) * np.ones(len(points))
        except Exception:  # noqa: BLE001 - a rival that cannot be evaluated predicts nothing anywhere
            values = None
        if values is not None:
            real = np.abs(values.imag) < 1e-12
            out[i] = np.where(real & np.isfinite(values.real), values.real, np.nan)
    return out


def disagreement(rivals: Sequence[str | sp.Expr], names: Sequence[str], points: np.ndarray,
                 sd: float | np.ndarray = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Per point: the largest difference between two rivals in units of ``sd``, and which two.

    A point where fewer than two rivals predict anything scores zero: nothing can be decided there.
    """
    values = predictions(rivals, names, points)
    usable = np.sum(np.isfinite(values), axis=0) >= 2
    high = np.where(np.isfinite(values), values, -np.inf)
    low = np.where(np.isfinite(values), values, np.inf)
    top, bottom = np.argmax(high, axis=0), np.argmin(low, axis=0)
    columns = np.arange(values.shape[1])
    scale = np.maximum(np.broadcast_to(np.asarray(sd, dtype=float), (values.shape[1],)), 1e-300)
    with np.errstate(all="ignore"):
        gap = np.where(usable, (high[top, columns] - low[bottom, columns]) / scale, 0.0)
    pairs: np.ndarray = np.column_stack([bottom, top])
    return np.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0), pairs


def deciding_points(rivals: Sequence[str | sp.Expr], names: Sequence[str], low: Sequence[float], high: Sequence[float],
                    sd: float = 1.0, count: int = 10, cloud: int = CLOUD, seed: int = 0,
                    separation: float = SEPARATION) -> list[DecidingPoint]:
    """Up to ``count`` points of the box ``low..high`` where the rivals disagree most, kept apart from each other.

    Fewer than ``count`` are returned if the rivals agree everywhere they all predict something.
    """
    lo: np.ndarray = np.asarray(low, dtype=float)
    hi: np.ndarray = np.asarray(high, dtype=float)
    if len(rivals) < 2 or np.any(hi < lo):
        return []
    points = lo + (hi - lo) * np.random.default_rng(seed).random((cloud, len(lo)))
    gap, pairs = disagreement(rivals, names, points, sd)
    width = np.where(hi > lo, hi - lo, 1.0)
    chosen: list[int] = []
    for index in np.argsort(-gap, kind="stable"):
        if gap[index] <= 0 or len(chosen) == count:
            break
        if all(np.linalg.norm((points[index] - points[j]) / width) >= separation * np.sqrt(len(lo)) for j in chosen):
            chosen.append(int(index))
    return [DecidingPoint(tuple(float(v) for v in points[i]), float(gap[i]), (int(pairs[i][0]), int(pairs[i][1])))
            for i in chosen]


def deciding_rows(rivals: Sequence[str | sp.Expr], names: Sequence[str], pool: np.ndarray, sd: float | np.ndarray = 1.0,
                  count: int = 10) -> list[int]:
    """The rows of ``pool`` (inputs that will be observed, not set) at which the rivals disagree most."""
    if len(rivals) < 2:
        return []
    gap, _ = disagreement(rivals, names, pool, sd)
    return [int(i) for i in np.argsort(-gap, kind="stable")[:count] if gap[i] > 0]


def random_points(low: Sequence[float], high: Sequence[float], count: int, seed: int = 0) -> list[tuple[float, ...]]:
    """As many points of the box drawn without regard to any rival: the comparison a chosen point has to beat."""
    lo: np.ndarray = np.asarray(low, dtype=float)
    hi: np.ndarray = np.asarray(high, dtype=float)
    points = lo + (hi - lo) * np.random.default_rng(seed).random((count, len(lo)))
    return [tuple(float(v) for v in p) for p in points]


def describe(points: Sequence[DecidingPoint], names: Sequence[str], rivals: Sequence[Any]) -> str:
    """What to measure, in a sentence."""
    if not points:
        return "The laws given agree wherever they all predict something in this range: no measurement here separates them."
    best = points[0]
    where = ", ".join(f"{n} = {v:.4g}" for n, v in zip(names, best.point, strict=True))
    a, b = best.pair
    return (f"Measure at {where}: there `{rivals[a]}` and `{rivals[b]}` differ by {best.spread:.3g} times the "
            f"measurement's error. {len(points)} such points were found.")
