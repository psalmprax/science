"""A follow-up experiment for a dynamical system: start it where the rival laws disagree, and confirm again.

Status: experimental

The ODE route sees a system through a few trajectories. On a chaotic system
they all end on the same attractor, and laws that agree there cannot be told
apart, however long one watches. Where a system can be restarted (a
simulation, a bench experiment), the informative places are off the beaten
track: states the trajectories never visit, where the rivals part.

``follow_up`` does that for one component:

1. the rivals are the candidates of the first run that fit the development
   rows best (``rivals_of``);
2. ``deciding_measurement.deciding_points`` finds the states, inside the range
   the system was seen in, where their predicted rates differ most;
3. the system is started at each and watched for a short time
   (``measured_rows``): the same sampling, the same noise, the same windows as
   every other row;
4. those rows join the pool, and the same candidates go through the route
   again. The confirmation picks its rows where rivals disagree, so it will
   pick these.

With ``chosen=False`` the starts are drawn at random in the same box: equal
data, no choice. That is the comparison a chosen experiment has to beat.

**Not measured blind.** No search runs here: the follow-up reuses the first
run's candidates, so it can settle between laws already proposed and cannot
find one that was not.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from apedp_dyn.discovery_l2 import deciding_measurement as dm
from apedp_dyn.ode.main_search_ode import state_names
from apedp_dyn.ode.window_rows import WindowRows, window_rows

STARTS = 10                 # short trajectories of one follow-up
DURATION = 1.0              # time units each is watched for
SAMPLES_PER_UNIT = 1000     # as the world builder samples: 10,000 over 10 time units
WINDOW = 100
RIVALS = 6                  # candidates of the first run that enter the choice
BOUND = 1e6


@dataclass
class FollowUp:
    starts: list[tuple[float, ...]]             #: where the system was restarted
    spreads: list[float]                        #: the rivals' disagreement at each start (empty for random starts)
    rivals: list[str] = field(default_factory=list)
    rows: int = 0                               #: rows added to the pool
    failed_starts: int = 0                      #: starts whose trajectory did not stay bounded
    chosen: bool = True
    note: str = ""


def rivals_of(candidates: Sequence[str], names: Sequence[str], dev_states: np.ndarray, dev_rates: np.ndarray,
              count: int = RIVALS) -> list[str]:
    """The ``count`` candidates that fit the development rows best, each structure once."""
    values = dm.predictions(list(candidates), names, dev_states)
    with np.errstate(all="ignore"):
        error = np.sqrt(np.nanmean((values - dev_rates[None, :]) ** 2, axis=1))
    error = np.where(np.all(np.isfinite(values), axis=1), error, np.inf)
    kept: list[str] = []
    seen: set[str] = set()
    for index in np.argsort(error, kind="stable"):
        if not np.isfinite(error[index]) or len(kept) == count:
            break
        key = str(sp.sympify(candidates[index], locals={n: sp.Symbol(n) for n in names}))
        if key not in seen:
            seen.add(key)
            kept.append(str(candidates[index]))
    return kept


def measured_rows(rate: Callable[[np.ndarray], np.ndarray], starts: Sequence[Sequence[float]], level: float,
                  spread: np.ndarray, rng: np.random.Generator, duration: float = DURATION) -> tuple[WindowRows | None, int]:
    """The system started at each of ``starts`` and watched for ``duration``; window rows of all that stayed bounded.

    ``rate`` maps states (rows) to their time derivatives. Noise is ``level`` times each variable's
    ``spread``, as in the rows the system was first seen through. Returns the rows and how many
    starts gave none.
    """
    samples = round(duration * SAMPLES_PER_UNIT)
    t = np.linspace(0.0, duration, samples)
    parts: list[WindowRows] = []
    failed = 0
    for start in starts:
        with np.errstate(all="ignore"):
            solution = solve_ivp(lambda _t, x: rate(x)[0], (0.0, duration), np.asarray(start, dtype=float), t_eval=t,
                                 method="DOP853", rtol=1e-12, atol=1e-14)
        ok = solution.success and solution.y.shape[1] == samples and np.all(np.isfinite(solution.y)) \
            and np.abs(solution.y).max() < BOUND
        if not ok:
            failed += 1
            continue
        clean = solution.y.T
        parts.append(window_rows(t, clean + level * spread * rng.standard_normal(clean.shape), window=WINDOW))
    if not parts:
        return None, failed
    return WindowRows(np.vstack([p.states for p in parts]), np.vstack([p.rates for p in parts]),
                      np.concatenate([p.times for p in parts]), parts[0].degree,
                      np.max([p.fit_rms for p in parts], axis=0), np.max([p.rate_sd for p in parts], axis=0)), failed


def joined(pool: WindowRows, extra: WindowRows) -> WindowRows:
    """``pool`` with the rows of ``extra`` after it."""
    return WindowRows(np.vstack([pool.states, extra.states]), np.vstack([pool.rates, extra.rates]),
                      np.concatenate([pool.times, extra.times]), pool.degree, np.maximum(pool.fit_rms, extra.fit_rms),
                      np.maximum(pool.rate_sd, extra.rate_sd))


def follow_up(dev: WindowRows, pool: WindowRows, component: int, candidates: Sequence[str],
              rate: Callable[[np.ndarray], np.ndarray], *, level: float, spread: np.ndarray, seed: int = 0,
              chosen: bool = True, starts: int = STARTS) -> tuple[WindowRows, FollowUp]:
    """The pool with the rows of one follow-up experiment added, and what the experiment was.

    ``candidates`` are the laws the first run was given for this component. If fewer than two of
    them can be told apart anywhere in the box (or none is usable), the pool comes back unchanged.
    """
    names = state_names(dev.states.shape[1])
    low, high = dev.states.min(axis=0), dev.states.max(axis=0)
    rivals = rivals_of(candidates, names, dev.states, dev.rates[:, component])
    if chosen:
        points = dm.deciding_points(rivals, names, low, high, sd=max(float(dev.rate_sd[component]), 1e-300),
                                    count=starts, seed=seed)
        where, spreads = [p.point for p in points], [p.spread for p in points]
        if not where:
            return pool, FollowUp([], [], rivals, chosen=True, note="the rivals agree everywhere in the range seen")
    else:
        where, spreads = dm.random_points(low, high, starts, seed=seed), []
    extra, failed = measured_rows(rate, where, level, spread, np.random.default_rng(seed + 1))
    if extra is None:
        return pool, FollowUp(where, spreads, rivals, failed_starts=failed, chosen=chosen, note="no start stayed bounded")
    return joined(pool, extra), FollowUp(where, spreads, rivals, rows=len(extra.states), failed_starts=failed, chosen=chosen)


def as_record(done: FollowUp) -> dict[str, Any]:
    return {"chosen": done.chosen, "starts": [list(s) for s in done.starts], "spreads": done.spreads, "rows": done.rows,
            "failed_starts": done.failed_starts, "rivals": done.rivals, "note": done.note}
