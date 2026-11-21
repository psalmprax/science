"""A follow-up experiment that decides between a reported law and its own shorter versions.

Status: experimental

**Why.** Two things the development replay of 2026-11-21 showed (the 98 systems
of PREREG-ODE-BLIND-2026-11 after that run, so development and not a result;
``benchmarks/ode_additions_dev_replay.json``):

1. *Adding an experiment's rows to the pool and confirming again made the route
   worse.* It recovered 167 and 128 of 310 components (no noise, 1 % noise)
   where the same route without the experiment recovered 202 and 232; with
   random restarts instead of chosen ones, 182 and 148. Of the 143 laws lost,
   139 ended as "no survivor". The path's last test compares the survivor's
   error on the rows it reads with its error on the development rows, and a
   short restart away from the attractor is measured less precisely than the
   long trajectories the development rows came from. The path took worse
   measurements for a worse law.
2. *Most reported laws that were not the law were the law with terms added.* Of
   78 such laws, 65: ``10.4*x0*x2 - 1.04*x1`` reported with
   ``-1.1e-4*x0**2*x1**2*x2**5*exp(-x0)`` beside it. On the attractor those
   terms are too small for any row to contradict, and they fit the error of the
   rate estimates, so neither confirmation nor pruning at the rows' precision
   removes all of them.

**What.** The rivals that matter are therefore the reported law and the same law
without one of its terms (or a group of them), and the measurement that decides
between them is one where those terms would be large:

1. ``deciding_measurement.deciding_points`` finds the states where the law with
   and without its terms predicts the most different rates, in a box ``EXPAND``
   times the range seen: a term that is tiny on the attractor is tiny at every
   state of the range seen as well.
2. The system is restarted at each and its states are recorded as before.
3. On each restart the law is compared with itself minus the terms, every other
   coefficient unchanged: how far the variable moved over a stretch of time
   against how far each version says it moved (``increment_error``). The terms
   are dropped when the version without them is the better one by an exact
   sign-flip test over the restarts, at the canonical level for comparing
   predictive accuracy divided by the number of versions tried.
4. The law that is left is refitted on the development rows, and the same is
   done again on the same restarts until nothing more can be dropped.

Three choices in this, each made on the replay's rows. *The comparison is
paired*, so how precisely a restart is measured cancels: it cannot fail a law
for being measured badly, which is what went wrong in (1). *Changes of the
state are compared, not rates*: a rate has to be estimated from a window of
samples, and on rows without noise the error of that estimate is the only
error there is; comparing estimated rates repaired 25 of 73 such laws where
comparing changes repaired more. *The other coefficients are left as they
are*: a shorter law refitted on rows whose rates carry an error takes part of
that error into its remaining coefficients, and then predicts worse than the
longer law wherever the dropped term is small. Terms are also tried in groups
(the smallest ones on the rows seen, and every two of a short law), because
added terms often cancel each other and removing one of such a pair predicts
no better.

The restarts are never used to fit anything. A term the law needs makes the
version without it worse wherever the two differ, so it is kept; a term can
only be dropped on the evidence of an experiment that contradicts it.

**Limits.** Only terms are removed: a law that is wrong in form stays as it was,
and nothing unreported becomes reported. It needs a system that can be
restarted where one chooses, which a simulation allows and most observed
systems do not. Developed on the first blind set; not yet measured in a
preregistered run.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import sympy as sp
from scipy.integrate import simpson, solve_ivp

from apedp_dyn.conf.canonical_spec import DIEBOLD_MARIANO_P_VALUE_THRESHOLD
from apedp_dyn.discovery_l2 import contrast_confirmation as cc
from apedp_dyn.discovery_l2 import deciding_measurement as dm
from apedp_dyn.discovery_l2 import refined_search
from apedp_dyn.ode import chosen_experiment, default_route
from apedp_dyn.ode.main_search_ode import state_names
from apedp_dyn.ode.window_rows import WindowRows

STARTS = 16                                  # restarts of one follow-up: an exact test over n restarts cannot go below 2**-n
ALPHA = DIEBOLD_MARIANO_P_VALUE_THRESHOLD    # the canonical level for a comparison of predictive accuracy
MAX_TERMS = 24                               # a longer law is left alone: its shorter versions would be too many to try
PAIRS_UP_TO = 8                              # laws of at most this many terms are also tried with every two terms removed
SEGMENT = 100                                # sampling steps per stretch over which a law's predicted change is compared
MAX_RATE_CALLS = 400_000                     # a restart whose integration needs more evaluations than this is given up


class _TooStiff(Exception):
    """A restart that the integrator cannot follow in reasonable time: counted as not bounded."""
#: The box restarts are chosen in, as a multiple of the range seen around its middle. Inside the range
#: seen (1.0) a term that is small on the attractor is small at every restart too. A search setting,
#: chosen on the development replay; not a threshold of the referee.
EXPAND = 2.0


@dataclass
class Decided:
    law: str | None                              #: the law after the experiment (the one given if nothing was dropped)
    dropped: list[str] = field(default_factory=list)
    starts: list[tuple[float, ...]] = field(default_factory=list)
    spreads: list[float] = field(default_factory=list)   #: the versions' disagreement at each start (empty for random starts)
    restarts_used: int = 0                       #: restarts that stayed bounded and gave rows
    rows: int = 0
    tests: list[dict[str, Any]] = field(default_factory=list)
    chosen: bool = True
    note: str = ""


def sign_flip_p(gains: np.ndarray) -> float:
    """Exact one-sided p-value that the mean of ``gains`` is positive, by flipping the sign of every one.

    Under "neither law predicts better" each restart's gain is as likely to have
    either sign. The p-value is the share of the ``2**n`` sign patterns whose
    sum is at least the observed one, so it cannot be smaller than ``2**-n``.
    """
    gains = np.asarray(gains, dtype=float)
    n = len(gains)
    if n == 0 or not np.all(np.isfinite(gains)):
        return 1.0
    if n > 20:
        raise ValueError("an exact test over more than 20 restarts is not provided")
    patterns = 1.0 - 2.0 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    sums = patterns @ gains
    tolerance = 1e-12 * float(np.sum(np.abs(gains))) if np.any(gains) else 0.0
    return float(np.mean(sums >= float(np.sum(gains)) - tolerance))


def _law(fit: cc.FittedCandidate, terms: Sequence[sp.Expr]) -> sp.Expr:
    return sp.Add(*[sp.Float(float(c)) * term for c, term in zip(fit.coef, terms, strict=True)])


def restarts(rate: Callable[[np.ndarray], np.ndarray], starts: Sequence[Sequence[float]], level: float,
             spread: np.ndarray, seed: int, duration: float = chosen_experiment.DURATION
             ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Times and measured states of each restart that stayed bounded: noise is ``level`` times each variable's spread."""
    rng = np.random.default_rng(seed)
    samples = round(duration * chosen_experiment.SAMPLES_PER_UNIT)
    t = np.linspace(0.0, duration, samples)
    kept: list[tuple[np.ndarray, np.ndarray]] = []
    for start in starts:
        calls = [0]

        def field(_t: float, x: np.ndarray, calls: list[int] = calls) -> np.ndarray:
            calls[0] += 1
            if calls[0] > MAX_RATE_CALLS:
                raise _TooStiff
            out: np.ndarray = rate(x)[0]
            return out

        try:
            with np.errstate(all="ignore"):
                solution = solve_ivp(field, (0.0, duration), np.asarray(start, dtype=float), t_eval=t, method="DOP853",
                                     rtol=1e-12, atol=1e-14)
        except _TooStiff:
            continue
        if not (solution.success and solution.y.shape[1] == samples and np.all(np.isfinite(solution.y))
                and float(np.abs(solution.y).max()) < chosen_experiment.BOUND):
            continue
        clean = solution.y.T
        kept.append((t, clean + level * spread * rng.standard_normal(clean.shape)))
    return kept


def increment_error(values: np.ndarray, t: np.ndarray, x: np.ndarray, component: int, segment: int = SEGMENT) -> float:
    """Mean squared difference between how far the variable moved and how far a law says it moved.

    ``values`` are the law's rates at the measured states of one restart. Over each stretch of
    ``segment`` sampling steps they are integrated (Simpson's rule) and compared with the measured
    change of the variable. No rate is estimated, so the comparison does not inherit the error of
    a rate estimate, which on rows without noise is the only error there is.
    """
    edges = np.arange(0, len(t) - segment, segment)
    if len(edges) == 0:
        return float("nan")
    if not np.all(np.isfinite(values)):
        return float("inf")
    step = float(t[1] - t[0])
    moved = np.array([x[a + segment, component] - x[a, component] for a in edges])
    said = np.array([simpson(values[a:a + segment + 1], dx=step) for a in edges])
    return float(np.mean((moved - said) ** 2))


def shorter_versions(terms: Sequence[sp.Expr], sizes: np.ndarray, pairs_up_to: int = PAIRS_UP_TO) -> list[tuple[int, ...]]:
    """Which terms to try removing together: each one, the smallest ones as a group, and (short laws) each two.

    ``sizes`` are the terms' root-mean-square contributions on the development rows. Added terms
    often come in groups that cancel each other, and removing one of a group predicts no better
    than keeping both; the terms that are smallest on the rows seen are the likeliest group.
    """
    count = len(terms)
    order = [int(k) for k in np.argsort(sizes, kind="stable")]
    tried: dict[tuple[int, ...], None] = {(k,): None for k in range(count)}
    for size in range(2, count):
        tried[tuple(sorted(order[:size]))] = None
    if count <= pairs_up_to:
        for a in range(count):
            for b in range(a + 1, count):
                if count > 2:
                    tried[(a, b)] = None
    return list(tried)


def prune_by_experiment(law: str, dev: WindowRows, component: int, rate: Callable[[np.ndarray], np.ndarray], *,
                        level: float, spread: np.ndarray, seed: int = 0, chosen: bool = True, starts: int = STARTS,
                        alpha: float = ALPHA, expand: float = EXPAND, pairs_up_to: int = PAIRS_UP_TO) -> Decided:
    """``law`` minus every term that a follow-up experiment contradicts; see the module docstring.

    ``rate`` is the system itself: states in (as rows), time derivatives out. ``level`` and
    ``spread`` are the measurement noise of the rows the system was first seen through.
    ``chosen=False`` restarts at random states of the same box: the comparison arm.
    """
    names = state_names(dev.states.shape[1])
    X, y = np.asarray(dev.states, dtype=float), np.asarray(dev.rates[:, component], dtype=float)
    terms = refined_search._terms(law, names)
    if len(terms) < 2:
        return Decided(law, chosen=chosen, note="one term: nothing shorter to compare with")
    if len(terms) > MAX_TERMS:
        return Decided(law, chosen=chosen, note=f"more than {MAX_TERMS} terms")

    def fitted(kept: Sequence[sp.Expr]) -> cc.FittedCandidate:
        return refined_search._fit(list(kept), names, X, y, "absolute")

    def sizes_of(fit: cc.FittedCandidate) -> np.ndarray:
        out: np.ndarray = np.sqrt(np.mean((fit.dev_basis * fit.coef) ** 2, axis=0))
        return out

    def without(coef: np.ndarray, removed: tuple[int, ...]) -> np.ndarray:
        out: np.ndarray = np.array(coef, dtype=float)
        out[list(removed)] = 0.0
        return out

    def written(coef: np.ndarray, kept: Sequence[sp.Expr]) -> str:
        return str(sp.Add(*[sp.Float(float(c)) * term for c, term in zip(coef, kept, strict=True) if c != 0.0]))

    current = fitted(terms)
    middle, half = 0.5 * (X.max(axis=0) + X.min(axis=0)), 0.5 * (X.max(axis=0) - X.min(axis=0))
    low, high = middle - expand * half, middle + expand * half
    if chosen:
        versions = [written(current.coef, terms),
                    *(written(without(current.coef, removed), terms)
                      for removed in shorter_versions(terms, sizes_of(current), pairs_up_to))]
        points = dm.deciding_points(versions, names, low, high, sd=max(float(dev.rate_sd[component]), 1e-300),
                                    count=starts, seed=seed)
        where, spreads = [p.point for p in points], [float(p.spread) for p in points]
        if not where:
            return Decided(law, chosen=True, note="the law and its shorter versions agree everywhere in the box")
    else:
        where, spreads = dm.random_points(low, high, starts, seed=seed), []
    measured = restarts(rate, where, level, spread, seed + 1)
    done = Decided(law, starts=[tuple(float(v) for v in p) for p in where], spreads=spreads, restarts_used=len(measured),
                   rows=sum(len(t) for t, _x in measured), chosen=chosen)
    if len(measured) < 2:
        done.note = "fewer than two restarts stayed bounded"
        return done
    kept = list(terms)
    while len(kept) > 1:
        with np.errstate(all="ignore"):
            bases = [np.asarray(current.basis(x), dtype=float) for _t, x in measured]

        def errors(coef: np.ndarray, bases: list[np.ndarray] = bases) -> np.ndarray:
            with np.errstate(all="ignore"):
                out: np.ndarray = np.array([increment_error(basis @ coef, t, x, component)
                                            for basis, (t, x) in zip(bases, measured, strict=True)])
            return out

        error = errors(current.coef)
        tried = shorter_versions(kept, sizes_of(current), pairs_up_to)
        best: tuple[float, tuple[int, ...]] | None = None
        for removed in tried:
            shorter_error = errors(without(current.coef, removed))
            gains = error - shorter_error                         # positive where the law predicts better without them
            p_value = sign_flip_p(gains) if np.all(np.isfinite(gains)) else 1.0
            done.tests.append({"terms": [str(kept[k]) for k in removed], "restarts_better": int(np.sum(gains > 0)),
                               "restarts": len(gains), "p": p_value})
            total = float(np.sum(shorter_error))
            if p_value < alpha / len(tried) and (best is None or (total, -len(removed)) < (best[0], -len(best[1]))):
                best = (total, removed)
        if best is None:
            break
        done.dropped.extend(str(kept[k]) for k in best[1])
        kept = [term for k, term in enumerate(kept) if k not in best[1]]
        current = fitted(kept)
    if done.dropped:
        done.law = str(_law(current, kept))
    return done


def decided(found: default_route.ComponentLaw, dev: WindowRows, component: int, rate: Callable[[np.ndarray], np.ndarray],
            *, level: float, spread: np.ndarray, seed: int = 0, chosen: bool = True, starts: int = STARTS
            ) -> tuple[default_route.ComponentLaw, dict[str, Any]]:
    """A route's outcome after the follow-up experiment: the reported law pruned by it, anything else unchanged."""
    if not found.law:
        return found, {"experiment": {"chosen": chosen, "note": "no reported law", "dropped": []}}
    try:
        done = prune_by_experiment(found.law, dev, component, rate, level=level, spread=spread, seed=seed, chosen=chosen,
                                   starts=starts)
    except (ValueError, TypeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        return found, {"experiment": {"chosen": chosen, "note": f"not run: {type(exc).__name__}: {exc}"[:200], "dropped": []}}
    record = {"chosen": done.chosen, "note": done.note, "dropped": done.dropped, "restarts_used": done.restarts_used,
              "rows": done.rows, "starts": [list(s) for s in done.starts], "spreads": done.spreads, "tests": done.tests}
    return (replace(found, law=done.law) if done.dropped else found), {"experiment": record}
