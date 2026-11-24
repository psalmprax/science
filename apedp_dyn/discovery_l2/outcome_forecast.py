"""The system's forecast of its own results, from the runs it has on record.

Status: experimental

Before each preregistered run the author wrote down what he expected, so that
he could be wrong. This module does the same thing mechanically. From the
runs on record it learns, per route, how the outcome of a run depends on what
the law looks like (its size, its functions, whether it is a polynomial, how
many inputs it has) and on the noise, and for a new set of problems it gives,
problem by problem, the chance that the law is recovered, that a law is
reported at all, and that a wrong one is reported, with the totals these add
up to and an interval around each total.

**What it is for.**

* A forecast filed before a run and scored after it says whether the system
  knows where it fails. A forecast that only repeats the average of earlier
  runs is the baseline it has to beat (``baseline``).
* Where the forecast is low is where practice problems are worth drawing
  (``hardest``).

**What it is not.** The features are read from the true law, which a
benchmark knows and the route does not. So this forecasts runs on benchmarks.
On real data, where the law is unknown, only the noise and the shape of the
table are available, and nothing here has been tested that way. A forecast is
no evidence about a law; it is a statement about the system, to be scored.

**Method.** One logistic regression per outcome on standardised features
(``scikit-learn``), with an indicator per route when several routes are
pooled. An outcome that never, or always, occurred in the runs on record is
forecast at its smoothed frequency. The interval of a total is that of a sum
of independent yes/no outcomes with the forecast chances, by simulation,
widened by how far whole runs have differed from their forecasts in backtests
(``between_runs``): without that the intervals are too narrow.

**What the backtest showed (2026-11-21, every outcome known when this was
written, so not a test).** Holding one recorded run out at a time: on the
table benchmark, where the same laws recur under other routes, the forecast
tells hard laws from easy ones far better than the average does (Brier 0.12
to 0.16 against 0.21 to 0.25). Across kinds of dynamical systems it does not:
in three of six cases it is worse than the average, and a whole run's recovery
rate differed from its forecast by ten percentage points (standard deviation).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp
from sklearn.linear_model import LogisticRegression

OUTCOMES = ("recovered", "reported", "wrong")
FEATURES = ("inputs", "noise_is_zero", "log10_noise", "cost", "constants", "terms", "depth", "polynomial", "degree",
            "rational", "exp", "log", "trig", "tanh", "root", "divides_by_inputs")
INTERVAL = 0.8
DRAWS = 20_000


def _depth(expr: sp.Basic) -> int:
    return 1 + max((_depth(arg) for arg in expr.args), default=0)


def law_features(law: sp.Expr | str, inputs: int, noise: float) -> dict[str, float]:
    """What a law looks like, as numbers: its size, its functions, and the noise it is seen through."""
    expr = sp.sympify(law) if isinstance(law, str) else law
    symbols = sorted(expr.free_symbols, key=str)
    polynomial = bool(symbols) and bool(expr.is_polynomial(*symbols))
    degree = float(sp.Poly(expr, *symbols).total_degree()) if polynomial else 0.0
    powers = list(expr.atoms(sp.Pow))
    numbers = {a for a in expr.atoms(sp.Number) if a not in (sp.S.One, sp.S.NegativeOne, sp.S.Zero)}
    return {
        "inputs": float(inputs), "noise_is_zero": float(noise == 0), "log10_noise": float(np.log10(noise)) if noise > 0 else -4.0,
        "cost": float(sp.count_ops(expr)), "constants": float(len(numbers)), "terms": float(len(sp.Add.make_args(expr))),
        "depth": float(_depth(expr)), "polynomial": float(polynomial), "degree": degree,
        "rational": float(bool(symbols) and bool(expr.is_rational_function(*symbols))),
        "exp": float(expr.has(sp.exp)), "log": float(expr.has(sp.log)), "trig": float(expr.has(sp.sin, sp.cos, sp.tan)),
        "tanh": float(expr.has(sp.tanh, sp.sinh, sp.cosh)),
        "root": float(any(p.exp.is_Rational and not p.exp.is_Integer for p in powers)),
        "divides_by_inputs": float(any(p.exp.is_negative and p.base.free_symbols for p in powers)),
    }


@dataclass
class Run:
    """One run on record, or one to be forecast: the law, how it was seen, by which route, and what came of it."""

    law: str
    inputs: int
    noise: float
    route: str
    recovered: bool | None = None
    reported: bool | None = None
    wrong: bool | None = None
    source: str = ""
    name: str = ""

    def outcome(self, which: str) -> bool:
        value = getattr(self, which)
        if value is None:
            raise ValueError(f"the run {self.name!r} has no recorded outcome")
        return bool(value)


@dataclass
class Forecast:
    """Chances per problem and the totals they add up to, for one route on one set of problems."""

    route: str
    chances: dict[str, list[float]]                       #: outcome -> one chance per problem
    names: list[str] = field(default_factory=list)

    def expected(self, outcome: str) -> float:
        return float(np.sum(self.chances[outcome]))

    def interval(self, outcome: str, level: float = INTERVAL, seed: int = 0, between_runs: float = 0.0) -> tuple[int, int]:
        """The range the total falls in with chance ``level``.

        With ``between_runs`` zero the problems' outcomes are taken as independent and the chances
        as exact, which backtests show is too narrow. ``between_runs`` is the spread (a standard
        deviation, as a share of the problems) by which a whole run has differed from its forecast.
        """
        p: np.ndarray = np.asarray(self.chances[outcome], dtype=float)
        rng = np.random.default_rng(seed)
        totals = (rng.random((DRAWS, len(p))) < p).sum(axis=1) + len(p) * between_runs * rng.standard_normal(DRAWS)
        low, high = np.quantile(np.clip(totals, 0, len(p)), [(1 - level) / 2, 1 - (1 - level) / 2])
        return int(np.floor(low)), int(np.ceil(high))

    def as_dict(self, between_runs: float = 0.0) -> dict[str, Any]:
        return {"route": self.route, "problems": len(self.names), "between_runs_spread": round(between_runs, 4),
                "totals": {o: {"expected": round(self.expected(o), 2),
                               "interval": list(self.interval(o, between_runs=between_runs)),
                               "level": INTERVAL} for o in self.chances},
                "chances": {o: [round(float(v), 4) for v in values] for o, values in self.chances.items()},
                "names": list(self.names)}


class OutcomeForecaster:
    """Learns from runs on record how the outcome of a run depends on the law and the route."""

    def __init__(self, strength: float = 1.0) -> None:
        self._strength = float(strength)
        self._routes: list[str] = []
        self._mean: np.ndarray = np.zeros(0)
        self._scale: np.ndarray = np.ones(0)
        self._models: dict[str, LogisticRegression | float] = {}
        self.trained_on = 0

    def _matrix(self, runs: Sequence[Run]) -> np.ndarray:
        rows = []
        for run in runs:
            features = law_features(run.law, run.inputs, run.noise)
            rows.append([features[name] for name in FEATURES] + [float(run.route == route) for route in self._routes])
        matrix: np.ndarray = np.asarray(rows, dtype=float).reshape(len(runs), len(FEATURES) + len(self._routes))
        return matrix

    def fit(self, runs: Sequence[Run]) -> OutcomeForecaster:
        if not runs:
            raise ValueError("no runs to learn from")
        self._routes = sorted({run.route for run in runs})
        raw = self._matrix(runs)
        self._mean, self._scale = raw.mean(axis=0), np.where(raw.std(axis=0) > 0, raw.std(axis=0), 1.0)
        X = (raw - self._mean) / self._scale
        self.trained_on = len(runs)
        for outcome in OUTCOMES:
            y = np.array([run.outcome(outcome) for run in runs], dtype=int)
            if y.min() == y.max():                 # never or always: its smoothed frequency
                self._models[outcome] = float((y.sum() + 1) / (len(y) + 2))
            else:
                self._models[outcome] = LogisticRegression(C=self._strength, max_iter=2000).fit(X, y)
        return self

    def forecast(self, runs: Sequence[Run], route: str | None = None) -> Forecast:
        """Chances for ``runs`` (their own routes, or all as ``route``). An unknown route gets the routes' average."""
        if not self._models:
            raise ValueError("fit the forecaster first")
        as_route = [Run(r.law, r.inputs, r.noise, route or r.route, name=r.name) for r in runs]
        raw = self._matrix(as_route)
        for i, run in enumerate(as_route):
            if run.route not in self._routes and self._routes:
                raw[i, len(FEATURES):] = 1.0 / len(self._routes)
        X = (raw - self._mean) / self._scale
        chances = {}
        for outcome, model in self._models.items():
            values = np.full(len(as_route), model) if isinstance(model, float) else model.predict_proba(X)[:, 1]
            chances[outcome] = [float(v) for v in values]
        return Forecast(route or (as_route[0].route if as_route else ""), chances, [r.name for r in as_route])


def baseline(runs: Sequence[Run], count: int, route: str = "") -> Forecast:
    """The forecast that knows nothing about a law: the frequency of each outcome in the runs on record."""
    chances = {o: [float((sum(r.outcome(o) for r in runs) + 1) / (len(runs) + 2))] * count for o in OUTCOMES}
    return Forecast(route, chances, [""] * count)


def brier(chances: Sequence[float], happened: Sequence[bool]) -> float:
    """Mean squared difference between the chance given and what happened; lower is better, 0.25 is a coin."""
    p: np.ndarray = np.asarray(chances, dtype=float)
    y: np.ndarray = np.asarray(happened, dtype=float)
    return float(np.mean((p - y) ** 2))


def score(forecast: Forecast, runs: Sequence[Run], between_runs: float = 0.0) -> dict[str, dict[str, Any]]:
    """A forecast against what happened: the Brier score per outcome, the total forecast and the total seen."""
    out = {}
    for outcome, chances in forecast.chances.items():
        happened = [run.outcome(outcome) for run in runs]
        low, high = forecast.interval(outcome, between_runs=between_runs)
        out[outcome] = {"brier": round(brier(chances, happened), 4), "expected": round(forecast.expected(outcome), 1),
                        "interval": [low, high], "seen": int(sum(happened)), "inside": bool(low <= sum(happened) <= high)}
    return out


def hardest(forecast: Forecast, count: int, outcome: str = "recovered") -> list[int]:
    """The positions of the ``count`` problems least likely to be recovered: where practice is worth most."""
    order = np.argsort(np.asarray(forecast.chances[outcome], dtype=float), kind="stable")
    return [int(i) for i in order[:count]]


def runs_from(records: Sequence[Mapping[str, Any]]) -> list[Run]:
    """Runs from plain records with the fields of ``Run``."""
    return [Run(**{k: r[k] for k in ("law", "inputs", "noise", "route", "recovered", "reported", "wrong", "source", "name")
                   if k in r}) for r in records]
