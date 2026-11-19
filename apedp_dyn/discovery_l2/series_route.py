"""The main search path for a noisy series: a law is reported only if it forecasts later data better than no law.

Status: experimental

``main_search.discover`` was built for data in which a law explains almost all
of the target. Its checks ask whether a law leaves structure in its residuals
and whether another candidate predicts differently; neither asks whether the
law predicts anything. On a return-like series, where noise is nearly all of
the variance, the residuals of any formula look like noise, the first pass
stops at once, and the path reported a law in five of six development worlds
that contained none.

This route adds the question. The rows are in time order. The search sees the
development rows; the pool is the period after them. A law that the path
returns is refitted on the development rows (in the form the path confirmed)
and its forecasts for the pool period are compared with the forecast that uses
no input at all, the development mean, by the referee's Diebold-Mariano test
(``apedp_referee.novelty_adjudication.evaluate_predictive_incrementality``:
squared error, Newey-West variance, one-sided, at the preregistered level in
the canonical specification). If the law does not win, the outcome is
``no_predictive_law`` and nothing is reported.

The test is the referee's; this module decides nothing by itself. It reads
every target of the pool period, so that period cannot be reused to test the
same law again.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from apedp_dyn.discovery_l2 import contrast_resolution, main_search
from apedp_dyn.grammar.proposer_interface import ProposerBase
from apedp_referee.novelty_adjudication import evaluate_predictive_incrementality

NO_PREDICTIVE_LAW = "no_predictive_law"


@dataclass
class SeriesResult:
    action: str                    #: the path's action, or ``no_predictive_law`` when the forecast test is not passed
    law: str | None                #: the law in the form that was tested, or ``None``
    ungated_action: str            #: what the path said before the forecast test
    ungated_law: str | None        #: the law the path returned, refitted, whether or not it passed
    forecast_test: dict[str, Any]  #: empty when the path returned no law
    path: main_search.MainSearchResult


def refitted(law: str, names: list[str], dev_X: np.ndarray, dev_y: np.ndarray) -> sp.Expr:
    """The law with one coefficient per additive term, fitted on the development rows (what the path confirmed)."""
    expr = sp.sympify(law, locals={n: sp.Symbol(n) for n in names})
    terms = [term.as_coeff_Mul()[1] for term in sp.Add.make_args(sp.expand(expr))]
    fitted = contrast_resolution.fitted_candidate(expr, names, dev_X, dev_y, "absolute")
    return sp.Add(*[sp.Float(float(c)) * term for c, term in zip(fitted.coef, terms, strict=True)])


def predict(law: sp.Expr, names: list[str], X: np.ndarray) -> np.ndarray:
    fn = sp.lambdify([sp.Symbol(n) for n in names], law, ["scipy", "numpy"], dummify=True)
    with np.errstate(all="ignore"):
        out: np.ndarray = np.asarray(fn(*np.asarray(X, dtype=float).T), dtype=float) * np.ones(len(X))
    return out


def discover_predictive_law(X: Mapping[str, np.ndarray], y: np.ndarray, later_X: np.ndarray, later_y: np.ndarray, *,
                            proposer: ProposerBase | None = None, seed: int = 0, **options: Any) -> SeriesResult:
    """``(X, y)`` are the development rows; ``later_X, later_y`` the period after them, columns in the order of ``X``."""
    names = list(X)
    dev_X = np.column_stack([np.asarray(X[n], dtype=float) for n in names])
    dev_y = np.asarray(y, dtype=float)
    later_X, later_y = np.asarray(later_X, dtype=float), np.asarray(later_y, dtype=float)
    options.setdefault("always_search", True)
    path = main_search.discover(X, dev_y, later_X, lambda rows: later_y[rows], proposer=proposer, noise="absolute",
                                seed=seed, **options)
    if path.law is None:
        return SeriesResult(path.action, None, path.action, None, {}, path)
    law = refitted(path.law, names, dev_X, dev_y)
    forecast = predict(law, names, later_X)
    if not np.all(np.isfinite(forecast)):
        test: dict[str, Any] = {"passed": False, "detail": "the law is not finite on the later period"}
    else:
        outcome = evaluate_predictive_incrementality(later_y, forecast, np.full(len(later_y), float(dev_y.mean())))
        test = {"passed": bool(outcome.is_significant), "p_value": outcome.p_value, "dm_statistic": outcome.dm_statistic,
                "loss_reduction": outcome.loss_reduction, "alpha": outcome.alpha, "rows": len(later_y)}
    passed = bool(test["passed"])
    return SeriesResult(path.action if passed else NO_PREDICTIVE_LAW, str(law) if passed else None, path.action, str(law), test, path)
