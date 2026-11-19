"""The main search path for a noisy series: a law is reported only if it forecasts later data better than no law.

Status: experimental

``main_search.discover`` was built for data in which a law explains almost all
of the target. Its checks ask whether a law leaves structure in its residuals
and whether another candidate predicts differently; neither asks whether the
law predicts anything, and both assume one noise level throughout. On a
return-like series, where noise is nearly all of the variance, heavy-tailed,
and changing in scale, that fails in both directions (development worlds,
``scripts/market_planted_worlds.py``): the path reported a law in ten of twelve
worlds that contained none when its first pass ran alone, and with PySR's
candidates beside it the confirmation step refused every candidate in seven of
twenty worlds that did contain one, because errors on the later rows did not
match the development error.

So on a series the path is used for what it is good at, producing candidates,
and the decision is made differently:

1. **Candidates.** ``main_search.discover`` runs on the development rows (the
   first pass and the search stage see nothing else). Its candidate list is
   taken; its own verdict is recorded as the ``ungated`` reading and not used.
2. **One candidate is chosen on the development rows alone.** Each candidate is
   refitted on the first two thirds of the development period (one coefficient
   per additive term) and scored on the last third. The lowest error wins, the
   simpler formula on a tie. If none beats the mean of the first two thirds,
   the outcome is ``no_predictive_law`` and the later period is not read.
3. **One test on the later period.** The chosen candidate is refitted on all
   development rows and its forecasts for the period after them are compared
   with the development mean by the referee's Diebold-Mariano test
   (``apedp_referee.novelty_adjudication.evaluate_predictive_incrementality``:
   squared error, Newey-West variance, one-sided, at the preregistered level in
   the canonical specification). If it does not win, ``no_predictive_law``.

Exactly one hypothesis is tested on the later period, and nothing about that
period entered its choice, so the test's level is the rate at which a law is
reported where there is none. (The path's own confirmation step reads up to 30
later rows for the ``ungated`` reading; that reading does not feed the choice.)

The test is the referee's; this module decides nothing by itself. A test
consumes the later period: it cannot be reused for the same question.
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
PREDICTIVE_LAW = "predictive_law"
FIT_SHARE = 2.0 / 3.0  # of the development period, in time order; the rest scores the candidates


@dataclass
class SeriesResult:
    action: str                    #: ``predictive_law`` or ``no_predictive_law``
    law: str | None                #: the chosen law, refitted on all development rows, if it passed the forecast test
    selected: dict[str, Any]       #: the candidate chosen on the development rows and every candidate's score
    forecast_test: dict[str, Any]  #: empty when no candidate earned a test
    ungated_action: str            #: what the path said on its own
    ungated_law: str | None        #: the law the path returned, refitted, whether or not anything else agrees
    path: main_search.MainSearchResult


def refitted(law: str | sp.Expr, names: list[str], dev_X: np.ndarray, dev_y: np.ndarray) -> sp.Expr:
    """The law with one coefficient per additive term, fitted on the rows given."""
    expr = sp.sympify(law, locals={n: sp.Symbol(n) for n in names})
    terms = [term.as_coeff_Mul()[1] for term in sp.Add.make_args(sp.expand(expr))]
    fitted = contrast_resolution.fitted_candidate(expr, names, dev_X, dev_y, "absolute")
    return sp.Add(*[sp.Float(float(c)) * term for c, term in zip(fitted.coef, terms, strict=True)])


def predict(law: sp.Expr, names: list[str], X: np.ndarray) -> np.ndarray:
    fn = sp.lambdify([sp.Symbol(n) for n in names], law, ["scipy", "numpy"], dummify=True)
    with np.errstate(all="ignore"):
        out: np.ndarray = np.asarray(fn(*np.asarray(X, dtype=float).T), dtype=float) * np.ones(len(X))
    return out


def select_on_development(formulas: list[str], names: list[str], dev_X: np.ndarray, dev_y: np.ndarray) -> dict[str, Any]:
    """Score every candidate on the last third of the development period; the best one, if it beats the mean."""
    cut = round(FIT_SHARE * len(dev_y))
    fit_X, fit_y, val_X, val_y = dev_X[:cut], dev_y[:cut], dev_X[cut:], dev_y[cut:]
    baseline = float(np.mean((val_y - fit_y.mean()) ** 2))
    scores: list[dict[str, Any]] = []
    for formula in formulas:
        try:
            law = refitted(formula, names, fit_X, fit_y)
            forecast = predict(law, names, val_X)
        except (ValueError, TypeError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError, sp.SympifyError):
            continue
        if not law.free_symbols or not np.all(np.isfinite(forecast)):
            continue
        scores.append({"formula": formula, "validation_mse": float(np.mean((val_y - forecast) ** 2)),
                       "size": int(sp.count_ops(law)) + len(law.atoms(sp.Float))})
    scores.sort(key=lambda s: (s["validation_mse"], s["size"]))
    chosen = scores[0]["formula"] if scores and scores[0]["validation_mse"] < baseline else None
    return {"chosen": chosen, "no_law_validation_mse": baseline, "candidates": scores, "fit_rows": cut, "validation_rows": len(val_y)}


def discover_predictive_law(X: Mapping[str, np.ndarray], y: np.ndarray, later_X: np.ndarray, later_y: np.ndarray, *,
                            proposer: ProposerBase | None = None, seed: int = 0, **options: Any) -> SeriesResult:
    """``(X, y)`` are the development rows in time order; ``later_X, later_y`` the period after them, columns as in ``X``."""
    names = list(X)
    dev_X = np.column_stack([np.asarray(X[n], dtype=float) for n in names])
    dev_y = np.asarray(y, dtype=float)
    later_X, later_y = np.asarray(later_X, dtype=float), np.asarray(later_y, dtype=float)
    options.setdefault("always_search", True)
    path = main_search.discover(X, dev_y, later_X, lambda rows: later_y[rows], proposer=proposer, noise="absolute",
                                seed=seed, **options)
    ungated = str(refitted(path.law, names, dev_X, dev_y)) if path.law else None
    selected = select_on_development([c["formula"] for c in path.candidates], names, dev_X, dev_y)
    if selected["chosen"] is None:
        return SeriesResult(NO_PREDICTIVE_LAW, None, selected, {}, path.action, ungated, path)
    law = refitted(selected["chosen"], names, dev_X, dev_y)
    forecast = predict(law, names, later_X)
    if not np.all(np.isfinite(forecast)):
        test: dict[str, Any] = {"passed": False, "detail": "the law is not finite on the later period"}
    else:
        outcome = evaluate_predictive_incrementality(later_y, forecast, np.full(len(later_y), float(dev_y.mean())))
        test = {"passed": bool(outcome.is_significant), "p_value": outcome.p_value, "dm_statistic": outcome.dm_statistic,
                "loss_reduction": outcome.loss_reduction, "alpha": outcome.alpha, "rows": len(later_y)}
    passed = bool(test["passed"])
    return SeriesResult(PREDICTIVE_LAW if passed else NO_PREDICTIVE_LAW, str(law) if passed else None, selected, test,
                        path.action, ungated, path)
