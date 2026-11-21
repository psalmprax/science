"""What a search does without a referee: report the candidate that fits best.

Status: experimental

Two ways of ending a search that are common practice, written down so that a
referee can be compared with them on the same worlds and the same candidates:

- ``best_fit``: the constants of every candidate are refitted on the
  development rows and the one with the lowest BIC is reported. Always.
- ``validated``: the same choice, then checked on rows it has not seen: it is
  reported unless its error there is larger than on the development rows by an
  F test at the canonical level. "It held up out of sample."

Neither can say "none of these": the first never refuses, and the second
refuses only a law that fits fresh rows worse than old ones, which an
approximation that fits both equally well does not.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy import stats

from apedp_dyn.conf.canonical_spec import DIEBOLD_MARIANO_P_VALUE_THRESHOLD
from apedp_dyn.discovery_l2 import main_search

ALPHA = DIEBOLD_MARIANO_P_VALUE_THRESHOLD


@dataclass
class Chosen:
    law: str | None
    bic: float | None = None
    constants: int = 0
    dev_mse: float | None = None
    p_value: float | None = None        #: of the check on fresh rows, where one was made


def _values(expr: sp.Expr, names: Sequence[str], X: np.ndarray) -> np.ndarray:
    fn = sp.lambdify([sp.Symbol(n) for n in names], expr, "numpy")
    with np.errstate(all="ignore"):
        out: np.ndarray = np.asarray(fn(*X.T), dtype=complex) * np.ones(len(X))
    real: np.ndarray = np.where(np.abs(out.imag) < 1e-12, out.real, np.nan)
    return real


def best_fit(candidates: Sequence[str], names: Sequence[str], X: np.ndarray, y: np.ndarray) -> Chosen:
    """The candidate with the lowest BIC on the rows, after its constants were refitted there."""
    known = {sp.Symbol(n) for n in names}
    best = Chosen(None)
    n = len(y)
    for text in candidates:
        try:
            expr = sp.sympify(text, locals={str(s): s for s in known})
        except (sp.SympifyError, TypeError, SyntaxError):
            continue
        if not expr.free_symbols or not expr.free_symbols <= known:
            continue
        expr = main_search.polish(expr, list(names), X, y)
        values = _values(expr, names, X)
        if not np.all(np.isfinite(values)):
            continue
        mse = float(np.mean((values - y) ** 2))
        constants = len(expr.atoms(sp.Float))
        floor = (main_search.NUMERIC_FLOOR * float(np.sqrt(np.mean(y ** 2)))) ** 2
        bic = n * float(np.log(max(mse, floor, 1e-300))) + constants * float(np.log(n))
        if best.bic is None or bic < best.bic:
            best = Chosen(str(expr), bic, constants, mse)
    return best


def validated(candidates: Sequence[str], names: Sequence[str], X: np.ndarray, y: np.ndarray, fresh_X: np.ndarray,
              fresh_y: np.ndarray, alpha: float = ALPHA) -> Chosen:
    """``best_fit``, reported only if its error on the fresh rows is not larger than on the rows it was fitted to."""
    chosen = best_fit(candidates, names, X, y)
    if chosen.law is None or chosen.dev_mse is None:
        return chosen
    values = _values(sp.sympify(chosen.law, locals={n: sp.Symbol(n) for n in names}), names, fresh_X)
    if not np.all(np.isfinite(values)):
        return Chosen(None, chosen.bic, chosen.constants, chosen.dev_mse, 0.0)
    fresh_mse = float(np.mean((values - fresh_y) ** 2))
    dof = max(1, len(y) - chosen.constants)
    floor = (main_search.NUMERIC_FLOOR * float(np.sqrt(np.mean(y ** 2)))) ** 2
    ratio = max(fresh_mse, floor) / max(chosen.dev_mse * len(y) / dof, floor)
    chosen.p_value = float(stats.f.sf(ratio, len(fresh_y), dof))
    if chosen.p_value < alpha:
        chosen.law = None
    return chosen
