"""The main search path with what its benchmarks showed it lacks, added around it.

Status: experimental

``main_search.discover`` is frozen by the preregistrations that measured it.
Those measurements also showed four things it does not do, and this module
does them without changing it:

1. **It ignores physical units.** With ``units`` and ``target_units`` the search
   stage is wrapped in ``proposers.composed.UnitsReducingProposer``: the search
   runs on the dimensionless products of the inputs, which are fewer than the
   inputs (113 of the 119 Feynman problems reduce; 26 of them to no variable at
   all), and its formulas come back in the original variables.
2. **It misses polynomials a library contains** (eleven components on
   ODEBench). ``PolynomialLibraryProposer`` adds that one candidate.
3. **It reports the search's text, not the fit it confirmed** (audit M-13), and
   an exact fit can carry a term that contributes nothing (M-15). The law
   returned here is the confirmed form: one coefficient per additive term,
   refitted on the development rows, after dropping every term whose removal
   the data do not notice (backward elimination by BIC, with fits below the
   numerical floor counted as equally good).
4. **An unchallenged law was reported like a confirmed one** (M-14; the two
   wrong laws of PREREG-MAIN-SEARCH that were wrong in substance were both
   unchallenged). The search stage is run on every problem, so rivals exist;
   and if a law still comes back ``unchallenged`` it is returned as
   ``tentative_law``, not as ``law``.

What is still the frozen path's: the first pass, constant polishing, contrast
confirmation, the adequacy check, and every verdict. This module only chooses
what is searched and how the outcome is written down.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from apedp_dyn.discovery_l2 import contrast_confirmation as cc
from apedp_dyn.discovery_l2 import main_search
from apedp_dyn.grammar.proposer_interface import ProposerBase
from apedp_dyn.proposers.composed import (
    CompositeProposer,
    PolynomialLibraryProposer,
    Units,
    UnitsReducingProposer,
)

REPORTED = ("confirmed", "equivalent_class")


@dataclass
class RefinedResult:
    action: str                     #: the path's action
    law: str | None                 #: reported law, in the confirmed and pruned form; ``None`` unless confirmed against rivals
    tentative_law: str | None       #: a law nothing challenged: weaker evidence, kept apart
    dropped_terms: list[str] = field(default_factory=list)
    units: dict[str, Any] = field(default_factory=dict)       #: what the units wrapper did, if it ran
    proposer_failures: list[str] = field(default_factory=list)
    path: main_search.MainSearchResult | None = None


def _terms(law: str | sp.Expr, names: Sequence[str]) -> list[sp.Expr]:
    expr = sp.sympify(law, locals={n: sp.Symbol(n) for n in names})
    return [term.as_coeff_Mul()[1] for term in sp.Add.make_args(sp.expand(expr))]


def _fit(terms: Sequence[sp.Expr], names: Sequence[str], dev_X: np.ndarray, dev_y: np.ndarray, noise: str) -> cc.FittedCandidate:
    symbols = [sp.Symbol(n) for n in names]
    fns = [sp.lambdify(symbols, term, ["scipy", "numpy"]) for term in terms]

    def basis(X: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            out: np.ndarray = np.column_stack([np.asarray(f(*X.T), dtype=float) * np.ones(len(X)) for f in fns])
        return out

    return cc.fit_candidate("+".join(map(str, terms)), basis, dev_X, dev_y, noise)


def confirmed_and_pruned(law: str, names: Sequence[str], dev_X: np.ndarray, dev_y: np.ndarray,
                         noise: str = "relative") -> tuple[sp.Expr, list[str]]:
    """The law with one fitted coefficient per additive term, minus the terms the data do not notice.

    Terms are removed one at a time while BIC falls. Residuals are in units of
    the noise model, and a mean square below the numerical floor of an exact
    fit counts as the floor, so that a term of size 1e-9 on exact data is
    dropped and a real term never is.
    """
    terms = _terms(law, names)
    full = _fit(terms, names, dev_X, dev_y, noise)
    weight = np.ones(len(dev_y)) if noise == "absolute" else np.maximum(np.abs(full.dev_basis @ full.coef), 1e-300)
    scale = float(np.sqrt(np.mean((dev_y / weight) ** 2)))
    floor = (main_search.NUMERIC_FLOOR * scale) ** 2
    n = len(dev_y)

    def bic(fit: cc.FittedCandidate, count: int) -> float:
        resid = (dev_y - fit.dev_basis @ fit.coef) / weight
        if not np.all(np.isfinite(resid)):
            return float("inf")
        return n * float(np.log(max(float(np.mean(resid ** 2)), floor, 1e-300))) + count * float(np.log(n))

    kept, fit = list(terms), full
    current = bic(fit, len(kept))
    dropped: list[str] = []
    while len(kept) > 1:
        trials = [(bic(candidate, len(kept) - 1), k, candidate)
                  for k in range(len(kept)) for candidate in [_fit(kept[:k] + kept[k + 1:], names, dev_X, dev_y, noise)]]
        best_bic, index, best_fit = min(trials, key=lambda t: (t[0], t[1]))
        if best_bic >= current:
            break
        dropped.append(str(kept.pop(index)))
        fit, current = best_fit, best_bic
    return sp.Add(*[sp.Float(float(c)) * term for c, term in zip(fit.coef, kept, strict=True)]), dropped


def discover_refined(
    X: Mapping[str, np.ndarray], y: np.ndarray, pool_X: np.ndarray, measure: Callable[[np.ndarray], np.ndarray], *,
    proposer: ProposerBase | None = None, units: Units | None = None, target_units: Mapping[str, Any] | None = None,
    polynomial_library: bool = True, noise: str = "relative", seed: int = 0, **options: Any,
) -> RefinedResult:
    """Run the frozen path with a composed search stage and write the outcome down in the confirmed form.

    ``proposer`` is the search (PySR, or ``None`` for the first passes alone).
    ``units`` maps each input to its exponents over the base units and
    ``target_units`` does the same for the target; without them nothing is
    reduced.
    """
    names = list(X)
    dev_X = np.column_stack([np.asarray(X[n], dtype=float) for n in names])
    dev_y = np.asarray(y, dtype=float)
    reducing = UnitsReducingProposer(proposer, units, target_units or {}, target_col="__target__") if units is not None else None
    members: list[ProposerBase] = [reducing if reducing is not None else proposer] if (reducing or proposer) is not None else []
    if polynomial_library:
        members.append(PolynomialLibraryProposer(target_col="__target__"))
    composite = CompositeProposer(members)
    options.setdefault("always_search", True)
    path = main_search.discover(X, dev_y, pool_X, measure, proposer=composite, noise=noise, seed=seed, **options)
    result = RefinedResult(path.action, None, None, units=dict(reducing.last_run) if reducing is not None else {},
                           proposer_failures=list(composite.failures), path=path)
    if path.law is None:
        return result
    law, result.dropped_terms = confirmed_and_pruned(path.law, names, dev_X, dev_y, noise)
    if path.action in REPORTED:
        result.law = str(law)
    else:  # unchallenged: nothing was there to contradict it
        result.tentative_law = str(law)
    return result
