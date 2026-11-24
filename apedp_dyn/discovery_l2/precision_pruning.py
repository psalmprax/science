"""Drop the terms of a confirmed law that are smaller than the data's own precision.

Status: experimental

**Why (audit M-15).** ``refined_search.confirmed_and_pruned`` removes a term
while BIC falls, with the mean square floored at a numerical constant. That
left nothing inaccurate on tables (PREREG-REFINED-SEARCH). On window rows of
trajectories it kept noise (PREREG-ODE-REFINED, 2026-11-20): ``dx/dt = -x**3``
was reported with ``-4.4e-9*exp(exp(x))``, and ``x0*x2 - 10*x1`` with two
terms of size 1e-12 and 1e-16. Measured afterwards: those terms contribute
8e-6 and 2e-4 to 8e-4 on the rows, and the rows' rates carry a standard error
of 8e-6 and 6e-4. The terms were fitting the error of the rate estimates, and
BIC rewarded them because the residual really did fall, below the precision
of the data.

**What.** The same greedy removal, with one change: the mean square cannot
count as lower than the variance the rows are known to carry. ``precision`` is
the standard error of each row's target (for window rows,
``WindowRows.rate_sd``). A term that only lowers the residual beneath that
level no longer pays for itself and is dropped, provided the term itself is
no larger than about the precision (``MAX_PRECISION_MULTIPLES``); a larger
term is left to the original rule's judgement and kept here.

**Limits.** It needs a precision; where none is known the caller should use
the original rule. Developed on the two recorded cases above, which are
held-out ODEBench systems, so those systems cannot test it blind. Not yet
measured in a preregistered run.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import sympy as sp

from apedp_dyn.discovery_l2 import contrast_confirmation as cc
from apedp_dyn.discovery_l2 import main_search, refined_search

#: A term may be dropped only if its root-mean-square contribution on the rows is at most this many
#: times the rows' precision. Without the bound the rule also drops a real term at 1 % noise when
#: the remaining terms can absorb most of it, which cost accuracy in one of six recorded cases
#: (ratios there: 0.2 to 1.5 for the noise terms, 7.7 for the real one). A search setting, chosen
#: on those six cases, not a threshold of the referee.
MAX_PRECISION_MULTIPLES = 2.0


def prune_to_precision(law: str, names: Sequence[str], dev_X: np.ndarray, dev_y: np.ndarray,
                       precision: float | np.ndarray, noise: str = "absolute") -> tuple[sp.Expr, list[str]]:
    """The law with one fitted coefficient per term, minus the terms below the rows' precision.

    ``precision`` is the standard error of the target on each row (a number or
    one per row). Returns the pruned law and the terms dropped, in order.
    """
    dev_X = np.asarray(dev_X, dtype=float)
    dev_y = np.asarray(dev_y, dtype=float)
    sd = np.broadcast_to(np.asarray(precision, dtype=float), dev_y.shape)
    if not np.all(np.isfinite(sd)) or np.any(sd < 0):
        raise ValueError("precision must be finite and not negative")
    terms = refined_search._terms(law, names)
    full = refined_search._fit(terms, names, dev_X, dev_y, noise)
    weight = np.ones(len(dev_y)) if noise == "absolute" else np.maximum(np.abs(full.dev_basis @ full.coef), 1e-300)
    scale = float(np.sqrt(np.mean((dev_y / weight) ** 2)))
    level = float(np.sqrt(np.mean((sd / weight) ** 2)))  # the rows' precision, in the fit's units
    floor = max((main_search.NUMERIC_FLOOR * scale) ** 2, level ** 2)
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
        size = np.sqrt(np.mean((fit.dev_basis * fit.coef / weight[:, None]) ** 2, axis=0))
        small = [k for k in range(len(kept)) if size[k] <= MAX_PRECISION_MULTIPLES * level]
        trials = [(bic(candidate, len(kept) - 1), k, candidate) for k in small
                  for candidate in [refined_search._fit(kept[:k] + kept[k + 1:], names, dev_X, dev_y, noise)]]
        if not trials:
            break
        best_bic, index, best_fit = min(trials, key=lambda t: (t[0], t[1]))
        if best_bic >= current:
            break
        dropped.append(str(kept.pop(index)))
        fit, current = best_fit, best_bic
    return sp.Add(*[sp.Float(float(c)) * term for c, term in zip(fit.coef, kept, strict=True)]), dropped
