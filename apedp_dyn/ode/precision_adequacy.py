"""A law refused for structure in a residual smaller than the rows' own error is not refused.

Status: experimental

**What was found (2026-11-21, on the 98 systems of PREREG-ODE-BLIND after its
run, so development and not a result).** The path's adequacy check passes a
law only if its residuals show no structure at all among neighbouring rows.
That is right for independent rows of a table. Window rows of a trajectory
are not that: each carries an estimated state and an estimated rate, their
errors are related, and without noise the error of a rate is the systematic
error of the window fit. Applied to the *true* law of each of the 310
components, the check fails it in 53 at 1 % noise. Without noise 173 of the
fits are exact to numerical precision, which the path accepts without this
check; of the other 137 the check fails 93. Of the 28 runs of that test in
which the route refused a law at this check, the law refused was the true one
in 20.

**What this does.** ``judged_at_precision`` looks at a result the path
refused at the adequacy check. If the refused law's residual on the
development rows is no larger than ``MULTIPLES`` times the rows' own standard
error of a rate, the structure in it is beneath what the rows can resolve and
is no evidence against the law: the law is reinstated, in the confirmed and
pruned form the refined path reports. ``MULTIPLES`` is the constant
``precision_pruning`` already uses for the same judgement (a term that small
is dropped), so no new setting is introduced.

**What it costs, on those systems.** Of the 22 laws it reinstates at 1 %
noise, 17 are the true law, 3 are accurate and not the law, and 2 are wrong.
The check it overrides was right about those two.

**Limits.** It applies where a refusal names a law; a run in which no
candidate survived confirmation is untouched. Without noise the rows' stated
error understates the systematic error of the rates, so few laws are
reinstated there. Not measured on systems it was not developed on.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import sympy as sp

from apedp_dyn.discovery_l2 import contrast_resolution as cr
from apedp_dyn.discovery_l2 import refined_search
from apedp_dyn.discovery_l2.precision_pruning import MAX_PRECISION_MULTIPLES
from apedp_dyn.ode.main_search_ode import state_names
from apedp_dyn.ode.window_rows import WindowRows

MULTIPLES = MAX_PRECISION_MULTIPLES
REFUSED = "widen_search"


def residual_in_precisions(law: str, dev: WindowRows, component: int) -> float:
    """The law's root-mean-square residual on the development rows, in units of the rows' error of a rate."""
    names = state_names(dev.states.shape[1])
    expr = sp.sympify(law, locals={n: sp.Symbol(n) for n in names})
    fit = cr.fitted_candidate(expr, names, dev.states, dev.rates[:, component], "absolute")
    precision = float(dev.rate_sd[component])
    rms = float(np.sqrt(np.mean(np.asarray(fit.dev_resid, dtype=float) ** 2)))
    return rms / precision if precision > 0 else float("inf")


def judged_at_precision(result: refined_search.RefinedResult, dev: WindowRows, component: int,
                        multiples: float = MULTIPLES) -> tuple[refined_search.RefinedResult, dict[str, Any]]:
    """``result``, with a law the adequacy check refused reinstated if its residual is within the rows' precision.

    Returns the result (unchanged unless reinstated) and what was done. A result that reports a law,
    or that was refused without naming one, comes back as it was.
    """
    path = result.path
    info: dict[str, Any] = {"reinstated": False}
    if path is None or path.action != REFUSED or not path.rejected_law or path.contrast is None:
        return result, info
    try:
        ratio = residual_in_precisions(path.rejected_law, dev, component)
    except Exception as exc:  # noqa: BLE001 - a refused law that cannot be fitted stays refused, with the reason
        return result, {**info, "error": f"{type(exc).__name__}: {exc}"[:200]}
    info.update({"refused_law": path.rejected_law, "residual_in_precisions": ratio, "limit": multiples,
                 "adequacy": dict(path.adequacy)})
    if not np.isfinite(ratio) or ratio > multiples:
        return result, info
    names = state_names(dev.states.shape[1])
    law, dropped = refined_search.confirmed_and_pruned(path.rejected_law, names, dev.states, dev.rates[:, component],
                                                       "absolute")
    action = cr.ACTIONS[path.contrast.verdict]
    if action not in refined_search.REPORTED:           # nothing had challenged it: weaker, and kept apart
        return replace(result, action=action, tentative_law=str(law), dropped_terms=dropped), {**info, "reinstated": True}
    return replace(result, action=action, law=str(law), dropped_terms=dropped), {**info, "reinstated": True}
