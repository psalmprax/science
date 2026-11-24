"""The route in force for discovering a law from tabular data. Since 2026-11-20: the refined path.

Status: experimental

**Why this module exists.** ``discovery_l2/route.py`` holds the switch between
the main search and the template ladder, with ``DEFAULT_ROUTE = "main_search"``.
That file is frozen by the digest of PREREG-LLM-PROPOSER-2026-11 and is not
edited. The change of default is made here, around it: this module adds the
third route, names it the default, and hands the other two to ``route.py``
unchanged. Code that wants "the default route" calls this module.

**Why the default moved.** PREREG-REFINED-SEARCH-2026-11 (run once): on SRBench
the refined path reported a law in 212 of 266 runs, none inaccurate, 174 the
law; the main search 186, 4 and 152. Feynman 78 and 77 of 119 against 69 and
70; Strogatz 19 of 28 against 13. On the 103 problems it was not built on it
recovered 61 and 61 against 55 and 52.

**What that run does not license.** One of its three expectations was missed:
61 held-out laws at noise 0 where 62 had been fixed in advance. The Feynman
figures use the problems' physical units, which the main search was not given.
The preregistration fixed no rule for moving the default, so this move is the
repository owner's decision of 2026-11-20 on that evidence, and is recorded as
such. ``APEDP_LAW_DISCOVERY_ROUTE=main_search`` restores the earlier default.

**What the refined path is.** ``discovery_l2/refined_search.discover_refined``:
the frozen main search with a composed search stage (the caller's proposer on
the units' dimensionless products where units are given, plus a polynomial
library), the search stage run on every problem, and the outcome written in
the confirmed form with every term dropped that the data do not notice. A law
that nothing challenged is returned as ``tentative_law`` and not as ``law``.

**Without a search stage nothing is reported.** In the measured path PySR ran
on every problem, so every one of the 212 reported laws had been confronted
with what a real search proposes. With no proposer the only rivals are the
first pass and the polynomial library, and on a few dozen noisy rows a
polynomial that approximates the law gets "confirmed" against them (the UI's
pendulum at 5 % noise: a cubic in the angle for the sine). So when no proposer
is given this module returns the law as ``tentative_law`` with the reason, and
``reported`` is false. The earlier routes are handed on as they were measured:
the main search reported laws its first pass had settled without the search.

**Since 2026-11-22 the route in force for a table is this one with a last
step**: ``discovery_l2/vouched_route.discover_law`` calls ``discover_law`` here
and reports a law of the refined path only if the checks of
``discovery_l2/vouching.py`` let it through (PREREG-COUNTERFEIT-2026-11). This
module is as it was measured.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from apedp_dyn.discovery_l2 import refined_search, route
from apedp_dyn.grammar.proposer_interface import ProposerBase

REFINED = "refined"
NOTHING_CHALLENGED = "nothing challenged it"
NO_SEARCH_STAGE = "the search stage did not run; its only rivals were the first pass and the polynomial library"
ROUTES = (REFINED, *route.ROUTES)
DEFAULT_ROUTE = REFINED
ENVIRONMENT_VARIABLE = route.ENVIRONMENT_VARIABLE
EVIDENCE = "preregistration/RESULT-REFINED-SEARCH-2026-11.md"


def default_route() -> str:
    """The route in force: the environment's choice if it is a known route, else ``DEFAULT_ROUTE``."""
    chosen = os.environ.get(ENVIRONMENT_VARIABLE, DEFAULT_ROUTE)
    if chosen not in ROUTES:
        raise ValueError(f"{ENVIRONMENT_VARIABLE} must be one of {ROUTES}, not {chosen!r}")
    return chosen


def discover_law(
    X: Mapping[str, np.ndarray], Y: np.ndarray, pool_X: np.ndarray, measure: Callable[[np.ndarray], np.ndarray], *,
    ledger: Any = None, search_id: str | None = None, route_name: str | None = None,
    proposer: ProposerBase | None = None, units: Mapping[str, Mapping[str, Any]] | None = None,
    target_units: Mapping[str, Any] | None = None, dimensions: Mapping[str, Any] | None = None,
    target_dimension: Any = None, noise: str = "relative", seed: int = 0, domain: str | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """Discover a law by the chosen route; the refined path unless another is named or set in the environment.

    The record has the keys ``route.discover_law`` returns (``route``,
    ``action``, ``law``, ``law_as_confirmed``, ``main_search``), so a caller of
    that function can move here without other changes. On the refined route:

    * ``law`` is the reported law, in the confirmed and pruned form, or ``None``;
    * ``tentative_law`` is a law that is not reported, with ``tentative_reason``: nothing challenged
      it, or no ``proposer`` was given so the search stage did not run;
    * ``law_as_confirmed`` is whichever of the two there is;
    * ``reported`` says whether ``law`` was confirmed against rivals with the search stage run;
    * ``complete`` says whether the search stage ran;
    * ``refined`` is the ``RefinedResult`` and ``main_search`` the frozen path's own record.

    ``units`` maps each input to its exponents over the base units and
    ``target_units`` does the same for the target; they are used by the refined
    route only. ``dimensions`` and ``target_dimension`` are the ``l2`` route's.
    """
    chosen = route_name or default_route()
    if chosen not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, not {chosen!r}")
    if chosen != REFINED:
        out: dict[str, Any] = route.discover_law(X, Y, pool_X, measure, ledger=ledger, search_id=search_id or "", route=chosen,
                                 proposer=proposer, dimensions=dimensions, target_dimension=target_dimension,
                                 noise=noise, seed=seed, domain=domain, dataset_id=dataset_id)
        tentative = out.get("law_as_confirmed") if out["action"] == "unchallenged" else None
        out.update({"tentative_law": tentative, "tentative_reason": NOTHING_CHALLENGED if tentative else None,
                    "reported": out["action"] in refined_search.REPORTED and bool(out.get("law")),
                    "complete": proposer is not None})
        return out
    result = refined_search.discover_refined(X, np.asarray(Y, dtype=float), pool_X, measure, proposer=proposer,
                                             units=units, target_units=target_units, noise=noise, seed=seed,
                                             ledger=ledger, search_id=search_id, domain=domain)
    complete = proposer is not None
    law = result.law if complete else None
    tentative = result.tentative_law if result.law is None or complete else result.law
    reason = None if tentative is None else NOTHING_CHALLENGED if result.law is None else NO_SEARCH_STAGE
    return {"route": REFINED, "action": result.action, "law": law, "tentative_law": tentative,
            "tentative_reason": reason, "law_as_confirmed": law or tentative, "reported": law is not None,
            "complete": complete, "dropped_terms": list(result.dropped_terms), "units": dict(result.units),
            "proposer_failures": list(result.proposer_failures), "rounds": [], "refined": result,
            "main_search": result.path}
