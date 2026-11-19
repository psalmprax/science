"""One entry point for discovering a law from tabular data, and the switch between the two routes.

Status: experimental

Two routes exist:

- ``main_search``: the exhaustive first pass, PySR, contrast confirmation and the
  adequacy check (:mod:`apedp_dyn.discovery_l2.main_search`).
- ``l2``: the template search's ladder (``contrast_resolution.climb``). When its
  rungs are exhausted and a proposer is supplied, the widening step is now the
  main search instead of a refusal, so the ladder no longer ends inside the
  template engine.

``DEFAULT_ROUTE`` is ``l2`` on purpose. The main search path was built on
evidence about its parts; whether what it reports can be relied on is what
PREREG-MAIN-SEARCH-2026-11 measures. The default changes to ``main_search``
only if that preregistration's expectations 1 to 3 hold, and the change is a
recorded edit here, not a setting someone can forget. Until then a caller
chooses the route explicitly, or a deployment sets
``APEDP_LAW_DISCOVERY_ROUTE``.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from apedp_dyn.discovery_l2 import contrast_resolution, main_search
from apedp_dyn.grammar.proposer_interface import ProposerBase

ROUTES = ("main_search", "l2")
DEFAULT_ROUTE = "l2"
ENVIRONMENT_VARIABLE = "APEDP_LAW_DISCOVERY_ROUTE"


def default_route() -> str:
    """The route in force: the environment's choice if it is a known route, else ``DEFAULT_ROUTE``."""
    chosen = os.environ.get(ENVIRONMENT_VARIABLE, DEFAULT_ROUTE)
    if chosen not in ROUTES:
        raise ValueError(f"{ENVIRONMENT_VARIABLE} must be one of {ROUTES}, not {chosen!r}")
    return chosen


def discover_law(
    X: Mapping[str, np.ndarray], Y: np.ndarray, pool_X: np.ndarray, measure: Callable[[np.ndarray], np.ndarray], *,
    ledger: Any, search_id: str, route: str | None = None, proposer: ProposerBase | None = None,
    dimensions: Mapping[str, Any] | None = None, target_dimension: Any = None, noise: str = "relative",
    seed: int = 0, domain: str | None = None, dataset_id: str | None = None,
) -> dict[str, Any]:
    """Discover a law by the chosen route. Returns the route taken, the action, the law and the route's own record.

    ``action`` is one of ``confirmed``, ``equivalent_class``, ``unchallenged`` or
    ``widen_search`` on either route. The ``l2`` route needs ``dimensions`` and
    ``target_dimension``; the main search does not use them.
    """
    route = route or default_route()
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, not {route!r}")
    if route == "l2":
        if dimensions is None or target_dimension is None:
            raise ValueError("the l2 route needs dimensions and target_dimension")
        rounds = contrast_resolution.climb(X, Y, dimensions, target_dimension, pool_X, measure, ledger=ledger,
                                           search_id=search_id, noise=noise, seed=seed, domain=domain, dataset_id=dataset_id)
        last = rounds[-1] if rounds else {"action": "widen_search", "law": None}
        if last["action"] != "widen_search" or proposer is None:
            return {"route": "l2", "action": last["action"], "law": last["law"], "rounds": rounds, "main_search": None}
    else:
        rounds = []
    result = main_search.discover(X, Y, pool_X, measure, proposer=proposer, noise=noise, seed=seed, ledger=ledger,
                                  search_id=search_id, domain=domain)
    return {"route": "main_search" if route == "main_search" else "l2+main_search", "action": result.action,
            "law": result.law, "rounds": rounds, "main_search": result}
