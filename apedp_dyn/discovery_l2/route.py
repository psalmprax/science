"""One entry point for discovering a law from tabular data, and the switch between the two routes.

Status: experimental

Two routes exist:

- ``main_search``: the exhaustive first pass, PySR, contrast confirmation and the
  adequacy check (:mod:`apedp_dyn.discovery_l2.main_search`).
- ``l2``: the template search's ladder (``contrast_resolution.climb``). When its
  rungs are exhausted and a proposer is supplied, the widening step is now the
  main search instead of a refusal, so the ladder no longer ends inside the
  template engine.

``DEFAULT_ROUTE`` is ``main_search`` since 2026-11-19. It was ``l2`` until
PREREG-MAIN-SEARCH-2026-11 had been run, and that preregistration fixed the
condition for the change: its expectations 1 to 3. They held (SRBench, 266
runs: 4 of the 186 laws the path reported were not accurate, 2.2 %; 152 were
the law; 69 and 70 of 119 Feynman laws recovered against 73 and 68 for PySR
alone; ``preregistration/RESULT-MAIN-SEARCH-2026-11.md``). A deployment can
still set ``APEDP_LAW_DISCOVERY_ROUTE=l2``.

``law`` in the returned record is the text the search produced.
``law_as_confirmed`` is the same law with one coefficient per additive term
refitted on the development rows, which is the form contrast confirmation and
the adequacy check were applied to. They are usually the same; in the benchmark
they differed materially in 2 of 186 reported laws (a search returned
``x0*(x2 + x3*x4/x1)**2``, and what was confirmed exact was its cross term
alone). Use ``law_as_confirmed`` for prediction.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import sympy as sp

from apedp_dyn.discovery_l2 import contrast_resolution, main_search
from apedp_dyn.grammar.proposer_interface import ProposerBase

ROUTES = ("main_search", "l2")
DEFAULT_ROUTE = "main_search"
ENVIRONMENT_VARIABLE = "APEDP_LAW_DISCOVERY_ROUTE"


def default_route() -> str:
    """The route in force: the environment's choice if it is a known route, else ``DEFAULT_ROUTE``."""
    chosen = os.environ.get(ENVIRONMENT_VARIABLE, DEFAULT_ROUTE)
    if chosen not in ROUTES:
        raise ValueError(f"{ENVIRONMENT_VARIABLE} must be one of {ROUTES}, not {chosen!r}")
    return chosen


def as_confirmed(law: str, X: Mapping[str, np.ndarray], Y: np.ndarray, noise: str = "relative") -> str:
    """``law`` with one coefficient per additive term, fitted on the development rows: the form that was confirmed."""
    names = list(X)
    dev_X = np.column_stack([np.asarray(X[n], dtype=float) for n in names])
    expr = sp.sympify(law, locals={n: sp.Symbol(n) for n in names})
    terms = [term.as_coeff_Mul()[1] for term in sp.Add.make_args(sp.expand(expr))]
    fitted = contrast_resolution.fitted_candidate(expr, names, dev_X, np.asarray(Y, dtype=float), noise)
    return str(sp.Add(*[sp.Float(float(c)) * term for c, term in zip(fitted.coef, terms, strict=True)]))


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
            "law": result.law, "law_as_confirmed": as_confirmed(result.law, X, Y, noise) if result.law else None,
            "rounds": rounds, "main_search": result}
