"""Systems of ordinary differential equations through the refined path.

Status: experimental

``ode/main_search_ode.py`` sends each component of ``dx/dt = f(x)`` through the
main search (PREREG-ODE-MAIN-SEARCH-2026-11: 80 and 69 of ODEBench's 117
components recovered at noise 0 and 1 %, 9 of 184 reported laws wrong). That
module is frozen by its preregistration's digest. This one sends the same rows
through ``discovery_l2.refined_search.discover_refined`` instead, which differs
in three ways that matter for vector fields:

* **A polynomial library is one more candidate.** Most of ODEBench's vector
  fields are polynomial. The recorded run's reference, a sparse polynomial
  regression, recovered 78 components without noise, and not the same ones as
  the main search; it also reported a wrong law in 29 % of its runs, because
  nothing checked it. Here its fit is a candidate like any other and has to
  survive contrast confirmation on a second trajectory.
* **The law is written in its confirmed form with unnoticed terms dropped.**
  The recorded run reported ``-x0**3 + 1.7e-8*x0 - 4.4e-9*exp(exp(x0))`` for
  ``dx/dt = -x**3`` (audit M-15): exact on the confirmed states and unbounded
  outside them.
* **A law that nothing challenged is not reported.** The recorded run counted
  ``unchallenged`` laws as reported; here they are returned as tentative.

The rows are the same: one state and one rate per disjoint window
(``window_rows``), development rows from two trajectories, the pool from a
third. ODEBench's states carry no units, so the units reduction does not run.
Nothing here integrates a candidate system forward.

**Without a search stage nothing is reported** (audit M-19). With no proposer
the only rivals are the first pass and the polynomial library. On the 15
development systems (31 components) that was enough without noise: 24 laws
found, all 24 the law, about four seconds each, against 24 for the main search
with PySR. It was not enough at 1 % noise: 23 laws found, 16 the law, and 4
not accurate, one of them a cubic in the wrong variable with test R² below
-18 000. So with no proposer the law comes back as ``tentative_law``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from apedp_dyn.discovery_l2 import refined_search
from apedp_dyn.grammar.proposer_interface import ProposerBase
from apedp_dyn.ode.main_search_ode import state_names
from apedp_dyn.ode.window_rows import WindowRows


def discover_component(dev: WindowRows, pool: WindowRows, component: int, *, pool_rows: np.ndarray | None = None,
                       proposer: ProposerBase | None = None, seed: int = 0,
                       **options: Any) -> refined_search.RefinedResult:
    """The law of ``d x_component / dt`` by the refined path; ``pool_rows`` restricts the pool.

    ``law`` in the result is set only for a law confirmed against rivals with a
    search stage run; ``tentative_law`` holds one that nothing challenged, or
    any law found with no ``proposer``.
    """
    names = state_names(dev.states.shape[1])
    rows = np.arange(len(pool.states)) if pool_rows is None else np.asarray(pool_rows, dtype=int)
    pool_rates = pool.rates[rows, component]
    result = refined_search.discover_refined(
        {n: dev.states[:, i] for i, n in enumerate(names)}, dev.rates[:, component], pool.states[rows],
        lambda picked: pool_rates[picked], proposer=proposer, noise="absolute", seed=seed, **options)
    if proposer is None and result.law is not None:
        return replace(result, law=None, tentative_law=result.law)
    return result


def discover_system(dev: WindowRows, pool: WindowRows, *, pool_rows: np.ndarray | None = None,
                    proposer_factory: Callable[[], ProposerBase | None] | None = None, seed: int = 0,
                    **options: Any) -> list[refined_search.RefinedResult]:
    """One result per component, in order. A proposer is built per component (a search keeps state)."""
    return [discover_component(dev, pool, k, pool_rows=pool_rows,
                               proposer=proposer_factory() if proposer_factory else None, seed=seed, **options)
            for k in range(dev.states.shape[1])]
