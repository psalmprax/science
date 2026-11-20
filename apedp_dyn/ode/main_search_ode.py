"""Systems of ordinary differential equations through the main search path.

Status: experimental

Each component of ``dx/dt = f(x)`` is one regression problem: inputs are the
state, the target is that variable's rate of change. The rows come from
``window_rows`` (one state and one rate per disjoint window, so row errors are
independent), and each component goes through
``apedp_dyn.discovery_l2.main_search.discover`` unchanged: first pass, search,
contrast confirmation, adequacy.

The development rows come from one trajectory and the pool from another. The
pool is where contrast confirmation reads its targets, so a law is confirmed on
states the search never saw, reached from a different initial condition. This
is the "second trajectory" that audit M-07 named as the missing selection step
of the older ODE path.

Nothing here integrates a candidate system forward. A component that passes is
a statement about the rate of change on the states visited; long-horizon
behaviour of the assembled system is not checked.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import sympy as sp

from apedp_dyn.discovery_l2 import contrast_resolution, main_search
from apedp_dyn.grammar.proposer_interface import ProposerBase
from apedp_dyn.ode.window_rows import WindowRows


def state_names(n: int) -> list[str]:
    return [f"x{i}" for i in range(n)]


def confirmed_form(law: str, names: list[str], dev_X: np.ndarray, dev_y: np.ndarray) -> sp.Expr:
    """The law as the path confirmed it: one coefficient per additive term, fitted on the development rows.

    ``MainSearchResult.law`` is the text a search returned. Confirmation and the
    adequacy check are made on a refit of that text with a free coefficient on
    every additive term (``contrast_resolution.fitted_candidate``), which can
    differ from the text: the first pass fitted ``a*(x0 + x0**3) + b*(x0 + x1)``
    and what was confirmed exact was ``-0.8*x0**3 - 1.2*x0 - 0.3*x1``. This
    returns the refit, so that what is reported is what was checked.
    """
    expr = sp.sympify(law, locals={n: sp.Symbol(n) for n in names})
    terms = [term.as_coeff_Mul()[1] for term in sp.Add.make_args(sp.expand(expr))]
    fitted = contrast_resolution.fitted_candidate(expr, names, dev_X, dev_y, "absolute")
    return sp.Add(*[sp.Float(float(c)) * term for c, term in zip(fitted.coef, terms, strict=True)])


def discover_component(dev: WindowRows, pool: WindowRows, component: int, *, pool_rows: np.ndarray | None = None,
                       proposer: ProposerBase | None = None, seed: int = 0, **options: Any) -> main_search.MainSearchResult:
    """The law of ``d x_component / dt``; ``pool_rows`` restricts the pool (the rest can be kept for testing)."""
    names = state_names(dev.states.shape[1])
    rows = np.arange(len(pool.states)) if pool_rows is None else np.asarray(pool_rows, dtype=int)
    pool_rates = pool.rates[rows, component]
    return main_search.discover(
        {n: dev.states[:, i] for i, n in enumerate(names)}, dev.rates[:, component], pool.states[rows],
        lambda picked: pool_rates[picked], proposer=proposer, noise="absolute", seed=seed, **options)


def discover_system(dev: WindowRows, pool: WindowRows, *, pool_rows: np.ndarray | None = None,
                    proposer_factory: Callable[[], ProposerBase | None] | None = None, seed: int = 0,
                    **options: Any) -> list[main_search.MainSearchResult]:
    """One result per component, in order. A proposer is built per component (a search keeps state)."""
    return [discover_component(dev, pool, k, pool_rows=pool_rows, proposer=proposer_factory() if proposer_factory else None,
                               seed=seed, **options) for k in range(dev.states.shape[1])]
