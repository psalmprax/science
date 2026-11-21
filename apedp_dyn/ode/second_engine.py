"""The ODE route with sparse regression as a second source of candidates.

Status: experimental

``ode/default_route.py`` is the route in force: four searches behind the fit
guard, the path's confirmation, pruning to the rows' precision. This module
puts one more member beside the searches, ``proposers/sparse_library``, the
reference engine of the ODE runs turned into a proposer, and changes nothing
else: the same confirmation decides, the same pruning follows.

**Not the default, and not measured blind.** The idea comes from the rows of
PREREG-ODE-BLIND-2026-11, so those systems cannot test it. Replaying that
run's recorded candidates with the sparse candidates added is development; a
test needs systems nobody here has looked at.

**What it costs.** The path confirms at most ten candidates of a stage. Each
sparse candidate takes the place of one of the searches' lower-ranked ones.
"""
from __future__ import annotations

from typing import Any

from apedp_dyn.discovery_l2.evolved_stage import SideBySide
from apedp_dyn.ode import default_route
from apedp_dyn.ode.window_rows import WindowRows
from apedp_dyn.proposers.fit_guard import FitGuardProposer
from apedp_dyn.proposers.sparse_library import SparseLibraryProposer

SPARSE_PROPOSALS = 2
DEGREE = 3


def with_sparse(searches: Any, degree: int = DEGREE, proposals: int = SPARSE_PROPOSALS) -> Any:
    """``searches`` and the sparse polynomial fits side by side, each behind the fit guard."""
    sparse = FitGuardProposer(SparseLibraryProposer(degree=degree, max_proposals=proposals, target_col="__target__"),
                              target_col="__target__")
    return sparse if searches is None else SideBySide(searches, sparse, max_proposals=default_route.MAX_PROPOSALS)


def search_stage(budget_seconds: float = default_route.SEARCH_BUDGET_S, seed: int = 0) -> Any:
    """The default route's four searches with the sparse fits beside them; the sparse fits alone without a budget."""
    return with_sparse(default_route.search_stage(budget_seconds, seed))


def discover_component(dev: WindowRows, pool: WindowRows, component: int, *,
                       budget_seconds: float = default_route.SEARCH_BUDGET_S, seed: int = 0,
                       proposer: Any = None) -> default_route.ComponentLaw:
    """The law of ``d x_component / dt`` by the default route with the second engine beside its searches.

    ``proposer`` replaces the searches (the sparse fits are still added).
    """
    stage = with_sparse(proposer) if proposer is not None else search_stage(budget_seconds, seed)
    return default_route.discover_component(dev, pool, component, budget_seconds=budget_seconds, seed=seed,
                                            proposer=stage, route_name=default_route.REPAIRED)
