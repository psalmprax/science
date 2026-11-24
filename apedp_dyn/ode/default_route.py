"""The route in force for the law of one component of a dynamical system. Since 2026-11-21: the repaired refined path.

Status: experimental

**Why this module exists.** PREREG-ODE-BLIND-2026-11 fixed, before its run, that
if its first three expectations held the repaired route "becomes the default
ODE route, in a new module around the frozen ones". They held. ``refined_ode``,
``main_search_ode``, ``precision_pruning``, ``fit_guard`` and ``portfolio`` are
frozen by that preregistration's digest and are not edited; the route is
composed here exactly as the frozen runner composed it
(``scripts/run_ode_blind.py``, ``run_repaired``), and a test holds the two to
the same answer. Code that wants "the ODE route" calls this module.

**The evidence, both runs, as the preregistration requires them to be stated
together.**

* PREREG-ODE-BLIND-2026-11 (run once, 2026-11-21): 98 chaotic polynomial
  systems from the ``dysts`` database, 310 components, that no route here was
  built on, run on or looked at. Of the 484 laws the route reported, 3 were
  wrong (0.6 %, where 5 % was allowed). It recovered 199 and 210 of the 310
  components at noise 0 and 1 %, the recorded route 132 and 201. Pruning lost
  no recovery and made 24.
* PREREG-ODE-REFINED-2026-11 (run once, 2026-11-20): ODEBench, 63 systems, the
  same route without the fit guard and the pruning. 10 of the 195 laws it
  reported were wrong, 5.1 % where 5 % was allowed, and it was not adopted.
  That result stands as recorded.

**What the blind run does not license.** Its systems are chaotic, polynomial
and have three or four variables; it says nothing about rational or
transcendental right-hand sides, where ODEBench's wrong laws were. The route
ran four searches on four cores for the wall time of the recorded route's one.
And one of its recoveries, by the benchmark's criterion, is a law that is
inaccurate as reported (audit M-26).

**What the route is.** ``refined_ode.discover_component`` with four PySR
searches side by side behind the fit guard, then
``precision_pruning.prune_to_precision`` with the rows' own rate error: a term
smaller than the error of the rate estimates is dropped. The law before
pruning is kept in the result. ``APEDP_ODE_ROUTE=main_search`` restores the
recorded route (one search, no pruning).

**Without a search stage nothing is reported**, as in ``refined_ode``: a law
found by the first passes alone is returned as ``tentative_law``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from apedp_dyn.discovery_l2.precision_pruning import prune_to_precision
from apedp_dyn.grammar.proposer_interface import ProposerBase
from apedp_dyn.ode import main_search_ode, refined_ode
from apedp_dyn.ode.window_rows import WindowRows

REPAIRED = "repaired"
RECORDED = "main_search"
ROUTES = (REPAIRED, RECORDED)
DEFAULT_ROUTE = REPAIRED
ENVIRONMENT_VARIABLE = "APEDP_ODE_ROUTE"
EVIDENCE = "preregistration/RESULT-ODE-BLIND-2026-11.md"
EARLIER_EVIDENCE = "preregistration/RESULT-ODE-REFINED-2026-11.md"

#: as run in PREREG-ODE-BLIND-2026-11
MEMBERS = 4
SEARCH_BUDGET_S = 240.0
MAX_PROPOSALS = 10
MEMBER_GRACE_S = 300.0

_UNSET: Any = object()


@dataclass
class ComponentLaw:
    """What the route says about one component."""

    route: str
    action: str                             #: the path's action
    law: str | None                         #: the reported law (pruned, on the repaired route), or ``None``
    tentative_law: str | None               #: a law that nothing challenged, or found with no search stage
    unpruned_law: str | None = None         #: the reported law before pruning to the rows' precision
    precision_dropped: list[str] = field(default_factory=list)   #: terms below the rows' precision
    prune_error: str | None = None          #: why the law is reported unpruned, if the pruning could not handle it
    fit_guard_dropped: list[list[str]] = field(default_factory=list)  #: proposals the path could not have fitted
    result: Any = None                      #: the underlying ``RefinedResult`` or ``MainSearchResult``


def default_route() -> str:
    """The route in force: the environment's choice if it is a known route, else ``DEFAULT_ROUTE``."""
    chosen = os.environ.get(ENVIRONMENT_VARIABLE, DEFAULT_ROUTE)
    if chosen not in ROUTES:
        raise ValueError(f"{ENVIRONMENT_VARIABLE} must be one of {ROUTES}, not {chosen!r}")
    return chosen


def search_stage(budget_seconds: float = SEARCH_BUDGET_S, seed: int = 0, members: int = MEMBERS) -> ProposerBase | None:
    """Four PySR searches side by side behind the fit guard, as measured; ``None`` without a budget."""
    if budget_seconds <= 0:
        return None
    from apedp_dyn.proposers.fit_guard import FitGuardProposer
    from apedp_dyn.proposers.portfolio import PortfolioProposer, pysr_members

    stage: ProposerBase = FitGuardProposer(PortfolioProposer(
        pysr_members(budget_seconds, tuple(seed + k for k in range(members))),
        max_proposals=MAX_PROPOSALS, parallel=True, timeout=budget_seconds + MEMBER_GRACE_S))
    return stage


def discover_component(dev: WindowRows, pool: WindowRows, component: int, *, budget_seconds: float = SEARCH_BUDGET_S,
                       seed: int = 0, proposer: Any = _UNSET, route_name: str | None = None) -> ComponentLaw:
    """The law of ``d x_component / dt`` by the route in force.

    ``proposer`` replaces the search stage (``None`` runs none, and then nothing is reported).
    On the recorded route the stage is one PySR search, nothing is pruned, and a law that nothing
    challenged is reported as well as named tentative: that route is handed on as it was measured.
    """
    chosen = route_name or default_route()
    if chosen not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, not {chosen!r}")
    names = main_search_ode.state_names(dev.states.shape[1])
    if chosen == RECORDED:
        if proposer is _UNSET:
            from apedp_dyn.proposers.pysr_proposer import PySRProposer

            proposer = PySRProposer(budget_seconds=budget_seconds, seed=seed) if budget_seconds > 0 else None
        found = main_search_ode.discover_component(dev, pool, component, proposer=proposer, seed=seed, always_search=True)
        # handed on as it was measured: the law in its confirmed form, reported also when nothing challenged it
        confirmed = str(main_search_ode.confirmed_form(found.law, names, dev.states, dev.rates[:, component])) \
            if found.law else None
        return ComponentLaw(RECORDED, found.action, confirmed, confirmed if found.action == "unchallenged" else None,
                            result=found)
    stage = search_stage(budget_seconds, seed) if proposer is _UNSET else proposer
    result = refined_ode.discover_component(dev, pool, component, proposer=stage, seed=seed)
    return pruned_to_precision(result, dev, component, stage)


def pruned_to_precision(result: Any, dev: WindowRows, component: int, stage: Any = None) -> ComponentLaw:
    """A refined-path result as the route reports it: its law pruned to the rows' precision, the unpruned law kept."""
    names = main_search_ode.state_names(dev.states.shape[1])
    law, pruned, dropped, prune_error = result.law, result.law, [], None
    if law:
        try:
            expression, terms = prune_to_precision(law, names, dev.states, dev.rates[:, component],
                                                   float(dev.rate_sd[component]), noise="absolute")
            pruned, dropped = str(expression), [str(t) for t in terms]
        except Exception as exc:  # noqa: BLE001 - a law the pruning cannot handle is reported unpruned, with the reason
            prune_error = f"{type(exc).__name__}: {exc}"[:200]
    return ComponentLaw(REPAIRED, result.action, pruned, result.tentative_law, unpruned_law=law, precision_dropped=dropped,
                        prune_error=prune_error, fit_guard_dropped=[list(d) for d in getattr(stage, "dropped", [])],
                        result=result)


def discover_system(dev: WindowRows, pool: WindowRows, *, budget_seconds: float = SEARCH_BUDGET_S, seed: int = 0,
                    route_name: str | None = None) -> list[ComponentLaw]:
    """One ``ComponentLaw`` per component, in order; a search stage is built for each (a search keeps state)."""
    return [discover_component(dev, pool, k, budget_seconds=budget_seconds, seed=seed, route_name=route_name)
            for k in range(dev.states.shape[1])]
