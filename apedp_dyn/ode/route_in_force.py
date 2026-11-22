"""The route in force for the law of one component of a dynamical system. Since 2026-11-22: with a second engine, and with an experiment where the system can be started again.

Status: experimental

**Why this module exists.** PREREG-ODE-ADDITIONS-2026-11 fixed, before its run,
what would follow from each of its expectations, "in a new module around the
frozen ones". ``default_route``, ``extended_route``, ``second_engine``,
``precision_adequacy`` and ``experiment_pruning`` are frozen by that
preregistration's digest and are not edited. Code that wants "the ODE route"
calls this module.

**The evidence** (``preregistration/RESULT-ODE-ADDITIONS-2026-11.md``, run once,
2026-11-22): 46 systems none of this was built on, 30 invented cubic ones and
16 with rational right-hand sides, 276 components and noise levels.

* The route in force until then (``default_route``) recovered 116 and reported
  164 laws, 4 of them wrong.
* With **the second engine** (sparse regression on a cubic library, as a source
  of candidates beside the four searches): 117, 165 reported, 3 wrong. The
  expectation that the additions cost no recovery held. It is part of the
  route.
* With **the follow-up experiment** (the system restarted sixteen times where
  the reported law and its shorter versions would move differently; terms
  dropped only when the version without them predicts better by an exact
  test): 123 recovered against 118 without it, no more wrong laws. The
  expectation held. It is the default follow-up wherever the caller can start
  the system again (``restart``). All five gains were without noise. Chosen
  restarts beat random ones by two components, and it is not claimed that
  choosing matters.

**What is not in force, and why (a deviation recorded with the result).** The
preregistration fixed that adequacy at the rows' precision would become part of
the route as well. Its expectations held for the arm as a whole; the rows show
that what the addition adds on its own is 23 reported laws of which one is the
law and three are wrong. It is available (``at_precision=True``) and off.

**What this route was and was not measured as.** The second engine followed by
the experiment, without that reinstatement, was not an arm of the run. Its
outcome there is nevertheless known unit by unit, because in every unit the
experiment was either run on the very law this route reports or had no law to
run on: 120 recovered, 165 reported, 3 wrong (1.8 %).

**The gains are small.** One component for the second engine, a handful for
the experiment, on 276. At 1 % noise two of every three laws any of these
routes reports are accurate on the test trajectory and are not the law.

**The experiment needs the system itself**: something that gives the rates at
any state, here a simulation. Most observed systems cannot be started again,
and a restart outside the range seen assumes the law holds there.

``APEDP_ODE_ROUTE=repaired`` or ``main_search`` restores an earlier route as
``default_route`` runs it.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from apedp_dyn.ode import default_route, experiment_pruning, extended_route
from apedp_dyn.ode.window_rows import WindowRows

SECOND_ENGINE = "second_engine"
ROUTES = (SECOND_ENGINE, *default_route.ROUTES)
DEFAULT_ROUTE = SECOND_ENGINE
ENVIRONMENT_VARIABLE = default_route.ENVIRONMENT_VARIABLE
EVIDENCE = "preregistration/RESULT-ODE-ADDITIONS-2026-11.md"
EARLIER_EVIDENCE = default_route.EVIDENCE

_UNSET: Any = object()


@dataclass(frozen=True)
class Restart:
    """A system the caller can start again at a state of its choosing, and how well it is measured."""

    rate: Callable[[np.ndarray], np.ndarray]     #: the rates at a state: the system itself, here a simulation
    level: float                                 #: the noise of a measurement, as a share of each variable's spread
    spread: np.ndarray                           #: each variable's spread


def route_in_force() -> str:
    """The environment's choice if it names a route, else ``DEFAULT_ROUTE``."""
    chosen = os.environ.get(ENVIRONMENT_VARIABLE, DEFAULT_ROUTE)
    if chosen not in ROUTES:
        raise ValueError(f"{ENVIRONMENT_VARIABLE} must be one of {ROUTES}, not {chosen!r}")
    return chosen


def discover_component(dev: WindowRows, pool: WindowRows, component: int, *,
                       budget_seconds: float = default_route.SEARCH_BUDGET_S, seed: int = 0, proposer: Any = _UNSET,
                       restart: Restart | None = None, at_precision: bool = False, route_name: str | None = None
                       ) -> tuple[default_route.ComponentLaw, dict[str, Any]]:
    """The law of ``d x_component / dt`` by the route in force, and what its additions did.

    ``proposer`` replaces the four searches. With ``None``, or with no budget, there is no search
    and nothing is reported, as before: the second engine stands beside searches and never in
    their place, because a route whose only candidates come from one library confirms what that
    library can write. With ``restart`` the reported law goes through the follow-up experiment.
    An earlier route, asked for by name or by the environment, is run as ``default_route`` runs
    it and takes no experiment: it is handed on as it was measured.
    """
    chosen = route_name or route_in_force()
    if chosen not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, not {chosen!r}")
    stage = default_route.search_stage(budget_seconds, seed) if proposer is _UNSET else proposer
    if chosen != SECOND_ENGINE or stage is None:
        earlier = default_route.discover_component(
            dev, pool, component, budget_seconds=budget_seconds, seed=seed,
            route_name=chosen if chosen != SECOND_ENGINE else default_route.REPAIRED,
            **({} if proposer is _UNSET and chosen != SECOND_ENGINE else {"proposer": stage}))
        return earlier, {"route": chosen, "second_engine": False}
    found, done = extended_route.discover_component(dev, pool, component, proposer=stage, seed=seed, second_engine=True,
                                                    at_precision=at_precision)
    done["route"] = SECOND_ENGINE
    if restart is not None:
        found, experiment = experiment_pruning.decided(found, dev, component, restart.rate, level=restart.level,
                                                       spread=restart.spread, seed=seed, chosen=True)
        done.update(experiment)
    return found, done
