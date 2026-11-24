"""The ODE route with three additions that can be switched on one by one. None is in force.

Status: experimental

``ode/default_route.py`` is the route in force. This module composes the same
route with any of:

* ``second_engine``: sparse regression on a cubic library beside the searches
  (``ode/second_engine.py``);
* ``at_precision``: a law the adequacy check refused is reinstated when its
  residual is within the rows' own precision (``ode/precision_adequacy.py``);
* a follow-up experiment (``follow_up``): the system is restarted where the
  first run's rival laws disagree most, or at random points for comparison,
  and the same candidates are confirmed again on the larger pool
  (``ode/chosen_experiment.py``).

Each came out of the rows of PREREG-ODE-BLIND-2026-11 after that run, so
those systems cannot test them. They are here so that one frozen comparison
on other systems can.

``Recording`` wraps a search stage and keeps what it proposed, so that every
variant can be given the same candidates: a comparison between variants is
then about the variants, not about what a second search happened to find.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.ode import chosen_experiment, default_route, refined_ode
from apedp_dyn.ode.precision_adequacy import judged_at_precision
from apedp_dyn.ode.second_engine import with_sparse
from apedp_dyn.ode.window_rows import WindowRows
from apedp_dyn.proposers.fit_guard import FitGuardProposer
from apedp_dyn.proposers.portfolio import FixedFormulasProposer


class Recording(ProposerBase):
    """A search stage that also remembers the formulas it proposed, in order."""

    def __init__(self, inner: ProposerBase) -> None:
        self._inner = inner
        self.formulas: list[str] = []

    @property
    def proposer_id(self) -> str:
        return str(self._inner.proposer_id)

    @property
    def dropped(self) -> list[tuple[str, str]]:
        return list(getattr(self._inner, "dropped", []) or [])

    @property
    def last_run(self) -> dict[str, Any]:
        return dict(getattr(self._inner, "last_run", {}) or {})

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Sequence[CandidateProposal]:
        proposals = list(self._inner.fit_and_propose(train_data, target_col=target_col, guesses=guesses, **kwargs))
        self.formulas = [str(p.formula) for p in proposals]
        return proposals


def replayed(formulas: Sequence[str]) -> ProposerBase | None:
    """A stage that proposes ``formulas`` again, behind the fit guard; nothing if there are none."""
    stage: ProposerBase = FitGuardProposer(FixedFormulasProposer(list(formulas)), target_col="__target__")
    return stage if formulas else None


def discover_component(dev: WindowRows, pool: WindowRows, component: int, *, proposer: Any, seed: int = 0,
                       second_engine: bool = False, at_precision: bool = False
                       ) -> tuple[default_route.ComponentLaw, dict[str, Any]]:
    """One component by the default route with the chosen additions; the law and what the additions did."""
    stage = with_sparse(proposer) if second_engine else proposer
    result = refined_ode.discover_component(dev, pool, component, proposer=stage, seed=seed)
    done: dict[str, Any] = {"second_engine": second_engine, "at_precision": at_precision}
    if at_precision:
        result, done["precision"] = judged_at_precision(result, dev, component)
    return default_route.pruned_to_precision(result, dev, component, stage), done


def candidates_of(found: default_route.ComponentLaw) -> list[str]:
    """The laws a run's confirmation was given."""
    path = getattr(found.result, "path", None)
    return [str(c["formula"]) for c in (path.candidates if path is not None else [])]


def follow_up(dev: WindowRows, pool: WindowRows, component: int, first: default_route.ComponentLaw,
              formulas: Sequence[str], rate: Callable[[np.ndarray], np.ndarray], *, level: float, spread: np.ndarray,
              seed: int = 0, chosen: bool = True, second_engine: bool = False, at_precision: bool = False
              ) -> tuple[default_route.ComponentLaw, dict[str, Any]]:
    """The route again after one follow-up experiment, with the first run's searched ``formulas`` replayed.

    ``first`` is the run whose rivals choose the experiment. ``rate`` is the system itself: states
    in, time derivatives out.
    """
    larger, experiment = chosen_experiment.follow_up(dev, pool, component, candidates_of(first), rate, level=level,
                                                     spread=spread, seed=seed, chosen=chosen)
    found, done = discover_component(dev, larger, component, proposer=replayed(formulas), seed=seed,
                                     second_engine=second_engine, at_precision=at_precision)
    return found, {**done, "experiment": chosen_experiment.as_record(experiment)}
