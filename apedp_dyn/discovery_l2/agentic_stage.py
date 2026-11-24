"""A search stage in which a language-model agent and the default searches take turns.

Status: experimental

``search_stage.default_search_stage`` is the route's search stage as measured:
four searches side by side where four cores are free, else one. This module
builds the same searches with ``proposers.agentic_variation`` around them: the
searches run for a share of the budget, an agent reads what they found with the
residuals in view and varies it, and the best structures seed the searches'
next round. The fit guard stays outermost, as in the default stage.

**Not the default, and not measured.** Nothing selects this stage unless a
caller asks for it. Whether it recovers more laws than the default stage at
equal time, without more wrong laws, is an open question that a preregistered
run has to answer; until then ``default_search_stage`` is what the route uses.

**Time.** The searches get ``budget_seconds`` in all, split over ``rounds``.
Each round of four searches starts four Julia runtimes again, and each agent
session may take up to ``session_seconds`` on top: the stage costs more wall
time than the default one for the same search budget, and a comparison at
equal time has to give the default stage that time too.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from apedp_dyn.discovery_l2.search_stage import (
    CORES_FOR_PORTFOLIO,
    MAX_PROPOSALS,
    MEMBER_GRACE_S,
    available_cores,
)
from apedp_dyn.proposers.agentic_variation import AgenticVariationProposer
from apedp_dyn.proposers.fit_guard import FitGuardProposer


def agentic_search_stage(client: Callable[[str], str] | None, budget_seconds: float, *, cores: int | None = None,
                         seed: int = 0, rounds: int = 2, steps: int = 6, session_seconds: float = 90.0,
                         description: str = "") -> tuple[Any, str]:
    """The default stage's searches with an agent taking turns with them, and a short description."""
    free = available_cores() if cores is None else int(cores)

    def searches(budget: float) -> Any:
        from apedp_dyn.proposers.deep_pass import mixed_members
        from apedp_dyn.proposers.portfolio import PortfolioProposer
        from apedp_dyn.proposers.pysr_proposer import PySRProposer

        if free >= CORES_FOR_PORTFOLIO:
            return PortfolioProposer(mixed_members(budget, seed), max_proposals=MAX_PROPOSALS, parallel=True,
                                     timeout=budget + MEMBER_GRACE_S, target_col="__target__")
        return PySRProposer(budget_seconds=budget, seed=seed)

    stage = AgenticVariationProposer(client, searches, budget_seconds=budget_seconds, rounds=rounds, steps=steps,
                                     session_seconds=session_seconds, max_proposals=MAX_PROPOSALS, seed=seed,
                                     description=description, target_col="__target__")
    kind = "four searches" if free >= CORES_FOR_PORTFOLIO else "PySR"
    return FitGuardProposer(stage), (f"{kind} and an agent taking turns, {rounds} rounds, "
                                     f"{budget_seconds:g} s of search")
