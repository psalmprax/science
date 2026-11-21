"""The default search stage with a checked, jailed candidate generator beside it.

Status: experimental

``search_stage.default_search_stage`` is the route's search stage as measured.
This module puts one more member next to it: a version of ``propose`` from
``apedp_dyn.evolution`` (the seed, or one a model wrote under the loop there),
run in the jail on the development rows. Its candidates and the searches' are
handed on in turn, each member's best first, and the fit guard screens the
generator's as it screens the searches'.

**Not the default, and not measured.** Nothing selects this stage unless a
caller asks for it, and no version of ``propose`` has been adopted. A version
enters the route only by a reviewed commit, after one preregistered test on
laws it was never scored on.

**What it costs.** The path confirms at most ``MAX_PROPOSALS`` candidates of a
stage. Each one the generator supplies takes the place of one of the searches'
lower-ranked candidates, so a generator that proposes badly can lose a law the
searches had found. That is what a comparison against the default stage has to
measure, in both directions.
"""
from __future__ import annotations

from typing import Any

from apedp_dyn.discovery_l2.search_stage import MAX_PROPOSALS, default_search_stage
from apedp_dyn.evolution.evolved_proposer import EvolvedProposer
from apedp_dyn.proposers.composed import CompositeProposer
from apedp_dyn.proposers.fit_guard import FitGuardProposer


class SideBySide(CompositeProposer):
    """The searches and the generator in turn; ``last_run`` says what each did."""

    def __init__(self, searches: Any, generator: Any, max_proposals: int = MAX_PROPOSALS) -> None:
        super().__init__([searches, generator], max_proposals=max_proposals)
        self._searches, self._generator = searches, generator

    @property
    def last_run(self) -> dict[str, Any]:
        return {"generator": dict(getattr(self._generator, "last_run", {}) or {}),
                "generator_dropped_by_fit_guard": [list(item) for item in getattr(self._generator, "dropped", [])],
                "searches": dict(getattr(self._searches, "last_run", {}) or {})}


def evolved_search_stage(code: str, budget_seconds: float, *, cores: int | None = None, seed: int = 0,
                         seconds: float = 10.0) -> tuple[Any, str]:
    """The default stage's searches and the generator ``code`` side by side, and a short description.

    Raises ``restricted.RejectedCodeError`` at once if ``code`` is outside the allowed subset.
    """
    searches, described = default_search_stage(budget_seconds, cores=cores, seed=seed)
    generator = FitGuardProposer(EvolvedProposer(code, target_col="__target__", seconds=seconds), target_col="__target__")
    return SideBySide(searches, generator), f"{described}; and a jailed candidate generator"
