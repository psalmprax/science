"""The default route's search stage: four different searches where four cores are free, else one.

Status: experimental

**Why.** PREREG-PORTFOLIO-SEARCH-2026-11 (run once, 2026-11-20) measured the
refined path with four searches side by side (``proposers.deep_pass.mixed_members``:
the default PySR search, the exhaustive pass at a larger size limit, PySR with
algebraic operators, a second seed) against one PySR search, on SRBench's 103
problems the portfolio was not built on: 73 and 63 recovered by form at noise 0
and 1 % against 61 and 61; 118 of the single search's 122 recoveries kept; 1 of
224 reported laws not accurate. All four expectations held. The preregistration
fixed that the portfolio then becomes the default route's search stage wherever
four cores are available. This module is that rule; the frozen route and
proposers are not edited.

**Cost.** Four processes for the wall time of one search, each starting its own
Julia runtime. Where fewer than ``CORES_FOR_PORTFOLIO`` cores are available the
stage is the single PySR search the refined run measured, so nothing slower is
imposed on a small machine.

**Not covered by the measurement.** ODE systems: four seeds of one search were
measured there (PREREG-ODE-REFINED-2026-11) and not adopted. Series data: not
measured.
"""
from __future__ import annotations

import os
from typing import Any

from apedp_dyn.proposers.deep_pass import mixed_members
from apedp_dyn.proposers.portfolio import PortfolioProposer

#: cores the four members need to run side by side (one each)
CORES_FOR_PORTFOLIO = 4
#: candidates handed to confirmation, as in the measured run
MAX_PROPOSALS = 10
#: extra seconds the portfolio waits for its members beyond the search budget, as in the measured run
MEMBER_GRACE_S = 300.0


def available_cores() -> int:
    """Cores this process may use (the scheduler's affinity where the platform reports it)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # not every platform has affinity
        return os.cpu_count() or 1


def default_search_stage(budget_seconds: float, *, cores: int | None = None, seed: int = 0) -> tuple[Any, str]:
    """The search stage for one discovery and a short description of it.

    The portfolio measured in PREREG-PORTFOLIO-SEARCH-2026-11 when ``cores`` (or
    the machine) allows four searches at once; otherwise one PySR search.
    """
    from apedp_dyn.proposers.pysr_proposer import PySRProposer

    free = available_cores() if cores is None else int(cores)
    if free >= CORES_FOR_PORTFOLIO:
        stage = PortfolioProposer(mixed_members(budget_seconds, seed), max_proposals=MAX_PROPOSALS, parallel=True,
                                  timeout=budget_seconds + MEMBER_GRACE_S)
        return stage, f"four searches side by side (PREREG-PORTFOLIO-SEARCH), {budget_seconds:g} s"
    return PySRProposer(budget_seconds=budget_seconds, seed=seed), f"PySR, {budget_seconds:g} s"
