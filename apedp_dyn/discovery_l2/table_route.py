"""The route in force for tables since 2026-11-23: the default route, with the last checks on request only.

Status: experimental

**Why this module exists.** The checks of ``discovery_l2/vouching.py`` became the
last step of the table route on 2026-11-22 (``vouched_route.py``, on
PREREG-COUNTERFEIT-2026-11). PREREG-VOUCHED-SRBENCH-2026-11 then measured what
they cost behind the four searches on the physics benchmark, and fixed in
advance that if they kept at least 75 % but less than 95 % of the laws the
route recovers, they would stay in force until the repository's owner
decided. They kept 78 %. **On 2026-11-23 the owner decided to take them out of
the route in force.** ``vouched_route`` and ``vouching`` are frozen by that
run's digest and are not edited; code that wants "the route for a table" calls
``discover_law`` here.

**The evidence behind the decision**
(``preregistration/RESULT-VOUCHED-SRBENCH-2026-11.md``): on SRBench, 133 laws at
noise 0 and 1 %, the checks kept 150 of the 193 laws the route recovered (47 of
90 at 1 % noise) and stopped no inaccurate law, because none of the route's 229
reports was inaccurate. 42 of the 43 losses came from one blindness (audit
M-35). Their measured benefit is elsewhere: on counterfeits of 60 new laws at
5 % noise, 2 wrong laws in 55 reports instead of 16 in 88
(``RESULT-COUNTERFEIT-2026-11``).

**So without the checks**, as before 2026-11-22: the table route's reports hold
where the preregistered runs measured (noise up to 1 %), and at 5 % noise they
did not (16 of 88 wrong on counterfeits; audit M-31, M-32).

``APEDP_TABLE_VOUCHING=on`` applies the checks, exactly as ``vouched_route``
does. A repaired version goes back in force only after it has been run on both
benchmarks and on laws drawn after it was written.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from apedp_dyn.discovery_l2 import default_route, vouched_route
from apedp_dyn.grammar.proposer_interface import ProposerBase

ENVIRONMENT_VARIABLE = vouched_route.ENVIRONMENT_VARIABLE
ON = "on"
DECISION = ("taken out of the route in force on 2026-11-23 by the owner's decision on "
            "preregistration/RESULT-VOUCHED-SRBENCH-2026-11.md")
EVIDENCE = "preregistration/RESULT-VOUCHED-SRBENCH-2026-11.md"


def checks_requested() -> bool:
    """Whether the last checks are applied: only when the environment asks for them."""
    return os.environ.get(ENVIRONMENT_VARIABLE, "").strip().lower() == ON


def with_checks_if_requested(record: Mapping[str, Any], X: Mapping[str, np.ndarray], Y: np.ndarray,
                             pool_X: np.ndarray) -> dict[str, Any]:
    """``vouched_route.checked`` when the checks are requested; otherwise the record as it is, saying why."""
    if checks_requested():
        checked: dict[str, Any] = vouched_route.checked(record, X, Y, pool_X)
        return checked
    out = dict(record)
    out["vouching"] = {"in_force": False, "applied": False, "evidence": EVIDENCE,
                       "why_not": f"{DECISION}; {ENVIRONMENT_VARIABLE}={ON} applies them"}
    return out


def discover_law(
    X: Mapping[str, np.ndarray], Y: np.ndarray, pool_X: np.ndarray, measure: Callable[[np.ndarray], np.ndarray], *,
    ledger: Any = None, search_id: str | None = None, route_name: str | None = None,
    proposer: ProposerBase | None = None, units: Mapping[str, Mapping[str, Any]] | None = None,
    target_units: Mapping[str, Any] | None = None, dimensions: Mapping[str, Any] | None = None,
    target_dimension: Any = None, noise: str = "relative", seed: int = 0, domain: str | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """``default_route.discover_law``, with ``vouching`` in the record; the checks run only on request."""
    record = default_route.discover_law(X, Y, pool_X, measure, ledger=ledger, search_id=search_id, route_name=route_name,
                                        proposer=proposer, units=units, target_units=target_units, dimensions=dimensions,
                                        target_dimension=target_dimension, noise=noise, seed=seed, domain=domain,
                                        dataset_id=dataset_id)
    return with_checks_if_requested(record, X, Y, pool_X)
