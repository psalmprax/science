"""The search stage in force for tables. Since 2026-11-22: the default searches with a memory of reported laws beside them.

Status: experimental

**Why this module exists.** PREREG-MEMORY-2026-11 fixed, before its run, that if
its two expectations held "the memory is put beside the default search stage
for tables, in a new module around the frozen ones, and this result is stated
wherever that stage is described". They held. ``search_stage``,
``memory_stage`` and ``proposers.structure_memory`` are frozen by that
preregistration's digest and are not edited. Code that wants "the search stage
for a table" calls ``search_stage`` here.

**The evidence** (``preregistration/RESULT-MEMORY-2026-11.md``, run once,
2026-11-22): 40 laws drawn after the memory file was written, at noise 0 and
1 %, both arms handed the same searched candidates. The four searches alone
recovered 17 of 80; with the memory's three laws beside them, 23. Six gained,
none lost, no wrong law in either arm (0 of 55 and 0 of 57 reported).

**What that does and does not say.**

* The gain is almost all without noise: 17 against 12 there, 6 against 5 at
  1 %. At 1 % the memory handed the route the law itself in 11 runs and the
  route reported it in 6; in the other five it could not tell its candidates
  apart and named a shorter one that is accurate and is not the law.
* The laws came from one grammar, and the way the memory joins parts was
  written knowing that grammar. Laws with several variables inside one
  function cannot be grown from one-variable parts, and most of the physics
  benchmarks' laws are such laws. No gain has been measured there.
* The memory holds laws the route reported, right or not, of that family.

**Cost.** Up to 60 seconds more per discovery.

``APEDP_TABLE_MEMORY=off`` gives the default searches alone, as before.
"""
from __future__ import annotations

import os
from typing import Any

from apedp_dyn.discovery_l2 import memory_stage
from apedp_dyn.discovery_l2.search_stage import default_search_stage

ENVIRONMENT_VARIABLE = "APEDP_TABLE_MEMORY"
OFF = "off"
EVIDENCE = "preregistration/RESULT-MEMORY-2026-11.md"


def memory_in_force() -> bool:
    """Whether the memory stands beside the searches: yes, unless the environment switches it off."""
    return os.environ.get(ENVIRONMENT_VARIABLE, "").strip().lower() != OFF


def search_stage(budget_seconds: float, *, cores: int | None = None, seed: int = 0) -> tuple[Any, str]:
    """The search stage for one discovery on a table, and a short description of it."""
    stage: Any
    described: str
    if memory_in_force():
        stage, described = memory_stage.memory_search_stage(budget_seconds, cores=cores, seed=seed)
    else:
        stage, described = default_search_stage(budget_seconds, cores=cores, seed=seed)
    return stage, described
