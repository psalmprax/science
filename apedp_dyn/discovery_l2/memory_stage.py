"""A search stage with the memory of confirmed laws beside the searches.

Status: experimental

``with_memory`` puts ``proposers.structure_memory.StructureMemoryProposer``
beside any search stage: the two take turns handing candidates to the path, so
a law grown from remembered parts takes the place of one of the searches'
lower-ranked candidates and never of their best. ``memory_search_stage`` does
that for the default stage. The memory's laws pass the same fit guard as any
proposal and are confirmed or refused by the same path; a remembered shape has
no standing of its own.

Not in force anywhere: the default stage is ``search_stage.default_search_stage``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from apedp_dyn.discovery_l2.evolved_stage import SideBySide
from apedp_dyn.discovery_l2.search_stage import default_search_stage
from apedp_dyn.proposers.fit_guard import FitGuardProposer
from apedp_dyn.proposers.structure_memory import Memory, StructureMemoryProposer

MEMORY_FILE = Path(__file__).resolve().parents[2] / "benchmarks" / "structure_memory.json"
PROPOSALS = 3            # laws the memory hands over
SECONDS = 60.0           # how long it may grow them


def recorded_memory() -> Memory:
    """The memory built from recorded runs (``scripts/build_structure_memory.py``)."""
    return Memory.load(MEMORY_FILE)


def with_memory(searches: Any, memory: Memory, seconds: float = SECONDS, proposals: int = PROPOSALS) -> Any:
    """``searches`` and the memory in turn; the memory alone if there are no searches."""
    grower = FitGuardProposer(StructureMemoryProposer(memory, max_proposals=proposals, seconds=seconds, target_col="__target__"),
                              target_col="__target__")
    return grower if searches is None else SideBySide(searches, grower)


def memory_search_stage(budget_seconds: float, *, memory: Memory | None = None, cores: int | None = None, seed: int = 0,
                        seconds: float = SECONDS) -> tuple[Any, str]:
    """The default stage's searches with the memory beside them, and a short description."""
    searches, described = default_search_stage(budget_seconds, cores=cores, seed=seed)
    memory = memory if memory is not None else recorded_memory()
    return with_memory(searches, memory, seconds), f"{described}; and laws grown from {len(memory.motifs)} remembered motifs"
