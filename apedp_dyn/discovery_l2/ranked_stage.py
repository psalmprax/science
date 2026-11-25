"""A search stage with the memory's list in a learned order beside it.

Status: experimental

**What it is.** ``with_ranked`` puts ``proposers.ranked_memory.RankedMemoryProposer`` beside any
search stage. The stage hands over what it hands over now; the ranked proposer builds the memory's
list from the same development rows, orders it with a learned order
(``proposers.learned_order.LearnedOrder``), and its first candidates **whose shape the stage did not
hand over** are added: at most ``EXTRA`` of them.

**What it costs, and why it is written this way.** The path confirms at most
``search_stage.MAX_PROPOSALS`` candidates of a stage, and the stage in force hands over that many on
nearly every table. So a candidate added is a candidate displaced: where the stage's list is full,
the added ones take the place of its **last** ones, which with the stage in force are the second
choices of three of its four searches. The memory in force keeps its own three (they stand among
the first six), and so does each search's first choice. What that trade gains and loses is what a
comparison has to measure, run for run; ``last_run`` names what was added and what was displaced.

**What it does not do.** It decides nothing: every candidate it adds passes the fit guard and is
confirmed, refused and checked by the path like any other (AGENTS.md 1.1). It reads no pool row.

**Not in force anywhere.** The stage for tables is ``stage_in_force.search_stage``. Whether this one
takes its place is the owner's decision after PREREG-RANKED-MEMORY-2026-11
(``docs/OWNER_DECISIONS_2026-11-25_AFTERNOON.md``, 1; AGENTS.md 1.5).
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import sympy as sp

from apedp_dyn.discovery_l2.search_stage import MAX_PROPOSALS
from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.proposers import ranked_memory
from apedp_dyn.proposers.fit_guard import FitGuardProposer
from apedp_dyn.proposers.learned_order import LearnedOrder
from apedp_dyn.proposers.structure_memory import Memory

ORDER_FILE = Path(__file__).resolve().parents[2] / "benchmarks" / "order_score" / "learned_order_in_force.json"
EXTRA = 3                #: candidates the ranked proposer may add
OFFERED = 12             #: how far down its own order it is read for shapes the stage did not hand over
_CONSTANT = sp.Symbol("C")


def shape(formula: str, names: Sequence[str]) -> str:
    """A law with every fitted constant made one symbol: two laws of one shape differ only in their constants."""
    try:
        expr = sp.sympify(str(formula), locals={name: sp.Symbol(name) for name in names})
    except (sp.SympifyError, TypeError, SyntaxError, AttributeError):
        return str(formula)
    return str(expr.xreplace(dict.fromkeys(expr.atoms(sp.Float), _CONSTANT)))


def recorded_order() -> LearnedOrder:
    """The order learned for the memory in force (``scripts/study_learned_order.py --model-out``)."""
    return ranked_memory.load_order(ORDER_FILE)


class WithRanked(ProposerBase):
    """A stage's candidates, and after them the ranked proposer's first new shapes, within the stage's limit."""

    def __init__(self, stage: Any, ranked: Any, extra: int = EXTRA, limit: int = MAX_PROPOSALS) -> None:
        self._stage, self._ranked = stage, ranked
        self._extra, self._limit = int(extra), int(limit)
        self.failures: list[str] = []
        self._report: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return str(getattr(self._stage, "proposer_id", "stage"))

    @property
    def dropped(self) -> list[tuple[str, str]]:
        """What the fit guards of both members dropped in the last call."""
        return [*(getattr(self._stage, "dropped", []) or []), *(getattr(self._ranked, "dropped", []) or [])]

    @property
    def last_run(self) -> dict[str, Any]:
        return {"ranked": dict(self._report), "ranked_proposer": dict(getattr(self._ranked, "last_run", {}) or {}),
                "stage": dict(getattr(self._stage, "last_run", {}) or {})}

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Sequence[CandidateProposal]:
        self.failures = list(getattr(self._stage, "failures", []) or [])
        handed = list(self._stage.fit_and_propose(train_data, target_col=target_col, guesses=guesses, **kwargs))
        try:
            offered = list(self._ranked.fit_and_propose(train_data, target_col=target_col, guesses=guesses, **kwargs))
        except Exception as exc:  # noqa: BLE001 - the ranked proposer failing must not cost the stage its candidates
            self.failures.append(f"{getattr(self._ranked, 'proposer_id', 'ranked')}: {type(exc).__name__}: {exc}"[:200])
            offered = []
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        names = [str(column) for column in frame.columns if column != target_col]
        shapes = {shape(str(proposal.formula), names) for proposal in handed}
        added: list[CandidateProposal] = []
        for proposal in offered:
            if len(added) == self._extra:
                break
            key = shape(str(proposal.formula), names)
            if key in shapes:
                continue
            shapes.add(key)
            added.append(proposal)
        kept = handed[: max(self._limit - len(added), 0)]
        self._report = {"offered": len(offered), "added": [str(p.formula) for p in added],
                        "displaced": [str(p.formula) for p in handed[len(kept):]], "handed_by_the_stage": len(handed)}
        return [*kept, *added]


def with_ranked(stage: Any, memory: Memory, order: LearnedOrder | None, *, extra: int = EXTRA, offered: int = OFFERED,
                limit: int = MAX_PROPOSALS) -> Any:
    """``stage`` with the memory's list in ``order`` beside it; the ranked proposer alone if there is no stage."""
    ranked = FitGuardProposer(ranked_memory.RankedMemoryProposer(memory, order, max_proposals=offered, target_col="__target__"),
                              target_col="__target__")
    return ranked if stage is None else WithRanked(stage, ranked, extra, limit)

