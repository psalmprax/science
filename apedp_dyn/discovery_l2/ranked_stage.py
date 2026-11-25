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

**What may be added.** Only a candidate that fits the development rows about as well as the best
candidate the stage handed over: one that misses them by more than ``GATE`` times what the stage's
best misses them by is left out. This was added after the dress rehearsal of
PREREG-RANKED-MEMORY-2026-11 (run 45658660944, sixteen development laws, 25 November): there the
stage without it added three candidates in every run, in 23 of 32 runs every one of them missed
the rows by more than ``GATE`` times the stage's best (often by thousands of times), and two reports
were lost against one gained. A candidate that fits far worse than what the searches already have
can only take a place; it cannot be the law.

**What it does not do.** It decides nothing: every candidate it adds passes the fit guard and is
confirmed, refused and checked by the path like any other (AGENTS.md 1.1). It reads no pool row;
the gate reads the development rows the proposers were given, and nothing else.

**Not in force anywhere.** The stage for tables is ``stage_in_force.search_stage``. Whether this one
takes its place is the owner's decision after PREREG-RANKED-MEMORY-2026-11
(``docs/OWNER_DECISIONS_2026-11-25_AFTERNOON.md``, 1; AGENTS.md 1.5).
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

from apedp_dyn.discovery_l2.search_stage import MAX_PROPOSALS
from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.proposers import ranked_memory
from apedp_dyn.proposers.fit_guard import FitGuardProposer
from apedp_dyn.proposers.learned_order import LearnedOrder
from apedp_dyn.proposers.structure_memory import Memory

ORDER_FILE = Path(__file__).resolve().parents[2] / "benchmarks" / "order_score" / "learned_order_model.json"
EXTRA = 3                #: candidates the ranked proposer may add
OFFERED = 12             #: how far down its own order it is read for shapes the stage did not hand over
GATE = 1.25              #: an added candidate may miss the development rows by at most this many times what the stage's best misses them by
_CONSTANT = sp.Symbol("C")


def shape(formula: str, names: Sequence[str]) -> str:
    """A law with every fitted constant made one symbol: two laws of one shape differ only in their constants."""
    try:
        expr = sp.sympify(str(formula), locals={name: sp.Symbol(name) for name in names})
    except (sp.SympifyError, TypeError, SyntaxError, AttributeError):
        return str(formula)
    return str(expr.xreplace(dict.fromkeys(expr.atoms(sp.Float), _CONSTANT)))


def recorded_order() -> LearnedOrder:
    """The order this stage ranks with: learned on the lists of the memory taught 80 curriculum laws.

    Of the two orders at hand it is the one that, on the second set of problems, brought the law
    forward further on the lists of the memory **in force** than the order learned on that memory's
    own lists did (``benchmarks/order_score/dev_learned_order_in_force.json``).
    """
    return ranked_memory.load_order(ORDER_FILE)


def miss(proposal: CandidateProposal, frame: pd.DataFrame, target: np.ndarray) -> float:
    """How far a candidate misses the rows it was proposed from (root mean square); without bound if it cannot be read on them."""
    try:
        with np.errstate(all="ignore"):
            value = np.asarray(proposal.compute_score(frame), dtype=float) * np.ones(len(target))
    except Exception:  # noqa: BLE001 - a candidate that cannot be evaluated on the rows fits them no better than nothing
        return float("inf")
    return float(np.sqrt(np.mean((value - target) ** 2))) if np.all(np.isfinite(value)) else float("inf")


class WithRanked(ProposerBase):
    """A stage's candidates, and after them the ranked proposer's first new shapes that fit as well, within the stage's limit."""

    def __init__(self, stage: Any, ranked: Any, extra: int = EXTRA, limit: int = MAX_PROPOSALS, gate: float = GATE) -> None:
        self._stage, self._ranked = stage, ranked
        self._extra, self._limit, self._gate = int(extra), int(limit), float(gate)
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
        # the gate: what the stage's best candidate misses the development rows by, and nothing added that misses them by far more
        target = frame[target_col].to_numpy(dtype=float) if target_col in frame.columns else None
        best = min((miss(proposal, frame, target) for proposal in handed), default=float("inf")) if target is not None else float("inf")
        allowed = self._gate * best if handed and target is not None else float("inf")
        added: list[CandidateProposal] = []
        worse = 0
        for proposal in offered:
            if len(added) == self._extra:
                break
            key = shape(str(proposal.formula), names)
            if key in shapes:
                continue
            if target is not None and not miss(proposal, frame, target) <= allowed:
                worse += 1
                continue
            shapes.add(key)
            added.append(proposal)
        kept = handed[: max(self._limit - len(added), 0)]
        self._report = {"offered": len(offered), "added": [str(p.formula) for p in added],
                        "displaced": [str(p.formula) for p in handed[len(kept):]], "handed_by_the_stage": len(handed),
                        "left_out_as_fitting_worse": worse, "gate": self._gate,
                        "best_miss_of_the_stage": None if not np.isfinite(best) else best}
        return [*kept, *added]


def with_ranked(stage: Any, memory: Memory, order: LearnedOrder | None, *, extra: int = EXTRA, offered: int = OFFERED,
                limit: int = MAX_PROPOSALS, gate: float = GATE) -> Any:
    """``stage`` with the memory's list in ``order`` beside it; the ranked proposer alone if there is no stage."""
    ranked = FitGuardProposer(ranked_memory.RankedMemoryProposer(memory, order, max_proposals=offered, target_col="__target__"),
                              target_col="__target__")
    return ranked if stage is None else WithRanked(stage, ranked, extra, limit, gate)

