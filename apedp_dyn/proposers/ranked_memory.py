"""The memory's candidates handed over in a learned order, from the rows a proposer really has.

Status: experimental

**Why.** The memory of confirmed laws hands the referee its first three candidates, best fit first.
Measured on the curriculum (``benchmarks/order_score/``), a law the judge would count solved is on
the memory's list far more often than among its first three: over-fitted laws stand ahead of it.
``learned_order.LearnedOrder`` learns an order from the system's own problems. This module is that
order made usable by a route: a proposer that builds the memory's list and hands over the
candidates the order puts first.

**The rows it has.** The order reads how a candidate fits rows it was *not* fitted on. A proposer
is given development rows and nothing else, so it sets a share of them aside (``HELD_SHARE``),
fits on the rest and reads the fit on the share set aside. The split is seeded by the table
itself, so the same table has the same rows set aside every time. The fits are made afresh at each
call: on one machine they have come out the same, and on GitHub's machines a badly conditioned
candidate's constants have ended a few digits apart between two calls (25 November), so the same
table gives the same shapes and not always the same last digits.

**What it does not do.** It decides nothing: every candidate it hands over is confirmed and checked
by the path like any other (AGENTS.md 1.1). It reads no confirmation rows. **Not in force**:
nothing builds it but its tests and the order-score measurement; putting it beside or in place of
the memory in force is a change for the owner to decide (AGENTS.md 1.5).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.proposers import learned_order
from apedp_dyn.proposers import structure_memory as timed
from apedp_dyn.proposers.composed import formula_proposal
from apedp_dyn.proposers.structure_memory import Memory
from apedp_dyn.proposers.structure_memory_counted import (
    GROW_FITS,
    WHOLE_FITS,
    grow_counted,
    remembered_whole_counted,
)
from apedp_dyn.recursion import mdl

POOL = 40                #: candidates read before the order chooses
HELD_SHARE = 0.2         #: share of the development rows set aside to read each candidate's fit on unseen rows
SMALL, LARGE = 0.02, 50.0


def _r2(y: np.ndarray, pred: np.ndarray) -> float | None:
    """The share of the rows' variance a prediction explains; ``None`` where it is not a number or the rows do not vary."""
    if not np.all(np.isfinite(pred)):
        return None
    total = float(np.sum((y - np.mean(y)) ** 2))
    return None if total <= 0 else 1.0 - float(np.sum((y - pred) ** 2)) / total


def pool_with_features(memory: Memory, names: Sequence[str], X: np.ndarray, y: np.ndarray, held_X: np.ndarray,
                       held_y: np.ndarray, keep: int = POOL, grow_fits: int = GROW_FITS, whole_fits: int = WHOLE_FITS
                       ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The memory's candidates fitted on ``X, y``, best BIC first, each with what the learned order reads of it."""
    best, motifs, report = grow_counted(memory, X, y, max_fits=grow_fits, keep=keep)
    whole, whole_report = remembered_whole_counted(memory, names, X, y, max_fits=whole_fits)
    pairs = [*[(g.bic, timed.written(g, names, motifs), len(g.steps)) for g in best], *[(bic, law, 0) for bic, law in whole]]
    symbols = [sp.Symbol(name) for name in names]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bic, law, steps in sorted(pairs, key=lambda item: item[0]):
        if law.free_symbols != set(symbols):
            continue
        key = str(law.xreplace(dict.fromkeys(law.atoms(sp.Float), timed.C)))
        if key in seen:
            continue
        seen.add(key)
        constants = [abs(float(value)) for value in law.atoms(sp.Float)]
        with np.errstate(all="ignore"):
            function = sp.lambdify(symbols, law, "numpy")
            train = np.asarray(function(*X.T), dtype=float) * np.ones(len(X))
            held = np.asarray(function(*held_X.T), dtype=float) * np.ones(len(held_X))
        out.append({"text": str(law), "bic": round(float(bic), 3), "steps": int(steps), "constants": len(constants),
                    "small_constants": sum(value < SMALL for value in constants),
                    "large_constants": sum(value > LARGE for value in constants),
                    "operations": int(sp.count_ops(law)),
                    "train_r2": _r2(y, train), "held_r2": _r2(held_y, held),
                    "law_bits": round(float(mdl.law_bits(law, len(names))), 1)})
        if len(out) == keep:
            break
    return out, {"fits": report.get("fits"), "motifs": report.get("motifs"), "whole_fits": whole_report.get("fits")}


def split(X: np.ndarray, y: np.ndarray, held_share: float = HELD_SHARE) -> tuple[np.ndarray, np.ndarray]:
    """Row numbers to fit on and to set aside: a permutation seeded by the table itself."""
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    y = np.ascontiguousarray(np.asarray(y, dtype=float))
    seed = int.from_bytes(hashlib.sha256(X.tobytes() + y.tobytes()).digest()[:8], "big")
    order = np.random.default_rng(seed).permutation(len(y))
    held = max(round(held_share * len(y)), 1)
    return np.sort(order[held:]), np.sort(order[:held])


def split_pool(memory: Memory, names: Sequence[str], X: np.ndarray, y: np.ndarray, keep: int = POOL,
               held_share: float = HELD_SHARE, grow_fits: int = GROW_FITS, whole_fits: int = WHOLE_FITS
               ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The list as a proposer can build it: fitted on the rows kept, read on the rows set aside."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    fit, held = split(X, y, held_share)
    pool, report = pool_with_features(memory, names, X[fit], y[fit], X[held], y[held], keep, grow_fits, whole_fits)
    return pool, {**report, "rows_fitted": len(fit), "rows_set_aside": len(held)}


def load_order(path: str | Path) -> learned_order.LearnedOrder:
    """A learned order from its record: the file itself, or a record that holds it under ``model``."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    return learned_order.LearnedOrder.from_dict(record.get("model", record))


class RankedMemoryProposer(ProposerBase):
    """The memory's list in a learned order. With no order, the shortest within a tolerance of the best fit on unseen rows."""

    def __init__(self, memory: Memory, order: learned_order.LearnedOrder | None = None, max_proposals: int = 3,
                 pool: int = POOL, held_share: float = HELD_SHARE, grow_fits: int = GROW_FITS,
                 whole_fits: int = WHOLE_FITS, target_col: str = "target") -> None:
        self._memory, self._order = memory, order
        self._max, self._pool, self._held_share = int(max_proposals), int(pool), float(held_share)
        self._grow_fits, self._whole_fits = int(grow_fits), int(whole_fits)
        self._target_col = target_col
        self.last_run: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return "structure_memory_ranked"

    def ordered(self, names: Sequence[str], X: np.ndarray, y: np.ndarray) -> list[str]:
        """Every candidate of the list, the one the order thinks most likely the law first."""
        pool, report = split_pool(self._memory, names, X, y, self._pool, self._held_share, self._grow_fits, self._whole_fits)
        places = self._order.order(pool) if self._order is not None else learned_order.simplest_within(pool)
        self.last_run = {**report, "laws_remembered": self._memory.laws, "candidates": len(pool),
                         "order": "learned" if self._order is not None else "shortest within tolerance",
                         "moved_to_the_front": [int(place) + 1 for place in places[: self._max]]}
        return [pool[place]["text"] for place in places]

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [c for c in frame.columns if c != target]
        texts = self.ordered(features, frame[features].to_numpy(dtype=float), frame[target].to_numpy(dtype=float))[: self._max]
        self.last_run["proposed"] = len(texts)
        return [formula_proposal(self.proposer_id, text, features, {"rank": rank}) for rank, text in enumerate(texts)]
