"""The exhaustive first pass, one or two levels deeper, as a member of a portfolio.

Status: experimental

**Why.** The route's own first pass fits every expression of at most six nodes
(five with more than five variables) and stops. It is deterministic and needs
no Julia runtime. Development on 2026-11-20 asked whether going deeper pays:
on the 18 development runs of SRBench that the first passes alone do not
recover, a pass of eight nodes for one variable and seven for two or three
recovered 3, among them one (``feynman_test_5`` without noise) that no PySR
run of this repository has recovered. It is not a replacement for the search:
PySR recovers 10 of the same 18, and with 1 % noise the deeper pass returned
four laws that fit and are not the law, built from stacked exponentials. So
it is offered as one more voice, to be confirmed against the others'
candidates, never as the only one.

**What.** ``DeepPassProposer`` runs ``confirmed_stop.search_confirmed`` with
the size prior at the depth given for the number of variables, and proposes
the one law it stops on. With more variables than the table covers it
proposes nothing; after the units reduction most problems have one to three.

**Cost** (one core, measured): 420 567 candidates for one variable at depth 8,
220 491 for two and 502 733 for three at depth 7; a pass that fits every one
of them took 35 to 75 seconds for two variables on SRBench's rows.

``mixed_members`` builds the portfolio this module was written for: the
default PySR search, this pass, a PySR search restricted to algebraic
operators, and a second seed of the default search.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from functools import lru_cache, partial
from typing import Any

import pandas as pd

from apedp_dyn.discovery_l2 import exhaustive_search as es
from apedp_dyn.discovery_l2.confirmed_stop import search_confirmed
from apedp_dyn.discovery_l2.guided_search import GuidedSearch
from apedp_dyn.discovery_l2.main_search import size_prior
from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.proposers.composed import formula_proposal
from apedp_dyn.proposers.portfolio import MemberFactory

#: nodes searched, by number of variables. A search size, not an acceptance threshold.
DEPTHS: Mapping[int, int] = {1: 8, 2: 7, 3: 7}
#: operators of the algebraic PySR member: no exponential, logarithm or trigonometric function
ALGEBRAIC_UNARY = ("sqrt", "square")


@lru_cache(maxsize=1)
def _guided() -> GuidedSearch:
    """One order per (variables, depth), kept for the life of the process."""
    return GuidedSearch(size_prior)


class DeepPassProposer(ProposerBase):
    """The first pass with a larger size limit. Proposes the law it stops on, or nothing."""

    def __init__(self, depths: Mapping[int, int] | None = None, target_col: str = "target") -> None:
        self._depths = dict(DEPTHS if depths is None else depths)
        self._target_col = target_col
        #: what the last call did
        self.last_run: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return "deep_pass"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [str(c) for c in frame.columns if c != target]
        depth = self._depths.get(len(features))
        if depth is None:
            self.last_run = {"ran": False, "variables": len(features), "reason": "no depth is set for this many variables"}
            return []
        started = time.perf_counter()
        found = search_confirmed(_guided(), frame[features].to_numpy(dtype=float), frame[target].to_numpy(dtype=float),
                                 features, depth)
        self.last_run = {"ran": True, "variables": len(features), "depth": depth, "fitted": int(found.evaluated),
                         "candidates": int(found.candidates), "stopped_early": bool(found.stopped_early),
                         "seconds": round(time.perf_counter() - started, 2)}
        return [formula_proposal(self.proposer_id, es.model_string(found.best), features, dict(self.last_run))]


def mixed_members(budget_seconds: float = 240.0, seed: int = 0) -> list[MemberFactory]:
    """Four different searches: default PySR, the deeper pass, algebraic PySR, default PySR with the next seed.

    The first member is the search the single-search route runs, with its seed,
    so the portfolio's candidates include that route's.
    """
    from apedp_dyn.proposers.pysr_proposer import PySRProposer

    return [partial(PySRProposer, budget_seconds=budget_seconds, seed=int(seed)),
            partial(DeepPassProposer),
            partial(PySRProposer, budget_seconds=budget_seconds, seed=int(seed), unary=ALGEBRAIC_UNARY),
            partial(PySRProposer, budget_seconds=budget_seconds, seed=int(seed) + 1)]
