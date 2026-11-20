"""Several independent searches on separate cores, merged into one proposer.

Status: experimental

**Why.** The search stage of the default route is one PySR search, serial, for
240 seconds on one core. Two things measured on 2026-11-20 say that this is
what limits recovery, and that the machine has room for more:

* Two runs of the same path that differ only in how the search was configured
  (PREREG-REFINED-SEARCH and PREREG-LLM-CONTROLLER) recovered 174 and 173 of 266
  SRBench runs, and 191 between them. Each search finds laws the other misses.
* The runners the benchmarks use have four cores, and three sat idle.

**What.** ``PortfolioProposer`` runs its members at the same time, each in its
own process, and merges what they propose: every member's best candidate
first, then every member's second, and so on, without duplicates, up to
``max_proposals``. It decides nothing. The merged candidates go to the same
contrast confirmation as a single search's would, which is where a wrong
candidate from one member meets the right one from another.

``pysr_members`` builds the usual portfolio: the same PySR search with
different random seeds.

**Processes.** Members run under the ``spawn`` start method, so each child
starts its own Julia runtime; a Julia runtime does not survive ``fork``. Each
member is therefore given as something that can be pickled: a class with its
keyword arguments (``functools.partial``), not a lambda. With ``parallel=False``
the members run one after another in this process, which is what the tests use
and what a machine with one core should use.

**Failures.** A member that raises, or does not answer within ``timeout``
seconds, is recorded in ``failures`` and contributes nothing. The portfolio
fails only if every member does.
"""
from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import pandas as pd

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.proposers.composed import formula_proposal

MemberFactory = Callable[[], ProposerBase]


class FixedFormulasProposer(ProposerBase):
    """A member that proposes the formulas it was given. For tests, and for adding known candidates to a portfolio."""

    def __init__(self, formulas: Sequence[str], fail: str = "") -> None:
        self._formulas, self._fail = tuple(formulas), fail

    @property
    def proposer_id(self) -> str:
        return "fixed_formulas"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Sequence[CandidateProposal]:
        if self._fail:
            raise RuntimeError(self._fail)
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        features = [c for c in frame.columns if c != (target_col or "target")]
        return [formula_proposal(self.proposer_id, formula, features) for formula in self._formulas]


def pysr_members(budget_seconds: float = 240.0, seeds: Sequence[int] = (0, 1, 2, 3), **options: Any) -> list[MemberFactory]:
    """The same PySR search with different seeds, as picklable member factories."""
    from apedp_dyn.proposers.pysr_proposer import PySRProposer

    return [partial(PySRProposer, budget_seconds=budget_seconds, seed=int(seed), **options) for seed in seeds]


def _member_formulas(factory: MemberFactory, columns: dict[str, list[float]], target_col: str | None,
                     guesses: Sequence[str] | None) -> list[str]:
    """One member's proposals as formula texts, best first. Runs in the child process."""
    proposals = factory().fit_and_propose(pd.DataFrame(columns), target_col=target_col, guesses=guesses)
    return [str(p.formula) for p in proposals]


class PortfolioProposer(ProposerBase):
    """Independent members run side by side; their proposals merged round-robin."""

    def __init__(self, members: Sequence[MemberFactory], max_proposals: int = 10, parallel: bool = True,
                 timeout: float = 900.0, target_col: str = "target") -> None:
        if not members:
            raise ValueError("a portfolio needs at least one member")
        self._members = list(members)
        self._max = int(max_proposals)
        self._parallel = bool(parallel)
        self._timeout = float(timeout)
        self._target_col = target_col
        self.failures: list[str] = []
        self.last_run: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return "portfolio"

    def _run_members(self, columns: dict[str, list[float]], target: str, guesses: Sequence[str] | None) -> list[list[str]]:
        per_member: list[list[str]] = [[] for _ in self._members]
        if not self._parallel or len(self._members) == 1:
            for i, factory in enumerate(self._members):
                try:
                    per_member[i] = _member_formulas(factory, columns, target, guesses)
                except Exception as exc:  # noqa: BLE001 - one member failing must not cost the others their candidates
                    self.failures.append(f"member {i}: {type(exc).__name__}: {exc}"[:200])
            return per_member
        context = multiprocessing.get_context("spawn")  # a Julia runtime does not survive fork
        pool = context.Pool(processes=len(self._members))
        try:
            pending = [pool.apply_async(_member_formulas, (factory, columns, target, guesses)) for factory in self._members]
            deadline = time.monotonic() + self._timeout
            for i, result in enumerate(pending):
                try:
                    per_member[i] = result.get(timeout=max(1.0, deadline - time.monotonic()))
                except multiprocessing.TimeoutError:
                    self.failures.append(f"member {i}: no answer within {self._timeout:g} s")
                except Exception as exc:  # noqa: BLE001 - one member failing must not cost the others their candidates
                    self.failures.append(f"member {i}: {type(exc).__name__}: {exc}"[:200])
        finally:
            pool.terminate()
            pool.join()
        return per_member

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [c for c in frame.columns if c != target]
        columns = {str(c): [float(v) for v in frame[c]] for c in frame.columns}
        self.failures = []
        started = time.perf_counter()
        per_member = self._run_members(columns, target, guesses)
        if len(self.failures) == len(self._members):
            raise RuntimeError("every member of the portfolio failed: " + "; ".join(self.failures))
        merged: list[CandidateProposal] = []
        seen: set[str] = set()
        sources: list[int] = []
        for rank in range(max((len(p) for p in per_member), default=0)):  # every member's best comes first
            for i, formulas in enumerate(per_member):
                if rank < len(formulas) and formulas[rank] not in seen and len(merged) < self._max:
                    seen.add(formulas[rank])
                    merged.append(formula_proposal(self.proposer_id, formulas[rank], features,
                                                   metadata={"member": i, "rank": rank}))
                    sources.append(i)
        self.last_run = {"members": len(self._members), "proposed": [len(p) for p in per_member],
                         "distinct": len({f for p in per_member for f in p}), "merged": len(merged),
                         "merged_from": sources, "failures": list(self.failures), "parallel": self._parallel,
                         "seconds": round(time.perf_counter() - started, 2)}
        return merged
