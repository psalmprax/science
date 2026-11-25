"""The four searches side by side in processes that are kept, not started again for every table.

Status: experimental

**Why.** The portfolio in force starts four fresh processes for every table and stops them when the
searches end. Each start loads Julia and compiles the search again: about 20 of the 110 seconds a
90-second search takes on GitHub's machines are spent before the search begins, and part of the 90
is compilation too. Stopping them is also where runs used to stop for ever (the cure for that is
the stage built by ``search_stage_that_cannot_hang``, the product's default).

**What this does.** One process for each member is started once and kept. A table's searches are
sent to them and their answers read back; the processes then wait for the next table with Julia
loaded and the search compiled. A process is ended only

* when its search has not answered by the portfolio's own deadline (it is killed and a fresh one
  takes its place for the next table: nothing is waited for);
* after ``RECYCLE_AFTER`` searches, so that whatever a long-lived runtime accumulates is dropped;
* when the main process closes down, by a kill (no exit path of Julia is ever run, which is the
  path that could wait for ever).

The members, their seeds, their budget and the order of proposals are those of the portfolio in
force. What can differ is the search itself: PySR's limit is a time, and a runtime that is already
compiled gets more search out of the same time. Whether that shows in laws recovered is a
measurement (the budget ladder, ``.github/workflows/budget-ladder.yml``).

**Not in force.** The product's reading builds it only where the environment says
``APEDP_SEARCH_PROCESSES=warm``. Making it the default would be a change to what is in force
(AGENTS.md 1.5). ``portfolio.py`` and ``search_stage.py`` are frozen and not edited.
"""
from __future__ import annotations

import atexit
import multiprocessing
import multiprocessing.connection
import os
import threading
import time
from collections.abc import Sequence
from typing import Any

from apedp_dyn.discovery_l2 import search_stage
from apedp_dyn.proposers.deep_pass import mixed_members
from apedp_dyn.proposers.fit_guard import FitGuardProposer
from apedp_dyn.proposers.portfolio import MemberFactory, PortfolioProposer, _member_formulas

ENVIRONMENT_VARIABLE = "APEDP_SEARCH_PROCESSES"
WARM = "warm"
RECYCLE_AFTER = 40          # searches a kept process runs before it is replaced
Task = tuple[MemberFactory, dict[str, list[float]], str | None, Sequence[str] | None]


def asked_for() -> bool:
    """Whether the environment asks for search processes that are kept. Off unless asked."""
    return os.environ.get(ENVIRONMENT_VARIABLE, "").strip().lower() == WARM


def _serve(connection: Any) -> None:
    """A kept process: run each search sent, answer, wait for the next. It never returns: it ends by ``os._exit`` or a kill."""
    while True:
        try:
            task = connection.recv()
        except (EOFError, OSError):
            os._exit(0)                      # the main process is gone
        if task is None:
            os._exit(0)
        try:
            reply: tuple[bool, Any] = (True, _member_formulas(*task))
        except Exception as exc:  # noqa: BLE001 - one member failing must not cost the others their candidates
            reply = (False, f"{type(exc).__name__}: {exc}"[:200])
        try:
            connection.send(reply)
        except (OSError, ValueError):
            os._exit(0)


class _Kept:
    """One kept process and the pipe to it."""

    def __init__(self) -> None:
        context = multiprocessing.get_context("spawn")      # a Julia runtime does not survive fork
        self.connection, theirs = context.Pipe(duplex=True)
        self.process = context.Process(target=_serve, args=(theirs,), daemon=True, name="kept-search")
        self.process.start()
        theirs.close()
        self.searches = 0

    def kill(self) -> None:
        try:
            self.process.kill()
            self.process.join(timeout=10.0)
        except (OSError, ValueError, AttributeError):
            pass
        try:
            self.connection.close()
        except OSError:
            pass


class WarmPool:
    """Kept search processes, one for each member's place in the portfolio."""

    def __init__(self) -> None:
        self._kept: list[_Kept | None] = []
        self._lock = threading.Lock()
        self.started = 0            #: processes started in all
        self.killed = 0             #: processes killed for not answering in time
        self.recycled = 0

    def _ready(self, index: int) -> _Kept:
        while len(self._kept) <= index:
            self._kept.append(None)
        kept = self._kept[index]
        if kept is not None and (not kept.process.is_alive() or kept.searches >= RECYCLE_AFTER):
            self.recycled += int(kept.process.is_alive())
            kept.kill()
            kept = None
        if kept is None:
            kept = _Kept()
            self._kept[index] = kept
            self.started += 1
        return kept

    def _drop(self, index: int) -> None:
        kept = self._kept[index]
        if kept is not None:
            kept.kill()
            self._kept[index] = None

    def run(self, tasks: Sequence[Task], timeout: float) -> list[tuple[bool, Any] | None]:
        """Each task's answer, ``(True, formulas)`` or ``(False, why)``, or ``None`` where none came in time."""
        with self._lock:
            waiting: dict[Any, int] = {}
            answers: list[tuple[bool, Any] | None] = [None] * len(tasks)
            for index, task in enumerate(tasks):
                kept = self._ready(index)
                try:
                    kept.connection.send(task)
                except (OSError, ValueError, TypeError) as exc:
                    answers[index] = (False, f"{type(exc).__name__}: {exc}"[:200])
                    self._drop(index)
                    continue
                kept.searches += 1
                waiting[kept.connection] = index
            deadline = time.monotonic() + float(timeout)
            while waiting:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                for ready in multiprocessing.connection.wait(list(waiting), timeout=left):
                    connection: Any = ready
                    index = waiting.pop(connection)
                    try:
                        answers[index] = connection.recv()
                    except (EOFError, OSError):
                        answers[index] = (False, "the search process ended without an answer")
                        self._drop(index)
            for index in waiting.values():          # still searching at the deadline: killed, replaced at the next table
                self._drop(index)
                self.killed += 1
            return answers

    def processes(self) -> list[int]:
        return [kept.process.pid for kept in self._kept if kept is not None and kept.process.pid is not None]

    def close(self) -> None:
        with self._lock:
            for index in range(len(self._kept)):
                self._drop(index)


_POOL = WarmPool()


def _close_at_exit() -> None:
    # registered after multiprocessing's own exit function, so run before it: the kept processes are killed here and
    # multiprocessing finds none left to send a stop signal to
    _POOL.close()


atexit.register(_close_at_exit)


class WarmPortfolio(PortfolioProposer):
    """The portfolio in force, its members run in kept processes."""

    pool: WarmPool = _POOL

    def _run_members(self, columns: dict[str, list[float]], target: str, guesses: Sequence[str] | None) -> list[list[str]]:
        if not self._parallel or len(self._members) == 1:
            return super()._run_members(columns, target, guesses)
        per_member: list[list[str]] = [[] for _ in self._members]
        answers = self.pool.run([(factory, columns, target, guesses) for factory in self._members], self._timeout)
        for index, answer in enumerate(answers):
            if answer is None:
                self.failures.append(f"member {index}: no answer within {self._timeout:g} s")
            elif answer[0]:
                per_member[index] = list(answer[1])
            else:
                self.failures.append(f"member {index}: {answer[1]}"[:200])
        return per_member


def search_stage_warm(budget_seconds: float, *, cores: int | None = None, seed: int = 0) -> tuple[Any, str]:
    """``search_stage.default_search_stage`` with its members in kept processes; the same searches.

    With fewer than four cores the stage is one PySR search in the caller's own process, already as
    warm as it can be, and it is returned as the frozen module builds it.
    """
    free = search_stage.available_cores() if cores is None else int(cores)
    if free < search_stage.CORES_FOR_PORTFOLIO:
        frozen: tuple[Any, str] = search_stage.default_search_stage(budget_seconds, cores=free, seed=seed)
        return frozen
    stage = WarmPortfolio(mixed_members(budget_seconds, seed), max_proposals=search_stage.MAX_PROPOSALS, parallel=True,
                          timeout=budget_seconds + search_stage.MEMBER_GRACE_S)
    return (FitGuardProposer(stage),
            f"four searches side by side (PREREG-PORTFOLIO-SEARCH), {budget_seconds:g} s; search processes kept warm")


def stage_in_force_warm(budget_seconds: float, *, cores: int | None = None, seed: int = 0) -> tuple[Any, str]:
    """``stage_in_force.search_stage`` with kept search processes, and the memory beside them unless it is switched off."""
    from apedp_dyn.discovery_l2 import memory_stage, stage_in_force

    searches, described = search_stage_warm(budget_seconds, cores=cores, seed=seed)
    if not stage_in_force.memory_in_force():
        return searches, described
    memory = memory_stage.recorded_memory()
    return (memory_stage.with_memory(searches, memory),
            f"{described}; and laws grown from {len(memory.motifs)} remembered motifs")
