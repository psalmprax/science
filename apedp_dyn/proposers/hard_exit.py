"""The four searches side by side, with search processes that cannot hang while leaving.

Status: experimental

**What went wrong.** A search of the portfolio runs in a process of its own, which holds a Julia
runtime. When the portfolio is done it tells its processes to stop and waits for each to go
(``PortfolioProposer._run_members``: ``pool.terminate(); pool.join()``). A process told to stop while
Python is already closing down is inside Julia's exit hook, which at that moment is compiling; the
stop signal makes Julia start its exit a second time, and the two can wait on each other for ever.
The portfolio then waits for ever too. Seen on GitHub on 2026-11-24: in the recursion run with the
searches on, all six jobs stopped this way after 1, 1, 1, 2, 2 and 7 searches and were ended by
their five-hour limits with nothing written; one search process had been told to stop at 15:48 and
was still there at 21:01.

**What this module does.** A member built here leaves its process, once its search has answered, by
``os._exit`` from Python's own exit hooks, before Julia's exit hook is reached. The answer has been
handed over by then. Nothing about the search changes: the same members, the same seeds, the same
budget, the same order of proposals. ``apedp_dyn/proposers/portfolio.py`` and
``apedp_dyn/discovery_l2/search_stage.py`` are frozen by earlier preregistrations and are not
edited; this stage is built from their own parts.

**Confirmed on GitHub** (2026-11-24, run 45493901958): the first job to finish with these members
ran 24 searches in a row; no search process was told to stop while leaving, and none was left
over. With the frozen stage 10 of 27 searches had stopped their job for ever.

**A second way to stop for ever, and its cure** (2026-11-25). With the members above, runs still
stopped: one job of the second recursion run (45493901958), and every shard of the second
wider-conditions run (45503100152) after 10 to 20 readings. What the logs of the second show: with
three inputs no search process was ever sent a stop signal while alive (40 readings); with four
inputs one was in 19 of 20 readings, each time at the moment the searches ended, and 4 of those 19
never went. Likely why: when the last answer arrives the portfolio stops its pool at once, and the
pool's stop competes with the search processes for the lock on their task queue; the process that
answered last can lose, is never asked to leave, and is sent the signal instead. With three inputs
the last to answer is usually the deep pass, which holds no Julia runtime and dies at once. With
four inputs the deep pass does not run and the last is always a PySR search: Julia handles the
signal itself ("signal 15: Terminated") and its exit can wait for ever. Leaving hard cannot help a
process that never reaches Python's exit hooks, and nothing inside such a process can be relied
on, so :class:`BoundedPortfolio` bounds the wait from outside: once the portfolio has waited
``STOP_GRACE_S`` seconds inside the pool's own stop, the search processes still there are killed,
and each kill is printed. The answers are all in before that wait begins, so none can be lost, and
the searches themselves are unchanged.

**Not in force.** The routes and the product still build ``search_stage.default_search_stage``.
Using this stage there is the owner's decision (AGENTS.md 1.5). The product's reading can be asked
for it: ``APEDP_SEARCH_PROCESSES=leave_hard`` (``apedp_dyn/ui/route_endpoint.py``); without that
variable nothing changes.
"""
from __future__ import annotations

import atexit
import multiprocessing
import multiprocessing.util
import os
import sys
import threading
import time
from collections.abc import Sequence
from typing import Any

from apedp_dyn.discovery_l2 import search_stage
from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.proposers.deep_pass import mixed_members
from apedp_dyn.proposers.fit_guard import FitGuardProposer
from apedp_dyn.proposers.portfolio import MemberFactory, PortfolioProposer

ENVIRONMENT_VARIABLE = "APEDP_SEARCH_PROCESSES"
LEAVE_HARD = "leave_hard"
STOP_GRACE_S = 20.0     # how long the portfolio may wait for its search processes to go before they are killed
STOP_FALLBACK_S = 60.0  # the same wait counted from the searches' deadline, should the wait itself never be seen
STOP_POLL_S = 0.5

_armed = False


def asked_for() -> bool:
    """Whether the environment asks for search processes that leave hard. Off unless asked."""
    return os.environ.get(ENVIRONMENT_VARIABLE, "").strip().lower() == LEAVE_HARD


def _leave() -> None:
    """Out of the process at once: what was queued or printed is flushed, and no later exit hook is run.

    Python's exit hooks run last registered first, and multiprocessing registered its own (which sends
    what is still waiting in a ``multiprocessing.Queue``) long before this one, so it would be skipped:
    it is run here first. A pool hands a result over before its worker goes on, so the portfolio's
    answers never depended on it.
    """
    flush_queues = getattr(multiprocessing.util, "_exit_function", None)
    if callable(flush_queues):
        try:
            flush_queues()
        except Exception:  # noqa: BLE001, S110 - leaving is what matters; whatever could not be flushed is lost either way  # nosec B110
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    os._exit(0)


def leave_without_the_exit_hooks() -> bool:
    """Arrange for this process, if it is a worker, to end by ``os._exit`` when Python closes down.

    Exit hooks run last registered first, so called after the search (when Julia's hook is already
    registered) this one runs before it. Does nothing in a main process, which must close down
    properly; returns whether it is now arranged.
    """
    global _armed
    if multiprocessing.parent_process() is None:
        return False
    if not _armed:
        atexit.register(_leave)
        _armed = True
    return True


class _LeavesHard(ProposerBase):
    """A member's search, unchanged; afterwards its worker process is set to leave without Julia's exit hook."""

    def __init__(self, inner: ProposerBase) -> None:
        self._inner = inner

    @property
    def proposer_id(self) -> str:
        return str(self._inner.proposer_id)

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, **kwargs: Any) -> Sequence[CandidateProposal]:
        try:
            found: Sequence[CandidateProposal] = self._inner.fit_and_propose(train_data, target_col=target_col, **kwargs)
            return found
        finally:
            leave_without_the_exit_hooks()


class HardExit:
    """A member factory (picklable if the one it wraps is) whose member leaves its worker process hard."""

    def __init__(self, factory: MemberFactory) -> None:
        self._factory = factory

    def __call__(self) -> ProposerBase:
        return _LeavesHard(self._factory())


def _has_ended(pid: int) -> bool:
    """Whether a child process has ended, asked without collecting it (the portfolio's own wait must still find it)."""
    waitid = getattr(os, "waitid", None)
    if waitid is None:
        return False
    try:
        ended = waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except OSError:      # no such child any more: it was collected in the meantime
        return True
    return ended


def _is_stopping(thread_id: int) -> bool:
    """Whether that thread is inside the pool's own ``terminate`` or ``join``: the answers are in, the processes told to go."""
    frame = sys._current_frames().get(thread_id)
    pool_file = os.path.join("multiprocessing", "pool.py")
    while frame is not None:
        code = frame.f_code
        if code.co_name in ("terminate", "_terminate_pool", "join") and code.co_filename.endswith(pool_file):
            return True
        if code.co_name == "_run_members":      # reached the portfolio itself: whatever called it is not looked at
            return False
        frame = frame.f_back
    return False


class _Reaper(threading.Thread):
    """Kills the search processes that have not gone, once the portfolio has waited for them long enough.

    Long enough is ``STOP_GRACE_S`` seconds inside the pool's ``terminate`` or ``join`` (seen in the
    waiting thread's own stack), or, should that never be seen, ``after`` seconds from the start.
    Only child processes that were not there before the searches began are touched.
    """

    def __init__(self, known: set[int | None], after: float, waiting_thread: int) -> None:
        super().__init__(name="search-process-reaper", daemon=True)
        self._known, self._after, self._waiting_thread = known, after, waiting_thread
        self._stopped = threading.Event()
        self.killed: list[int] = []

    def stop(self) -> None:
        self._stopped.set()

    def _waited_long_enough(self) -> bool:
        """Blocks until the portfolio has waited ``STOP_GRACE_S`` for its processes; false if it returned first."""
        started = time.monotonic()
        stopping_since: float | None = None
        while not self._stopped.wait(STOP_POLL_S):
            now = time.monotonic()
            if now - started >= self._after:
                return True
            if _is_stopping(self._waiting_thread):
                stopping_since = now if stopping_since is None else stopping_since
                if now - stopping_since >= STOP_GRACE_S:
                    return True
            else:
                stopping_since = None
        return False

    def run(self) -> None:
        if not self._waited_long_enough():
            return
        while not self._stopped.is_set():
            # the processes are read from multiprocessing's own list and not through active_children(), which
            # would collect a finished process from under the portfolio's own wait for it
            for process in list(getattr(multiprocessing.process, "_children", ())):
                pid = process.pid
                if pid is None or pid in self._known or pid in self.killed or _has_ended(pid):
                    continue
                try:
                    process.kill()
                except (OSError, ValueError, AttributeError):
                    continue
                self.killed.append(pid)
                print(f"search process {pid} was still there {STOP_GRACE_S:g} s after it was told to stop and was killed",
                      file=sys.stderr, flush=True)
            self._stopped.wait(1.0)


class BoundedPortfolio(PortfolioProposer):
    """The portfolio as frozen, except that its wait for the search processes to go has an end.

    The frozen ``_run_members`` takes each answer or gives up on it by ``timeout`` (and at most a
    second a member later), then tells the processes to stop and waits for them inside the pool's
    ``terminate`` and ``join``. Here a watcher kills whatever search process is still there once
    that wait has lasted ``STOP_GRACE_S`` seconds, so it returns. Nothing more is taken from any
    process after the wait has begun, so what the portfolio proposes cannot change;
    ``killed_processes`` says which processes it was done to.
    """

    killed_processes: list[int]

    def _run_members(self, columns: dict[str, list[float]], target: str, guesses: Sequence[str] | None) -> list[list[str]]:
        self.killed_processes = []
        if not self._parallel or len(self._members) == 1:
            return super()._run_members(columns, target, guesses)
        known: set[int | None] = {process.pid for process in multiprocessing.active_children()}
        reaper = _Reaper(known, after=self._timeout + len(self._members) + STOP_FALLBACK_S, waiting_thread=threading.get_ident())
        reaper.start()
        try:
            return super()._run_members(columns, target, guesses)
        finally:
            reaper.stop()
            reaper.join(timeout=5.0)
            self.killed_processes = list(reaper.killed)


def search_stage_that_cannot_hang(budget_seconds: float, *, cores: int | None = None, seed: int = 0) -> tuple[Any, str]:
    """``search_stage.default_search_stage`` with members that leave hard; the same searches and description.

    With fewer than four cores the stage is one PySR search in the caller's own process, where
    nothing is waited for, and it is returned as the frozen module builds it.
    """
    free = search_stage.available_cores() if cores is None else int(cores)
    if free < search_stage.CORES_FOR_PORTFOLIO:
        frozen: tuple[Any, str] = search_stage.default_search_stage(budget_seconds, cores=free, seed=seed)
        return frozen
    members = [HardExit(member) for member in mixed_members(budget_seconds, seed)]
    stage = BoundedPortfolio(members, max_proposals=search_stage.MAX_PROPOSALS, parallel=True,
                             timeout=budget_seconds + search_stage.MEMBER_GRACE_S)
    return (FitGuardProposer(stage),
            f"four searches side by side (PREREG-PORTFOLIO-SEARCH), {budget_seconds:g} s; search processes leave hard")


def stage_in_force_that_cannot_hang(budget_seconds: float, *, cores: int | None = None, seed: int = 0) -> tuple[Any, str]:
    """``stage_in_force.search_stage`` with members that leave hard: the same searches, and the memory beside them
    unless the environment switches it off, composed by the frozen modules' own functions."""
    from apedp_dyn.discovery_l2 import memory_stage, stage_in_force

    searches, described = search_stage_that_cannot_hang(budget_seconds, cores=cores, seed=seed)
    if not stage_in_force.memory_in_force():
        return searches, described
    memory = memory_stage.recorded_memory()
    return (memory_stage.with_memory(searches, memory),
            f"{described}; and laws grown from {len(memory.motifs)} remembered motifs")
