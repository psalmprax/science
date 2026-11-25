"""Search processes that cannot hang while leaving: same proposals, and a worker that skips later exit hooks."""
from __future__ import annotations

import multiprocessing
import signal
import threading
import time
from collections.abc import Sequence
from functools import partial
from typing import Any

import pandas as pd

from apedp_dyn.discovery_l2 import search_stage
from apedp_dyn.proposers import hard_exit
from apedp_dyn.proposers.portfolio import FixedFormulasProposer, PortfolioProposer

FRAME = pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})


class _WillNotGo(FixedFormulasProposer):
    """A member standing for a search process caught in Julia's exit: it ignores being told to stop and never answers."""

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Any:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(3600.0)
        return super().fit_and_propose(train_data, target_col=target_col, guesses=guesses, **kwargs)


def _formulas(proposer: PortfolioProposer) -> list[str]:
    return [str(p.formula) for p in proposer.fit_and_propose(FRAME, target_col="target")]


def test_the_wrapped_members_propose_what_the_members_propose() -> None:
    members = [partial(FixedFormulasProposer, ["2*x0", "x0 + x0"]), partial(FixedFormulasProposer, ["x0**2", "2*x0"])]
    plain = _formulas(PortfolioProposer(members, parallel=False))
    wrapped = _formulas(PortfolioProposer([hard_exit.HardExit(m) for m in members], parallel=False))
    assert wrapped == plain and plain


def test_in_worker_processes_the_wrapped_members_still_answer() -> None:
    members = [partial(FixedFormulasProposer, ["2*x0"]), partial(FixedFormulasProposer, ["x0**2"])]
    plain = _formulas(PortfolioProposer(members, parallel=True, timeout=120.0))
    started = time.monotonic()
    wrapped = _formulas(PortfolioProposer([hard_exit.HardExit(m) for m in members], parallel=True, timeout=120.0))
    assert wrapped == plain == ["2*x0", "x0**2"]
    assert time.monotonic() - started < 120.0


def test_a_main_process_is_never_set_to_leave_hard() -> None:
    """A search run in the caller's own process must leave that process to close down properly."""
    assert multiprocessing.parent_process() is None
    assert hard_exit.leave_without_the_exit_hooks() is False
    member = hard_exit.HardExit(partial(FixedFormulasProposer, ["2*x0"]))()
    assert [str(p.formula) for p in member.fit_and_propose(FRAME, target_col="target")] == ["2*x0"]
    assert member.proposer_id == "fixed_formulas"
    assert hard_exit._armed is False


def _slow_hook_then_search(armed: bool, done: multiprocessing.queues.Queue) -> None:  # type: ignore[name-defined]
    """In a worker: an exit hook that would take a minute (standing for Julia's), then a member's search."""
    import atexit

    atexit.register(time.sleep, 60.0)
    factory = partial(FixedFormulasProposer, ["2*x0"])
    member = hard_exit.HardExit(factory)() if armed else factory()
    done.put([str(p.formula) for p in member.fit_and_propose(FRAME, target_col="target")])


def test_a_worker_leaves_at_once_and_skips_an_exit_hook_registered_before_its_search() -> None:
    context = multiprocessing.get_context("spawn")
    done = context.Queue()
    process = context.Process(target=_slow_hook_then_search, args=(True, done))
    process.start()
    try:
        assert done.get(timeout=120.0) == ["2*x0"]              # the answer is handed over first
        process.join(timeout=30.0)                              # then it is gone, well inside the hook's minute
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join()


def test_a_search_process_that_will_not_go_is_killed_and_the_other_answers_are_kept(monkeypatch, capfd) -> None:  # type: ignore[no-untyped-def]
    """The frozen portfolio would wait for this process for ever; the bounded one ends the wait from outside."""
    monkeypatch.setattr(hard_exit, "STOP_GRACE_S", 2.0)
    members = [partial(FixedFormulasProposer, ["2*x0"]), partial(_WillNotGo, ["x0**2"])]
    portfolio = hard_exit.BoundedPortfolio(members, parallel=True, timeout=6.0)
    started = time.monotonic()
    assert _formulas(portfolio) == ["2*x0"]                      # the member that answered is kept
    # the wait ended because it was seen to be a wait (2 s into it), not by the fallback a minute after the deadline
    assert time.monotonic() - started < 6.0 + 2.0 + 40.0
    assert portfolio.failures == ["member 1: no answer within 6 s"]
    assert len(portfolio.killed_processes) == 1                  # only the process that was still there
    assert "was still there 2 s after it was told to stop and was killed" in capfd.readouterr().err
    assert not [thread for thread in threading.enumerate() if thread.name == "search-process-reaper"]
    assert not multiprocessing.active_children()


def test_the_wait_also_ends_when_it_cannot_be_seen(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Should the waiting thread's stack never show the pool's stop, the deadline plus a fallback ends the wait."""
    monkeypatch.setattr(hard_exit, "_is_stopping", lambda thread_id: False)
    monkeypatch.setattr(hard_exit, "STOP_FALLBACK_S", 2.0)
    members = [partial(FixedFormulasProposer, ["2*x0"]), partial(_WillNotGo, ["x0**2"])]
    portfolio = hard_exit.BoundedPortfolio(members, parallel=True, timeout=5.0)
    started = time.monotonic()
    assert _formulas(portfolio) == ["2*x0"]
    assert 5.0 + 2.0 <= time.monotonic() - started < 90.0
    assert len(portfolio.killed_processes) == 1 and not multiprocessing.active_children()


def test_a_thread_that_is_not_stopping_a_pool_is_not_taken_for_one() -> None:
    assert hard_exit._is_stopping(threading.get_ident()) is False
    assert hard_exit._is_stopping(-1) is False                   # no such thread


def test_the_bounded_portfolio_kills_nothing_when_every_process_goes() -> None:
    members = [partial(FixedFormulasProposer, ["2*x0"]), partial(FixedFormulasProposer, ["x0**2"])]
    plain = _formulas(PortfolioProposer(members, parallel=True, timeout=120.0))
    bounded = hard_exit.BoundedPortfolio([hard_exit.HardExit(m) for m in members], parallel=True, timeout=120.0)
    assert _formulas(bounded) == plain == ["2*x0", "x0**2"]
    assert bounded.killed_processes == [] and bounded.failures == []
    assert not [thread for thread in threading.enumerate() if thread.name == "search-process-reaper"]
    in_this_process = hard_exit.BoundedPortfolio(members, parallel=False)
    assert _formulas(in_this_process) == plain and in_this_process.killed_processes == []


def test_the_stage_is_the_stage_in_force_with_wrapped_members() -> None:
    stage, described = hard_exit.search_stage_that_cannot_hang(90.0, cores=4, seed=3)
    reference, reference_described = search_stage.default_search_stage(90.0, cores=4, seed=3)
    assert described.startswith(reference_described) and described.endswith("search processes leave hard")
    portfolio, frozen = stage._inner, reference._inner
    assert isinstance(portfolio, hard_exit.BoundedPortfolio) and type(frozen) is PortfolioProposer
    assert len(portfolio._members) == len(frozen._members) == 4
    assert all(isinstance(member, hard_exit.HardExit) for member in portfolio._members)
    assert (portfolio._timeout, portfolio._max, portfolio._parallel) == (frozen._timeout, frozen._max, frozen._parallel)
    # the members wrapped are the members in force: the same factories with the same settings
    for wrapped, plain in zip(portfolio._members, frozen._members, strict=True):
        inner = wrapped._factory
        assert (inner.func, inner.args, inner.keywords) == (plain.func, plain.args, plain.keywords)


def test_with_fewer_than_four_cores_the_stage_is_the_frozen_one() -> None:
    stage, described = hard_exit.search_stage_that_cannot_hang(30.0, cores=2, seed=0)
    _reference, reference_described = search_stage.default_search_stage(30.0, cores=2, seed=0)
    assert described == reference_described
    assert not isinstance(stage._inner, PortfolioProposer)


def test_nothing_in_force_builds_the_hard_exit_stage() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    users = sorted(path.relative_to(root).as_posix() for path in (root / "apedp_dyn").rglob("*.py")
                   if "hard_exit" in path.read_text(encoding="utf-8") and path.name != "hard_exit.py")
    # the recursion package, which is not in force, and the product's reading, which builds it only where asked
    assert users == ["apedp_dyn/recursion/levels.py", "apedp_dyn/ui/route_endpoint.py"]


def test_the_products_reading_builds_it_only_where_the_environment_asks(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from apedp_dyn.discovery_l2 import stage_in_force

    monkeypatch.delenv(hard_exit.ENVIRONMENT_VARIABLE, raising=False)
    assert hard_exit.asked_for() is False                       # off unless asked: what is in force is unchanged
    monkeypatch.setenv(hard_exit.ENVIRONMENT_VARIABLE, "LEAVE_HARD ")
    assert hard_exit.asked_for() is True
    monkeypatch.setenv(hard_exit.ENVIRONMENT_VARIABLE, "yes")
    assert hard_exit.asked_for() is False

    stage, described = hard_exit.stage_in_force_that_cannot_hang(90.0, cores=4, seed=1)
    _reference, reference_described = stage_in_force.search_stage(90.0, cores=4, seed=1)
    # the same description as the stage in force, with the one difference named in the middle of it
    assert described.replace("; search processes leave hard", "") == reference_described
    searches = stage._searches._inner if hasattr(stage, "_searches") else stage._inner
    assert all(isinstance(member, hard_exit.HardExit) for member in searches._members)

    monkeypatch.setenv(stage_in_force.ENVIRONMENT_VARIABLE, stage_in_force.OFF)
    alone, alone_described = hard_exit.stage_in_force_that_cannot_hang(90.0, cores=4, seed=1)
    assert alone_described.endswith("search processes leave hard") and isinstance(alone._inner, PortfolioProposer)
