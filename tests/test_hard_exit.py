"""Search processes that cannot hang while leaving: same proposals, and a worker that skips later exit hooks."""
from __future__ import annotations

import multiprocessing
import time
from functools import partial

import pandas as pd

from apedp_dyn.discovery_l2 import search_stage
from apedp_dyn.proposers import hard_exit
from apedp_dyn.proposers.portfolio import FixedFormulasProposer, PortfolioProposer

FRAME = pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})


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


def test_the_stage_is_the_stage_in_force_with_wrapped_members() -> None:
    stage, described = hard_exit.search_stage_that_cannot_hang(90.0, cores=4, seed=3)
    reference, reference_described = search_stage.default_search_stage(90.0, cores=4, seed=3)
    assert described.startswith(reference_described) and described.endswith("search processes leave hard")
    portfolio, frozen = stage._inner, reference._inner
    assert isinstance(portfolio, PortfolioProposer) and isinstance(frozen, PortfolioProposer)
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
    assert users == ["apedp_dyn/recursion/levels.py"]            # the recursion package, which is not in force
