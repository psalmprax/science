"""Pins for search processes that are kept: the same proposals, the same processes, and no waiting for one that will not answer."""
from __future__ import annotations

import multiprocessing
import time
from collections.abc import Sequence
from functools import partial
from typing import Any

import pandas as pd
import pytest

from apedp_dyn.discovery_l2 import search_stage
from apedp_dyn.proposers import warm_pool
from apedp_dyn.proposers.portfolio import FixedFormulasProposer, PortfolioProposer

FRAME = pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})


class _NeverAnswers(FixedFormulasProposer):
    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Any:
        time.sleep(3600.0)
        return super().fit_and_propose(train_data, target_col=target_col, guesses=guesses, **kwargs)


def _formulas(proposer: PortfolioProposer) -> list[str]:
    return [str(p.formula) for p in proposer.fit_and_propose(FRAME, target_col="target")]


@pytest.fixture
def pool():
    fresh = warm_pool.WarmPool()
    yield fresh
    fresh.close()


def _portfolio(members: list[Any], pool: warm_pool.WarmPool, timeout: float = 120.0) -> warm_pool.WarmPortfolio:
    portfolio = warm_pool.WarmPortfolio(members, parallel=True, timeout=timeout)
    portfolio.pool = pool
    return portfolio


def test_kept_processes_propose_what_the_portfolio_proposes_and_are_used_again(pool) -> None:
    members = [partial(FixedFormulasProposer, ["2*x0", "x0 + x0"]), partial(FixedFormulasProposer, ["x0**2", "2*x0"])]
    plain = _formulas(PortfolioProposer(members, parallel=True, timeout=120.0))
    kept = _portfolio(members, pool)
    assert _formulas(kept) == plain and kept.failures == []
    first = pool.processes()
    assert len(first) == 2 and pool.started == 2
    assert _formulas(kept) == plain                               # a second table: the same two processes, none started
    assert pool.processes() == first and pool.started == 2 and pool.killed == 0


def test_a_member_that_fails_costs_only_its_own_candidates(pool) -> None:
    members = [partial(FixedFormulasProposer, ["2*x0"]), partial(FixedFormulasProposer, ["x0**2"], fail="no Julia here")]
    kept = _portfolio(members, pool)
    assert _formulas(kept) == ["2*x0"]
    assert kept.failures == ["member 1: RuntimeError: no Julia here"]
    assert len(pool.processes()) == 2                              # a failed search leaves its process in place


def test_a_search_that_does_not_answer_in_time_is_killed_and_replaced_not_waited_for(pool) -> None:
    answering = [partial(FixedFormulasProposer, ["2*x0"]), partial(FixedFormulasProposer, ["x0**2"])]
    kept = _portfolio(answering, pool)
    assert _formulas(kept) == ["2*x0", "x0**2"] and pool.started == 2   # both processes are up and have answered once
    kept._members = [answering[0], partial(_NeverAnswers, ["x0**2"])]
    kept._timeout = 4.0
    started = time.monotonic()
    assert _formulas(kept) == ["2*x0"]
    assert time.monotonic() - started < 60.0
    assert kept.failures == ["member 1: no answer within 4 s"] and pool.killed == 1
    assert len(pool.processes()) == 1                              # the silent one is gone; the other is kept
    kept._members, kept._timeout = answering, 120.0
    assert _formulas(kept) == ["2*x0", "x0**2"] and pool.started == 3   # one fresh process took the killed one's place


def test_a_kept_process_is_replaced_after_its_share_of_searches(pool, monkeypatch) -> None:
    monkeypatch.setattr(warm_pool, "RECYCLE_AFTER", 2)
    kept = _portfolio([partial(FixedFormulasProposer, ["2*x0"]), partial(FixedFormulasProposer, ["x0**2"])], pool)
    for _ in range(3):
        assert _formulas(kept) == ["2*x0", "x0**2"]
    assert pool.started == 4 and pool.recycled == 2 and pool.killed == 0


def test_closing_leaves_no_process_behind(pool) -> None:
    kept = _portfolio([partial(FixedFormulasProposer, ["2*x0"]), partial(FixedFormulasProposer, ["x0**2"])], pool)
    assert _formulas(kept) == ["2*x0", "x0**2"]
    pool.close()
    assert pool.processes() == [] and not [p for p in multiprocessing.active_children() if p.name == "kept-search"]


def test_one_member_or_not_parallel_runs_in_the_caller_s_process(pool) -> None:
    alone = _portfolio([partial(FixedFormulasProposer, ["2*x0"])], pool)
    assert _formulas(alone) == ["2*x0"] and pool.started == 0


def test_the_stage_is_the_stage_in_force_with_kept_processes(monkeypatch) -> None:
    stage, described = warm_pool.search_stage_warm(90.0, cores=4, seed=3)
    reference, reference_described = search_stage.default_search_stage(90.0, cores=4, seed=3)
    assert described.startswith(reference_described) and described.endswith("search processes kept warm")
    portfolio, frozen = stage._inner, reference._inner
    assert isinstance(portfolio, warm_pool.WarmPortfolio) and type(frozen) is PortfolioProposer
    assert (portfolio._timeout, portfolio._max, portfolio._parallel) == (frozen._timeout, frozen._max, frozen._parallel)
    for kept, plain in zip(portfolio._members, frozen._members, strict=True):
        assert (kept.func, kept.args, kept.keywords) == (plain.func, plain.args, plain.keywords)
    _few, few_described = warm_pool.search_stage_warm(30.0, cores=2, seed=0)
    assert few_described == search_stage.default_search_stage(30.0, cores=2, seed=0)[1]
    monkeypatch.delenv(warm_pool.ENVIRONMENT_VARIABLE, raising=False)
    assert warm_pool.asked_for() is False
    monkeypatch.setenv(warm_pool.ENVIRONMENT_VARIABLE, " Warm")
    assert warm_pool.asked_for() is True


def test_the_products_reading_builds_it_only_where_the_environment_asks(monkeypatch) -> None:
    import sys

    from apedp_dyn.proposers import hard_exit
    from apedp_dyn.ui import route_endpoint

    monkeypatch.setitem(sys.modules, "pysr", type(sys)("pysr"))          # the endpoint only asks whether it can be loaded
    monkeypatch.setattr(hard_exit, "stage_in_force_that_cannot_hang", lambda budget, **kw: ("bounded", f"bounded, {budget:g} s"))
    monkeypatch.setattr(warm_pool, "stage_in_force_warm", lambda budget, **kw: ("kept", f"kept, {budget:g} s"))
    monkeypatch.delenv(warm_pool.ENVIRONMENT_VARIABLE, raising=False)
    assert route_endpoint._proposer(12.0) == ("bounded", "bounded, 12 s")           # the default is unchanged
    monkeypatch.setenv(warm_pool.ENVIRONMENT_VARIABLE, warm_pool.WARM)
    assert route_endpoint._proposer(12.0) == ("kept", "kept, 12 s")
    assert route_endpoint.status_block()["search_processes"]["in_force"].startswith("kept from one table to the next")


def test_a_measurement_can_raise_the_reading_s_search_ceiling_for_its_own_process(monkeypatch) -> None:
    from apedp_dyn.ui import route_endpoint

    monkeypatch.delenv(route_endpoint.MAX_SEARCH_SECONDS_VARIABLE, raising=False)
    assert route_endpoint._search_budget(7200.0) == route_endpoint.MAX_SEARCH_SECONDS == 300.0      # a web request's ceiling
    monkeypatch.setenv(route_endpoint.MAX_SEARCH_SECONDS_VARIABLE, "7200")
    assert route_endpoint._search_budget(7200.0) == 7200.0 and route_endpoint._search_budget(90.0) == 90.0
    monkeypatch.setenv(route_endpoint.MAX_SEARCH_SECONDS_VARIABLE, "not a number")
    assert route_endpoint._search_budget(7200.0) == 300.0
