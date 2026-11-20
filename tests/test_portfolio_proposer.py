"""Several searches side by side, merged: every member's best first, a failing member is not fatal."""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.discovery_l2 import default_route
from apedp_dyn.proposers.portfolio import FixedFormulasProposer, PortfolioProposer, pysr_members

FRAME = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 1.0, 4.0, 3.0], "target": [2.0, 2.0, 12.0, 12.0]})


def _members(*lists, fail=()):
    return [partial(FixedFormulasProposer, formulas, fail="broken" if i in fail else "") for i, formulas in enumerate(lists)]


def test_every_members_best_comes_before_anyones_second_and_duplicates_are_merged():
    portfolio = PortfolioProposer(_members(["a*b", "a + b", "a"], ["a/b", "a*b", "b"]), parallel=False)
    formulas = [p.formula for p in portfolio.fit_and_propose(FRAME)]
    assert formulas == ["a*b", "a/b", "a + b", "a", "b"]
    assert portfolio.last_run["proposed"] == [3, 3] and portfolio.last_run["distinct"] == 5
    assert portfolio.last_run["merged_from"] == [0, 1, 0, 0, 1]


def test_the_cap_keeps_the_best_of_each_member():
    portfolio = PortfolioProposer(_members(["a", "a + 1", "a + 2", "a + 3"], ["b", "b + 1", "b + 2", "b + 3"]),
                                  max_proposals=4, parallel=False)
    assert [p.formula for p in portfolio.fit_and_propose(FRAME)] == ["a", "b", "a + 1", "b + 1"]


def test_a_failing_member_is_recorded_and_the_others_still_count():
    portfolio = PortfolioProposer(_members(["a*b"], ["a/b"], fail=(0,)), parallel=False)
    assert [p.formula for p in portfolio.fit_and_propose(FRAME)] == ["a/b"]
    assert len(portfolio.failures) == 1 and "member 0" in portfolio.failures[0] and "broken" in portfolio.failures[0]


def test_it_fails_only_when_every_member_does():
    portfolio = PortfolioProposer(_members(["a"], ["b"], fail=(0, 1)), parallel=False)
    with pytest.raises(RuntimeError, match="every member"):
        portfolio.fit_and_propose(FRAME)
    with pytest.raises(ValueError):
        PortfolioProposer([])


def test_members_run_in_separate_processes_and_give_the_same_answer():
    """The real mode: spawn, one process per member. Same merge as in-process."""
    lists = (["a*b", "a + b"], ["a/b", "a*b"], ["a - b"])
    side_by_side = PortfolioProposer(_members(*lists), parallel=True, timeout=120.0)
    in_process = PortfolioProposer(_members(*lists), parallel=False)
    assert [p.formula for p in side_by_side.fit_and_propose(FRAME)] == [p.formula for p in in_process.fit_and_propose(FRAME)]
    assert side_by_side.last_run["parallel"] is True and not side_by_side.failures


def test_a_proposal_scores_rows_like_any_other():
    proposal = PortfolioProposer(_members(["a*b"]), parallel=False).fit_and_propose(FRAME)[0]
    assert np.allclose(proposal.compute_score(FRAME), FRAME["a"] * FRAME["b"])
    assert proposal.metadata == {"member": 0, "rank": 0}


def test_the_default_route_reports_a_law_one_member_found_and_another_did_not():
    """The point of the portfolio: the right candidate from one search meets the wrong one from another."""
    rng = np.random.default_rng(0)
    X, pool = rng.uniform(1.0, 5.0, (300, 2)), rng.uniform(1.0, 5.0, (600, 2))
    law = lambda a, b: a * np.exp(-0.37 * b)                      # outside the first pass and the polynomial library
    y = law(*X.T) * (1 + 0.005 * rng.standard_normal(300))
    measure_rng = np.random.default_rng(1)
    measure = lambda rows: law(*pool[rows].T) * (1 + 0.005 * measure_rng.standard_normal(len(rows)))
    portfolio = PortfolioProposer(_members(["a/b", "a - b"], ["a*exp(-0.37*b)", "a*b"]), parallel=False)
    out = default_route.discover_law({"a": X[:, 0], "b": X[:, 1]}, y, pool, measure, proposer=portfolio)
    assert out["complete"] is True and out["law_as_confirmed"] and "exp" in out["law_as_confirmed"], out["action"]


def test_the_usual_portfolio_is_the_same_search_with_different_seeds():
    members = pysr_members(budget_seconds=30.0, seeds=(5, 6, 7))
    assert [m.keywords["seed"] for m in members] == [5, 6, 7]
    assert all(m.keywords["budget_seconds"] == 30.0 for m in members)
    import pickle

    assert len(pickle.dumps(members)) > 0          # members must cross a process boundary
