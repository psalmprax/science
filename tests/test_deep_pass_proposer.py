"""The exhaustive pass at a larger size limit, as a portfolio member: finds what the shallow pass cannot, decides nothing."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import sympy as sp

from apedp_dyn.discovery_l2 import exhaustive_search as es
from apedp_dyn.proposers import deep_pass
from apedp_dyn.proposers.deep_pass import DeepPassProposer, mixed_members
from apedp_dyn.proposers.portfolio import PortfolioProposer
from apedp_dyn.proposers.pysr_proposer import PySRProposer

RNG = np.random.default_rng(7)
X1 = RNG.uniform(0.5, 2.5, 200)
X2 = RNG.uniform(0.5, 2.5, 200)


def _frame(y: np.ndarray, **columns: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({**columns, "target": y})


def _same(formula: str, truth: sp.Expr, names: list[str]) -> bool:
    symbols = {n: sp.Symbol(n) for n in names}
    found = sp.sympify(formula, locals=symbols)
    points = RNG.uniform(0.6, 2.4, (40, len(names)))
    a = np.asarray(sp.lambdify(list(symbols.values()), found, "numpy")(*points.T), dtype=float)
    b = np.asarray(sp.lambdify(list(symbols.values()), truth, "numpy")(*points.T), dtype=float)
    return bool(np.allclose(a, b, rtol=1e-6, atol=1e-9))


def test_a_law_one_node_past_the_shallow_limit_is_found_by_the_deeper_pass():
    a, b = sp.symbols("a b")
    truth = 3.0 * sp.sqrt(a * a + b) + 1.0          # five nodes: a, its square, b, the sum, the root
    y = 3.0 * np.sqrt(X1 * X1 + X2) + 1.0
    frame = _frame(y, a=X1, b=X2)
    shallow = DeepPassProposer(depths={2: 4}).fit_and_propose(frame)
    deeper = DeepPassProposer(depths={2: 5})
    found = deeper.fit_and_propose(frame)
    assert not _same(shallow[0].formula, truth, ["a", "b"])
    assert len(found) == 1 and _same(found[0].formula, truth, ["a", "b"])
    assert found[0].proposer_id == "deep_pass"
    assert deeper.last_run["ran"] and deeper.last_run["depth"] == 5 and deeper.last_run["variables"] == 2
    assert 0 < deeper.last_run["fitted"] <= deeper.last_run["candidates"]


def test_it_proposes_nothing_when_no_depth_is_set_for_the_number_of_variables():
    frame = _frame(X1 + X2, a=X1, b=X2, c=X1 * 0 + 1.0, d=X2)
    proposer = DeepPassProposer()
    assert proposer.fit_and_propose(frame) == []
    assert proposer.last_run == {"ran": False, "variables": 4, "reason": "no depth is set for this many variables"}


def test_the_default_depths_are_the_measured_ones_and_stay_enumerable():
    assert dict(deep_pass.DEPTHS) == {1: 8, 2: 7, 3: 7}
    assert len(es.enumerate_expressions(1, 5)) < len(es.enumerate_expressions(1, 6))  # each level is a larger space


def test_the_target_column_can_be_named_by_the_caller():
    frame = pd.DataFrame({"p0": X1, "__target__": 2.0 * np.exp(X1) - 1.0})
    found = DeepPassProposer(depths={1: 3}).fit_and_propose(frame, target_col="__target__")
    assert _same(found[0].formula, 2.0 * sp.exp(sp.Symbol("p0")) - 1.0, ["p0"])


def test_the_mixed_portfolio_has_four_different_picklable_members_and_the_baseline_search_first():
    members = mixed_members(budget_seconds=12.0, seed=5)
    assert len(members) == 4
    for factory in members:
        assert pickle.dumps(factory)  # spawn needs members that can be pickled
    assert members[0].func is PySRProposer and members[0].keywords == {"budget_seconds": 12.0, "seed": 5}
    assert members[1].func is DeepPassProposer
    assert members[2].func is PySRProposer and members[2].keywords["unary"] == ("sqrt", "square")
    assert members[3].func is PySRProposer and members[3].keywords["seed"] == 6
    assert not {"exp", "log", "sin", "cos"} & set(deep_pass.ALGEBRAIC_UNARY)


def test_in_a_portfolio_it_is_one_voice_among_the_members():
    from functools import partial

    from apedp_dyn.proposers.portfolio import FixedFormulasProposer

    y = 3.0 * np.sqrt(X1 * X1 + X2) + 1.0
    portfolio = PortfolioProposer([partial(FixedFormulasProposer, ["a + b"]), partial(DeepPassProposer, depths={2: 5})],
                                  parallel=False)
    formulas = [p.formula for p in portfolio.fit_and_propose(_frame(y, a=X1, b=X2))]
    assert formulas[0] == "a + b" and len(formulas) == 2
    assert portfolio.last_run["proposed"] == [1, 1] and portfolio.failures == []
