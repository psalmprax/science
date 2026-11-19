"""PySR as a proposer, and the main search path around it (first pass, search, confirmation, adequacy)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
import sympy as sp

from apedp_dyn.discovery_l2 import main_search
from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.proposers import pysr_proposer
from apedp_dyn.proposers.pysr_proposer import PySRProposer, PySRUnavailableError, hall_of_fame


class FakeRegressor:
    """Stands in for ``PySRRegressor``: returns a fixed hall of fame and records what it was given."""

    def __init__(self, equations, **settings):
        self.settings = settings
        self._equations = equations
        self.fitted_with = None

    def fit(self, X, y, variable_names=None):
        self.fitted_with = (X.shape, list(variable_names))
        self.equations_ = pd.DataFrame([{"sympy_format": sp.sympify(text), "complexity": c, "loss": loss, "score": score}
                                        for text, c, loss, score in self._equations])
        return self


def _factory(equations, made):
    def build(**settings):
        made.append(FakeRegressor(equations, **settings))
        return made[-1]
    return build


def test_hall_of_fame_puts_pysrs_pick_first_then_orders_by_loss():
    model = FakeRegressor([("x0", 1, 5.0, 0.0), ("x0*x1", 3, 0.011, 2.0), ("x0*x1 + 0.01*x0", 7, 0.010, 0.1),
                           ("x0 + x1", 3, 2.0, 0.4)]).fit(np.zeros((2, 2)), np.zeros(2), ["x0", "x1"])
    order = [str(r["expr"]) for r in hall_of_fame(model)]
    assert order[0] == "x0*x1"  # within 1.5 times the lowest loss, and the highest score
    assert [sp.sympify(o) for o in order[1:]] == [sp.sympify(t) for t in ("x0*x1 + 0.01*x0", "x0 + x1", "x0")]


def test_proposer_emits_candidates_in_the_callers_feature_names_and_decides_nothing():
    made: list[FakeRegressor] = []
    proposer = PySRProposer(budget_seconds=5, max_proposals=2,
                            regressor_factory=_factory([("x0*exp(-0.5*x1)", 6, 0.01, 1.0), ("x0", 1, 3.0, 0.0), ("x1", 1, 4.0, 0.0)], made))
    frame = pd.DataFrame({"mass": [1.0, 2.0, 3.0], "sin": [0.0, 1.0, 2.0], "target": [1.0, 1.2, 1.1]})
    proposals = proposer.fit_and_propose(frame, guesses=["mass/sin"])
    assert made[0].fitted_with == ((3, 2), ["x0", "x1"])  # neutral names: a column called "sin" cannot collide
    assert made[0].settings["guesses"] == ["x0/x1"] and made[0].settings["budget_seconds"] == 5.0
    assert len(proposals) == 2 and all(isinstance(p, CandidateProposal) for p in proposals)
    best = proposals[0]
    assert best.proposer_id == "pysr_proposer" and "mass" in best.formula and "x0" not in best.formula
    assert best.compute_score(frame) == pytest.approx(frame["mass"] * np.exp(-0.5 * frame["sin"]))
    assert best.metadata["selection_pool_size"] == 3 and best.parameters["complexity"] == 6
    assert not hasattr(best, "accepted")


def test_missing_pysr_is_an_explicit_error_not_a_silent_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "pysr", None)
    with pytest.raises(PySRUnavailableError):
        pysr_proposer.default_regressor(budget_seconds=1, binary=("+",), unary=(), seed=0, guesses=None)


class _Ledger:
    def __init__(self):
        self.records = []

    def record_trial(self, **fields):
        self.records.append(fields)


class _Scripted(ProposerBase):
    """A proposer that returns fixed formulas, or fails the test if it is called at all."""

    def __init__(self, formulas=None):
        self.formulas, self.calls = formulas, []

    @property
    def proposer_id(self) -> str:
        return "scripted"

    def fit_and_propose(self, train_data, target_col=None, guesses=None, **kwargs):
        if self.formulas is None:
            raise AssertionError("the search stage must not run after a confirmed early stop")
        self.calls.append({"columns": list(train_data.columns), "guesses": list(guesses or ())})
        return tuple(CandidateProposal(proposer_id="scripted", candidate_id=f"s{i}", formula=f, strategy_archetype=None,
                                       declared_scale_invariant=False) for i, f in enumerate(self.formulas))


def _problem(law, noise=0.01, seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, (n, 2))
    pool = rng.uniform(1.0, 5.0, (600, 2))
    fn = sp.lambdify([sp.Symbol("a"), sp.Symbol("b")], sp.sympify(law), "numpy")
    y = fn(*X.T) * (1 + noise * rng.standard_normal(n))
    read: list[int] = []

    def measure(rows):
        read.extend(int(r) for r in rows)
        return fn(*pool[rows].T) * (1 + noise * np.random.default_rng(seed + 1).standard_normal(len(rows)))

    return {"a": X[:, 0], "b": X[:, 1]}, y, pool, measure, read


def _same_shape(law: str, truth: str) -> bool:
    """True when ``law`` is ``truth`` up to a scale and an offset (which the searches fit)."""
    grid = np.random.default_rng(9).uniform(1.0, 5.0, (200, 2))
    symbols = [sp.Symbol("a"), sp.Symbol("b")]
    u = sp.lambdify(symbols, sp.sympify(law), "numpy")(*grid.T)
    v = sp.lambdify(symbols, sp.sympify(truth), "numpy")(*grid.T)
    return bool(abs(np.corrcoef(u, v)[0, 1]) > 0.999999)


def test_first_pass_settles_a_law_the_enumeration_can_express_without_calling_the_search():
    X, y, pool, measure, read = _problem("a/(a + b)")
    ledger = _Ledger()
    result = main_search.discover(X, y, pool, measure, proposer=_Scripted(None), ledger=ledger, search_id="t1", domain="chemistry")
    assert result.action in ("confirmed", "unchallenged") and result.found_by == "first_pass"
    assert _same_shape(result.law, "a/(a + b)")
    assert result.first_pass["stopped_early"] and not result.search["ran"]
    assert result.adequacy["passed"]
    record = ledger.records[0]
    assert record["status"] == result.action and record["proposer_id"] == "main_search"
    assert record["metadata"]["resolved_by"] == "contrast" and record["metadata"]["domain"] == "chemistry"
    assert len(read) == len(set(read))  # every confirmation target was read once


def test_search_is_seeded_by_the_first_pass_and_its_law_is_confirmed_against_a_rival():
    X, y, pool, measure, _read = _problem("a*exp(-0.37*b)")  # a constant inside: outside the enumeration
    proposer = _Scripted(["a*exp(-0.37*b)", "a/b"])
    result = main_search.discover(X, y, pool, measure, proposer=proposer, always_search=True)
    assert proposer.calls and proposer.calls[0]["columns"] == ["a", "b", "__target__"] and len(proposer.calls[0]["guesses"]) == 1
    assert result.search == {"ran": True, "proposals": 2, "proposer_id": "scripted"}
    assert result.action in ("confirmed", "equivalent_class", "unchallenged")
    assert result.found_by == "search" and _same_shape(result.law, "a*exp(-0.37*b)")
    assert result.adequacy["passed"]


def test_a_wrong_law_that_wins_the_contest_is_still_refused_when_its_residuals_have_structure():
    """With the true law absent, confirmation alone names the least bad rival; the adequacy check refuses it."""
    X, y, pool, measure, _read = _problem("a*sin(3*b)", noise=0.005)
    result = main_search.discover(X, y, pool, measure, proposer=_Scripted(["a + b", "a*b"]), always_search=True)
    assert result.action == "widen_search" and result.law is None
    if result.adequacy:
        assert result.adequacy["passed"] is False and result.rejected_law is not None


def test_without_a_proposer_the_path_is_the_first_pass_alone():
    X, y, pool, measure, _read = _problem("a/b")
    result = main_search.discover(X, y, pool, measure)
    assert result.search == {"ran": False, "proposals": 0}
    assert result.found_by == "first_pass" and result.action in ("confirmed", "unchallenged")


def test_competition_runs_the_main_search_and_no_longer_runs_the_retired_engine(monkeypatch):
    from apedp_dyn.orchestration import competition

    rng = np.random.default_rng(0)
    X = {"a": rng.uniform(1, 5, 120), "b": rng.uniform(1, 5, 120)}
    Y = X["a"] / X["b"]
    monkeypatch.setitem(sys.modules, "pysr", None)  # as on a machine without PySR
    results = {r.engine: r for r in competition.run_competition(X, Y)}
    assert list(results) == list(competition.ENGINE_IDS) and "pysr" in results
    assert results["pysr"].status == competition.NOT_APPLICABLE and "PySR could not be loaded" in results["pysr"].reason
    assert results["rime_ad"].status == competition.NOT_APPLICABLE and "retired" in results["rime_ad"].reason
    assert "PREREG-SRBENCH" in competition.RETIRED_ENGINES["rime_ad"]


def test_among_laws_that_cannot_be_told_apart_the_simplest_is_reported():
    """A long search returns accurate approximations beside the law; on the domain they are equal."""
    X, y, pool, measure, _read = _problem("a*exp(-0.37*b)", noise=0.02)
    padded = "a*exp(-0.37*b)*(1 + 0.00001*sin(a))"  # the same predictions to five decimals, more operations
    result = main_search.discover(X, y, pool, measure, proposer=_Scripted([padded, "a*exp(-0.37*b)"]), always_search=True)
    assert result.action == "equivalent_class"
    assert "sin" not in result.law and _same_shape(result.law, "a*exp(-0.37*b)")
    assert result.adequacy["passed"]
