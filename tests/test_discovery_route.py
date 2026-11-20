"""The switch between the two law-discovery routes, and the ladder's new widening step."""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.discovery_l2 import route
from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.epistemic.trial_ledger import CandidateTrialLedger
from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase


class _Scripted(ProposerBase):
    def __init__(self, formulas):
        self.formulas, self.calls = formulas, 0

    @property
    def proposer_id(self) -> str:
        return "scripted"

    def fit_and_propose(self, train_data, target_col=None, guesses=None, **kwargs):
        self.calls += 1
        return tuple(CandidateProposal(proposer_id="scripted", candidate_id=f"s{i}", formula=f, strategy_archetype=None,
                                       declared_scale_invariant=False) for i, f in enumerate(self.formulas))


def _problem(fn, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, (300, 2))
    pool = rng.uniform(1.0, 5.0, (600, 2))
    y = fn(X[:, 0], X[:, 1]) * (1 + noise * rng.standard_normal(300))
    measure_rng = np.random.default_rng(seed + 1)
    return ({"a": X[:, 0], "b": X[:, 1]}, y, pool,
            lambda rows: fn(pool[rows, 0], pool[rows, 1]) * (1 + noise * measure_rng.standard_normal(len(rows))))


def test_default_route_is_the_recorded_one_and_the_environment_can_choose(monkeypatch):
    monkeypatch.delenv(route.ENVIRONMENT_VARIABLE, raising=False)
    # main_search since PREREG-MAIN-SEARCH-2026-11's expectations 1 to 3 held; the change is a recorded edit
    assert route.default_route() == route.DEFAULT_ROUTE == "main_search"
    monkeypatch.setenv(route.ENVIRONMENT_VARIABLE, "l2")
    assert route.default_route() == "l2"
    monkeypatch.setenv(route.ENVIRONMENT_VARIABLE, "something_else")
    with pytest.raises(ValueError):
        route.default_route()


def test_main_search_route_finds_a_simple_law_and_records_one_trial(tmp_path):
    X, y, pool, measure = _problem(lambda a, b: a / b)
    ledger = CandidateTrialLedger(tmp_path / "ledger.jsonl")
    out = route.discover_law(X, y, pool, measure, ledger=ledger, search_id="r1", route="main_search", domain="demo")
    assert out["route"] == "main_search" and out["action"] in ("confirmed", "unchallenged") and out["law"]
    assert out["law_as_confirmed"] and "a" in out["law_as_confirmed"] and "b" in out["law_as_confirmed"]
    records = ledger.get_records()
    assert len(records) == 1 and records[0].metadata["resolved_by"] == "contrast" and records[0].metadata["domain"] == "demo"


def test_when_the_ladder_is_exhausted_the_widening_step_is_the_main_search(tmp_path):
    """exp(-0.37*b) is outside the template rungs; with a proposer the ladder hands over instead of refusing."""
    X, y, pool, measure = _problem(lambda a, b: a * np.exp(-0.37 * b))
    dims = {"a": Dimension(), "b": Dimension()}
    alone = route.discover_law(X, y, pool, measure, ledger=CandidateTrialLedger(tmp_path / "l1.jsonl"), search_id="r2",
                               route="l2", dimensions=dims, target_dimension=Dimension())
    proposer = _Scripted(["a*exp(-0.37*b)", "a/b"])
    widened = route.discover_law(X, y, pool, measure, ledger=CandidateTrialLedger(tmp_path / "l2.jsonl"), search_id="r3",
                                 route="l2", dimensions=dims, target_dimension=Dimension(), proposer=proposer)
    assert alone["route"] == "l2" and alone["main_search"] is None
    if alone["action"] == "widen_search":
        assert widened["route"] == "l2+main_search" and proposer.calls == 1
        assert widened["action"] in ("confirmed", "equivalent_class", "unchallenged") and "exp" in widened["law"]
    else:  # the ladder settled it on its own: the main search must not have been called
        assert widened["route"] == "l2" and proposer.calls == 0


def test_l2_route_needs_dimensions():
    X, y, pool, measure = _problem(lambda a, b: a / b)
    with pytest.raises(ValueError):
        route.discover_law(X, y, pool, measure, ledger=None, search_id="r4", route="l2")


def test_the_confirmed_form_frees_every_additive_term():
    """A search can return a tied form whose refit is a different law; the refit is what was confirmed."""
    rng = np.random.default_rng(3)
    X = {"a": rng.uniform(1.0, 3.0, 200), "b": rng.uniform(1.0, 3.0, 200)}
    y = 2.0 * X["a"] * X["b"]                      # the cross term of (a + b)**2 alone
    refit = route.as_confirmed("(a + b)**2", X, y, noise="absolute")
    import sympy as sp
    a, b = sp.symbols("a b")
    coefficients = sp.Poly(sp.sympify(refit), a, b).as_dict()
    assert coefficients[(1, 1)] == pytest.approx(2.0, abs=1e-9)
    assert abs(coefficients.get((2, 0), 0.0)) < 1e-9 and abs(coefficients.get((0, 2), 0.0)) < 1e-9


def test_the_default_route_is_taken_when_none_is_named(tmp_path, monkeypatch):
    monkeypatch.delenv(route.ENVIRONMENT_VARIABLE, raising=False)
    X, y, pool, measure = _problem(lambda a, b: a / b)
    out = route.discover_law(X, y, pool, measure, ledger=CandidateTrialLedger(tmp_path / "ledger.jsonl"), search_id="r9")
    assert out["route"] == "main_search"
