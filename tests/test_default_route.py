"""The route in force for law discovery: the refined path, made the default around the frozen ``route.py``."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from apedp_dyn.discovery_l2 import default_route, route
from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase

ROOT = Path(__file__).resolve().parents[1]


class _Scripted(ProposerBase):
    """A search stage that proposes fixed formulas: stands in for PySR."""

    def __init__(self, formulas):
        self.formulas = formulas

    @property
    def proposer_id(self) -> str:
        return "scripted"

    def fit_and_propose(self, train_data, target_col=None, guesses=None, **kwargs):
        return tuple(CandidateProposal(proposer_id="scripted", candidate_id=f"s{i}", formula=f, strategy_archetype=None,
                                       declared_scale_invariant=False) for i, f in enumerate(self.formulas))


def _problem(fn, seed=0, noise=0.01, names=("a", "b")):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, (300, len(names)))
    pool = rng.uniform(1.0, 5.0, (600, len(names)))
    y = fn(*X.T) * (1 + noise * rng.standard_normal(300))
    measure_rng = np.random.default_rng(seed + 1)
    return ({n: X[:, i] for i, n in enumerate(names)}, y, pool,
            lambda rows: fn(*pool[rows].T) * (1 + noise * measure_rng.standard_normal(len(rows))))


def test_the_default_is_the_refined_path_and_the_earlier_routes_can_still_be_chosen(monkeypatch):
    monkeypatch.delenv(default_route.ENVIRONMENT_VARIABLE, raising=False)
    assert default_route.default_route() == default_route.DEFAULT_ROUTE == "refined"
    assert default_route.ROUTES == ("refined", "main_search", "l2")
    for earlier in ("main_search", "l2"):
        monkeypatch.setenv(default_route.ENVIRONMENT_VARIABLE, earlier)
        assert default_route.default_route() == earlier
    monkeypatch.setenv(default_route.ENVIRONMENT_VARIABLE, "something_else")
    with pytest.raises(ValueError):
        default_route.default_route()


def test_the_frozen_route_module_is_not_edited_to_move_the_default():
    """route.py is in PREREG-LLM-PROPOSER's digest; the default moved in a module around it."""
    assert route.DEFAULT_ROUTE == "main_search"
    digest = json.loads((ROOT / "preregistration" / "DIGEST-LLM-PROPOSER.json").read_text())["sha256"]
    path = "apedp_dyn/discovery_l2/route.py"
    assert digest[path] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_the_refined_route_finds_a_two_term_law_the_first_pass_alone_does_not():
    """-4a - 0.2b needs two fitted coefficients: the polynomial library has it, the exhaustive first pass does not."""
    X, y, pool, measure = _problem(lambda a, b: -4.0 * a - 0.2 * b, noise=0.002)
    refined = default_route.discover_law(X, y, pool, measure)
    assert refined["route"] == "refined" and refined["law_as_confirmed"], refined["action"]
    fitted = refined["law_as_confirmed"].replace(" ", "")
    assert "a" in fitted and "b" in fitted
    assert refined["main_search"] is refined["refined"].path


def test_without_a_search_stage_a_law_is_tentative_and_the_record_says_why():
    """The measured path had PySR on every problem; a polynomial 'confirmed' against the first pass alone is not that."""
    X, y, pool, measure = _problem(lambda a, b: -4.0 * a - 0.2 * b, noise=0.002)
    alone = default_route.discover_law(X, y, pool, measure)
    assert alone["complete"] is False and alone["reported"] is False and alone["law"] is None
    assert alone["tentative_law"] == alone["law_as_confirmed"]
    assert alone["tentative_reason"] in (default_route.NO_SEARCH_STAGE, default_route.NOTHING_CHALLENGED)
    if alone["action"] in ("confirmed", "equivalent_class"):
        assert alone["tentative_reason"] == default_route.NO_SEARCH_STAGE


def test_with_a_search_stage_a_confirmed_law_is_reported():
    X, y, pool, measure = _problem(lambda a, b: -4.0 * a - 0.2 * b, noise=0.002)
    out = default_route.discover_law(X, y, pool, measure, proposer=_Scripted(["a + b", "a*b", "a - b**2"]))
    assert out["complete"] is True
    assert out["reported"] == (out["action"] in ("confirmed", "equivalent_class")) == (out["law"] is not None)
    if out["reported"]:
        assert out["tentative_law"] is None and out["tentative_reason"] is None
        assert "a" in out["law"] and "b" in out["law"]


def test_a_law_nothing_challenged_is_tentative_on_every_route():
    X, y, pool, measure = _problem(lambda a, b: a / b)
    for name in ("refined", "main_search"):
        out = default_route.discover_law(X, y, pool, measure, route_name=name)
        assert out["route"] == name
        if out["action"] == "unchallenged":
            assert out["reported"] is False and out["tentative_law"]
            assert out["tentative_reason"] == default_route.NOTHING_CHALLENGED
        if out["reported"]:
            assert out["action"] in ("confirmed", "equivalent_class") and out["law"]


def test_the_earlier_routes_are_handed_to_the_frozen_module_unchanged():
    X, y, pool, measure = _problem(lambda a, b: a * b)
    here = default_route.discover_law(X, y, pool, measure, route_name="main_search", seed=3)
    there = route.discover_law(X, y, pool, _problem(lambda a, b: a * b)[3], ledger=None, search_id="", route="main_search", seed=3)
    assert (here["action"], here["law"], here["law_as_confirmed"]) == (there["action"], there["law"], there["law_as_confirmed"])
    with pytest.raises(ValueError):
        default_route.discover_law(X, y, pool, measure, route_name="l2")          # needs dimensions, as before
    with pytest.raises(ValueError):
        default_route.discover_law(X, y, pool, measure, route_name="nonsense")


def test_units_reach_the_refined_route():
    """With units the search works on dimensionless products; the record says what was reduced."""
    X, y, pool, measure = _problem(lambda m, v: 0.5 * m * v**2, names=("m", "v"))
    units = {"m": {"kg": 1}, "v": {"m": 1, "s": -1}}
    out = default_route.discover_law(X, y, pool, measure, units=units, target_units={"kg": 1, "m": 2, "s": -2})
    assert out["route"] == "refined" and out["law_as_confirmed"]
    assert "m*v**2" in out["law_as_confirmed"].replace(" ", "")
