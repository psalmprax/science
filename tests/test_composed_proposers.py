"""Units reduction, the polynomial library, composition, the refined path and the search controller."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import sympy as sp

from apedp_dyn.discovery_l2 import refined_search
from apedp_dyn.proposers import composed, llm_controller


def _frame(columns: dict[str, np.ndarray], target: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({**columns, "target": target})


class Fixed(composed.ProposerBase):
    """Returns the formulas it was built with and remembers the frame it was shown."""

    def __init__(self, formulas: list[str], proposer_id: str = "fixed", fail: bool = False) -> None:
        self.formulas, self._id, self.fail, self.seen = formulas, proposer_id, fail, None

    @property
    def proposer_id(self) -> str:
        return self._id

    def fit_and_propose(self, train_data, target_col=None, **kwargs):
        if self.fail:
            raise ValueError("array must not contain infs or NaNs")
        self.seen = train_data
        features = [c for c in train_data.columns if c != (target_col or "target")]
        return [composed.formula_proposal(self._id, f, features) for f in self.formulas]


UNITS = {"m": {"m": 1}, "v": {"l": 1, "t": -1}, "r": {"l": 1}, "w": {"l": 1}}


def test_dimensional_reduction_finds_the_scale_and_the_products():
    red = composed.dimensional_reduction(["m", "v", "r", "w"], UNITS, {"m": 1, "l": 1, "t": -2})
    assert red is not None and len(red["groups"]) == 1
    m, v, r, w = (sp.Symbol(n, positive=True) for n in "mvrw")
    # the scale has the units of a force, and the one dimensionless product is a ratio of the two lengths
    assert sp.simplify(red["scale"] / (m * v ** 2 / r)) in (1, w / r, r / w) or sp.simplify(red["scale"] * w / (m * v ** 2)) == 1
    assert sp.simplify(red["groups"][0] * w / r) == 1 or sp.simplify(red["groups"][0] * r / w) == 1
    assert composed.dimensional_reduction(["a", "b"], {"a": {}, "b": {}}, {}) is None
    assert composed.dimensional_reduction(["m"], {"m": {"m": 1}}, {"t": 1}) is None       # no product of the inputs is a time


def test_the_units_wrapper_searches_fewer_variables_and_answers_in_the_original_ones():
    rng = np.random.default_rng(0)
    cols = {n: rng.uniform(1.0, 3.0, 300) for n in "mvrw"}
    y = cols["m"] * cols["v"] ** 2 / cols["r"] * (1.0 + 0.5 * cols["w"] / cols["r"])
    inner = Fixed(["1 + 0.5*p0", "1 + 0.5/p0"])
    wrapper = composed.UnitsReducingProposer(inner, UNITS, {"m": 1, "l": 1, "t": -2}, own_first_pass=False)
    proposals = wrapper.fit_and_propose(_frame(cols, y), target_col="target")
    assert wrapper.last_run["reduced"] and wrapper.last_run["variables_before"] == 4 and wrapper.last_run["variables_after"] == 1
    assert list(inner.seen.columns) == ["p0", "target"]                     # the search saw one dimensionless input
    errors = [float(np.sqrt(np.mean((p.compute_score(_frame(cols, y)) - y) ** 2))) for p in proposals]
    assert min(errors) < 1e-9 and all(set(map(str, sp.sympify(p.formula).free_symbols)) <= set("mvrw") for p in proposals)


def test_without_usable_units_the_inner_search_runs_on_the_inputs_as_they_are():
    rng = np.random.default_rng(1)
    cols = {"a": rng.uniform(1, 2, 50), "b": rng.uniform(1, 2, 50)}
    inner = Fixed(["a*b"])
    wrapper = composed.UnitsReducingProposer(inner, {"a": {}, "b": {}}, {})
    proposals = wrapper.fit_and_propose(_frame(cols, cols["a"] * cols["b"]), target_col="target")
    assert not wrapper.last_run["reduced"] and [p.formula for p in proposals] == ["a*b"] and list(inner.seen.columns) == ["a", "b", "target"]


def test_polynomial_library_recovers_a_sparse_polynomial_and_keeps_out_of_large_problems():
    rng = np.random.default_rng(2)
    cols = {"x0": rng.uniform(-2, 2, 200), "x1": rng.uniform(-2, 2, 200)}
    y = -0.8 * cols["x0"] ** 3 - 1.2 * cols["x0"] - 0.3 * cols["x1"]
    (proposal,) = composed.PolynomialLibraryProposer().fit_and_propose(_frame(cols, y), target_col="target")
    x0, x1 = sp.symbols("x0 x1")
    found = sp.Poly(sp.sympify(proposal.formula), x0, x1).as_dict()
    assert set(found) == {(3, 0), (1, 0), (0, 1)} and found[(3, 0)] == pytest.approx(-0.8) and proposal.metadata["terms"] == 3
    wide = {f"x{i}": rng.uniform(1, 2, 60) for i in range(8)}
    assert composed.PolynomialLibraryProposer().fit_and_propose(_frame(wide, wide["x0"]), target_col="target") == []


def test_a_failing_member_does_not_cost_the_others_their_candidates():
    rng = np.random.default_rng(3)
    cols = {"a": rng.uniform(1, 2, 40)}
    composite = composed.CompositeProposer([Fixed(["a", "a**2"], "one"), Fixed([], "broken", fail=True), Fixed(["a", "a**3"], "two")])
    proposals = composite.fit_and_propose(_frame(cols, cols["a"]), target_col="target")
    assert [p.formula for p in proposals] == ["a", "a**2", "a**3"]          # round-robin, no duplicates
    assert len(composite.failures) == 1 and composite.failures[0].startswith("broken: ValueError")


def test_the_confirmed_form_drops_terms_the_data_do_not_notice():
    rng = np.random.default_rng(4)
    X = rng.uniform(0.5, 1.5, (200, 1))
    y = -X[:, 0] ** 3
    # the ODEBench case: a correct law with a term at the level of numerical error that explodes out of range
    law, dropped = refined_search.confirmed_and_pruned("-x0**3 + 1.7e-8*x0 - 4.4e-9*exp(exp(x0))", ["x0"], X, y, "absolute")
    assert sorted(dropped) == ["exp(exp(x0))", "x0"]
    assert sp.Poly(law, sp.Symbol("x0")).as_dict() == pytest.approx({(3,): -1.0})
    noisy = 2.0 * X[:, 0] + 3.0 * X[:, 0] ** 2 + 0.01 * rng.standard_normal(200)
    kept, none = refined_search.confirmed_and_pruned("x0 + x0**2", ["x0"], X, noisy, "absolute")
    assert none == [] and len(sp.Add.make_args(kept)) == 2                   # a real term is never dropped


def test_the_refined_path_reports_only_what_was_confirmed_against_rivals():
    rng = np.random.default_rng(5)
    cols = {n: rng.uniform(1.0, 3.0, 300) for n in "mvr"}
    pool = rng.uniform(1.0, 3.0, (300, 3))
    law = lambda m, v, r: m * v ** 2 / r
    y = law(cols["m"], cols["v"], cols["r"])
    result = refined_search.discover_refined(cols, y, pool, lambda rows: law(*pool[rows].T),
                                             units={k: UNITS[k] for k in "mvr"}, target_units={"m": 1, "l": 1, "t": -2}, noise="absolute")
    assert result.action in refined_search.REPORTED and result.law is not None and result.tentative_law is None
    assert result.units["reduced"] and result.units["variables_after"] == 0   # nothing dimensionless is left: the law is the scale
    m, v, r = sp.symbols("m v r")
    assert sp.simplify(sp.sympify(result.law) / (m * v ** 2 / r) - 1).evalf() == pytest.approx(0.0, abs=1e-6)


def test_the_controller_uses_what_it_can_check_and_ignores_the_rest():
    reply = ('Here you go:\n```json\n{"unary": ["exp", "tan", "log"], "features": ["x0*x1", "x0", "__import__(\\"os\\")", "x1/c0"], '
             '"guesses": ["c0*x0*exp(-c1*x1)", "open(1)"]}\n```')
    config = llm_controller.read_configuration(reply, 2)
    assert config["readable"] and config["unary"] == ["exp", "log"] and [str(e) for e in config["features"]] == ["x0*x1"]
    assert len(config["guesses"]) == 1 and any("tan" in r for r in config["rejected"]) and any("open" in r for r in config["rejected"])
    assert not llm_controller.read_configuration("I think exp is a good idea.", 2)["readable"]
    assert llm_controller.read_configuration(json.dumps({"unary": []}), 2)["unary"] == []   # arithmetic only is a valid choice


def test_the_controlled_search_runs_with_the_checked_configuration_and_survives_a_dead_model():
    rng = np.random.default_rng(6)
    cols = {"a": rng.uniform(1, 3, 200), "b": rng.uniform(1, 3, 200)}
    y = cols["a"] * np.exp(-0.37 * cols["b"])
    built = {}

    def factory(unary):
        built["unary"] = unary
        built["search"] = Fixed(["f0*0.5", "a*exp(-0.37*b)"], "search")
        return built["search"]

    reply = json.dumps({"unary": ["exp"], "features": ["x0/x1"], "guesses": ["c0*x0*exp(-c1*x1)"]})
    proposer = llm_controller.ControlledSearchProposer(lambda prompt: reply, factory)
    proposals = proposer.fit_and_propose(_frame(cols, y), target_col="target", guesses=["a/b"])
    assert built["unary"] == ("exp",) and list(built["search"].seen.columns) == ["a", "b", "target", "f0"]
    formulas = [p.formula for p in proposals]
    assert any("exp" in f for f in formulas) and all(set(map(str, sp.sympify(f).free_symbols)) <= {"a", "b"} for f in formulas)
    assert proposer.last_run["features"] == ["x0/x1"] and proposer.last_run["guesses"] and proposals[0].metadata["found_by"] == "model_guess"

    def dead(prompt):
        raise ConnectionError("no route")

    silent = llm_controller.ControlledSearchProposer(dead, factory)
    assert silent.fit_and_propose(_frame(cols, y), target_col="target") and silent.last_run["model_failure"].startswith("ConnectionError")
    assert built["unary"] == llm_controller.DEFAULT_UNARY                  # the default search, not no search
