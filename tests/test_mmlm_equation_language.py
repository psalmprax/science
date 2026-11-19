"""MMLM Track S: the equation language model (apedp_mmlm/equation_language.py)."""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from apedp_dyn.discovery_l2 import exhaustive_search as es
from apedp_mmlm import equation_language as el

m, v, r, q, t = sp.symbols("m v r q t")
PHYSICS = [m * v, m * v**2 / 2, m * v**2 / r, q * v / r, m / (r * t), sp.sqrt(m / r), m * r / t**2, q**2 / r**2,
           m * v / t, v * t, v / t, m * v * r, q / (4 * sp.pi * r**2), m * sp.exp(-t / r)]


def test_expressions_become_shapes_not_names_or_values():
    assert el.tokens(el.to_tree(m * v**2 / 2)) == el.tokens(el.to_tree(7.3 * q * r**2)) == ["*", "v0", "sq", "v1"]
    assert el.tokens(el.to_tree(m / (r * t))) == ["/", "v0", "*", "v1", "v2"]
    assert el.tokens(el.to_tree(sp.sqrt(m / r))) == ["sqrt", "/", "v0", "v1"]
    assert el.tokens(el.to_tree(m - v)) == ["-", "v0", "v1"]
    assert el.tokens(el.to_tree(sp.exp(-t / r))) == ["exp", "/", "v0", "v1"]
    assert el.tokens(el.to_tree(1 + m)) == ["+", "C", "v0"] or el.tokens(el.to_tree(1 + m)) == ["+", "v0", "C"]
    assert set(el.tokens(el.to_tree(sp.tan(m)))) == {"/", "sin", "cos", "v0"}


def test_search_nodes_and_formulas_share_one_dialect():
    node = ("/", ("*", ("x", 0), ("sq", ("x", 1))), ("x", 2))
    x0, x1, x2 = sp.symbols("x0 x1 x2")
    assert el.tokens(el.from_search_node(node)) == el.tokens(el.to_tree(x0 * x1**2 / x2))


def test_tree_contexts_pair_each_token_with_its_parent_and_position():
    assert el.tree_contexts(el.to_tree(m / (r * t))) == [
        ("root", "/"), ("/:0", "v0"), ("/:1", "*"), ("*:0", "v1"), ("*:1", "v2")]


@pytest.mark.parametrize("engine", ["unigram", "tree", "automaton", "bayesian"])
def test_a_trained_model_finds_physics_shaped_laws_cheaper_than_arbitrary_ones(engine):
    model = el.EquationLanguageModel(engine).fit([el.to_tree(e) for e in PHYSICS])
    a, b = sp.symbols("a b")
    plausible = model.bits(el.to_tree(a * b**2))          # not in the training set
    odd = model.bits(el.to_tree(sp.sin(sp.cos(sp.log(a))) - sp.exp(b)))
    assert plausible < odd
    assert plausible < el.EquationLanguageModel("uniform").bits(el.to_tree(a * b**2))


def test_every_engine_gives_finite_positive_code_lengths():
    trees = [el.to_tree(e) for e in PHYSICS]
    for engine in el.ENGINES:
        bits = el.EquationLanguageModel(engine).fit(trees).bits(trees[0])
        assert np.isfinite(bits) and bits > 0, engine
    with pytest.raises(ValueError, match="engine must be"):
        el.EquationLanguageModel("transformer")


def test_the_model_can_price_structure_for_the_exhaustive_search():
    """With equal fits, the search prefers the form the model finds more plausible."""
    model = el.EquationLanguageModel("automaton").fit([el.to_tree(e) for e in PHYSICS])
    prior = el.search_prior(model)
    assert prior(("*", ("x", 0), ("x", 1))) < prior(("sin", ("cos", ("x", 0))))
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 5, (300, 2))
    y = X[:, 0] * X[:, 1] ** 2 * (1 + 0.01 * rng.standard_normal(300))
    best = es.search(X, y, ["x0", "x1"], 5, structure_bits=prior)[0]
    assert best.expression == "((x0) * (((x1)**2)))"
    plain = es.search(X, y, ["x0", "x1"], 5)[0]
    assert plain.expression == best.expression  # the prior does not change a clear answer
