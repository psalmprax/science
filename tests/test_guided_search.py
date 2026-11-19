"""Prior-ordered exhaustive search with an early stop (apedp_dyn/discovery_l2/guided_search.py)."""
from __future__ import annotations

import numpy as np
import sympy as sp

from apedp_dyn.discovery_l2 import exhaustive_search as es
from apedp_dyn.discovery_l2 import guided_search as gs
from apedp_mmlm import equation_language as el

a, b, c = sp.symbols("a b c")
MODEL = el.SelfAdjustingEquationModel().fit([el.to_tree(e) for e in (
    a * b, a / b, a * b**2, a * b / c, sp.sqrt(a / b), a / b**2, a * b * c, a**2 * b, sp.exp(a / b))])
RNG = np.random.default_rng(0)
X = RNG.uniform(1, 5, (300, 2))
Y = X[:, 0] / X[:, 1] ** 2 * (1 + 0.02 * RNG.standard_normal(300))


def test_it_stops_early_on_the_law_the_full_search_finds():
    guided = gs.GuidedSearch(el.search_prior(MODEL))
    result = guided.search(X, Y, ["x0", "x1"], 5)
    full = es.search(X, Y, ["x0", "x1"], 5)[0]
    assert result.stopped_early and result.best.expression == full.expression
    assert result.evaluated < 0.05 * result.candidates
    assert result.residual_structure <= 0.0


def test_the_order_is_priced_once_and_reused():
    calls = []
    prior = el.search_prior(MODEL)

    def counting(node):
        calls.append(1)
        return prior(node)

    guided = gs.GuidedSearch(counting)
    guided.search(X, Y, ["x0", "x1"], 4)
    first = len(calls)
    guided.search(X, Y * 2.0, ["x0", "x1"], 4)
    assert first > 0 and len(calls) == first


def test_without_a_stop_it_returns_the_full_searchs_answer():
    """A law the prior finds implausible and that needs a sum: no single candidate clears the stop test."""
    y = (3 * np.sin(X[:, 0]) + 2 * np.log(X[:, 1])) * (1 + 0.005 * RNG.standard_normal(300))
    prior = el.search_prior(MODEL)
    result = gs.GuidedSearch(prior).search(X, y, ["x0", "x1"], 4)
    full = es.search(X, y, ["x0", "x1"], 4, structure_bits=prior)[0]
    assert not result.stopped_early and result.evaluated == result.candidates
    assert result.best.expression == full.expression


def test_the_residual_test_tells_structure_from_noise():
    nn = gs.neighbours(X, 8)
    noise = RNG.standard_normal(300)
    assert gs.residual_structure(noise, nn) < 0.05
    assert gs.residual_structure(np.sin(X[:, 0]) + 0.1 * noise, nn) > 0.5
    assert gs.residual_structure(np.zeros(300), nn) == -1.0
