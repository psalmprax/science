"""The confirmed early stop: same order and residual check, a later and better-founded return."""
from __future__ import annotations

import numpy as np

from apedp_dyn.discovery_l2 import exhaustive_search as es
from apedp_dyn.discovery_l2 import guided_search as gs
from apedp_dyn.discovery_l2.confirmed_stop import search_confirmed


def _size_prior(node) -> float:
    return float(1 if node[0] == "x" else 1 + sum(_size_prior(child) for child in node[1:]))


def _data(noise: float, seed: int = 0, simple: bool = False):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, (300, 2))
    clean = X[:, 0] / X[:, 1] if simple else X[:, 0] / (X[:, 0] + X[:, 1])
    return X, clean * (1 + noise * rng.standard_normal(300))


def test_confirmed_stop_recovers_the_law_and_fits_a_fraction_of_the_candidates():
    X, y = _data(0.01, simple=True)
    guided = gs.GuidedSearch(_size_prior)
    result = search_confirmed(guided, X, y, ["a", "b"], 5)
    assert result.stopped_early
    assert result.evaluated < 0.5 * result.candidates
    assert result.best.expression.replace("(", "").replace(")", "").replace(" ", "") == "a/b"


def test_confirmed_stop_looks_further_than_the_first_stop_and_never_returns_a_longer_description():
    X, y = _data(0.05, seed=4)
    guided = gs.GuidedSearch(_size_prior)
    first = guided.search(X, y, ["a", "b"], 5)
    confirmed = search_confirmed(guided, X, y, ["a", "b"], 5)
    assert confirmed.evaluated >= first.evaluated
    if first.stopped_early:
        assert confirmed.best.description_length <= first.best.description_length + 1e-9


def test_window_grows_with_patience_and_floor():
    X, y = _data(0.01)
    guided = gs.GuidedSearch(_size_prior)
    short = search_confirmed(guided, X, y, ["a", "b"], 5, patience=1.0, floor=10)
    long = search_confirmed(guided, X, y, ["a", "b"], 5, patience=4.0, floor=500)
    assert short.evaluated <= long.evaluated


def test_falls_back_to_the_full_search_when_nothing_is_structure_free():
    rng = np.random.default_rng(1)
    X = rng.uniform(1.0, 5.0, (200, 2))
    y = np.sin(7.0 * X[:, 0] * X[:, 1]) + np.cos(11.0 * X[:, 0])  # outside the enumerated space
    guided = gs.GuidedSearch(_size_prior)
    result = search_confirmed(guided, X, y, ["a", "b"], 3)
    full = es.search(X, y, ["a", "b"], 3, structure_bits=_size_prior)[0]
    assert result.evaluated == result.candidates and not result.stopped_early
    assert result.residual_structure == 0.0 or result.best.expression == full.expression
