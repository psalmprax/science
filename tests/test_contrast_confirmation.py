"""Contrast confirmation (apedp_dyn/discovery_l2/contrast_confirmation.py).

Synthetic laws only; none is from a preregistered benchmark.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.discovery_l2 import contrast_confirmation as cc

NOISE = 0.02


def _world(seed: int, truth, n_dev: int = 120, n_pool: int = 4000):
    rng = np.random.default_rng(seed)
    dev = rng.uniform(1, 10, (n_dev, 2))
    pool = rng.uniform(1, 10, (n_pool, 2))
    y = truth(dev) * (1 + NOISE * rng.standard_normal(n_dev))
    calls: list[np.ndarray] = []

    def measure(rows):
        calls.append(np.array(rows))
        return truth(pool[rows]) * (1 + NOISE * rng.standard_normal(len(rows)))

    return dev, y, pool, measure, calls


BASES = {
    "x**2 + 1/y": lambda X: np.column_stack([X[:, 0] ** 2, 1 / X[:, 1]]),
    "x**2": lambda X: X[:, :1] ** 2,
    "x**2 + y": lambda X: np.column_stack([X[:, 0] ** 2, X[:, 1]]),
    "x**3": lambda X: X[:, :1] ** 3,
}


def _fit(names, dev, y):
    return [cc.fit_candidate(n, BASES[n], dev, y) for n in names]


def test_a_small_term_is_confirmed_where_a_random_holdout_cannot_see_it():
    """3 x^2 + 2/y: the second term is under 5 % of the signal on most rows."""
    truth = lambda X: 3 * X[:, 0] ** 2 + 2 / X[:, 1]
    wins = 0
    for seed in range(10):
        dev, y, pool, measure, _ = _world(seed, truth)
        result = cc.contrast_confirm(_fit(["x**2", "x**2 + 1/y", "x**2 + y", "x**3"], dev, y), pool, measure)
        wins += result.verdict == cc.UNIQUE and result.survivor == "x**2 + 1/y"
        assert result.survivor in ("x**2 + 1/y", None)  # never a wrong law
    assert wins >= 9


def test_a_law_missing_from_the_candidates_is_reported_as_no_survivor():
    truth = lambda X: 3 * X[:, 0] ** 2 + 2 / X[:, 1]
    refuted = 0
    for seed in range(10):
        dev, y, pool, measure, _ = _world(seed, truth)
        refuted += cc.contrast_confirm(_fit(["x**2", "x**2 + y", "x**3"], dev, y), pool, measure).verdict == cc.NO_SURVIVOR
    assert refuted >= 9


def test_indistinguishable_rivals_are_declared_equivalent_without_reading_a_target():
    """A term a thousand times below the noise cannot be decided, and no target is spent finding that out."""
    truth = lambda X: 3 * X[:, 0] ** 2
    dev, y, pool, _measure, _ = _world(0, truth)
    with_term = cc.fit_candidate("x**2 + 1/y", BASES["x**2 + 1/y"], dev, y)
    tiny = cc.FittedCandidate("x**2 + 1/y", with_term.basis, np.array([3.0, 1e-4]), with_term.dev_basis, with_term.dev_y)
    plain = cc.fit_candidate("x**2", BASES["x**2"], dev, y)
    read_for_pairs: list[int] = []

    def measure(rows):
        read_for_pairs.append(len(rows))
        return truth(pool[rows])

    result = cc.contrast_confirm([plain, tiny], pool, measure)
    assert result.verdict == cc.EQUIVALENT_CLASS and result.equivalent == ["x**2 + 1/y"] and not result.eliminated
    assert read_for_pairs == [30]  # only the survivor's own refutation draw


def test_a_lone_candidate_is_unchallenged_not_unique():
    truth = lambda X: 3 * X[:, 0] ** 2
    dev, y, pool, measure, _ = _world(1, truth)
    assert cc.contrast_confirm(_fit(["x**2"], dev, y), pool, measure).verdict == cc.UNCHALLENGED


def test_rows_are_chosen_from_inputs_alone_and_each_is_read_once():
    truth = lambda X: 3 * X[:, 0] ** 2 + 2 / X[:, 1]
    dev, y, pool, measure, calls = _world(2, truth)
    cands = _fit(["x**2", "x**2 + 1/y", "x**3"], dev, y)
    first = cc.contrast_confirm(cands, pool, measure)
    rows = np.concatenate(calls)
    assert len(rows) == len(set(rows.tolist())) == first.rows_read
    # Different targets, same candidates and pool: the first design is identical.
    calls2: list[np.ndarray] = []

    def other(r):
        calls2.append(np.array(r))
        return np.full(len(r), 1.0)

    cc.contrast_confirm(cands, pool, other)
    assert np.array_equal(calls[0], calls2[0])


def test_the_true_law_is_rarely_refuted():
    """Calibration of the refutation test: alpha is 1 %, so 40 worlds should lose at most a couple."""
    truth = lambda X: 3 * X[:, 0] ** 2 + 2 / X[:, 1]
    lost = 0
    for seed in range(100, 140):
        dev, y, pool, measure, _ = _world(seed, truth)
        lost += cc.contrast_confirm(_fit(["x**2 + 1/y", "x**2", "x**2 + y", "x**3"], dev, y), pool, measure).survivor != "x**2 + 1/y"
    assert lost <= 3


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError, match="at least one"):
        cc.contrast_confirm([], np.ones((5, 2)), lambda r: r)
    truth = lambda X: X[:, 0]
    dev, y, pool, measure, _ = _world(3, truth)
    with pytest.raises(ValueError, match="noise must be"):
        cc.contrast_confirm(_fit(["x**2"], dev, y), pool, measure, noise="lognormal")
