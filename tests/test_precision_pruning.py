"""A term smaller than the rows' own precision is dropped; a real term is kept (audit M-15, trajectory rows)."""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from apedp_dyn.discovery_l2 import refined_search
from apedp_dyn.discovery_l2.precision_pruning import MAX_PRECISION_MULTIPLES, prune_to_precision

RNG = np.random.default_rng(11)
X = RNG.uniform(-1.5, 1.5, (300, 1))
SD = 1e-5
Y = -X[:, 0] ** 3 + SD * RNG.standard_normal(300)   # dx/dt = -x**3, rates known to about 1e-5
LAW_WITH_NOISE_TERM = "-1.0*x0**3 - 4.4e-9*exp(exp(x0)) + 1e-8"


def test_the_recorded_case_a_term_that_fits_the_rate_error_is_dropped():
    law, dropped = prune_to_precision(LAW_WITH_NOISE_TERM, ["x0"], X, Y, SD)
    assert "exp(exp(x0))" in dropped
    assert sp.simplify(law / sp.Symbol("x0") ** 3).is_number and float(law.coeff(sp.Symbol("x0") ** 3)) == pytest.approx(-1.0, abs=1e-4)


def test_the_original_rule_can_keep_that_term_and_the_new_one_never_keeps_more():
    old, _ = refined_search.confirmed_and_pruned(LAW_WITH_NOISE_TERM, ["x0"], X, Y, "absolute")
    new, _ = prune_to_precision(LAW_WITH_NOISE_TERM, ["x0"], X, Y, SD)
    assert len(sp.Add.make_args(new)) <= len(sp.Add.make_args(old))


def test_a_real_term_far_above_the_precision_is_kept():
    y = -X[:, 0] ** 3 + 0.05 * X[:, 0] + SD * RNG.standard_normal(300)
    law, dropped = prune_to_precision("-1.0*x0**3 + 0.05*x0 + 1e-9", ["x0"], X, y, SD)
    assert "x0" not in dropped and float(law.coeff(sp.Symbol("x0"), 1)) == pytest.approx(0.05, rel=1e-2)


def test_a_term_larger_than_the_bound_is_not_a_candidate_for_removal_even_if_others_could_absorb_it():
    """At coarse precision a real cubic term could be traded away; the bound keeps it."""
    coarse = 0.02
    y = 0.7 * X[:, 0] ** 2 - 0.5 * X[:, 0] + 0.1 * X[:, 0] ** 3 + coarse * RNG.standard_normal(300)
    size = float(np.sqrt(np.mean((0.1 * X[:, 0] ** 3) ** 2)))
    assert size > MAX_PRECISION_MULTIPLES * coarse
    _, dropped = prune_to_precision("0.7*x0**2 - 0.5*x0 + 0.1*x0**3", ["x0"], X, y, coarse)
    assert "x0**3" not in dropped


def test_precision_can_be_given_per_row_and_must_be_usable():
    law, dropped = prune_to_precision(LAW_WITH_NOISE_TERM, ["x0"], X, Y, np.full(300, SD))
    assert "exp(exp(x0))" in dropped and law.free_symbols == {sp.Symbol("x0")}
    with pytest.raises(ValueError, match="precision"):
        prune_to_precision(LAW_WITH_NOISE_TERM, ["x0"], X, Y, -1.0)
    with pytest.raises(ValueError, match="precision"):
        prune_to_precision(LAW_WITH_NOISE_TERM, ["x0"], X, Y, float("nan"))
