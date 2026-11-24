"""The second version of the last checks: its tests are calibrated, and it repairs the three faults the first version showed."""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.discovery_l2 import vouching, vouching_v2

LAW = "0.8*x0 + 1.9/(x0 + 0.6)"


def _rows(fn, n: int = 700, noise: float = 0.01, inputs: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, (n, inputs))
    clean = fn(*X.T)
    return X, clean + noise * float(np.sqrt(np.mean(clean ** 2))) * rng.standard_normal(n), clean


def _law(x):
    return 0.8 * x + 1.9 / (x + 0.6)


def test_its_settings_come_from_the_canonical_ones() -> None:
    assert vouching_v2.LEVEL == vouching.LEVEL == 0.01 and vouching_v2.POWER == vouching.POWER == 0.8
    assert vouching_v2.ACCURACY_R2 == vouching.ACCURACY_R2 and vouching_v2.TESTS == 2


def test_the_gradient_holds_what_the_constants_can_follow() -> None:
    X = np.random.default_rng(1).uniform(1.0, 5.0, (50, 1))
    G = vouching_v2.gradient("2.0*x0 + 3.0*x0**2 + 1.5", ["x0"], X)
    assert G.shape == (50, 4)                                    # a column of ones and one per constant
    assert np.allclose(G[:, 1], X[:, 0] ** 2) or np.allclose(G[:, 1], X[:, 0]) or np.allclose(G[:, 1], 1.0)
    B = vouching_v2.basis(G)
    assert B.shape[1] == 3                                       # the constant 1.5 and the ones column are one direction
    assert np.allclose(B.T @ B, np.eye(3), atol=1e-10)
    # a shape the constants can follow is dropped; one they cannot is kept with unit root mean square
    shapes = vouching_v2.kept_shapes([X[:, 0] ** 2, np.sin(3 * X[:, 0])], B)
    assert len(shapes) == 1 and np.sqrt(np.mean(shapes[0] ** 2)) == pytest.approx(1.0)


def test_a_right_law_is_refused_about_as_often_as_the_level_allows() -> None:
    refused = 0
    trials = 60
    for seed in range(trials):
        X, y, _ = _rows(_law, n=400, seed=seed)
        done = vouching_v2.vouch(LAW, [], ["x0"], X, y, flips=511, seed=seed)
        refused += done.members[0].refused
    assert refused <= 3                                          # one in a hundred expected; three in sixty is far out


def test_a_wrong_law_is_refused_and_gives_way_to_the_law_in_its_class() -> None:
    X, y, _ = _rows(_law)
    far = "0.52*x0 + 2.2"
    assert vouching_v2.vouch(far, [], ["x0"], X, y).law is None
    passed = vouching_v2.vouch(far, [LAW], ["x0"], X, y)
    assert passed.law == LAW and passed.switched
    right = vouching_v2.vouch(LAW, [], ["x0"], X, y)
    assert right.law == LAW and right.reason == "" and right.power is not None and right.power >= vouching_v2.POWER


def test_m35_a_rival_that_parts_from_the_law_on_a_few_rows_does_not_cost_the_law() -> None:
    """The first version withheld the law: it could not see an error shaped like such a rival. Rivals are now judged by the class rule only."""
    X, y, _ = _rows(_law)
    spiky = "0.8*x0 + 1.9/(x0 + 0.6) + 1e-6/(x0 - 0.999)**2"           # inside the accuracy line, all of it next to x0 = 1
    f, g = vouching._values(LAW, ["x0"], X), vouching._values(spiky, ["x0"], X)
    gap = np.sort((g - f) ** 2)[::-1]
    assert np.mean((g - f) ** 2) < (1 - vouching.ACCURACY_R2) * np.var(f) and gap[:7].sum() > 0.9 * gap.sum()
    first = vouching.vouch(LAW, [spiky], ["x0"], X, y)
    assert first.law is None and "would have been seen" in first.reason      # the fault, as the first version shows it
    second = vouching_v2.vouch(LAW, [spiky], ["x0"], X, y)
    assert second.law == LAW and second.reason == ""


def test_a_pole_a_little_off_is_refused_where_rows_lie_close_to_it() -> None:
    """Rows near the pole make the error smooth enough for the neighbour test. M-34 is the case where they do not."""
    def truth(x):
        return 1.0 / (x - 0.95)

    X, y, clean = _rows(truth, n=300, noise=0.002)
    values = 1.003 / (X[:, 0] - 0.948)
    assert np.max(np.abs(values - clean)) > 20 * 0.002 * np.sqrt(np.mean(clean ** 2))   # wrong on the rows next to it
    done = vouching_v2.vouch("1.003/(x0 - 0.948)", [], ["x0"], X, y)
    assert done.law is None and done.members[0].p_structure < vouching_v2.LEVEL / vouching_v2.TESTS


def test_m33_a_polynomial_that_bends_around_the_error_is_judged_on_what_it_cannot_follow() -> None:
    """A cubic fitted to a law it is not: the shapes it can follow are dropped, and the fourth degree is still tried."""
    def truth(x):
        return np.exp(1.1 * x)

    X, y, clean = _rows(truth, noise=0.002)
    coef = np.polyfit(X[:, 0], y, 3)
    cubic = " + ".join(f"{float(c)!r}*x0**{3 - k}" for k, c in enumerate(coef))
    r2 = 1.0 - np.mean((np.polyval(coef, X[:, 0]) - clean) ** 2) / np.var(clean)
    assert 0.99 < r2 < vouching_v2.ACCURACY_R2                    # wrong by a small margin
    assert vouching_v2.vouch(cubic, [], ["x0"], X, y).law is None


def test_a_law_its_constants_let_follow_every_shape_is_not_vouched_for() -> None:
    X, y, _ = _rows(_law, n=60, noise=0.01)
    terms = " + ".join(f"0.1*x0**{k}" for k in range(1, 9))
    done = vouching_v2.vouch(terms, [], ["x0"], X, y)
    assert done.law is None


def test_power_follows_the_noncentral_f_distribution() -> None:
    weak = vouching_v2.power_for(10, 600, 1.0, 0.05, 700, 0.01 / 3)
    strong = vouching_v2.power_for(10, 600, 1.0, 0.5, 700, 0.01 / 3)
    assert 0.0 <= weak < 0.2 < 0.99 < strong <= 1.0
    assert vouching_v2.power_for(0, 600, 1.0, 0.5, 700, 0.01) == 0.0


def test_text_that_is_not_a_formula_is_refused_not_raised() -> None:
    X, y, _ = _rows(_law)
    done = vouching_v2.vouch("np.float64(2.0)*x0", [LAW], ["x0"], X, y)
    assert done.members[0].usable is False and done.law == LAW and done.switched
