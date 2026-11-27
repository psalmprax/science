"""The tensor train fitted to a table's rows, and its reading as a formula where it is a product."""
from __future__ import annotations

import numpy as np
import sympy as sp

from apedp_dyn.proposers import tensor_train_reading as reading

LOW, HIGH = 1.0, 5.0


def _bases(X: np.ndarray) -> list[np.ndarray]:
    return [reading.basis(X[:, i], LOW, HIGH) for i in range(X.shape[1])]


def _noisy(clean: np.ndarray, rng: np.random.Generator, share: float = 0.01) -> tuple[np.ndarray, float]:
    sigma = share * float(np.sqrt(np.mean(clean ** 2)))
    return clean + sigma * rng.standard_normal(len(clean)), sigma


def test_the_sum_of_two_trains_is_their_sum() -> None:
    X = np.random.default_rng(0).uniform(LOW, HIGH, (50, 3))
    bases = _bases(X)
    one, two = reading.Train(3, [1, 1], seed=1), reading.Train(3, [2, 1], seed=2)
    both = reading.add(one, two)
    assert both.ranks == [1, 3, 2, 1]
    assert np.allclose(both.predict(bases), one.predict(bases) + two.predict(bases))


def test_a_product_is_fitted_at_rank_one_and_a_sum_of_two_products_at_rank_two() -> None:
    rng = np.random.default_rng(3)
    X = rng.uniform(LOW, HIGH, (700, 3))
    product = X[:, 0] ** 2 * np.log(X[:, 1] + 1.0) / (X[:, 2] + 0.7)
    two = product + 3.0 * np.sin(1.3 * X[:, 0]) * X[:, 1] * np.sqrt(X[:, 2] + 1.0)
    for clean, rank in ((product, 1), (two, 2)):
        y, sigma = _noisy(clean, rng)
        chosen = reading.choose(_bases(X), y, rank_max=2, folds=3)
        assert chosen["rank"] == rank
        assert chosen["left_out_rmse"] <= 1.5 * sigma                # the fit reaches the table's noise


def test_the_stand_in_is_the_simplest_within_the_yardstick() -> None:
    scored = [(0.90, 3, True, "long"), (0.95, 1, True, "one block and a constant"), (0.96, 1, False, "one block"),
              (2.00, 1, False, "one block that misses")]
    assert reading.stand_in(scored, yard=1.0) == ("one block", True)
    assert reading.stand_in(scored, yard=0.5) == ("long", False)     # none within it: the best fit, and it says so
    assert reading.stand_in([], yard=1.0) == (None, False)


def test_polish_refits_the_constants_and_leaves_the_exponents() -> None:
    x0, x1 = sp.symbols("x0 x1")
    X = np.random.default_rng(7).uniform(LOW, HIGH, (200, 2))
    read, error = reading.polish(1.5 * x0 ** 2.0 / (x1 + 1.0), ["x0", "x1"], X, 2.0 * X[:, 0] ** 2 / (X[:, 1] + 0.7))
    assert error < 1.0e-6
    assert {float(e) for power in read.atoms(sp.Pow) for e in power.exp.atoms(sp.Float)} == {2.0}
    assert {round(float(f), 3) for f in read.atoms(sp.Float)} == {2.0, 0.7}


def test_a_product_of_two_inputs_is_read_as_an_accurate_formula_in_the_table_s_own_names() -> None:
    rng = np.random.default_rng(11)
    X = rng.uniform(LOW, HIGH, (700, 2))
    clean = 1.7 * X[:, 0] ** 2 / X[:, 1]
    y, _ = _noisy(clean, rng)
    out = reading.read_product(X, y, ["width", "depth"])
    assert out["rank"] == 1 and out["why_not"] is None
    law = sp.sympify(out["law"], locals={"width": sp.Symbol("width"), "depth": sp.Symbol("depth")})
    assert law.free_symbols == {sp.Symbol("width"), sp.Symbol("depth")}
    fresh = rng.uniform(LOW, HIGH, (300, 2))
    predicted = np.asarray(sp.lambdify([sp.Symbol("width"), sp.Symbol("depth")], law, "numpy")(*fresh.T), dtype=float)
    truth = 1.7 * fresh[:, 0] ** 2 / fresh[:, 1]
    assert float(np.sqrt(np.mean((predicted - truth) ** 2)) / np.sqrt(np.mean(truth ** 2))) < 0.01


def test_what_is_not_a_product_is_not_read_and_says_why() -> None:
    rng = np.random.default_rng(13)
    X = rng.uniform(LOW, HIGH, (700, 2))
    y, _ = _noisy(X[:, 0] ** 2 + 3.0 * np.sin(1.3 * X[:, 0]) * X[:, 1] + 5.0 / X[:, 1], rng)
    out = reading.read_product(X, y, ["x0", "x1"])
    assert out["law"] is None and out["rank"] > 1 and "only a product" in out["why_not"]
    one = reading.read_product(X[:, :1], y, ["x0"])
    assert one["law"] is None and "one input" in one["why_not"]
