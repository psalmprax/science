"""The study of a tensor train read from a wide table: what its fit, its rank and its reading are made of."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_wider_conditions as wider
import study_symbolic_tensor_train as study

LOW, HIGH = 1.0, 5.0


def _bases(X: np.ndarray) -> list[np.ndarray]:
    return [study.basis(X[:, i], LOW, HIGH) for i in range(X.shape[1])]


def _noisy(clean: np.ndarray, rng: np.random.Generator, share: float = 0.01) -> tuple[np.ndarray, float]:
    sigma = share * float(np.sqrt(np.mean(clean ** 2)))
    return clean + sigma * rng.standard_normal(len(clean)), sigma


def test_the_sum_of_two_trains_is_their_sum() -> None:
    X = np.random.default_rng(0).uniform(LOW, HIGH, (50, 3))
    bases = _bases(X)
    one, two = study.Train(3, [1, 1], seed=1), study.Train(3, [2, 1], seed=2)
    both = study.add(one, two)
    assert both.ranks == [1, 3, 2, 1]
    assert np.allclose(both.predict(bases), one.predict(bases) + two.predict(bases))


def test_a_product_is_fitted_at_rank_one_and_a_sum_of_two_products_at_rank_two() -> None:
    rng = np.random.default_rng(3)
    X = rng.uniform(LOW, HIGH, (study.ROWS, 3))
    product = X[:, 0] ** 2 * np.log(X[:, 1] + 1.0) / (X[:, 2] + 0.7)
    two = product + 3.0 * np.sin(1.3 * X[:, 0]) * X[:, 1] * np.sqrt(X[:, 2] + 1.0)
    for clean, rank in ((product, 1), (two, 2)):
        y, sigma = _noisy(clean, rng)
        chosen = study.choose(_bases(X), y, rank_max=2, folds=3)
        assert chosen["rank"] == rank
        assert chosen["left_out_rmse"] <= study.AT_THE_FLOOR * sigma     # the fit reaches the table's noise


def test_a_core_holds_its_own_function_and_not_another() -> None:
    rng = np.random.default_rng(5)
    X = rng.uniform(LOW, HIGH, (400, 3))
    y, _ = _noisy(X[:, 0] ** 2 * np.log(X[:, 1] + 1.0) / (X[:, 2] + 0.7), rng)
    model = study.grown(_bases(X), y, rank_max=1)[0]
    grid = np.linspace(LOW, HIGH, study.GRID)
    spaces = study.core_spaces(model, grid, LOW, HIGH)
    for space, own in zip(spaces, (grid ** 2, np.log(grid + 1.0), 1.0 / (grid + 0.7)), strict=True):
        assert study.miss(space, own) < 0.02
    assert study.miss(spaces[0], np.sin(2.9 * grid)) > 0.2


def test_the_stand_in_is_the_simplest_within_the_yardstick() -> None:
    scored = [(0.90, 3, True, "long"), (0.95, 1, True, "one block and a constant"), (0.96, 1, False, "one block"),
              (2.00, 1, False, "one block that misses")]
    assert study.stand_in(scored, yard=1.0) == ("one block", True)
    assert study.stand_in(scored, yard=0.5) == ("long", False)      # none within it: the best fit, and it says so
    assert study.stand_in([], yard=1.0) == (None, False)


def test_polish_refits_the_constants_and_leaves_the_exponents() -> None:
    x0, x1 = sp.symbols("x0 x1")
    X = np.random.default_rng(7).uniform(LOW, HIGH, (200, 2))
    read, error = study.polish(1.5 * x0 ** 2.0 / (x1 + 1.0), ["x0", "x1"], X, 2.0 * X[:, 0] ** 2 / (X[:, 1] + 0.7))
    assert error < 1.0e-6
    assert {float(e) for power in read.atoms(sp.Pow) for e in power.exp.atoms(sp.Float)} == {2.0}
    assert {round(float(f), 3) for f in read.atoms(sp.Float)} == {2.0, 0.7}


def test_the_table_is_the_one_the_wide_runner_reads() -> None:
    job = {"index": 40, "law": 0, "text": "2.0*x0*log(1 + x1) + 3.0*sin(x2)*x3", "inputs": 4, "noise": 0.05}
    t = study.table(job)
    key = hashlib.sha256(b"wider-conditions|40").digest()[:8]
    X = np.random.default_rng(int.from_bytes(key, "big")).uniform(LOW, HIGH, (wider.ROWS + wider.TEST_ROWS, 4))
    assert np.array_equal(t["X"], X[: study.ROWS]) and np.array_equal(t["X_clean"], X[wider.ROWS:])
    clean = 2.0 * X[:, 0] * np.log(1.0 + X[:, 1]) + 3.0 * np.sin(X[:, 2]) * X[:, 3]
    assert np.allclose(t["y_clean"], clean[wider.ROWS:])
    assert abs(float(np.std(t["y"] - clean[: study.ROWS])) / t["sigma"] - 1.0) < 0.1
    assert t["sigma"] == 0.05 * float(np.sqrt(np.mean(clean[: wider.ROWS] ** 2)))
