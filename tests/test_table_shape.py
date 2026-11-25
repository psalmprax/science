"""The shape of a law read from its table: what is split, what is left whole, and that the parts give back the whole."""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from apedp_dyn.proposers import table_shape as sep


def _table(law, inputs: int, rows: int = 320, noise: float = 0.01, seed: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, (rows, inputs))
    clean = law(*X.T)
    return X, clean + noise * float(np.sqrt(np.mean(clean**2))) * rng.standard_normal(rows), clean


def test_every_way_to_put_a_group_in_two_is_offered_once():
    ways = sep.bipartitions((0, 1, 2, 3))
    assert len(ways) == 7 and len(set(ways)) == 7
    assert all(first[0] == 0 and set(first) | set(second) == {0, 1, 2, 3} and not set(first) & set(second) for first, second in ways)
    assert sep.bipartitions((4, 7)) == [((4,), (7,))]


def test_a_product_is_read_as_a_product_even_where_the_target_changes_sign():
    X, y, _ = _table(lambda a, b: (1.0 + a * a) * np.sin(1.3 * b), 2)
    assert np.any(y > 0) and np.any(y < 0)
    reading = sep.read(X, y)
    assert reading.report["read"] and reading.shape.form == sep.PRODUCT
    assert [group for _, group in reading.shape.leaves()] == [(0,), (1,)]
    assert reading.shape.widest() == 1 and reading.shape.text() == "f(x0)*f(x1)"


def test_a_sum_is_read_as_a_sum():
    X, y, _ = _table(lambda a, b: np.sqrt(a + 0.7) + 2.0 * np.sin(1.7 * b), 2)
    reading = sep.read(X, y)
    assert reading.shape.form == sep.SUM and reading.shape.text() == "f(x0) + f(x1)"


def test_inputs_that_meet_inside_a_function_are_left_as_one_table():
    X, y, _ = _table(lambda a, b: np.sin(0.7 * a * b), 2, rows=400)
    reading = sep.read(X, y)
    assert reading.report["read"] and reading.shape.form == sep.LEAF and reading.shape.widest() == 2
    assert np.array_equal(reading.tables[()], y[reading.rows])


def test_a_product_and_a_term_of_one_group_is_read_as_three_parts():
    X, y, _ = _table(lambda a, b: a * np.log(1.0 + b) + 0.5 * b * b, 2, rows=400)
    reading = sep.read(X, y)
    assert reading.shape.form in (sep.WITH_SECOND, sep.WITH_FIRST)
    assert len(reading.shape.leaves()) == 3 and reading.shape.widest() == 1


def test_the_parts_give_back_the_whole_and_a_formula_for_each_part_gives_one_formula():
    X, y, clean = _table(lambda a, b, c: 0.5 * c + a * np.sqrt(b + 1.0), 3, rows=400)
    reading = sep.read(X, y)
    assert reading.shape.widest() == 1
    whole = reading.recomposed()
    spread = float(np.std(clean))
    assert float(np.sqrt(np.mean((whole - clean[reading.rows]) ** 2))) < 0.03 * spread
    laws = {path: f"p{''.join(map(str, path))}" for path, _ in reading.shape.leaves()}
    formula = reading.compose(laws)
    names = {name: sp.Symbol(name) for name in laws.values()}
    parsed = sp.sympify(formula, locals=names)
    assert parsed.free_symbols == set(names.values())
    values = {name: reading.tables[path] for path, name in laws.items()}
    by_formula = sp.lambdify(list(names.values()), parsed, "numpy")(*[values[str(s)] for s in names.values()])
    assert np.allclose(by_formula, whole)


def test_a_table_no_smooth_model_fits_is_not_read():
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 5.0, (200, 3))
    reading = sep.read(X, rng.standard_normal(200))
    assert not reading.report["read"] and reading.shape.form == sep.LEAF and reading.report["steps"] == []


def test_one_input_is_returned_as_it_came_and_a_bad_table_is_refused():
    X = np.linspace(1.0, 5.0, 50)[:, None]
    reading = sep.read(X, np.sin(X[:, 0]))
    assert reading.shape == sep.Shape(sep.LEAF, (0,)) and not reading.report["read"]
    with pytest.raises(ValueError):
        sep.read(np.ones((5, 2)), np.ones(4))


def test_the_same_table_is_read_the_same_way_twice_and_more_rows_than_the_limit_are_thinned():
    X, y, _ = _table(lambda a, b: (1.0 + a) / (0.5 + b), 2, rows=360)
    first, second = sep.read(X, y, max_rows=300), sep.read(X, y, max_rows=300)
    assert first.shape == second.shape and np.array_equal(first.rows, second.rows) and len(first.rows) == 300
    assert all(np.array_equal(first.tables[path], second.tables[path]) for path in first.tables)


def test_a_stricter_limit_never_splits_further():
    X, y, _ = _table(lambda a, b, c: a * np.sqrt(b + 1.0) * (1.0 + 0.2 * np.sin(b * c)), 3, rows=400, noise=0.05)
    loose, strict = sep.read(X, y, z_refused=3.0), sep.read(X, y, z_refused=-50.0)
    assert strict.shape.form == sep.LEAF
    assert len(loose.shape.leaves()) >= len(strict.shape.leaves())
