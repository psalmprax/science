"""What the tables of one family tell each other: shapes, their order by all rows, constants as a table, a wider law."""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from apedp_dyn.discovery_l2 import across_tables as at

NAMES = ["x0", "x1"]


def _member(c: float, rng: np.random.Generator, noise: float = 0.02, rows: int = 400, pool: int = 100) -> at.Table:
    """y = (0.7 + 0.5 c) x0^2 / (x1 + 2.3): the first constant follows the condition, the second is one number."""
    X = rng.uniform(1.0, 5.0, (rows + pool, 2))
    clean = (0.7 + 0.5 * c) * X[:, 0] ** 2 / (X[:, 1] + 2.3)
    y = clean + noise * float(np.sqrt(np.mean(clean ** 2))) * rng.standard_normal(len(X))
    return at.Table(f"c={c:g}", list(NAMES), X[:rows], y[:rows], X[rows:], lambda asked, y=y, rows=rows: y[rows:][asked],
                    {"T": float(c)})


@pytest.fixture(scope="module")
def family() -> at.Family:
    rng = np.random.default_rng(0)
    return at.Family("demo", [_member(c, rng) for c in (1.0, 2.0, 3.0, 4.0, 5.0)])


def test_a_shape_gives_each_fitted_constant_a_place_of_its_own_and_keeps_whole_numbers() -> None:
    shape = at.shape_of("0.7*x0**2/(x1 + 0.7) - 3*x1", NAMES)
    assert shape is not None
    assert shape.constants == 2 and sorted(shape.start) == [0.7, 0.7]      # the same value twice is two constants
    assert "x0**2" in shape.text and "3*x1" in shape.text                   # the power and the whole number stay
    assert sp.simplify(sp.sympify(shape.law(NAMES, shape.start)) - sp.sympify("0.7*x0**2/(x1 + 0.7) - 3*x1")) == 0


def test_laws_of_one_shape_have_one_key_and_what_is_not_a_law_of_the_inputs_has_none() -> None:
    one, other = at.shape_of("1.9*x0**2/(x1 + 2.0)", NAMES), at.shape_of("0.2*x0**2/(x1 + 9.5)", NAMES)
    assert one is not None and other is not None and one.key == other.key
    assert at.shape_of("1.9*x0**2 + 0.3*x1", NAMES).key != one.key
    for text in ("1.5", "2.0*z + x0", "x0 +* x1", "0.5*x0" + "".join(f" + {k}.5*x1**{k}" for k in range(2, 14))):
        assert at.shape_of(text, NAMES) is None     # a number, an unknown name, not a formula, too many constants


def test_a_refit_finds_a_table_s_own_constants_from_a_sibling_s(family: at.Family) -> None:
    shape = at.shape_of("1.2*x0**2/(x1 + 2.0)", NAMES)
    last = family.tables[-1]
    fit = at.refit(shape, NAMES, last.X, last.y)
    assert fit is not None and fit.relative < 0.03
    assert fit.theta == pytest.approx((0.7 + 0.5 * 5.0, 2.3), rel=0.05)
    assert at.refit(at.shape_of("1.0*log(x0 - 9.0)", NAMES), NAMES, last.X, last.y, extra=0) is None    # never finite


def test_shapes_are_ordered_by_all_rows_and_a_reported_one_comes_first(family: at.Family) -> None:
    right, wrong = "1.0*x0**2/(x1 + 2.0)", "0.4*x0**2 - 0.3*x1"
    scored = at.score_shapes(family, [[wrong, right], [wrong], [], [right], []], [None] * 5)
    assert [item.shape.key for item in scored] == [at.shape_of(right, NAMES).key, at.shape_of(wrong, NAMES).key]
    assert scored[0].from_members == {0, 3} and scored[1].from_members == {0, 1}
    assert scored[0].score < scored[1].score and all(fit is not None for fit in scored[0].fits)
    # the referee's word on one member outranks the fit of all rows
    again = at.score_shapes(family, [[wrong, right], [wrong], [], [right], []], [None, wrong, None, None, None])
    assert again[0].shape.key == at.shape_of(wrong, NAMES).key and again[0].reported_on == {1}


def test_a_member_is_handed_only_what_came_from_a_sibling_refitted_on_its_own_rows(family: at.Family) -> None:
    right, wrong = "1.0*x0**2/(x1 + 2.0)", "0.4*x0**2 - 0.3*x1"
    scored = at.score_shapes(family, [[right], [wrong], [], [], []], [None] * 5)
    assert at.sibling_laws(family, scored, 0) == [at.shape_of(wrong, NAMES).law(NAMES, scored[1].fits[0].theta)]
    handed = at.sibling_laws(family, scored, 4)
    assert len(handed) == 2
    first = at.shape_of(handed[0], NAMES)
    assert first.key == at.shape_of(right, NAMES).key and first.start == pytest.approx((3.2, 2.3), rel=0.05)
    assert at.sibling_laws(family, scored, 4, handed=1) == handed[:1]


def test_constants_a_table_cannot_tell_apart_are_held_and_the_rest_get_standard_errors(family: at.Family) -> None:
    table = family.tables[0]
    redundant = at.shape_of("2.0*(0.6*x0**2/(x1 + 2.3))", NAMES)        # sympy multiplies the two numbers into one
    assert redundant.constants == 2
    written = at.Shape("_k0*(_k1*x0**2/(x1 + _k2))", 3, (2.0, 0.6, 2.3), "three for two")
    fit = at.refit(written, NAMES, table.X, table.y)
    assert at.standard_errors(written, NAMES, table.X, table.y, fit.theta) is None
    held = at.told_apart(written, NAMES, table.X, table.y, fit.theta)
    assert held.constants == 2
    again = at.refit(held, NAMES, table.X, table.y)
    errors = at.standard_errors(held, NAMES, table.X, table.y, again.theta)
    assert errors is not None and np.all(errors > 0) and np.all(errors < 0.2 * np.abs(again.theta))
    plain = at.shape_of("1.2*x0**2/(x1 + 2.3)", NAMES)
    assert at.told_apart(plain, NAMES, table.X, table.y, plain.start) is plain


def test_a_constant_that_moves_is_told_from_one_that_does_not() -> None:
    errors = np.full(5, 0.05)
    assert not at.moves(np.array([2.3, 2.25, 2.36, 2.31, 2.28]), errors)
    assert at.moves(np.array([1.2, 1.7, 2.2, 2.7, 3.2]), errors)


def test_a_constant_gets_one_number_or_a_short_law_of_the_condition() -> None:
    c, T = np.array([1.0, 2.0, 3.0, 4.0, 5.0]), sp.Symbol("T")
    errors = np.full(5, 0.01)
    still = at.laws_of_a_constant(np.array([2.3, 2.29, 2.31, 2.3, 2.3]), errors, c, T)
    assert len(still) == 1 and still[0][0].is_Float and float(still[0][0]) == pytest.approx(2.3, abs=0.01)
    line = at.laws_of_a_constant(0.7 + 0.5 * c, errors, c, T)
    assert len(line) == at.FORMS_KEPT and line[0][1] < 1.0e-6
    assert float(line[0][0].subs(T, 10.0)) == pytest.approx(5.7, rel=1e-4)
    power = at.laws_of_a_constant(1.3 * c ** -1.5, errors, c, T)
    assert float(power[0][0].subs(T, 2.5)) == pytest.approx(1.3 * 2.5 ** -1.5, rel=1e-3)
    # a condition that is not positive is given no root, logarithm, power or reciprocal
    signed = at.laws_of_a_constant(0.7 + 0.5 * (c - 3.0), errors, c - 3.0, T)
    assert all(not law.has(sp.log) and not law.has(sp.Pow) for law, _ in signed)
    # rows without noise fit exactly and leave no standard error: a constant still moves, or does not
    exact = np.zeros(5)
    assert at.moves(0.7 + 0.5 * c, exact) and not at.moves(np.full(5, 2.3), exact)
    assert float(at.laws_of_a_constant(0.7 + 0.5 * c, exact, c, T)[0][0].subs(T, 10.0)) == pytest.approx(5.7, rel=1e-4)
    assert at.laws_of_a_constant(np.full(5, 2.3), exact, c, T)[0][0] == sp.Float(2.3)


def test_the_constants_are_read_along_the_condition(family: at.Family) -> None:
    table = at.constants_of(family, at.shape_of("1.0*x0**2/(x1 + 2.0)", NAMES), "T")
    assert table is not None and list(table.at) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert table.values[:, 0] == pytest.approx(0.7 + 0.5 * table.at, rel=0.05)
    assert table.values[:, 1] == pytest.approx(np.full(5, 2.3), rel=0.05)
    assert table.moving == [True, False]
    assert set(table.as_record()) == {"shape", "condition", "at", "values", "standard_errors", "moving"}


def test_the_pooled_table_has_the_condition_as_an_input_and_measures_each_row_through_its_member(family: at.Family) -> None:
    whole = at.pooled(family, "T")
    assert whole.names == ["x0", "x1", "T"] and whole.X.shape == (2000, 3) and whole.pool_X.shape == (500, 3)
    assert sorted(set(whole.X[:, 2])) == [1.0, 2.0, 3.0, 4.0, 5.0] and not whole.conditions
    asked = np.array([0, 99, 100, 250, 499])
    expected = [family.tables[0].measure(np.array([0]))[0], family.tables[0].measure(np.array([99]))[0],
                family.tables[1].measure(np.array([0]))[0], family.tables[2].measure(np.array([50]))[0],
                family.tables[4].measure(np.array([99]))[0]]
    assert list(whole.measure(asked)) == expected
    assert list(whole.pool_X[asked, 2]) == [1.0, 1.0, 2.0, 3.0, 5.0]
    with pytest.raises(ValueError, match="already is an input"):
        at.pooled(family, "x0")


def test_the_wider_law_is_the_shape_with_the_constants_laws_put_in(family: at.Family) -> None:
    scored = at.score_shapes(family, [["1.0*x0**2/(x1 + 2.0)", "0.4*x0**2 - 0.3*x1"], [], [], [], []], [None] * 5)
    laws, how = at.wider_laws(family, scored, "T")
    assert laws and len(laws) == len(how) <= at.WIDER_LAWS
    best = sp.sympify(laws[0], locals={n: sp.Symbol(n) for n in ("x0", "x1", "T")})
    assert best.free_symbols == {sp.Symbol("x0"), sp.Symbol("x1"), sp.Symbol("T")}
    rng = np.random.default_rng(5)
    X = rng.uniform(1.0, 5.0, (300, 3))
    predicted = np.asarray(sp.lambdify([sp.Symbol(n) for n in ("x0", "x1", "T")], best, "numpy")(*X.T), dtype=float)
    truth = (0.7 + 0.5 * X[:, 2]) * X[:, 0] ** 2 / (X[:, 1] + 2.3)        # also between the members' values of T
    assert float(np.sqrt(np.mean((predicted - truth) ** 2)) / np.sqrt(np.mean(truth ** 2))) < 0.01
    assert how[0]["relative_error"] < 0.03 and how[0]["constants"]["moving"] == [True, False]


def test_laws_handed_in_take_turns_with_a_table_s_own_and_each_law_comes_once() -> None:
    assert at.beside(["a", "b", "c"], ["x", "y"]) == ["a", "x", "b", "y", "c"]
    assert at.beside([], ["x", "y"]) == ["x", "y"] and at.beside(["a"], []) == ["a"]
    assert at.beside(["a", "x"], ["x", "y", "z"]) == ["a", "x", "y", "z"]


def test_a_shape_a_search_proposed_too_low_to_reach_the_referee_is_scored_and_can_be_handed_back(family: at.Family) -> None:
    names = family.names
    right, wrong = "1.0*x0**2/(x1 + 2.0)", "0.5*x0**2 - 0.3*x1"
    listed = [[wrong] for _ in family.tables]                       # every referee had only the wrong law before it
    unseen = [[right] if i == 2 else [] for i in range(len(family.tables))]     # one search had the shape, low on its list
    scored = at.score_shapes(family, listed, [None] * len(family.tables), unseen=unseen)
    best = scored[0]
    assert best.shape.key == at.shape_of(right, names).key and best.from_members == {2} and best.seen_on == set()
    for member in range(len(family.tables)):                        # handed to all, the member whose search had it too
        handed = at.sibling_laws(family, scored, member)
        assert at.shape_of(handed[0], names).key == best.shape.key
    assert at.score_shapes(family, listed, [None] * len(family.tables))[0].shape.key == at.shape_of(wrong, names).key
    assert len(at.score_shapes(family, listed, [None] * len(family.tables), unseen=unseen, max_unseen=0)) == 1


def test_too_few_members_give_no_wider_law(family: at.Family) -> None:
    small = at.Family("small", family.tables[:3])
    scored = at.score_shapes(small, [["1.0*x0**2/(x1 + 2.0)"], [], []], [None] * 3)
    laws, how = at.wider_laws(small, scored, "T")
    assert laws == [] and "fewer than 4 members" in how[0]["why_not"]


def test_a_table_is_split_by_an_input_that_takes_few_values() -> None:
    rng = np.random.default_rng(3)
    rows, pool = 400, 200
    X = np.column_stack([rng.uniform(1.0, 5.0, rows + pool), rng.choice([1.0, 2.0, 4.0, 8.0], rows + pool)])
    y = X[:, 0] ** 2 * X[:, 1]
    table = at.Table("grid", ["x", "level"], X[:rows], y[:rows], X[rows:], lambda asked: y[rows:][asked])
    family = at.by_levels(table)
    assert family is not None and family.names == ["x"] and family.conditions == ["level"]
    assert [t.conditions["level"] for t in family.tables] == [1.0, 2.0, 4.0, 8.0]
    assert sum(len(t.X) for t in family.tables) == rows and sum(len(t.pool_X) for t in family.tables) == pool
    for member in family.tables:                      # each member measures its own pool rows of the whole table
        level = member.conditions["level"]
        assert np.allclose(member.measure(np.arange(len(member.pool_X))), member.pool_X[:, 0] ** 2 * level)
    assert at.by_levels(table, "x") is None           # an input with a value a row is not a condition
    smooth = at.Table("smooth", ["a", "b"], rng.uniform(1, 5, (rows, 2)), y[:rows], rng.uniform(1, 5, (pool, 2)), lambda a: a)
    assert at.by_levels(smooth) is None
    assert at.by_levels(table, max_levels=3) is None and at.by_levels(table, min_rows=200) is None


def test_a_family_has_one_set_of_inputs_and_conditions(family: at.Family) -> None:
    first = family.tables[0]
    other = at.Table("other", ["x0", "z"], first.X, first.y, first.pool_X, first.measure, {"T": 1.0})
    with pytest.raises(ValueError, match="same inputs"):
        at.Family("bad", [first, other])
    bare = at.Table("bare", list(NAMES), first.X, first.y, first.pool_X, first.measure)
    with pytest.raises(ValueError, match="same conditions"):
        at.Family("bad", [first, bare])
    with pytest.raises(ValueError, match="at least two"):
        at.Family("bad", [first])
    clash = at.Table("clash", list(NAMES), first.X, first.y, first.pool_X, first.measure, {"x0": 1.0})
    with pytest.raises(ValueError, match="name of an input"):
        at.Family("bad", [clash, clash])
    assert at.conditions_table(family) == {"T": [1.0, 2.0, 3.0, 4.0, 5.0]}
