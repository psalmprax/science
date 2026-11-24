"""The small-table reading and what is known beforehand: forms, leaving rows out, the plainest within one standard error."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from apedp_dyn.discovery_l2 import known_beforehand as known
from apedp_dyn.discovery_l2 import small_table as st

ROOT = Path(__file__).resolve().parents[1]


def _real_data() -> object:
    spec = importlib.util.spec_from_file_location("real_data", ROOT / "scripts" / "real_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -------------------------------------------------------------------- forms
def test_a_candidates_floats_become_constants_to_refit_and_exponents_stay_unless_asked() -> None:
    form = st.form_of("2.5*x0**1.5/(0.7 + x1) + 3.0", ["x0", "x1"], "test")
    assert form is not None and len(form.params) == 3 and form.key == "C*x0**1.5/(C + x1) + C"
    free = st.form_of("2.5*x0**1.5", ["x0"], "test", free_exponents=True)
    assert free is not None and len(free.params) == 2
    bare = st.form_of("x0**Rational(3, 2)", ["x0"], "test")
    assert bare is not None and len(bare.params) == 1                      # a bare shape gets a scale
    assert st.form_of("2.0*x0 + q", ["x0"], "test") is None and st.form_of("3.0", ["x0"], "test") is None


def test_a_form_linear_in_its_constants_is_fitted_exactly_and_its_left_out_errors_match_refitting() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 5, (12, 1))
    y = 2.0 * np.sqrt(X[:, 0]) + 1.0 + 0.05 * rng.standard_normal(12)
    form = st.form_of("1.0*sqrt(x0) + 1.0", ["x0"], "test")
    assert form is not None and form.columns is not None
    theta = st.fit(form, X, y)
    assert theta is not None and sorted(theta) == pytest.approx([1.0, 2.0], abs=0.2)
    quick = st.left_out_errors(form, X, y, theta, seed=0)
    form.columns = None                                                     # the same form through the general path
    slow = st.left_out_errors(form, X, y, st.fit(form, X, y), seed=0)
    assert quick is not None and slow is not None and quick == pytest.approx(slow, rel=1e-4, abs=1e-9)
    inner = st.form_of("1.0*x0/(2.0 + x0)", ["x0"], "test")
    assert inner is not None and inner.columns is None                      # a constant inside: not linear


def test_nesting_counts_one_transcendental_function_inside_another() -> None:
    x = sp.Symbol("x")
    assert st.nesting(sp.exp(-0.3 * x) / (1 + x)) == 1 and st.nesting(sp.sqrt(sp.log(x))) == 1
    assert st.nesting(sp.sin(sp.exp(sp.sqrt(x)))) == 2 and st.nesting(x ** 2 + 1) == 0


def test_errors_are_relative_only_when_the_target_keeps_its_sign_and_spans_a_decade() -> None:
    assert np.all(st.weights_of(np.array([1.0, 2.0, 5.0]), "auto") == 1.0)
    assert st.weights_of(np.array([0.1, 2.0, 50.0]), "auto") == pytest.approx([10.0, 0.5, 0.02])
    assert np.all(st.weights_of(np.array([-1.0, 2.0, 50.0]), "auto") == 1.0)
    assert st.weights_of(np.array([1.0, 2.0]), "relative") == pytest.approx([1.0, 0.5])
    with pytest.raises(ValueError):
        st.weights_of(np.array([0.0, 2.0]), "relative")


def test_the_choice_is_the_plainest_form_within_one_standard_error_of_the_best() -> None:
    def scored(bits: float, errors: list[float]) -> st.Scored:
        form = st.Form(f"form{bits}", sp.Symbol("x"), [], np.zeros(0), "test", bits)
        return st.Scored(form, np.zeros(0), np.array(errors))

    best = scored(90.0, [1.0, 1.2, 0.8, 1.0])                               # mean 1.0, standard error about 0.08
    plain_and_close = scored(40.0, [1.05, 1.1, 1.0, 1.05])
    plain_and_far = scored(20.0, [2.0, 2.2, 1.8, 2.0])
    chosen, members = st.choose([best, plain_and_close, plain_and_far])
    assert chosen is plain_and_close and members == [plain_and_close, best]


# ------------------------------------------------------------------ reading
def test_a_power_law_on_eight_clean_rows_is_reported_with_its_exponent_as_a_simple_fraction() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(0.3, 30.0, 8)
    reading = st.read({"d": x}, 0.36525 * x ** 1.5, memory=False)
    assert reading["action"] == "read" and reading["reported"] is True and reading["errors"] == "relative"
    assert reading["chosen"]["shape"] == "C*d**(3/2)" and reading["stability"]["stable"] is True
    assert reading["accuracy"]["reached"] is True


def test_scatter_keeps_a_reading_tentative_and_says_why() -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(1.0, 5.0, 14)
    y = 3.0 * x + 2.0 + 1.5 * rng.standard_normal(14)
    reading = st.read({"x0": x}, y, memory=False)
    assert reading["action"] == "read" and reading["reported"] is False and reading["tentative"] is True
    assert "its error on left-out rows is above the accuracy line" in reading["tentative_reasons"]
    assert reading["law"] and reading["class_size"] >= 1


def test_too_few_rows_are_not_read_and_candidates_respect_both_limits() -> None:
    assert st.read({"x0": [1.0, 2.0, 3.0]}, [1.0, 2.0, 3.0], memory=False)["action"] == "not_run"
    rng = np.random.default_rng(3)
    x = rng.uniform(1.0, 5.0, 9)
    forms, report = st.candidates({"x0": x}, np.sin(1.3 * x) + 0.2 * x, memory=False)
    assert forms and all(len(form.params) <= 9 // st.ROWS_PER_CONSTANT for form in forms)
    assert all(st.nesting(form.expr) <= st.MAX_NESTING for form in forms)
    assert set(report["left_out"]) == {"more_constants_than_the_rows_allow", "one_function_inside_another"}


def test_two_inputs_with_the_same_unit_may_be_added_in_a_power_law() -> None:
    rng = np.random.default_rng(4)
    X = rng.uniform(0.5, 3.0, (14, 3))
    y = 1.7 * np.sqrt(X[:, 2] ** 3 / (X[:, 0] + X[:, 1]))
    columns = {"m1": X[:, 0], "m2": X[:, 1], "d": X[:, 2]}
    with_units = st.read(columns, y, units={"m1": "solar masses", "m2": "solar masses", "d": "au"}, memory=False)
    assert with_units["reported"] is True and "m1 + m2" in with_units["law"]
    assert sp.simplify(sp.sympify(with_units["chosen"]["shape"].replace("C", "1")) - sp.sympify("d**(3/2)/sqrt(m1 + m2)")) == 0


# --------------------------------------------------------- known beforehand
RANGES = {"p": (2.7, 122.0)}


def test_a_rise_is_proved_for_a_ratio_of_polynomials_and_checked_on_a_grid_otherwise() -> None:
    pytest.importorskip("z3")
    p = sp.Symbol("p")
    rational = known.check(107.9 * p / (p + 8.19), ["p"], {"kind": "increasing", "input": "p", "over": [0.0, 500.0]}, RANGES, 104.0)
    assert rational["holds"] is True and rational["how"] == "proved"
    cosine = known.check(46.3 - 53.3 * sp.cos(p ** 0.25), ["p"], {"kind": "increasing", "input": "p", "over": [0.0, 500.0]}, RANGES, 104.0)
    assert cosine["holds"] is False and cosine["how"] == "grid" and cosine["fails"]["at"] > 90.0     # it turns at pi**4
    falling = known.check(5.0 - 2.0 * p, ["p"], {"kind": "increasing", "input": "p"}, RANGES, 104.0)
    assert falling["holds"] is False and falling["how"] == "proved"


def test_zero_at_zero_levelling_off_and_bending_tell_langmuirs_form_from_its_rivals() -> None:
    p = sp.Symbol("p")
    langmuir, root, log = 107.9 * p / (p + 8.19), 9.4 * sp.sqrt(p), 19.8 * sp.log(p) + 11.2
    statements = [{"kind": "zero_at_zero"}, {"kind": "concave", "input": "p", "over": [0.0, 500.0]}, {"kind": "levels_off", "input": "p"}]
    assert known.agrees(known.check_all(langmuir, ["p"], statements, RANGES, 104.0))
    by_kind = {c["kind"]: c["holds"] for c in known.check_all(root, ["p"], statements, RANGES, 104.0)}
    assert by_kind == {"zero_at_zero": True, "concave": True, "levels_off": False}        # a root keeps growing
    assert known.check(log, ["p"], statements[0], RANGES, 104.0)["holds"] is False        # a logarithm has no value at zero
    with pytest.raises(ValueError):
        known.check(langmuir, ["p"], {"kind": "pretty"}, RANGES, 104.0)


def test_an_interval_can_be_stated_as_multiples_of_how_far_the_table_goes() -> None:
    p = sp.Symbol("p")
    cosine = 46.3 - 53.3 * sp.cos(p ** 0.25)                                 # rises until p = pi**4, about 97
    inside = known.check(cosine, ["p"], {"kind": "increasing", "input": "p", "over_times": [0.1, 0.5]}, RANGES, 104.0)
    beyond = known.check(cosine, ["p"], {"kind": "increasing", "input": "p", "over_times": [0.0, 4.0]}, RANGES, 104.0)
    assert inside["holds"] is True and inside["over"] == pytest.approx([12.2, 61.0])
    assert beyond["holds"] is False and beyond["over"] == pytest.approx([0.0, 488.0])


def test_stated_errors_weigh_the_rows_and_a_larger_table_is_read_in_folds() -> None:
    rng = np.random.default_rng(5)
    x = rng.uniform(0.1, 2.0, 60)
    sigma = np.where(x > 1.5, 30.0, 1.0)                                     # the far rows are barely measured
    y = 70.0 * (1.0 + x) + sigma * rng.standard_normal(60)
    reading = st.read({"z": x}, y, sigma=sigma, memory=False)
    assert reading["action"] == "read" and reading["errors"] == "in units of each row's stated error"
    assert reading["stability"]["rows_removed"] == st.FOLDS                  # sixty rows: ten folds, not sixty removals
    assert reading["chosen"]["shape"] in ("C*z + C", "C*(z + C)") or reading["class_size"] > 1
    assert st.read({"z": x}, y, sigma=sigma[:-1], memory=False)["action"] == "not_run"


def test_a_statement_that_cannot_be_checked_does_not_exclude_a_law() -> None:
    assert known.agrees([{"kind": "increasing", "holds": None}, {"kind": "concave", "holds": True}])
    assert not known.agrees([{"kind": "increasing", "holds": False}])


# ------------------------------------------------ the development real tables
def test_keplers_eight_planets_give_keplers_law_reported() -> None:
    loaded = _real_data().load("kepler_solar_system")
    reading = st.read(loaded["X"], loaded["y"], memory=False)
    assert reading["reported"] is True and reading["chosen"]["shape"] == "C*d**(3/2)"
    assert float(sp.sympify(reading["law"]).subs("d", 1.0)) == pytest.approx(0.36525, rel=0.01)   # a year, in thousands of days


def test_langmuirs_ten_rows_give_langmuirs_form_only_with_what_is_known_about_adsorption() -> None:
    pytest.importorskip("z3")
    loaded = _real_data().load("langmuir_table_ix")
    statements = [{"kind": "zero_at_zero"}, {"kind": "increasing", "input": "p", "over": [0.0, 500.0]},
                  {"kind": "concave", "input": "p", "over": [0.0, 500.0]}, {"kind": "levels_off", "input": "p"}]
    plain = st.read(loaded["X"], loaded["y"], memory=False)
    informed = st.read(loaded["X"], loaded["y"], known=statements, memory=False)
    assert plain["chosen"]["shape"] != "C*p/(C + p)"                        # the rows alone do not single it out
    assert informed["chosen"]["shape"] == "C*p/(C + p)" and informed["stability"]["stable"] is True
    assert informed["reported"] is False and informed["tentative_reasons"] == ["its error on left-out rows is above the accuracy line"]
    assert informed["set_aside_by_what_is_known"]["forms"] > 0
    proved = [c for c in informed["known_checks_on_the_law"] if c["kind"] == "increasing"]
    assert proved and proved[0]["how"] == "proved" and proved[0]["holds"] is True
