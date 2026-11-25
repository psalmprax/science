"""The study of the table split: exact shapes are what they should be, a wrong shape is scored as far off, traps are traps."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_wider_conditions as wider
import study_table_split as study

from apedp_dyn.proposers import table_shape as sep

LEAF = sep.Shape(sep.LEAF, (0, 1))


def _shape(text: str, inputs: int) -> tuple[np.ndarray, sep.Shape]:
    tensor = study.exact_tensor(text, inputs)
    return tensor, study.exact_shape(tensor, tuple(range(inputs)))


@pytest.mark.parametrize("text, inputs, expected", [
    ("x0**2*sin(x1)", 2, "f(x0)*f(x1)"),
    ("sqrt(x0) + log(x1)", 2, "f(x0) + f(x1)"),
    ("2.5 + x0*exp(-x1)", 2, "c + f(x0)*f(x1)"),
    ("x0*log(1 + x1) + 0.5*x1**2", 2, "f(x0)*f(x1) + f(x1)"),
    ("x1*log(1 + x0) + 0.5*x0**2", 2, "f(x0)*f(x1) + f(x0)"),
    ("sin(0.7*x0*x1)", 2, "f(x0, x1)"),
])
def test_exact_shapes_of_laws_whose_shape_is_known(text, inputs, expected):
    assert _shape(text, inputs)[1].text() == expected


def test_a_law_with_its_own_shape_departs_from_it_by_nothing_and_from_a_wrong_one_by_much():
    tensor, shape = _shape("x0*log(1 + x1) + 0.5*x1**2 + sin(x2)", 3)
    assert shape.widest() == 1 and study.departure(tensor, shape) < study.EXACT
    assert study.verdict(study.departure(tensor, shape)) == "exact"
    product = sep.Shape(sep.PRODUCT, (0, 1, 2), (sep.Shape(sep.LEAF, (0, 1)), sep.Shape(sep.LEAF, (2,))))
    assert study.departure(tensor, product) > study.CLOSE and study.verdict(study.departure(tensor, product)) == "wrong"
    assert study.departure(tensor, sep.Shape(sep.LEAF, (0, 1, 2))) == 0.0


def test_a_sum_is_not_passed_off_as_a_product_with_a_huge_constant():
    tensor = study.exact_tensor("sqrt(x0) + log(x1)", 2)
    assert study.project(study._matrix(tensor, (0, 1), (0,), (1,)), sep.PRODUCT_AND_CONSTANT) is None


def test_every_trap_has_inputs_no_form_separates():
    for text in study.TRAPS:
        inputs = max(job["inputs"] for job in study.jobs("traps") if job["text"] == text)
        assert _shape(text, inputs)[1].widest() >= 2, text


def test_the_sets_are_what_they_are_called():
    first, second, traps = study.jobs("first"), study.jobs("second"), study.jobs("traps")
    assert first == wider.draw_laws() and len(second) == len(first) == 120
    assert not {job["text"] for job in first} & {job["text"] for job in second}
    assert len(traps) == len(study.TRAPS) * len(wider.NOISES) and {job["noise"] for job in traps} == set(wider.NOISES)
    with pytest.raises(ValueError):
        study.jobs("third")


def test_the_first_sets_tables_are_the_rows_the_wider_conditions_run_read():
    job = study.jobs("first")[0]
    X, y = study.table(job, "first")
    assert X.shape == (wider.ROWS, job["inputs"]) and y.shape == (wider.ROWS,)
    other, _ = study.table(job, "second")
    assert not np.array_equal(X, other)


def test_the_summary_counts_what_it_says():
    rows = [
        {"noise": 0.01, "inputs": 3, "read": True, "verdict": "exact", "parts": 3, "widest": 1, "exact_widest": 1, "seconds": 2.0,
         "departure": 0.0},
        {"noise": 0.01, "inputs": 4, "read": True, "verdict": "wrong", "parts": 4, "widest": 1, "exact_widest": 2, "seconds": 4.0,
         "departure": 0.2},
        {"noise": 0.05, "inputs": 4, "read": False, "verdict": "exact", "parts": 1, "widest": 4, "exact_widest": 1, "seconds": 6.0,
         "departure": 0.0},
    ]
    got = study.summarise(rows)
    assert got["all"]["split_and_within_the_accuracy_line"] == 1 and got["all"]["split_and_beyond_the_accuracy_line"] == 1
    assert 0.03 < study.ACCURACY_LINE < 0.04
    assert got["all"]["tables"] == 3 and got["all"]["exact"] == 2 and got["all"]["wrong"] == 1 and got["all"]["not_read"] == 1
    assert got["all"]["split_at_all"] == 2 and got["all"]["exact_and_as_fine"] == 1 and got["all"]["finer_than_the_law_allows"] == 1
    assert got["by_noise"]["0.01"]["tables"] == 2 and got["by_inputs"]["4"]["tables"] == 2 and got["all"]["median_seconds"] == 4.0


def test_the_older_packages_analyzer_is_read_as_a_shape_and_scored_the_same_way():
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 5.0, (600, 3))
    assert study.earlier_shape(X, X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2).text() == "f(x0) + (f(x1) + f(x2))"
    product = study.earlier_shape(X, X[:, 0] * X[:, 1] * X[:, 2])
    assert product.form == sep.PRODUCT and len(product.leaves()) >= 2
    assert study.departure(study.exact_tensor("x0*x1*x2", 3), product) < study.EXACT
    whole = study.earlier_shape(X, np.sin(X[:, 0] * X[:, 1] * X[:, 2]))
    assert whole == sep.Shape(sep.LEAF, (0, 1, 2))
    tensor = study.exact_tensor("x0**2 + x1**2 + x2**2", 3)
    assert study.departure(tensor, study.earlier_shape(X, X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2)) < study.EXACT
