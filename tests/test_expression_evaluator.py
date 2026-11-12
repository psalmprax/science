from __future__ import annotations

import numpy as np
import pytest

from apedp.expression_evaluator import evaluate_expression


def test_evaluates_declared_variables_functions_and_arithmetic():
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([1.0, 2.0, 3.0])
    result = evaluate_expression("+sin(x) + y - 2 * x / 2 + x**2", ["x", "y"], [x, y])
    assert result == pytest.approx(np.sin(x) + y - x + x**2)


@pytest.mark.parametrize(
    "expression",
    [
        "unknown",
        "x.__class__",
        "__import__('os').system('true')",
        "sin(x, x)",
        "sin(x, offset=1)",
        "True",
        "1 / 0",
        "x +",
    ],
)
def test_refuses_unsupported_or_malformed_expressions(expression: str):
    assert evaluate_expression(expression, ["x"], [np.arange(3.0)]) is None


def test_accepts_scalar_numeric_expression():
    assert evaluate_expression("sqrt(9) + exp(0) + log(1)", [], []) == pytest.approx(4.0)


def test_accepts_unary_negative_expression():
    x = np.array([1.0, 2.0])
    assert evaluate_expression("-x", ["x"], [x]) == pytest.approx(-x)


def test_missing_columns_do_not_create_implicit_variables():
    assert evaluate_expression("x + y", ["x", "y"], [np.arange(3.0)]) is None


@pytest.mark.parametrize("error", [TypeError, ValueError])
def test_rejects_values_that_cannot_be_converted_to_numeric(monkeypatch, error):
    def fail_conversion(value, *args, **kwargs):
        raise error("not numeric")

    monkeypatch.setattr("apedp.expression_evaluator.np.asarray", fail_conversion)
    assert evaluate_expression("1", [], []) is None
