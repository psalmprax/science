"""Restricted numeric evaluator for symbolic-search expressions."""
from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Any

import numpy as np

_EXPR_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "tanh": np.tanh,
    "abs": np.abs,
}


def evaluate_expression(
    expr: str,
    var_names: Sequence[str],
    data: Sequence[Any],
) -> np.ndarray | None:
    """Evaluate the supported arithmetic expression, or return ``None``.

    The interpreter accepts declared variable names, numeric literals, basic
    arithmetic, and one-argument calls to the functions in ``_EXPR_FUNCS``.
    Python evaluation and attribute access are never used.
    """
    scope = {
        name: np.asarray(column, float)
        for name, column in zip(var_names, data, strict=False)
    }
    try:
        tree = ast.parse(expr, mode="eval")
        with np.errstate(all="ignore"):
            value = _evaluate_node(tree.body, scope)
    except (
        SyntaxError,
        NameError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        OverflowError,
        KeyError,
    ):
        return None
    try:
        return np.asarray(value, float)
    except (TypeError, ValueError):
        return None


def _evaluate_node(node: ast.expr, scope: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise TypeError("boolean literals are not supported")
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in scope:
            raise NameError(node.id)
        return scope[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_node(node.operand, scope)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, scope)
        right = _evaluate_node(node.right, scope)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = _EXPR_FUNCS.get(node.func.id)
        if function is not None and len(node.args) == 1 and not node.keywords:
            return function(_evaluate_node(node.args[0], scope))
    raise ValueError(f"unsupported expression syntax: {type(node).__name__}")
