"""Typed Physical Dimensional Homogeneity Engine (Option 3).

Enforces SI dimensional homogeneity and Buckingham-Pi constraints on discovered
candidate hypotheses and expressions.

Guarantees:
1. Terms in additions and subtractions must have identical physical dimensions.
2. Arguments to transcendental functions (sin, cos, exp, log, tanh) must be
   strictly dimensionless.
3. Powers must have rational exponents.
4. Output dimension must match the target quantity's physical dimension.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from apedp_dyn.discovery_l2.dimensional_types import (
    Dimension,
    DimensionalError,
    _to_fraction,
)
from apedp_dyn.grammar.proposer_interface import CandidateProposal

# Standard SI base and derived dimensional constants
DIMENSIONLESS = Dimension.dimensionless()
MASS = Dimension(m=1)
LENGTH = Dimension(l=1)
TIME = Dimension(t=1)
CURRENT = Dimension(i=1)
TEMPERATURE = Dimension(th=1)
AMOUNT = Dimension(n=1)
LUMINOUS_INTENSITY = Dimension(j=1)
PRICE = Dimension(p=1)
QUANTITY = Dimension(q=1)

VELOCITY = Dimension(l=1, t=-1)
ACCELERATION = Dimension(l=1, t=-2)
FORCE = Dimension(m=1, l=1, t=-2)
ENERGY = Dimension(m=1, l=2, t=-2)
POWER = Dimension(m=1, l=2, t=-3)
PRESSURE = Dimension(m=1, l=-1, t=-2)
MOMENTUM = Dimension(m=1, l=1, t=-1)
FREQUENCY = Dimension(t=-1)


def _eval_node_dimension(node: ast.AST, var_dims: Mapping[str, Dimension]) -> Dimension:
    """Recursively infer physical dimension from an AST node."""
    if isinstance(node, ast.Constant):
        # Numeric literals and strings are dimensionless scalars
        return DIMENSIONLESS

    if isinstance(node, ast.Name):
        if node.id not in var_dims:
            raise DimensionalError(f"Variable '{node.id}' has no registered physical dimension")
        return var_dims[node.id]

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, (ast.UAdd, ast.USub)):
            return _eval_node_dimension(node.operand, var_dims)
        raise DimensionalError(f"Unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        left_dim = _eval_node_dimension(node.left, var_dims)
        right_dim = _eval_node_dimension(node.right, var_dims)

        if isinstance(node.op, (ast.Add, ast.Sub)):
            if left_dim != right_dim:
                raise DimensionalError(
                    f"Dimensional mismatch in addition/subtraction: {left_dim} vs {right_dim}"
                )
            return left_dim

        if isinstance(node.op, ast.Mult):
            return left_dim * right_dim

        if isinstance(node.op, ast.Div):
            return left_dim / right_dim

        if isinstance(node.op, ast.Pow):
            # Power exponent must evaluate to a constant rational number
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                exp_val = _to_fraction(node.right.value)
                return left_dim ** exp_val
            if isinstance(node.right, ast.UnaryOp) and isinstance(node.right.op, ast.USub):
                if isinstance(node.right.operand, ast.Constant) and isinstance(node.right.operand.value, (int, float)):
                    exp_val = -_to_fraction(node.right.operand.value)
                    return left_dim ** exp_val
            raise DimensionalError("Exponentiation exponent must be a constant rational number")

        raise DimensionalError(f"Unsupported binary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr.lower()

        # Transcendental functions require strictly dimensionless inputs
        if func_name in {
            "sin", "cos", "tan", "asin", "acos", "atan",
            "sinh", "cosh", "tanh", "exp", "log", "ln", "log10", "safelog"
        }:
            if not node.args:
                raise DimensionalError(f"Function '{func_name}' expects at least 1 argument")
            arg_dim = _eval_node_dimension(node.args[0], var_dims)
            if not arg_dim.is_dimensionless():
                raise DimensionalError(
                    f"Transcendental function '{func_name}' requires dimensionless argument, got {arg_dim}"
                )
            return DIMENSIONLESS

        if func_name == "sqrt":
            if not node.args:
                raise DimensionalError("sqrt expects 1 argument")
            arg_dim = _eval_node_dimension(node.args[0], var_dims)
            return arg_dim ** Fraction(1, 2)

        if func_name in {"abs", "sign", "diff"}:
            if not node.args:
                raise DimensionalError(f"{func_name} expects 1 argument")
            return _eval_node_dimension(node.args[0], var_dims)

        if func_name == "safediv":
            if len(node.args) != 2:
                raise DimensionalError("safediv expects 2 arguments (numerator, denominator)")
            num_dim = _eval_node_dimension(node.args[0], var_dims)
            den_dim = _eval_node_dimension(node.args[1], var_dims)
            if not den_dim.is_dimensionless():
                raise DimensionalError(
                    f"safediv denominator must be dimensionless for 1+|y| pole protection, got {den_dim}"
                )
            return num_dim

        raise DimensionalError(f"Unsupported function call in dimensional analysis: '{func_name}'")

    raise DimensionalError(f"Unsupported syntax in expression: {type(node).__name__}")


def validate_expression_dimensions(
    expr: str,
    variable_dimensions: Mapping[str, Dimension],
    target_dimension: Dimension | None = None,
) -> tuple[bool, str, Dimension | None]:
    """Validate that an expression is dimensionally homogeneous and matches target_dimension.

    Returns
    -------
    tuple[bool, str, Dimension | None]
        (is_valid, error_message, inferred_dimension)
    """
    clean_expr = expr.strip()
    if not clean_expr:
        return False, "Empty expression", None

    # Handle lag indexing notation like v0[t-1] -> v0
    import re
    norm_expr = re.sub(r"\[\s*t\s*-\s*\d+\s*\]", "", clean_expr)

    try:
        tree = ast.parse(norm_expr, mode="eval")
    except SyntaxError as e:
        return False, f"Syntax error in expression: {e}", None

    try:
        inferred = _eval_node_dimension(tree.body, variable_dimensions)
    except DimensionalError as err:
        return False, str(err), None

    if target_dimension is not None and inferred != target_dimension:
        return (
            False,
            f"Target dimension mismatch: inferred {inferred} != target {target_dimension}",
            inferred,
        )

    return True, "", inferred


class DimensionalHomogeneityGate:
    """Epistemic gate checking dimensional homogeneity of candidate proposals."""

    def __init__(
        self,
        variable_dimensions: Mapping[str, Dimension],
        target_dimension: Dimension | None = None,
    ) -> None:
        self._var_dims = dict(variable_dimensions)
        self._target_dim = target_dimension

    def adjudicate(self, proposal: CandidateProposal) -> tuple[bool, str, Dimension | None]:
        """Check if proposal.formula satisfies dimensional homogeneity."""
        formula = proposal.formula
        return validate_expression_dimensions(
            formula,
            self._var_dims,
            self._target_dim,
        )
