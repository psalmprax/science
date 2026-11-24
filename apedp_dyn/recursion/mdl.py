"""Description length in bits: what a law costs to write down, and what its misses cost.

Status: experimental

Two parts, as in the minimum description length principle:

* **the law**: every node of its tree is one choice among the symbols available
  (the operations, the variables, "a constant", and the library's blocks), so it
  costs ``log2`` of how many there are; a constant then costs
  ``CONSTANT_BITS`` for its value (a small whole number costs
  ``INTEGER_BITS``). A library block used whole costs one choice, plus its
  holes and constants. That is why a good library shortens laws;
* **the rows given the law**: the Gaussian code length of the residuals,
  ``n/2 * log2(2 pi e mse)``, floored so an exact fit is not worth unbounded bits.

These bits choose the library (``library.learn``) and order a library's
candidates. They are not a threshold of the referee and decide nothing it decides.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import sympy as sp

OPERATIONS = ("add", "mul", "pow", "exp", "log", "sin", "cos", "tanh", "tan", "sqrt")
CONSTANT_BITS = 16.0        # a fitted constant, to about five significant digits
INTEGER_BITS = 3.0          # a small whole number (an exponent, a 2)
MSE_FLOOR = 1e-12           # relative to the target's variance: below this a fit is exact


def alphabet(variables: int, blocks: int = 0) -> int:
    """How many symbols a node is chosen among."""
    return len(OPERATIONS) + int(variables) + 1 + int(blocks)


def _nodes(expr: sp.Expr) -> tuple[int, int, int]:
    """(choices, fitted constants, small whole numbers) in an expression tree, n-ary sums and products written as binary."""
    if expr.is_Symbol:
        return 1, 0, 0
    if expr.is_Number:
        whole = bool(expr.is_Integer or (expr.is_Rational and abs(expr.q) <= 4)) and abs(float(expr)) <= 8
        return 1, int(not whole), int(whole)
    choices, constants, wholes = max(len(expr.args) - 1, 1), 0, 0
    for arg in expr.args:
        c, k, w = _nodes(arg)
        choices, constants, wholes = choices + c, constants + k, wholes + w
    return choices, constants, wholes


def law_bits(expr: sp.Expr, variables: int, blocks: int = 0) -> float:
    """What the law costs to write with no block used."""
    choices, constants, wholes = _nodes(sp.sympify(expr))
    return choices * math.log2(alphabet(variables, blocks)) + constants * CONSTANT_BITS + wholes * INTEGER_BITS


def block_use_bits(holes: int, constants: int, variables: int, blocks: int) -> float:
    """What one use of a block costs: the choice of the block, a variable for each hole, a value for each constant."""
    per = math.log2(alphabet(variables, blocks))
    return per + holes * per + constants * CONSTANT_BITS


def data_bits(residual: Sequence[float] | np.ndarray, target: Sequence[float] | np.ndarray) -> float:
    """The Gaussian code length of the residuals, in units where the target has variance one."""
    r: np.ndarray = np.asarray(residual, dtype=float)
    y: np.ndarray = np.asarray(target, dtype=float)
    spread = float(np.var(y)) or 1.0
    mse = max(float(np.mean(r ** 2)) / spread, MSE_FLOOR)
    return 0.5 * len(r) * math.log2(2 * math.pi * math.e * mse)


def total_bits(expr: sp.Expr, residual: Sequence[float] | np.ndarray, target: Sequence[float] | np.ndarray,
               variables: int, blocks: int = 0) -> float:
    return law_bits(expr, variables, blocks) + data_bits(residual, target)
