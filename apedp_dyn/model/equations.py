"""D1 model equations: one explicit first-order ODE per state."""
from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class DynamicalEquation:
    """d(state)/dt = expression. Parsed once, stored as text + sympy tree."""

    state: str
    expression: str
    sympy_expr: sp.Basic = None  # type: ignore[assignment]

    def __init__(self, state: str, expression: str) -> None:
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "sympy_expr", sp.sympify(expression))
