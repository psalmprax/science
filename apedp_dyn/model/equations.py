"""D1 model equations: one explicit first-order ODE per state."""
from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp

# Callables an ODE right-hand side may legitimately invoke. Everything else
# that looks like an identifier is bound to a plain symbol -- see
# _parse_physics_expr for why that distinction is load-bearing.
_ALLOWED_CALLS = frozenset({
    "exp", "log", "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "sqrt", "Abs", "sign",
    "erf", "erfc", "gamma", "atan2", "hypot", "Min", "Max", "Piecewise",
})


def _parse_physics_expr(expression: str) -> sp.Basic:
    """Parse an ODE right-hand side with unlisted identifiers bound to symbols.

    SymPy's parser pre-binds a long list of short names: ``I`` is the
    imaginary unit, ``E`` is Euler's number, ``N`` is nan, ``O`` is the
    order-of-growth symbol, and ``beta``/``gamma``/``zeta`` are special
    functions. None of those meanings is ever intended on the right-hand
    side of a physical ODE, but the binding happened silently: the RC
    circuit's ``(I - V/R)/C`` -- current in amps -- parsed to a
    complex-valued expression and failed far downstream with "float()
    argument must be a string or a real number, not 'complex'".

    Shadowing every identifier that is not an allowed function keeps the
    written meaning. It also makes a typo'd function name a NameError at
    parse time rather than an attribute lookup deeper in the stack.
    """
    local: dict[str, object] = {
        ident: sp.Symbol(ident)
        for ident in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression))
        if ident not in _ALLOWED_CALLS
    }
    return sp.parse_expr(expression, local_dict=local)


@dataclass(frozen=True)
class DynamicalEquation:
    """d(state)/dt = expression. Parsed once, stored as text + sympy tree."""

    state: str
    expression: str
    sympy_expr: sp.Basic = None  # type: ignore[assignment]

    def __init__(self, state: str, expression: str) -> None:
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "sympy_expr", _parse_physics_expr(expression))
