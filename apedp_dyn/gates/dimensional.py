"""Gate 1 — mathematical/dimensional validity. Reject immediately."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp
from pint import UnitRegistry

_ureg = UnitRegistry()

ALLOWED_FUNCTIONS = frozenset({
    "exp", "log", "sin", "cos", "tan", "asin", "acos", "atan",
    "sqrt", "Abs", "sign", "Pow", "sinh", "cosh", "tanh",
})

DIMENSIONLESS_FUNCS = frozenset({
    "exp", "log", "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh",
})


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


def _unit_of(expr: sp.Basic, units: dict[str, str]) -> Any:
    """Pint Unit of a sympy expression. Raises TypeError on violation."""
    if isinstance(expr, sp.Symbol):
        name = str(expr)
        if name not in units:
            raise TypeError(f"unknown symbol {name!r}")
        return _ureg.Unit(units[name])
    if isinstance(expr, sp.Number):
        return _ureg.dimensionless
    if isinstance(expr, sp.Add):
        sub = [_unit_of(a, units) for a in expr.args]
        first = sub[0].dimensionality
        if any(u.dimensionality != first for u in sub[1:]):
            raise TypeError(f"dimension mismatch in Add: {expr}")
        return sub[0]
    if isinstance(expr, sp.Mul):
        out: Any = _ureg.dimensionless
        for a in expr.args:
            out = out * _unit_of(a, units)
        return out
    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        exp_unit = _unit_of(exp, units)
        if exp_unit.dimensionality != _ureg.dimensionless.dimensionality:
            raise TypeError(f"dimensional exponent in {expr}")
        if not exp.is_number:
            raise TypeError(f"non-constant exponent in {expr}")
        return _unit_of(base, units) ** float(exp)
    func = type(expr).__name__
    if func in DIMENSIONLESS_FUNCS:
        (arg,) = expr.args
        if _unit_of(arg, units).dimensionality != _ureg.dimensionless.dimensionality:
            raise TypeError(f"dimensional argument to {func} in {expr}")
        return _ureg.dimensionless
    if func in ("sqrt", "Abs", "sign"):
        (arg,) = expr.args
        u = _unit_of(arg, units)
        if func == "sqrt":
            return u ** 0.5
        return u
    raise TypeError(f"unsupported function {func} in {expr}")


def _dims_of(expr: sp.Basic, units: dict[str, str]) -> Any:
    """Pint dimensionality of a sympy expression. Raises TypeError on violation."""
    return _unit_of(expr, units).dimensionality


def dimensional_validity(model) -> GateResult:
    """Gate 1: RHS units must equal state_units / t_units for every state."""
    units = model.symbol_units()
    state_units = {s.name: s.units for s in model.states}
    for eq in model.equations:
        if eq.state not in state_units:
            return GateResult("dimensional_validity", False,
                              f"equation for unknown state {eq.state!r}")
        try:
            got = _dims_of(eq.sympy_expr, units)
        except TypeError as e:
            return GateResult("dimensional_validity", False,
                              f"{eq.state}: {e}")
        want = (_ureg.Unit(state_units[eq.state]) / _ureg.Unit(model.t_units)).dimensionality
        if got != want:
            return GateResult(
                "dimensional_validity", False,
                f"{eq.state}: RHS dims {got} != state-rate dims {want}")
    return GateResult("dimensional_validity", True,
                      f"{len(model.equations)} equations dimensionally consistent")
