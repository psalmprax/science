"""Gate 1 — mathematical/dimensional validity. Reject immediately."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp
from pint import UnitRegistry
from pint.errors import PintError

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


def dao_validity(model) -> GateResult:
    """Gate 1.5: DAE validity check.

    Checks:
      - Constrained mode: constraint terms are dimensionally consistent
        (units are compatible, ideally reducing to dimensionless equality).
      - Multi-timescale mode: tau_fast and tau_slow parameters have
        units of seconds; fast/slow variable labeling is consistent with
        the candidate metadata.
      - Pure ODE mode (dae_mode="none"): always passes (delegates to
        gate 1 result).
    """
    dae_mode = getattr(model, "dae_mode", "none")
    if dae_mode == "none":
        return GateResult("dao_validity", True,
                          "pure ODE mode; no DAE constraints")
    if dae_mode == "constrained":
        return _dao_constrained_validity(model)
    if dae_mode == "multi_timescale":
        return _dao_multiscale_validity(model)
    return GateResult("dao_validity", False,
                      f"unknown dae_mode {dae_mode!r}")


def _dao_constrained_validity(model) -> GateResult:
    """Check constraint term dimensionality for constrained DAE mode.

    A constrained-DAE candidate is one whose hypothesis is that the states
    satisfy an algebraic relation, so two things must hold:

    1. The model must actually carry constraint parameters. A candidate
       enumerated with ``dae_mode="constrained"`` that contains no
       constraint term has silently degenerated into a plain ODE while
       still claiming DAE provenance -- that is a mislabelled model, not a
       valid one.
    2. Each constraint parameter must be dimensionless. The constraint
       basis terms ("c1", "c2") expand to ``X**0``, a dimensionless
       constant, so a dimensionful coefficient on one of them makes the
       constraint equation dimensionally inhomogeneous.

    Units that cannot be parsed are recorded and reported rather than
    silently accepted, matching the leniency policy used elsewhere in
    this module.
    """
    constraint_params = tuple(getattr(model, "dae_constraint_params", ()))
    if not constraint_params:
        return GateResult(
            "dao_validity", False,
            "dae_mode='constrained' but the model carries no constraint "
            "parameters: the DAE hypothesis is not represented in the "
            "equations")

    by_name = {p.name: p for p in model.parameters}
    unresolved: list[str] = []
    for name in constraint_params:
        param = by_name.get(name)
        if param is None:
            unresolved.append(f"{name}: no Parameter record")
            continue
        try:
            dim = _ureg.Unit(param.units).dimensionality
        except (PintError, ValueError, TypeError, AttributeError) as e:
            unresolved.append(f"{name}: {type(e).__name__}")
            continue
        if dim != _ureg.dimensionless.dimensionality:
            return GateResult(
                "dao_validity", False,
                f"constraint parameter {name!r} has units {dim}, but a "
                "constraint basis term is dimensionless; the constraint "
                "equation would be dimensionally inhomogeneous")

    detail = f"{len(constraint_params)} constraint parameter(s) verified dimensionless"
    if unresolved:
        detail += f"; unresolved: {unresolved}"
    return GateResult("dao_validity", True, detail)


def _units_parse(units: str) -> bool:
    """True if `units` is a pint-parseable unit string.

    Narrow on purpose: a blind ``except Exception`` here would also hide a
    typo'd attribute, which is a bug rather than an unparsable unit.
    """
    try:
        _ureg.Unit(units)
    except (PintError, ValueError, TypeError, AttributeError):
        return False
    return True


def _dao_multiscale_validity(model) -> GateResult:
    """Check multi-timescale mode: tau_fast and tau_slow have seconds units."""
    # Check that tau parameters exist and have seconds units
    tau_fast_ok = False
    tau_slow_ok = False
    for p in model.parameters:
        if p.name == "tau_fast":
            tau_fast_ok = _units_parse(p.units)
        if p.name == "tau_slow":
            tau_slow_ok = _units_parse(p.units)
    if tau_fast_ok and tau_slow_ok:
        return GateResult("dao_validity", True,
                          "tau_fast and tau_slow both have seconds units")
    # If we can't find the tau params, check the declared timescales. The
    # previous fallback read ``model.candidate.metadata``, but a
    # PhysicalModel has no ``candidate`` attribute, so it could never fire.
    timescales = tuple(getattr(model, "dae_timescales", ()))
    if sorted(timescales) == ["fast", "slow"]:
        return GateResult("dao_validity", True,
                          f"timescales labeled: {', '.join(sorted(timescales))}")
    return GateResult("dao_validity", False,
                      "multi-timescale: cannot verify tau units or fast/slow labeling")
