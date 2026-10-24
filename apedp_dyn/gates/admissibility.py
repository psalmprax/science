"""Gate 2 — physical admissibility: structure, symbols, functions, bounds."""
from __future__ import annotations

import numpy as np
import sympy as sp

from apedp_dyn.gates.dimensional import ALLOWED_FUNCTIONS, GateResult
from apedp_dyn.model.variables import TIME_SYMBOL


def equation_admissibility_structure(model,
                                     allowed: frozenset[str] = ALLOWED_FUNCTIONS
                                     ) -> GateResult:
    """Gate 2 structural part: no parameter values needed (pre-fit pruning)."""
    names = [s.name for s in model.states]
    eq_states = [e.state for e in model.equations]
    if sorted(eq_states) != sorted(names):
        return GateResult("equation_admissibility", False,
                          f"equation coverage {sorted(eq_states)} != states {sorted(names)}")
    if len(set(eq_states)) != len(eq_states):
        return GateResult("equation_admissibility", False, "duplicate equations")
    known = set(names) | {u.name for u in model.inputs} | \
        {p.name for p in model.parameters} | {TIME_SYMBOL}
    func_names = {type(n).__name__ for e in model.equations
                  for n in e.sympy_expr.atoms(sp.Function)}
    unknown_funcs = {f for f in func_names if f not in allowed}
    if unknown_funcs:
        return GateResult("equation_admissibility", False,
                          f"disallowed functions: {sorted(unknown_funcs)}")
    for eq in model.equations:
        syms = {str(s) for s in eq.sympy_expr.atoms(sp.Symbol)}
        unknown = syms - known
        if unknown:
            return GateResult("equation_admissibility", False,
                              f"{eq.state}: unknown symbols {sorted(unknown)}")
        if eq.sympy_expr.atoms(sp.Derivative):
            return GateResult("equation_admissibility", False,
                              f"{eq.state}: derivatives in RHS (D1 is explicit first-order)")
    for p in model.parameters:
        if p.bounds is not None:
            lo, hi = p.bounds
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                return GateResult("equation_admissibility", False,
                                  f"parameter {p.name}: non-finite bounds {p.bounds}")
    return GateResult("equation_admissibility", True,
                      f"{len(model.equations)} explicit first-order equations (structure)")


def equation_admissibility(model,
                           allowed: frozenset[str] = ALLOWED_FUNCTIONS) -> GateResult:
    """Gate 2: structural admissibility of the equation set."""
    structural = equation_admissibility_structure(model, allowed)
    if not structural.passed:
        return structural
    missing = [p.name for p in model.parameters if p.value is None]
    if missing:
        return GateResult("equation_admissibility", False,
                          f"unidentified parameters: {missing}")
    try:
        r0 = model.rhs(0.0, model.initial_vector(),
                       lambda t: np.zeros(len(model.inputs)))
    except (ArithmeticError, ValueError, TypeError, ZeroDivisionError) as e:
        return GateResult("equation_admissibility", False,
                          f"RHS not finite at initial condition: {e}")
    if not np.all(np.isfinite(r0)):
        return GateResult("equation_admissibility", False,
                          "RHS not finite at initial condition")
    return GateResult("equation_admissibility", True,
                      f"{len(model.equations)} explicit first-order equations")
