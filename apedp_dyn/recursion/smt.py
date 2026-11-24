"""Proofs with an SMT solver (z3) about laws on a box of inputs: no pole there, or two laws equal there.

Status: experimental

A numerical check samples rows; a solver decides for every point of the box at
once. That is only possible for the part of the grammar a solver can reason
about exactly: polynomials with rational coefficients (z3's nonlinear real
arithmetic). So every answer is three-valued:

* ``True`` / ``False``: proved, with a witness point where there is one;
* ``None``: not decided, with the reason (a denominator that is not a
  polynomial, the solver's time limit, or z3 not installed).

Floats are read as the decimal they print as (``1.3`` is 13/10), so a proof is
about the law as written, not about its binary rounding.

``pole_in_box`` looks at what a law divides by and takes logarithms of. It is a
proposer-side filter and a development tool: the referee and the last checks do
not call it, and it changes nothing they decide.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sympy as sp

TIMEOUT_MS = 2000
Box = Mapping[str, tuple[float, float]]


def _z3() -> Any:
    try:
        import z3
    except ImportError:
        return None
    return z3


def _rational(expr: sp.Expr) -> sp.Expr:
    """The expression with every float replaced by the decimal fraction it prints as."""
    return expr.xreplace({f: sp.Rational(str(f)) for f in expr.atoms(sp.Float)})


def singular_parts(expr: sp.Expr) -> list[tuple[str, sp.Expr]]:
    """What the law divides by (``"denominator"``) and takes the logarithm of (``"log"``), where it depends on an input."""
    out: list[tuple[str, sp.Expr]] = []
    for node in sp.preorder_traversal(expr):
        if node.is_Pow and node.exp.is_number and node.exp.is_negative and node.base.free_symbols:
            out.append(("denominator", node.base))
        elif isinstance(node, sp.log) and node.args[0].free_symbols:
            out.append(("log", node.args[0]))
    return out


def _to_z3(z3: Any, poly: sp.Poly, variables: dict[str, Any]) -> Any:
    total: Any = z3.RealVal(0)
    for powers, coefficient in poly.terms():
        term: Any = z3.RealVal(str(sp.Rational(coefficient)))
        for symbol, power in zip(poly.gens, powers, strict=True):
            for _ in range(int(power)):
                term = term * variables[str(symbol)]
        total = total + term
    return total


def _solver(z3: Any, box: Box, names: Sequence[str]) -> tuple[Any, dict[str, Any]]:
    solver = z3.Solver()
    solver.set("timeout", TIMEOUT_MS)
    variables = {name: z3.Real(name) for name in names}
    for name in names:
        low, high = box[name]
        solver.add(variables[name] >= z3.RealVal(str(sp.Rational(str(low)))),
                   variables[name] <= z3.RealVal(str(sp.Rational(str(high)))))
    return solver, variables


def _witness(z3: Any, model: Any, variables: dict[str, Any]) -> dict[str, float]:
    out = {}
    for name, variable in variables.items():
        value = model.eval(variable, model_completion=True)
        try:
            out[name] = float(value.as_fraction())
        except (AttributeError, z3.Z3Exception):
            out[name] = float(value.approx(12).as_fraction())
    return out


def zero_in_box(expr: sp.Expr, box: Box, *, sign_too: bool = False) -> dict[str, Any]:
    """Whether a polynomial reaches zero (or, with ``sign_too``, zero or below) somewhere in the box."""
    z3 = _z3()
    if z3 is None:
        return {"decided": None, "reason": "z3 is not installed"}
    expr = _rational(sp.sympify(expr))
    symbols = sorted(expr.free_symbols, key=str)
    if not symbols:                                   # a number: nothing to solve
        value = sp.nsimplify(expr)
        return {"decided": bool(value <= 0) if sign_too else bool(value == 0)}
    missing = [str(s) for s in symbols if str(s) not in box]
    if missing:
        return {"decided": None, "reason": f"no bounds for {missing}"}
    try:
        poly = sp.Poly(expr, *symbols)
    except sp.polys.polyerrors.BasePolynomialError:
        return {"decided": None, "reason": "not a polynomial in the inputs"}
    if not all(c.is_Rational for c in poly.coeffs()):
        return {"decided": None, "reason": "a coefficient is not rational"}
    solver, variables = _solver(z3, box, [str(s) for s in symbols])
    value = _to_z3(z3, poly, variables)
    solver.add(value <= 0 if sign_too else value == 0)
    verdict = solver.check()
    if verdict == z3.sat:
        return {"decided": True, "witness": _witness(z3, solver.model(), variables)}
    if verdict == z3.unsat:
        return {"decided": False}
    return {"decided": None, "reason": f"the solver did not decide in {TIMEOUT_MS} ms"}


def pole_in_box(expr: sp.Expr, box: Box) -> dict[str, Any]:
    """Whether the law has a pole in the box: a denominator that reaches zero, or a logarithm's argument that reaches zero or below.

    ``{"pole": True, ...}`` names the part and a point; ``{"pole": False}`` means every
    such part was proved clear; ``{"pole": None}`` means at least one part was not decided
    and none was proved to have a pole.
    """
    undecided: list[str] = []
    for kind, part in singular_parts(sp.sympify(expr)):
        found = zero_in_box(part, box, sign_too=kind == "log")
        if found["decided"] is True:
            return {"pole": True, "part": str(part), "kind": kind, "witness": found["witness"]}
        if found["decided"] is None:
            undecided.append(f"{part}: {found['reason']}")
    if undecided:
        return {"pole": None, "undecided": undecided}
    return {"pole": False}


def equal_in_box(a: sp.Expr, b: sp.Expr, box: Box) -> dict[str, Any]:
    """Whether two rational laws are equal at every point of the box where both are defined.

    The difference is written as one fraction; the laws are equal where defined exactly when
    its numerator is zero there. ``True`` is a proof; ``False`` comes with a point where the
    numerator is not zero.
    """
    z3 = _z3()
    if z3 is None:
        return {"equal": None, "reason": "z3 is not installed"}
    difference = sp.together(_rational(sp.sympify(a)) - _rational(sp.sympify(b)))
    numerator = sp.fraction(difference)[0]
    symbols = sorted(numerator.free_symbols, key=str)
    if not symbols:
        return {"equal": bool(sp.simplify(numerator) == 0)}
    try:
        poly = sp.Poly(sp.expand(numerator), *symbols)
    except sp.PolynomialError:
        return {"equal": None, "reason": "the difference is not a rational function of the inputs"}
    if any(str(s) not in box for s in symbols):
        return {"equal": None, "reason": "no bounds for an input"}
    if not all(c.is_Rational for c in poly.coeffs()):
        return {"equal": None, "reason": "a coefficient is not rational"}
    solver, variables = _solver(z3, box, [str(s) for s in symbols])
    solver.add(_to_z3(z3, poly, variables) != 0)
    verdict = solver.check()
    if verdict == z3.unsat:
        return {"equal": True}
    if verdict == z3.sat:
        return {"equal": False, "witness": _witness(z3, solver.model(), variables)}
    return {"equal": None, "reason": f"the solver did not decide in {TIMEOUT_MS} ms"}
