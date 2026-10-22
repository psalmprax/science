"""DE arm, shared-infrastructure: AD service (external-review step 2).

Not a discovery arm: RIME static repairs, ODE systems and PDE models may all
use this derivative service. It differentiates the IMPLEMENTED candidate
computation symbolically over a closed typed vocabulary — two honest limits
are declared in the module itself, per the review:

  1. This differentiates the candidate's implemented computation, NOT the
     unknown physical system: a perfect derivative of the wrong hypothesis
     stays wrong. Correctness here is only ever a SYMBOLIC property.
  2. Measured data is NOT maximally differentiable: derivative estimates of
     data keep flowing through the existing finite-difference stencils
     (Fornberg / 4th-order), which stay the source of truth for residuals.

Vocabulary (closed, refuse-unknown): Const, Var, Add, Sub, Mul, Div,
Pow(constant exponent), Tanh, Sin, Cos, Exp, Log (natural, x > 0 domain at
evaluation time). Every unsupported construction raises ValueError with the
reason — never silently approximated.

Status: experimental (DE-discovery arm phase 1).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Expr",
    "differentiate",
    "evaluate",
    "numerical_check",
]


class Expr:
    """Base of the closed typed expression vocabulary (assume no-op)."""


@dataclass(frozen=True)
class Const(Expr):
    value: float


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class Add(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Sub(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Mul(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Div(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Pow(Expr):
    base: Expr
    exponent: float  # CONSTANT exponent only: symbolic exponents refuse.


@dataclass(frozen=True)
class Tanh(Expr):
    arg: Expr


@dataclass(frozen=True)
class Sin(Expr):
    arg: Expr


@dataclass(frozen=True)
class Cos(Expr):
    arg: Expr


@dataclass(frozen=True)
class Exp(Expr):
    arg: Expr


@dataclass(frozen=True)
class Log(Expr):
    arg: Expr  # natural log; evaluation domain x > 0 enforced honestly


def _diff(e: Expr, var: str) -> Expr:
    """Symbolic derivative (exact rules; every unknown raises ValueError)."""
    if isinstance(e, Const):
        return Const(0.0)
    if isinstance(e, Var):
        return Const(1.0) if e.name == var else Const(0.0)
    if isinstance(e, Add):
        return Add(_diff(e.left, var), _diff(e.right, var))
    if isinstance(e, Sub):
        return Sub(_diff(e.left, var), _diff(e.right, var))
    if isinstance(e, Mul):
        # product rule: (fg)' = f'g + fg'  (MUTATION TARGET: f'g is WRONG)
        return Add(Mul(_diff(e.left, var), e.right), Mul(e.left, _diff(e.right, var)))
    if isinstance(e, Div):
        # quotient rule: (f/g)' = (f'g - fg') / g²
        num = Sub(Mul(_diff(e.left, var), e.right), Mul(e.left, _diff(e.right, var)))
        return Div(num, Pow(e.right, 2.0))
    if isinstance(e, Pow):
        if e.exponent == 0.0:
            return Const(0.0)
        return Mul(Mul(Const(e.exponent), Pow(e.base, e.exponent - 1.0)), _diff(e.base, var))
    if isinstance(e, Tanh):
        return Mul(Sub(Const(1.0), Pow(Tanh(e.arg), 2.0)), _diff(e.arg, var))
    if isinstance(e, Sin):
        return Mul(Cos(e.arg), _diff(e.arg, var))
    if isinstance(e, Cos):
        return Mul(Mul(Const(-1.0), Sin(e.arg)), _diff(e.arg, var))
    if isinstance(e, Exp):
        return Mul(Exp(e.arg), _diff(e.arg, var))
    if isinstance(e, Log):
        return Mul(Div(Const(1.0), e.arg), _diff(e.arg, var))
    raise ValueError(
        f"AD service refuses unknown expression node {type(e).__name__}: the "
        "vocabulary is closed and unsupported constructs never approximate")


def differentiate(e: Expr, var: str) -> Expr:
    """Exact symbolic derivative of a typed candidate computation."""
    if not isinstance(var, str) or not var:
        raise ValueError("differentiate needs a declared variable name")
    return _diff(e, var)


def evaluate(e: Expr, values: Mapping[str, float]) -> float:
    """Evaluate a typed expression; domain violations fail honestly."""
    if isinstance(e, Const):
        return float(e.value)
    if isinstance(e, Var):
        if e.name not in values:
            raise ValueError(f"unbound variable {e.name!r}")
        return float(values[e.name])
    if isinstance(e, (Add, Sub, Mul, Div)):
        a = evaluate(e.left, values)
        b = evaluate(e.right, values)
        if isinstance(e, Add):
            return a + b
        if isinstance(e, Sub):
            return a - b
        if isinstance(e, Mul):
            return a * b
        if b == 0.0:
            raise ValueError("division by zero: expression evaluation refused")
        return a / b
    if isinstance(e, Pow):
        return _eval_pow(e, values)
    if isinstance(e, Tanh):
        return float(np.tanh(evaluate(e.arg, values)))
    if isinstance(e, Sin):
        return float(np.sin(evaluate(e.arg, values)))
    if isinstance(e, Cos):
        return float(np.cos(evaluate(e.arg, values)))
    if isinstance(e, Exp):
        return float(np.exp(evaluate(e.arg, values)))
    if isinstance(e, Log):
        v = evaluate(e.arg, values)
        if v <= 0.0:
            raise ValueError("natural log evaluated outside its declared domain (arg > 0)")
        return float(np.log(v))
    raise ValueError(f"unknown expression node {type(e).__name__}")


def _eval_pow(e: Pow, values: Mapping[str, float]) -> float:
    return float(evaluate(e.base, values)) ** float(e.exponent)


def numerical_check(
    e: Expr, d_expr: Expr, values: Mapping[str, float], var: str, eps: float = 1e-6
) -> float:
    """|exact − central-difference| as an independent numerical audit helper.

    Central difference on the SAME implemented computation; the returned
    gap is reported only (a checker's input, never a verdict).
    """
    lo = dict(values)
    hi = dict(values)
    base = float(values[var])
    lo[var], hi[var] = base - eps, base + eps
    numeric = (evaluate(e, hi) - evaluate(e, lo)) / (2.0 * eps)
    return abs(evaluate(d_expr, values) - numeric)
