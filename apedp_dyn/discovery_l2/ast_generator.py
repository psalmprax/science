"""Physically Typed AST Construction and Generator for L2 Open Discovery.

Constructs mathematical expression trees while enforcing physical dimensional types
at every AST node:
  - BinaryOpNode:
      * '+' and '-': enforce left.dimension == right.dimension
      * '*': computes left.dimension * right.dimension
      * '/': computes left.dimension / right.dimension
  - UnaryOpNode:
      * 'exp', 'log': enforce child.dimension.is_dimensionless()
  - PowNode:
      * 'pow': requires constant rational exponent p (Fraction), computes base.dimension ** p
"""

from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from fractions import Fraction

import numpy as np
import sympy as sp

from apedp_dyn.discovery_l2.dimensional_types import (
    DIMENSIONLESS,
    Dimension,
    DimensionalError,
    _to_fraction,
)


class ASTNode(ABC):
    """Abstract base class for dimensionally typed AST expression nodes."""

    dimension: Dimension
    complexity: int

    @abstractmethod
    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        """Evaluate expression against variable environment returning a numpy array."""

    @abstractmethod
    def to_str(self) -> str:
        """Return human-readable algebraic string representation."""

    @abstractmethod
    def to_sympy(self) -> sp.Expr:
        """Convert AST to a SymPy expression for canonical algebraic reduction."""

    def canonical_str(self) -> str:
        """Return simplified canonical algebraic representation."""
        try:
            return str(self.to_sympy())
        except (TypeError, ValueError, ArithmeticError, sp.SympifyError):
            return self.to_str()

    def __repr__(self) -> str:
        return self.to_str()

    def __hash__(self) -> int:
        return int(zlib.crc32(self.canonical_str().encode()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ASTNode):
            return False
        return self.canonical_str() == other.canonical_str()


class VariableNode(ASTNode):
    """Leaf node representing an input variable with known physical dimension."""

    def __init__(self, name: str, dimension: Dimension) -> None:
        self.name = name
        self.dimension = dimension
        self.complexity = 1

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.name not in env:
            raise KeyError(f"Variable '{self.name}' not found in data environment")
        return np.asarray(env[self.name], dtype=float)

    def to_str(self) -> str:
        return self.name

    def to_sympy(self) -> sp.Expr:
        return sp.Symbol(self.name)


class ConstantNode(ASTNode):
    """Leaf node representing a constant numeric value (dimensionless by default)."""

    def __init__(self, value: float, dimension: Dimension = DIMENSIONLESS) -> None:
        self.value = float(value)
        self.dimension = dimension
        self.complexity = 1

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        if env:
            first = next(iter(env.values()))
            return np.full_like(first, self.value, dtype=float)
        return np.array(self.value, dtype=float)

    def to_str(self) -> str:
        return f"{self.value:g}"

    def to_sympy(self) -> sp.Expr:
        return sp.Float(self.value)


class BinaryOpNode(ASTNode):
    """Binary operator node (+, -, *, /) with physical dimensional type checking."""

    def __init__(self, op: str, left: ASTNode, right: ASTNode) -> None:
        if op not in ("+", "-", "*", "/"):
            raise ValueError(f"Unsupported binary operator: '{op}'")

        if op in ("+", "-"):
            if left.dimension != right.dimension:
                raise DimensionalError(
                    f"Cannot perform '{op}' on incompatible dimensions: "
                    f"[{left.dimension}] vs [{right.dimension}]"
                )
            dim = left.dimension
        elif op == "*":
            dim = left.dimension * right.dimension
        elif op == "/":
            dim = left.dimension / right.dimension
        else:
            raise ValueError(f"Unknown operator {op}")

        self.op = op
        self.left = left
        self.right = right
        self.dimension = dim
        self.complexity = left.complexity + right.complexity + 1

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        lv = self.left.evaluate(env)
        rv = self.right.evaluate(env)
        with np.errstate(all="ignore"):
            if self.op == "+":
                return lv + rv
            if self.op == "-":
                return lv - rv
            if self.op == "*":
                return lv * rv
            if self.op == "/":
                safe_rv = np.where(np.abs(rv) < 1e-15, 1.0, rv)
                res = lv / safe_rv
                return np.where(np.abs(rv) < 1e-15, np.nan, res)
        raise ValueError(f"Unknown binary operator '{self.op}'")

    def to_str(self) -> str:
        return f"({self.left.to_str()} {self.op} {self.right.to_str()})"

    def to_sympy(self) -> sp.Expr:
        ls = self.left.to_sympy()
        rs = self.right.to_sympy()
        if self.op == "+":
            return ls + rs
        if self.op == "-":
            return ls - rs
        if self.op == "*":
            return ls * rs
        if self.op == "/":
            return ls / rs
        raise ValueError(f"Unknown operator {self.op}")


class UnaryOpNode(ASTNode):
    """Unary operator node ('exp', 'log', 'sin', 'cos') strictly requiring dimensionless argument."""

    def __init__(self, op: str, child: ASTNode) -> None:
        if op not in ("exp", "log", "sin", "cos"):
            raise ValueError(f"Unsupported unary operator: '{op}'")
        if not child.dimension.is_dimensionless():
            raise DimensionalError(
                f"Transcendental operator '{op}' requires strictly dimensionless input [0, 0, ...], "
                f"got [{child.dimension}]"
            )
        self.op = op
        self.child = child
        self.dimension = DIMENSIONLESS
        self.complexity = child.complexity + 1

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        cv = self.child.evaluate(env)
        if self.op == "exp":
            with np.errstate(over="ignore", under="ignore"):
                clipped = np.clip(cv, -100.0, 100.0)
                return np.exp(clipped)
        if self.op == "log":
            with np.errstate(divide="ignore", invalid="ignore"):
                safe_cv = np.where(cv > 1e-15, cv, 1.0)
                res = np.log(safe_cv)
                return np.where(cv > 1e-15, res, np.nan)
        if self.op == "sin":
            with np.errstate(all="ignore"):
                return np.sin(cv)
        if self.op == "cos":
            with np.errstate(all="ignore"):
                return np.cos(cv)
        raise ValueError(f"Unknown unary operator '{self.op}'")

    def to_str(self) -> str:
        return f"{self.op}({self.child.to_str()})"

    def to_sympy(self) -> sp.Expr:
        cs = self.child.to_sympy()
        if self.op == "exp":
            return sp.exp(cs)
        if self.op == "log":
            return sp.log(cs)
        if self.op == "sin":
            return sp.sin(cs)
        if self.op == "cos":
            return sp.cos(cs)
        raise ValueError(f"Unknown operator {self.op}")


class PowNode(ASTNode):
    """Rational power node requiring constant rational exponent p."""

    def __init__(self, base: ASTNode, exponent: int | float | str | Fraction) -> None:
        try:
            p_frac = _to_fraction(exponent)
        except (ValueError, TypeError) as exc:
            raise TypeError(f"pow requires constant rational exponent p, got {exponent!r}") from exc
        self.base = base
        self.exponent = p_frac
        self.dimension = base.dimension**p_frac
        self.complexity = base.complexity + 1

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        bv = np.asarray(self.base.evaluate(env), dtype=float)
        p_flt = float(self.exponent)
        with np.errstate(invalid="ignore", divide="ignore"):
            if p_flt.is_integer():
                p_int = int(p_flt)
                if p_int < 0:
                    safe_bv = np.where(np.abs(bv) < 1e-15, 1.0, bv)
                    res = np.power(safe_bv, p_int)
                    return np.where(np.abs(bv) < 1e-15, np.nan, res)
                return np.power(bv, p_int)
            safe_bv = np.where(bv >= 0, bv, 0.0)
            res = np.power(safe_bv, p_flt)
            return np.where(bv >= 0, res, np.nan)

    def to_str(self) -> str:
        if self.exponent.denominator == 1:
            p_str = str(self.exponent.numerator)
        else:
            p_str = f"{self.exponent.numerator}/{self.exponent.denominator}"
        return f"({self.base.to_str()})^({p_str})"

    def to_sympy(self) -> sp.Expr:
        bs = self.base.to_sympy()
        p = sp.Rational(self.exponent.numerator, self.exponent.denominator)
        return sp.Pow(bs, p)


DEFAULT_POWERS: tuple[Fraction, ...] = (
    Fraction(-2, 1),
    Fraction(-1, 1),
    Fraction(-1, 2),
    Fraction(1, 2),
    Fraction(2, 1),
    Fraction(3, 1),
)


def _canonical_key_and_validity(node: ASTNode) -> tuple[bool, str]:
    """Return whether node is algebraically non-trivial, and its canonical monic SymPy key."""
    try:
        expr = node.to_sympy()
        if expr.is_number:
            return False, str(expr)
        cancelled = sp.cancel(expr)
        if cancelled.has(sp.sin, sp.cos):
            try:
                cancelled = sp.trigsimp(cancelled)
            except Exception:
                pass
        if cancelled.is_number:
            return False, str(cancelled)
        _coeff, monic = cancelled.as_coeff_Mul()
        if monic.is_number:
            return False, str(monic)
        return True, str(monic)
    except (TypeError, ValueError, ArithmeticError, sp.SympifyError):
        return True, node.to_str()


def generate_dimensionally_typed_ast(
    variables: Mapping[str, Dimension],
    operators: Sequence[str] = ("+", "-", "*", "/", "pow", "exp", "log", "sin", "cos"),
    max_depth: int = 2,
    max_complexity: int = 6,
    allowed_powers: Sequence[Fraction] = DEFAULT_POWERS,
) -> list[ASTNode]:
    """Generate dimensionally typed AST expressions iteratively.

    Enforces physical dimensional type checking at instantiation to prune
    unphysical expressions during construction, and canonical algebraic
    deduplication to prevent spurious collinear multi-hypothesis copies.
    """
    ops_set = set(operators)
    leaves: list[ASTNode] = [VariableNode(name, dim) for name, dim in variables.items()]

    canonical_map: dict[str, ASTNode] = {}
    for leaf in leaves:
        is_valid, c_key = _canonical_key_and_validity(leaf)
        if is_valid:
            canonical_map[c_key] = leaf

    current_frontier: list[ASTNode] = list(canonical_map.values())

    def try_register(candidate: ASTNode, next_f: list[ASTNode]) -> None:
        if candidate.complexity > max_complexity:
            return
        is_valid, c_key = _canonical_key_and_validity(candidate)
        if not is_valid:
            return
        existing = canonical_map.get(c_key)
        if existing is None:
            canonical_map[c_key] = candidate
            next_f.append(candidate)
        elif candidate.complexity < existing.complexity:
            canonical_map[c_key] = candidate
            next_f.append(candidate)

    unary_candidates = [op for op in ("exp", "log", "sin", "cos") if op in ops_set]

    for _ in range(max_depth):
        next_frontier: list[ASTNode] = []

        # 1. Unary operators (exp, log, sin, cos) on dimensionless nodes
        if unary_candidates:
            for node in current_frontier:
                if node.dimension.is_dimensionless():
                    for uop in unary_candidates:
                        if not (
                            isinstance(node, UnaryOpNode)
                            and node.op in ("exp", "log", "sin", "cos")
                        ):
                            try:
                                un = UnaryOpNode(uop, node)
                                try_register(un, next_frontier)
                            except DimensionalError:
                                pass

        # 2. Pow / Sqrt operator with constant rational exponents
        if "pow" in ops_set or "sqrt" in ops_set:
            powers_to_use = list(allowed_powers) if "pow" in ops_set else [Fraction(1, 2)]
            if "sqrt" in ops_set and Fraction(1, 2) not in powers_to_use:
                powers_to_use.append(Fraction(1, 2))
            for node in current_frontier:
                if not isinstance(node, (PowNode, ConstantNode)):
                    for p in powers_to_use:
                        if p in (Fraction(1, 1), Fraction(0, 1)):
                            continue
                        try:
                            pn = PowNode(node, p)
                            try_register(pn, next_frontier)
                        except (DimensionalError, TypeError):
                            pass

        # 3. Binary operators (+, -, *, /)
        candidate_pool = list(canonical_map.values())
        for n1 in current_frontier:
            for n2 in candidate_pool:
                if n1.complexity + n2.complexity + 1 > max_complexity:
                    continue

                # Multiplication: commutative deduplication
                if "*" in ops_set:
                    if n1.to_str() <= n2.to_str() and n1.to_str() != n2.to_str():
                        try:
                            bn = BinaryOpNode("*", n1, n2)
                            try_register(bn, next_frontier)
                        except DimensionalError:
                            pass

                # Division
                if "/" in ops_set:
                    if n1.to_str() != n2.to_str():
                        try:
                            bn = BinaryOpNode("/", n1, n2)
                            try_register(bn, next_frontier)
                        except DimensionalError:
                            pass

                # Addition: requires identical dimensions [u] == [v]
                if "+" in ops_set:
                    if n1.dimension == n2.dimension and n1.to_str() < n2.to_str():
                        try:
                            bn = BinaryOpNode("+", n1, n2)
                            try_register(bn, next_frontier)
                        except DimensionalError:
                            pass

                # Subtraction: requires identical dimensions [u] == [v]
                if "-" in ops_set:
                    if n1.dimension == n2.dimension and n1.to_str() != n2.to_str():
                        try:
                            bn = BinaryOpNode("-", n1, n2)
                            try_register(bn, next_frontier)
                        except DimensionalError:
                            pass

        current_frontier = next_frontier
        if not current_frontier:
            break

    return list(canonical_map.values())
