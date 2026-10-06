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
from apedp_dyn.discovery_l2.fingerprint import (
    FingerprintDeduplicator,
    make_bipolar_probe_grid,
    make_probe_grid,
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
    """Unary operator node ('exp', 'log', 'sin', 'cos', 'abs').

    Transcendentals (exp/log/sin/cos) strictly require dimensionless
    arguments. 'abs' is dimensionally transparent (|x| keeps x's
    dimension) and therefore takes any dimension -- it exists so that
    sign-preserving physics (quadratic drag v*|v|, Coulomb friction)
    is expressible at all. Nesting abs(abs(x)) is pointless but
    harmless (idempotent); generation bounds depth regardless.
    """

    def __init__(self, op: str, child: ASTNode) -> None:
        if op not in ("exp", "log", "sin", "cos", "abs", "sat", "inv_quad"):
            raise ValueError(f"Unsupported unary operator: '{op}'")
        if op == "abs":
            self.dimension = child.dimension
        else:
            if not child.dimension.is_dimensionless():
                raise DimensionalError(
                    f"Operator '{op}' requires strictly dimensionless input [0, 0, ...], "
                    f"got [{child.dimension}]"
                )
            self.dimension = DIMENSIONLESS
        self.op = op
        self.child = child
        self.complexity = child.complexity + 1

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        cv = self.child.evaluate(env)
        if self.op == "abs":
            return np.abs(cv)
        if self.op == "sat":
            with np.errstate(all="ignore"):
                return cv / (1.0 + np.abs(cv))
        if self.op == "inv_quad":
            with np.errstate(all="ignore"):
                return 1.0 / (1.0 + cv**2)
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
        if self.op == "sat":
            return f"sat({self.child.to_str()})"
        if self.op == "inv_quad":
            return f"inv_quad({self.child.to_str()})"
        return f"{self.op}({self.child.to_str()})"

    def to_sympy(self) -> sp.Expr:
        cs = self.child.to_sympy()
        if self.op == "abs":
            return sp.Abs(cs)
        if self.op == "sat":
            return cs / (1 + sp.Abs(cs))
        if self.op == "inv_quad":
            return 1 / (1 + cs**2)
        if self.op == "exp":
            return sp.exp(cs)
        if self.op == "log":
            return sp.log(cs)
        if self.op == "sin":
            return sp.sin(cs)
        if self.op == "cos":
            return sp.cos(cs)
        raise ValueError(f"Unknown operator {self.op}")



# Frozen causal lookback for rolling operators (matches the validated
# volume-surprise experiment: trailing-40 bars; do not tune per search).
ROLL_WINDOW = 40

# Minimum history for the expanding z-score, matching the frozen H-A
# construction (grammar_space.MIN_HISTORY); distinct from ROLL_WINDOW
# because an expanding window is unbounded but must still warm up.
MIN_ZSCORE_HISTORY = 60

# Ops whose value is normalized (unitless), so any input dimension is legal.
NORMALIZING_ROLL_OPS = ("rollrank", "rollzscore", "expanding_zscore")


class RollingOpNode(ASTNode):
    """Causal rolling operator ('rollmean', 'rollstd', 'rollrank', 'rollzscore',
    'expanding_zscore').

    Dimensionally transparent for moments (rollmean/rollstd keep the child's
    dimension); normalizing ops (rollrank/rollzscore/expanding_zscore) output
    DIMENSIONLESS from any input dimension -- this is what makes the
    volume-family forms (H-RV rank, H-A z-scored components) generatable.
    Strictly causal: output[t] uses child[t-W+1 .. t] for fixed-window ops and
    child[0 .. t] for expanding_zscore; warm-up positions are NaN, so searches
    must tolerate leading-NaN predictions exactly as for log-guards.
    Only one rolling application per branch (no roll-of-roll): nesting adds
    smoothing without new information and explodes the search.

    expanding_zscore carries window=MIN_ZSCORE_HISTORY as its minimum-history
    requirement (not a lookback): it is NaN until that many finite values are
    available, then standardizes against the whole causal history.
    """

    def __init__(self, op: str, child: ASTNode, window: int = ROLL_WINDOW) -> None:
        if op not in ("rollmean", "rollstd", "rollrank", "rollzscore", "expanding_zscore"):
            raise ValueError(f"Unsupported rolling operator: '{op}'")
        if isinstance(child, RollingOpNode):
            raise ValueError("rolling operators do not nest (roll-of-roll rejected)")
        if window < 2:
            raise ValueError(f"rolling window must be >= 2, got {window}")
        self.op = op
        self.child = child
        self.window = window
        self.dimension = DIMENSIONLESS if op in NORMALIZING_ROLL_OPS else child.dimension
        self.complexity = child.complexity + 1

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        cv = np.asarray(self.child.evaluate(env), dtype=float)
        w = self.window
        out = np.full_like(cv, np.nan)
        if len(cv) >= w:
            c = np.concatenate([[0.0], np.nancumsum(np.where(np.isfinite(cv), cv, 0.0))])
            n = np.concatenate([[0], np.cumsum(np.isfinite(cv).astype(float))])
            tot = c[w:] - c[:-w]
            cnt = n[w:] - n[:-w]
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = tot / np.where(cnt > 0, cnt, np.nan)
            if self.op == "rollmean":
                out[w - 1:] = mean
            elif self.op == "rollrank":
                for t in range(w - 1, len(cv)):
                    win = cv[t - w + 1:t + 1]
                    if np.isfinite(win).all() and len(win) == w:
                        out[t] = float(np.sum(win <= cv[t]) - 1) / (w - 1)
            elif self.op == "rollzscore":
                c2 = np.concatenate([[0.0], np.nancumsum(np.where(np.isfinite(cv), cv * cv, 0.0))])
                tot2 = c2[w:] - c2[:-w]
                with np.errstate(invalid="ignore", divide="ignore"):
                    var = tot2 / np.where(cnt > 0, cnt, np.nan) - mean * mean
                sd = np.sqrt(np.where(var > 0, var, np.nan))
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[w - 1:] = (cv[w - 1:] - mean) / np.where(sd > 0, sd, np.nan)
            elif self.op == "expanding_zscore":
                # Causal expanding standardization with a minimum-history
                # warm-up (window carries MIN_ZSCORE_HISTORY, not a lookback).
                # ddof=1 matches pandas .expanding().std() used by the frozen
                # grammar_space H-A construction, so generated forms stay
                # comparable with the established panel.
                fin = np.isfinite(cv)
                cnt_e = np.cumsum(fin.astype(float))
                # [1:] aligns element i to the inclusive sum through cv[i];
                # the prepended 0.0 otherwise shifts the window by one.
                cs = np.concatenate([[0.0], np.nancumsum(np.where(fin, cv, 0.0))])[1:]
                cs2 = np.concatenate([[0.0], np.nancumsum(np.where(fin, cv * cv, 0.0))])[1:]
                k = w
                idx = np.arange(len(cv))
                ready = (idx + 1 >= k) & (cnt_e >= k)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mu_e = cs / np.where(cnt_e > 0, cnt_e, np.nan)
                    var_e = ((cs2 - cnt_e * mu_e**2) / np.where(cnt_e > 1, cnt_e - 1, np.nan))
                    sd_e = np.sqrt(np.where(var_e > 0, var_e, np.nan))
                    z_e = (cv - mu_e) / np.where(sd_e > 0, sd_e, np.nan)
                out[ready] = z_e[ready]
            else:
                c2 = np.concatenate([[0.0], np.nancumsum(np.where(np.isfinite(cv), cv * cv, 0.0))])
                tot2 = c2[w:] - c2[:-w]
                with np.errstate(invalid="ignore", divide="ignore"):
                    var = tot2 / np.where(cnt > 0, cnt, np.nan) - mean * mean
                out[w - 1:] = np.sqrt(np.where(var > 0, var, np.nan))
        return out

    def to_str(self) -> str:
        if self.op == "expanding_zscore":
            return f"expanding_zscore{self.window}({self.child.to_str()})"
        return f"{self.op}{self.window}({self.child.to_str()})"

    def to_sympy(self) -> sp.Expr:
        cs = self.child.to_sympy()
        names = {"rollmean": "RollMean", "rollstd": "RollStd",
                 "rollrank": "RollRank", "rollzscore": "RollZscore",
                 "expanding_zscore": "ExpandingZscore"}
        return sp.Function(names[self.op])(cs, sp.Integer(self.window))


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
            except (TypeError, ValueError, ArithmeticError, sp.SympifyError):
                pass
        if cancelled.is_number:
            return False, str(cancelled)
        _coeff, monic = cancelled.as_coeff_Mul()
        if monic.is_number:
            return False, str(monic)
        return True, str(monic)
    except (TypeError, ValueError, ArithmeticError, sp.SympifyError):
        return True, node.to_str()


def _has_transcendental(n: ASTNode) -> bool:
    if isinstance(n, UnaryOpNode):
        return True
    if isinstance(n, BinaryOpNode):
        return _has_transcendental(n.left) or _has_transcendental(n.right)
    if isinstance(n, PowNode):
        return _has_transcendental(n.base)
    if isinstance(n, RollingOpNode):
        return _has_transcendental(n.child)
    return False


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

    # Rolling operators need probes longer than ROLL_WINDOW: with the default
    # 32-point grid every rolling response is all-NaN and dedup rejects it.
    # expanding_zscore needs its full minimum history to be finite at all.
    n_probe = 32
    if {"rollmean", "rollstd", "rollrank", "rollzscore"} & ops_set:
        n_probe = max(n_probe, ROLL_WINDOW + 8)
    if "expanding_zscore" in ops_set:
        n_probe = max(n_probe, MIN_ZSCORE_HISTORY + 8)
    probe_grid = make_probe_grid(variables, n_points=n_probe)
    bipolar_grid = make_bipolar_probe_grid(variables, n_points=n_probe)
    dedup = FingerprintDeduplicator(probe_grid, bipolar_grid)

    canonical_map: dict[str, ASTNode] = {}
    for leaf in leaves:
        dedup.is_novel(leaf)
        is_valid, c_key = _canonical_key_and_validity(leaf)
        if is_valid:
            canonical_map[c_key] = leaf

    current_frontier: list[ASTNode] = list(canonical_map.values())

    def try_register(candidate: ASTNode, next_f: list[ASTNode]) -> None:
        if candidate.complexity > max_complexity:
            return
        if not dedup.is_novel(candidate):
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

    unary_candidates = [op for op in ("exp", "log", "sin", "cos", "sat", "inv_quad") if op in ops_set]

    for _ in range(max_depth):
        next_frontier: list[ASTNode] = []

        # 1a. Unary abs operator (dimensionally transparent, e.g. for drag |v|v)
        if "abs" in ops_set:
            for node in current_frontier:
                if not (isinstance(node, UnaryOpNode) and node.op == "abs"):
                    try:
                        an = UnaryOpNode("abs", node)
                        try_register(an, next_frontier)
                    except DimensionalError:
                        pass

        # 1b. Unary transcendental operators (exp, log, sin, cos) on dimensionless nodes
        if unary_candidates:
            for node in current_frontier:
                if node.dimension.is_dimensionless() and not _has_transcendental(node):
                    for uop in unary_candidates:
                        try:
                            un = UnaryOpNode(uop, node)
                            try_register(un, next_frontier)
                        except DimensionalError:
                            pass

        # 1c. Causal rolling operators (rollmean, rollstd, rollrank, rollzscore)
        # on non-rolling nodes. Restricted to leaves and depth-1 transforms
        # (complexity <= 2): rolling is a *feature constructor* (validated use:
        # rolling moments/ranks of raw/transformed series, including the
        # volume-family H-RV/H-A forms); rolling compound expressions adds
        # smoothing without information and explodes the frontier.
        _roll_ops = (("rollmean", ROLL_WINDOW), ("rollstd", ROLL_WINDOW),
                     ("rollrank", ROLL_WINDOW), ("rollzscore", ROLL_WINDOW),
                     ("expanding_zscore", MIN_ZSCORE_HISTORY))
        if {name for name, _ in _roll_ops} & ops_set:
            for node in current_frontier:
                if isinstance(node, RollingOpNode) or node.complexity > 2:
                    continue
                for rop, rop_window in _roll_ops:
                    if rop not in ops_set:
                        continue
                    try:
                        rn = RollingOpNode(rop, node, rop_window)
                        try_register(rn, next_frontier)
                    except (DimensionalError, ValueError):
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
        pairs = [
            (n1, n2)
            for n1 in current_frontier
            for n2 in candidate_pool
            if n1.complexity + n2.complexity + 1 <= max_complexity
        ]
        pairs.sort(key=lambda p: p[0].complexity + p[1].complexity)

        for n1, n2 in pairs:
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
