"""Dimensionally typed context-free production grammar for symbolic expression generation.

Guarantees physical dimensional homogeneity by construction:
- No illegal additions (e.g. mass + velocity) can ever be generated.
- Arguments to transcendental functions (exp, log, sin) are guaranteed dimensionless.
- Subtree replacements during mutation preserve the exact dimensionality of the replaced node.
- Dimensional basis synthesis solves for exact physical monomials when decomposing.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import pint
import sympy as sp
from pint.facets.plain.unit import PlainUnit

from apedp.core import DimensionalAssuranceEngine

logger = logging.getLogger(__name__)

_PROBE_ERRORS = (
    sp.SympifyError,
    TypeError,
    ValueError,
    ArithmeticError,
    ZeroDivisionError,
    OverflowError,
)

# Standard SI base dimension keys in canonical order:
# [Mass (M), Length (L), Time (T), Temperature (Theta), Substance (N), Current (I), Luminosity (J)]
SI_BASE_KEYS = ("mass", "length", "time", "temperature", "substance", "current", "luminosity")


@dataclass(frozen=True)
class DimVector:
    exponents: tuple[int, ...]  # Length 7

    ZERO: ClassVar[DimVector]

    def __post_init__(self):
        if len(self.exponents) != 7:
            raise ValueError(f"DimVector must have 7 components, got {len(self.exponents)}")

    def __add__(self, other: DimVector) -> DimVector:
        return DimVector(tuple(a + b for a, b in zip(self.exponents, other.exponents, strict=True)))

    def __sub__(self, other: DimVector) -> DimVector:
        return DimVector(tuple(a - b for a, b in zip(self.exponents, other.exponents, strict=True)))

    def __neg__(self) -> DimVector:
        return DimVector(tuple(-a for a in self.exponents))

    def __mul__(self, scalar: int) -> DimVector:
        return DimVector(tuple(a * scalar for a in self.exponents))

    @property
    def is_dimensionless(self) -> bool:
        return all(x == 0 for x in self.exponents)

    def to_pint_str(self) -> str:
        """Convert DimVector to a valid Pint unit expression string."""
        units = ["kilogram", "meter", "second", "kelvin", "mole", "ampere", "candela"]
        parts = []
        for exp, u in zip(self.exponents, units, strict=True):
            if exp == 1:
                parts.append(u)
            elif exp != 0:
                parts.append(f"{u}**{exp}")
        return " * ".join(parts) if parts else "dimensionless"

    @classmethod
    def from_pint(cls, unit: pint.Unit | pint.Quantity | PlainUnit) -> DimVector:
        """Convert a Pint unit/quantity into a 7-element canonical DimVector."""
        dims = unit.dimensionality
        exp_list = []
        for key in SI_BASE_KEYS:
            pint_key = f"[{key}]"
            exp_list.append(int(cast(Any, dims).get(pint_key, 0)))
        return cls(tuple(exp_list))


DimVector.ZERO = DimVector((0, 0, 0, 0, 0, 0, 0))


@dataclass(frozen=True)
class TypedTerminal:
    symbol: sp.Symbol
    dim: DimVector
    name: str


class TypedProductionGrammar:
    """Grammar engine generating dimensionally valid symbolic expression trees."""

    def __init__(self, variable_units: Mapping[str, str | pint.Unit]):
        self.engine = DimensionalAssuranceEngine()
        self.terminals: list[TypedTerminal] = []
        self._dim_to_terminals: dict[DimVector, list[TypedTerminal]] = {}

        for var_name, unit_val in variable_units.items():
            sym = self.engine.register_variable(var_name, unit_val)
            unit_obj = self.engine._get_unit(sym)
            dim_vec = DimVector.from_pint(unit_obj)
            term = TypedTerminal(symbol=sym, dim=dim_vec, name=var_name)
            self.terminals.append(term)
            self._dim_to_terminals.setdefault(dim_vec, []).append(term)
        self._param_counter = 0

    def _new_param(self, unit_dim: DimVector | None = None) -> sp.Symbol:
        """Create a scalar parameter with specified or dimensionless unit."""
        self._param_counter += 1
        name = f"p_{self._param_counter}"
        unit_str = unit_dim.to_pint_str() if unit_dim is not None else "dimensionless"
        sym = self.engine.register_variable(name, unit_str)
        return sym

    def _synthesize_monomial(self, target_dim: DimVector, rng: random.Random) -> sp.Expr:
        """Find an exact monomial product/ratio of terminals matching target_dim using linear algebra."""
        matching = self._dim_to_terminals.get(target_dim)
        if matching:
            return rng.choice(matching).symbol

        if target_dim.is_dimensionless:
            return self._new_param()

        if not self.terminals:
            return self._new_param(target_dim)

        # Solve M * c = target_dim
        cols = [list(t.dim.exponents) for t in self.terminals]
        M = sp.Matrix(cols).T  # 7 x len(terminals)
        b = sp.Matrix(list(target_dim.exponents))

        try:
            sol, free_vars = M.gauss_jordan_solve(b)
            subs_map = {fv: rng.choice([-1, 0, 1]) for fv in free_vars}
            eval_sol = sol.subs(subs_map)

            monomial = sp.S.One
            has_term = False
            for i, term in enumerate(self.terminals):
                power = eval_sol[i]
                if power.is_integer:
                    p_int = int(power)
                    if p_int != 0:
                        monomial = monomial * (term.symbol ** p_int)
                        has_term = True
                else:
                    return self._new_param(target_dim)

            if has_term:
                unit = self.engine.check_expression_consistency(monomial)
                if DimVector.from_pint(unit) == target_dim:
                    return monomial
        except _PROBE_ERRORS as exc:
            logger.debug("Monomial basis synthesis fallback: %s", exc)

        return self._new_param(target_dim)

    def sample_expression(
        self,
        target_dim: DimVector,
        depth: int,
        rng: random.Random,
        allow_constants: bool = True,
    ) -> sp.Expr:
        """Recursively sample an expression guaranteed to have dimensionality `target_dim`."""
        matching = self._dim_to_terminals.get(target_dim)
        if depth <= 0 or (rng.random() < 0.35 and matching):
            if allow_constants and rng.random() < 0.4:
                c = self._new_param()
                base_expr = self._synthesize_monomial(target_dim, rng)
                return c * base_expr
            return self._synthesize_monomial(target_dim, rng)

        options = ["add", "mul"]
        if target_dim.is_dimensionless:
            options.extend(["transcendental", "power"])

        op = rng.choice(options)

        if op == "add":
            left = self.sample_expression(target_dim, depth - 1, rng, allow_constants=False)
            right = self.sample_expression(target_dim, depth - 1, rng, allow_constants=False)
            res = left + right if rng.random() < 0.5 else left - right
            if res == 0:
                res = left + right
            return res

        if op == "mul":
            t = rng.choice(self.terminals) if self.terminals else None
            d1 = t.dim if t else DimVector.ZERO
            d2 = target_dim - d1
            left = self.sample_expression(d1, depth - 1, rng, allow_constants=False)
            right = self.sample_expression(d2, depth - 1, rng, allow_constants=False)
            return left * right

        if op == "transcendental" and target_dim.is_dimensionless:
            arg = self.sample_expression(DimVector.ZERO, depth - 1, rng, allow_constants=False)
            func_choice = rng.choice(["exp", "log", "sin", "cos"])
            if func_choice == "exp":
                return sp.exp(arg)
            if func_choice == "log":
                return sp.log(sp.Abs(arg) + 1)
            if func_choice == "sin":
                return sp.sin(arg)
            return sp.cos(arg)

        return self._synthesize_monomial(target_dim, rng)

    def mutate_expression(
        self, expr: sp.Expr, rng: random.Random, max_depth: int = 2
    ) -> sp.Expr:
        """Mutate a subtree while preserving exact dimensional consistency."""
        try:
            expr_unit = self.engine.check_expression_consistency(expr)
            root_dim = DimVector.from_pint(expr_unit)
        except _PROBE_ERRORS:
            return self.sample_expression(DimVector.ZERO, max_depth, rng)

        subtrees = list(expr.args)
        if not subtrees or rng.random() < 0.25:
            return self.sample_expression(root_dim, max_depth, rng)

        idx = rng.randint(0, len(subtrees) - 1)
        sub = subtrees[idx]
        try:
            sub_unit = self.engine.check_expression_consistency(sub)
            sub_dim = DimVector.from_pint(sub_unit)
            new_sub = self.sample_expression(sub_dim, max_depth - 1, rng)
            new_args = list(subtrees)
            new_args[idx] = new_sub
            if isinstance(expr, sp.Add):
                return sp.Add(*new_args)
            if isinstance(expr, sp.Mul):
                return sp.Mul(*new_args)
            return expr.func(*new_args)
        except _PROBE_ERRORS:
            return expr


__all__ = [
    "SI_BASE_KEYS",
    "DimVector",
    "TypedProductionGrammar",
    "TypedTerminal",
]
