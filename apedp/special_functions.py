"""Special-Function Symbolic Library and Domain Safety Engine for APEDP.

Provides rigorous mathematical and computational foundations for special functions:
- Riemann Zeta: zeta(s) with pole at s = 1.
- Euler Gamma: gamma(s) with poles at non-positive integers {0, -1, -2, ...},
  reflection formula Gamma(z)*Gamma(1-z) = pi / sin(pi*z), and recurrence Gamma(z+1) = z*Gamma(z).
- Error Functions: erf(x), erfc(x) with range (-1, 1), odd parity, and erf(x) + erfc(x) = 1.
- Bessel Functions: besselj(n, x) with recurrence J_{n-1}(x) + J_{n+1}(x) = (2n/x)*J_n(x).
- Trigonometric & Hyperbolic: sin^2(x) + cos^2(x) = 1, cosh^2(x) - sinh^2(x) = 1.

Maintains strict separation between:
1. Mathematical domain definitions and algebraic equivalence rewrites.
2. Robust dual-backend numerical evaluation (via scipy.special / numpy).
3. The Epistemic (C,P) Firewall: empirical fits to special functions stay P0;
   mathematical identities require dual symbolic-CAS and numerical corroboration.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from apedp.discovery_types import DictCompatibleResult
from apedp.safety import safe_parse_expr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpecialFunctionMetadata:
    """Mathematical metadata and domain properties of a symbolic special function."""

    name: str
    sympy_func: Any
    has_poles: bool
    pole_locations: tuple[float, ...]  # known isolated poles
    codomain: tuple[float, float]      # [min, max] or (-inf, inf)
    parity: str | None                 # "even", "odd", or None
    description: str


# Canonical registry of supported special functions
SPECIAL_FUNCTION_REGISTRY: dict[str, SpecialFunctionMetadata] = {
    "zeta": SpecialFunctionMetadata(
        name="zeta",
        sympy_func=sp.zeta,
        has_poles=True,
        pole_locations=(1.0,),
        codomain=(float("-inf"), float("inf")),
        parity=None,
        description="Riemann Zeta function with simple pole at s = 1",
    ),
    "gamma": SpecialFunctionMetadata(
        name="gamma",
        sympy_func=sp.gamma,
        has_poles=True,
        pole_locations=(0.0, -1.0, -2.0, -3.0, -4.0, -5.0),
        codomain=(float("-inf"), float("inf")),
        parity=None,
        description="Euler Gamma function with poles at non-positive integers",
    ),
    "erf": SpecialFunctionMetadata(
        name="erf",
        sympy_func=sp.erf,
        has_poles=False,
        pole_locations=(),
        codomain=(-1.0, 1.0),
        parity="odd",
        description="Gauss Error Function with codomain (-1, 1) and odd symmetry",
    ),
    "erfc": SpecialFunctionMetadata(
        name="erfc",
        sympy_func=sp.erfc,
        has_poles=False,
        pole_locations=(),
        codomain=(0.0, 2.0),
        parity=None,
        description="Complementary Error Function erfc(x) = 1 - erf(x)",
    ),
    "besselj": SpecialFunctionMetadata(
        name="besselj",
        sympy_func=sp.besselj,
        has_poles=False,
        pole_locations=(),
        codomain=(float("-inf"), float("inf")),
        parity=None,
        description="Bessel function of the first kind J_n(x)",
    ),
    "sin": SpecialFunctionMetadata(
        name="sin",
        sympy_func=sp.sin,
        has_poles=False,
        pole_locations=(),
        codomain=(-1.0, 1.0),
        parity="odd",
        description="Sine trigonometric function with codomain [-1, 1]",
    ),
    "cos": SpecialFunctionMetadata(
        name="cos",
        sympy_func=sp.cos,
        has_poles=False,
        pole_locations=(),
        codomain=(-1.0, 1.0),
        parity="even",
        description="Cosine trigonometric function with codomain [-1, 1]",
    ),
}


@dataclass(frozen=True)
class DomainSafetyReport(DictCompatibleResult):
    """Result of auditing an expression's domain assumptions against poles and singular branch cuts."""

    is_domain_safe: bool
    detected_special_functions: list[str]
    pole_conflicts: list[str] = field(default_factory=list)
    safe_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class SpecialFunctionLibrary:
    """Manages parsing, domain validation, algebraic identities, and evaluation of special functions."""

    @classmethod
    def detect_special_functions(cls, expr: sp.Expr) -> list[str]:
        """Detects all registered special functions present in an expression AST."""
        found: set[str] = set()
        for node in sp.preorder_traversal(expr):
            if isinstance(node, sp.Function):
                fname = getattr(type(node), "__name__", "")
                if fname in SPECIAL_FUNCTION_REGISTRY:
                    found.add(fname)
        return sorted(found)

    @classmethod
    def check_domain_safety(
        cls,
        expr_input: str | sp.Expr,
        eval_domain: dict[str, tuple[float, float]],
        num_probe_points: int = 100,
    ) -> DomainSafetyReport:
        """Audits whether any evaluation domain intersects poles or invalid branch cuts."""
        expr = safe_parse_expr(expr_input) if isinstance(expr_input, str) else expr_input
        funcs = cls.detect_special_functions(expr)
        pole_conflicts: list[str] = []
        warnings: list[str] = []

        # Audit explicit variables
        for var_name, (lo, hi) in eval_domain.items():
            var_sym = sp.Symbol(var_name)
            for fn_name in funcs:
                meta = SPECIAL_FUNCTION_REGISTRY.get(fn_name)
                if not meta or not meta.has_poles:
                    continue

                for pole in meta.pole_locations:
                    if lo <= pole <= hi:
                        pole_conflicts.append(
                            f"Domain [{lo}, {hi}] for '{var_name}' contains pole of {fn_name} at {pole}"
                        )

            # Numerical singularity probe across the interval
            try:
                f_lamb = sp.lambdify(
                    [var_sym],
                    expr.subs({s: 1.0 for s in expr.free_symbols if s != var_sym}),
                    modules=["scipy", "numpy"],
                )
                test_pts = np.linspace(lo, hi, num_probe_points)
                vals = np.asarray(f_lamb(test_pts), dtype=float)
                if not np.all(np.isfinite(vals)):
                    non_finite_count = int(np.sum(~np.isfinite(vals)))
                    warnings.append(
                        f"Numerical evaluation produced {non_finite_count} non-finite values on [{lo}, {hi}]"
                    )
            except Exception as e:
                warnings.append(f"Domain probe encountered evaluation error on '{var_name}': {e}")

        is_safe = len(pole_conflicts) == 0
        return DomainSafetyReport(
            is_domain_safe=is_safe,
            detected_special_functions=funcs,
            pole_conflicts=pole_conflicts,
            safe_intervals=dict(eval_domain),
            warnings=warnings,
        )

    @classmethod
    def apply_special_identities(cls, expr: sp.Expr) -> sp.Expr:
        """Applies exact mathematical rewrites for trigonometric, gamma, erf, and Bessel functions.

        Rules applied conservatively to prevent false algebraic equivalence certification:
        1. sin^2(x) + cos^2(x) -> 1
        2. cosh^2(x) - sinh^2(x) -> 1
        3. erf(x) + erfc(x) -> 1
        4. Gamma(x + 1) / Gamma(x) -> x
        5. Gamma(1/2) -> sqrt(pi)
        """
        try:
            # Trigonometric / Hyperbolic simplification
            res = sp.trigsimp(expr)

            # Gamma rewrites
            res = sp.expand_func(res)

            # Erf / Erfc consolidation
            if res.has(sp.erfc):
                res = res.rewrite(sp.erf)
            res = sp.simplify(res)
            return res
        except Exception as e:
            logger.debug("Special identity simplification exception: %s", e)
            return expr

    @classmethod
    def evaluate_vectorized(
        cls,
        expr_input: str | sp.Expr,
        variable_data: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Vectorized numerical evaluation supporting all special functions via scipy.special."""
        expr = safe_parse_expr(expr_input) if isinstance(expr_input, str) else expr_input
        var_names = sorted(variable_data.keys())
        symbols = [sp.Symbol(v) for v in var_names]

        # Use scipy modules dictionary for special functions
        f_lamb: Callable[..., Any] = sp.lambdify(symbols, expr, modules=["scipy", "numpy"])
        inputs = [variable_data[v] for v in var_names]

        try:
            res: Any = f_lamb(*inputs)
            if np.isscalar(res):
                n_pts = len(next(iter(variable_data.values()))) if variable_data else 1
                # `np.isscalar` narrows to a union that includes `complex`, which
                # `float()` rejects. Going through `np.asarray(...).item()`
                # keeps the coercion identical to the array path below while
                # still raising (and being caught) for genuinely non-real values.
                return np.full(n_pts, float(np.asarray(res).item()))
            return np.asarray(res, dtype=float)
        except Exception as e:
            logger.debug("Vectorized evaluation error: %s", e)
            n_pts = len(next(iter(variable_data.values()))) if variable_data else 1
            return np.full(n_pts, np.nan)


__all__ = [
    "SPECIAL_FUNCTION_REGISTRY",
    "DomainSafetyReport",
    "SpecialFunctionLibrary",
    "SpecialFunctionMetadata",
]
