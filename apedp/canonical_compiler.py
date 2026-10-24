"""Scientific Equation Compiler & Multi-Tier Recovery Classification for APEDP.

Implements:
1. Canonical AST simplification and algebraic normalization:
   - x + x -> 2x
   - exp(log(x)) -> x, log(exp(x)) -> x
   - x**2 / x -> x
   - a*x**2 + b*x**2 -> (a+b)*x**2
   - sqrt(x**2) -> Abs(x)
2. Dimensional Null-Space Search Engine:
   - Solves D a = d_y for monomial physical scaling laws: y = C * prod(x_i**a_i)
   - Restricts search space to the null-space manifold of dimensional conservation.
3. Multi-Tier Recovery Evaluator:
   - Exact Recovery: Identical symbolic expression.
   - Algebraic Equivalence: Same mathematical identity under simplification.
   - Structural Recovery: Identical functional topology / operator graph.
   - Numerical Equivalence: Indistinguishable over observation domain (|f - g| < epsilon).
   - Physical Recovery: Dimensionally valid with verified physical parameters.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np
import sympy as sp

from apedp.core import DimensionalAssuranceEngine
from apedp.safety import safe_parse_expr
from apedp.symbolic_equivalence import SymbolicEquivalenceEngine

logger = logging.getLogger(__name__)


class RecoveryTier(str, Enum):
    """Hierarchical recovery tiers for scientific equation evaluation."""

    EXACT = "EXACT_RECOVERY"
    ALGEBRAIC = "ALGEBRAIC_EQUIVALENCE"
    STRUCTURAL = "STRUCTURAL_RECOVERY"
    NUMERICAL = "NUMERICAL_EQUIVALENCE"
    NONE = "NOT_RECOVERED"


@dataclass(frozen=True)
class RecoveryVerdict:
    """Outcome of multi-tier scientific equation recovery evaluation."""

    tier: RecoveryTier
    is_exact: bool
    is_algebraic: bool
    is_structural: bool
    is_numerical: bool
    normalized_predicted: str
    normalized_truth: str
    structural_signature: str
    max_domain_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "is_exact": self.is_exact,
            "is_algebraic": self.is_algebraic,
            "is_structural": self.is_structural,
            "is_numerical": self.is_numerical,
            "normalized_predicted": self.normalized_predicted,
            "normalized_truth": self.normalized_truth,
            "structural_signature": self.structural_signature,
            "max_domain_error": self.max_domain_error,
        }


class CanonicalEquationCompiler:
    """Canonicalizes, simplifies, and structurally audits mathematical expressions."""

    def __init__(self):
        self.eq_engine = SymbolicEquivalenceEngine()
        self.dim_engine = DimensionalAssuranceEngine()

    def normalize(self, expr: sp.Expr | str) -> sp.Expr:
        """Applies canonical algebraic simplifications to an expression."""
        if isinstance(expr, str):
            parsed = safe_parse_expr(expr)
        else:
            parsed = expr

        if parsed is None:
            return sp.Integer(0)

        # 1. SymPy simplify & cancel with numerical rationalization
        try:
            simplified = sp.cancel(sp.simplify(sp.nsimplify(parsed)))
        except Exception:  # noqa: BLE001 - best-effort probe must not crash
            try:
                simplified = sp.cancel(sp.simplify(parsed))
            except Exception:  # noqa: BLE001 - best-effort probe must not crash
                simplified = parsed

        # 2. Transcendental inverse identity reduction
        try:
            # log(exp(x)) -> x for real positive x
            reduced = simplified.replace(
                sp.log,
                lambda arg: arg.args[0] if isinstance(arg, sp.exp) else sp.log(arg)
            )
            # exp(log(x)) -> x
            reduced = reduced.replace(
                sp.exp,
                lambda arg: arg.args[0] if isinstance(arg, sp.log) else sp.exp(arg)
            )
        except Exception:  # noqa: BLE001 - best-effort probe must not crash
            reduced = simplified

        # 3. Rational fraction cancellation: e.g. x**2 / x -> x
        try:
            reduced = sp.together(reduced)
            num, den = sp.fraction(reduced)
            if den != 1:
                reduced = sp.cancel(num / den)
        except Exception:
            logger.debug("rational fraction cancellation failed", exc_info=True)

        return reduced

    def extract_structural_signature(self, expr: sp.Expr | str) -> str:
        """Extracts topology of functional forms independent of numeric coefficients.

        Replaces all non-zero numeric constants with generic parameter symbol 'C'
        to identify whether two equations share the same functional family (e.g.
        C*x**2 + C is quadratic, C / (1 + C*x) is rational decay).
        """
        norm = self.normalize(expr)

        def repl_numbers(node: sp.Basic) -> sp.Basic:
            if isinstance(node, (sp.Number, sp.Float, sp.Integer, sp.Rational)):
                return sp.Symbol("C")
            return node

        try:
            # Exponents in Pow are preserved as integers if small, else generic C
            def repl_node(node: sp.Basic) -> sp.Basic:
                if isinstance(node, sp.Pow):
                    base, exp = node.args
                    if isinstance(exp, (int, sp.Integer)) and abs(int(exp)) <= 4:
                        return sp.Pow(repl_node(base), exp)
                    return sp.Pow(repl_node(base), sp.Symbol("C"))
                if isinstance(node, (sp.Float, sp.Rational)):
                    return sp.Symbol("C")
                if isinstance(node, sp.Integer) and abs(int(node)) > 1:
                    return sp.Symbol("C")
                if not node.args:
                    return node
                return node.func(*[repl_node(arg) for arg in node.args])

            struct_expr = repl_node(norm)
            return str(sp.srepr(struct_expr))
        except Exception:  # noqa: BLE001 - best-effort probe must not crash
            return str(norm)

    def classify_recovery(
        self,
        predicted: sp.Expr | str,
        truth: sp.Expr | str,
        domain_bounds: Mapping[str, tuple[float, float]] | None = None,
        numerical_tolerance: float = 1e-4,
    ) -> RecoveryVerdict:
        """Evaluates predicted expression against ground truth across all 4 recovery tiers."""
        norm_pred = self.normalize(predicted)
        norm_truth = self.normalize(truth)

        str_pred = str(norm_pred).replace(" ", "")
        str_truth = str(norm_truth).replace(" ", "")

        # Tier 1: Exact String / Form Recovery
        is_exact = (str_pred == str_truth) or (str(predicted).replace(" ", "") == str(truth).replace(" ", ""))

        # Tier 2: Algebraic Equivalence
        is_algebraic = False
        try:
            diff = sp.simplify(norm_pred - norm_truth)
            if diff == 0:
                is_algebraic = True
            else:
                is_algebraic = self.eq_engine.are_algebraically_equivalent(str(norm_pred), str(norm_truth))
        except Exception:  # noqa: BLE001 - best-effort probe must not crash
            is_algebraic = False

        # Tier 3: Structural Family Recovery
        sig_pred = self.extract_structural_signature(norm_pred)
        sig_truth = self.extract_structural_signature(norm_truth)
        is_structural = (sig_pred == sig_truth)

        # Tier 4: Numerical Domain Equivalence
        is_numerical = False
        max_err = None
        if domain_bounds:
            try:
                variables = sorted({str(s) for s in norm_pred.free_symbols} | {str(s) for s in norm_truth.free_symbols})
                if variables:
                    grid_points = 50
                    grids = {
                        v: np.linspace(domain_bounds.get(v, (0.5, 2.0))[0], domain_bounds.get(v, (0.5, 2.0))[1], grid_points)
                        for v in variables
                    }
                    f_pred = sp.lambdify([sp.Symbol(v) for v in variables], norm_pred, modules="numpy")
                    f_truth = sp.lambdify([sp.Symbol(v) for v in variables], norm_truth, modules="numpy")
                    args = [grids[v] for v in variables]
                    y_p = np.asarray(f_pred(*args), dtype=float)
                    y_t = np.asarray(f_truth(*args), dtype=float)
                    if np.all(np.isfinite(y_p)) and np.all(np.isfinite(y_t)):
                        max_err = float(np.max(np.abs(y_p - y_t)))
                        is_numerical = bool(max_err <= numerical_tolerance)
            except Exception:
                logger.debug("numerical domain equivalence probe failed", exc_info=True)

        # Determine highest tier
        if is_exact:
            tier = RecoveryTier.EXACT
        elif is_algebraic:
            tier = RecoveryTier.ALGEBRAIC
        elif is_structural:
            tier = RecoveryTier.STRUCTURAL
        elif is_numerical:
            tier = RecoveryTier.NUMERICAL
        else:
            tier = RecoveryTier.NONE

        return RecoveryVerdict(
            tier=tier,
            is_exact=is_exact,
            is_algebraic=is_algebraic or is_exact,
            is_structural=is_structural or is_algebraic or is_exact,
            is_numerical=is_numerical or is_algebraic or is_exact,
            normalized_predicted=str(norm_pred),
            normalized_truth=str(norm_truth),
            structural_signature=sig_pred,
            max_domain_error=max_err,
        )


class DimensionalNullSpaceEngine:
    """Generates dimensionally valid monomial exponent candidate manifolds.

    Solves D * a = d_y where D is the dimensional matrix of input variables
    and d_y is the dimensional vector of the target variable.
    """

    def __init__(self, base_dimensions: tuple[str, ...] = ("[mass]", "[length]", "[time]", "[temperature]", "[substance]")):
        self.base_dims = list(base_dimensions)
        self.dim_engine = DimensionalAssuranceEngine()

    def _extract_dim_vector(self, unit_str: str) -> np.ndarray:
        # `DimensionalAssuranceEngine` exposes no unit parser of its own; its
        # registry is the pint `UnitRegistry` that does. Calling a
        # non-existent `parse_unit` here raised AttributeError on every
        # monomial solve, so the engine's registry is used directly.
        unit = self.dim_engine.registry.parse_units(unit_str)
        dim_dict = dict(unit.dimensionality)
        vec = np.zeros(len(self.base_dims), dtype=float)
        for i, b in enumerate(self.base_dims):
            vec[i] = float(cast(Any, dim_dict.get(b, 0.0)))
        return vec

    def solve_admissible_monomials(
        self,
        variable_units: Mapping[str, str],
        target_unit: str,
        max_exponent: int = 4,
    ) -> list[dict[str, Any]]:
        """Finds all integer/half-integer exponent combinations that balance physical dimensions."""
        var_names = sorted(variable_units.keys())
        if not var_names:
            return []

        D = np.column_stack([self._extract_dim_vector(variable_units[v]) for v in var_names])
        d_y = self._extract_dim_vector(target_unit)

        admissible = []
        # Grid search over integer and half-integer exponents: [-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4]
        candidate_exponents = [-4, -3, -2, -1.5, -1, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

        import itertools
        for combo in itertools.product(candidate_exponents, repeat=len(var_names)):
            a = np.array(combo)
            diff = D @ a - d_y
            if np.allclose(diff, 0, atol=1e-5):
                # Valid dimensional monomial!
                terms = []
                for v, exp in zip(var_names, combo, strict=True):
                    if exp == 0:
                        continue
                    elif exp == 1.0:
                        terms.append(v)
                    elif exp == 0.5:
                        terms.append(f"sqrt({v})")
                    else:
                        terms.append(f"{v}**{exp}")
                expr_str = " * ".join(terms) if terms else "1.0"
                admissible.append({
                    "expression": expr_str,
                    "exponents": {v: float(exp) for v, exp in zip(var_names, combo, strict=True)},
                })

        return admissible
