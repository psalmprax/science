"""Symbolic Equivalence and Structural Canonicalization Engine.

Solves the equation deduplication and structural normalization problem:
Two equations can look syntactically distinct but represent identical physical laws
(e.g., log(exp(x)) vs x, a*x^2 + b*x^2 vs c*x^2, or arbitrary coefficient permutations).

Pipeline:
Expression -> Safe Parsing -> Algebraic Simplification -> Parameter Consolidation
          -> Canonical Skeleton -> Structural Family Classification -> Fingerprint
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import numpy as np
import sympy as sp

from apedp.safety import safe_parse_expr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CanonicalForm:
    """Canonical structural representation of an equation or candidate model."""

    raw_expression: str
    simplified_expression: str
    canonical_skeleton: str
    structural_family: str
    fingerprint: str  # SHA-256 hash of canonical skeleton
    complexity: float
    variables: tuple[str, ...]
    parameters: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["variables"] = list(self.variables)
        d["parameters"] = list(self.parameters)
        return d


class SymbolicEquivalenceEngine:
    """Normalizes symbolic expressions and identifies structural family equivalence."""

    STANDARD_VARIABLES: ClassVar[set[str]] = {
        "x", "y", "z", "t", "T", "r", "v", "p", "P", "q", "u", "w",
        "x1", "x2", "x3", "x4", "dx", "dy", "dz", "theta", "phi",
    }

    @classmethod
    def _extract_variables(
        cls,
        expr: sp.Expr,
        known_variables: list[str] | set[str] | None = None,
    ) -> tuple[set[str], set[str]]:
        free_names = {str(s) for s in expr.free_symbols}
        if known_variables is not None:
            var_set = set(known_variables) & free_names
            param_set = free_names - var_set
            return var_set, param_set

        # Heuristic: symbols matching common scientific variables are independent vars
        var_set = {s for s in free_names if s in cls.STANDARD_VARIABLES}
        if not var_set and free_names:
            # If none match standard names, sort alphabetically and treat first as var
            sorted_names = sorted(free_names)
            var_set = {sorted_names[0]}
        param_set = free_names - var_set
        return var_set, param_set

    @classmethod
    def _simplify_identities(cls, expr: sp.Expr) -> sp.Expr:
        """Applies mathematical and algebraic simplifications."""
        try:
            # 1. Expand logarithmic/exponential inverses
            expanded = sp.expand_log(expr, force=True)
            # 2. Rational cancellation and trigonometric simplification
            simplified = sp.cancel(expanded)
            simplified = sp.simplify(simplified)
            return simplified
        except Exception as exc:  # noqa: BLE001 - best-effort probe must not crash
            logger.debug("Simplification exception for %r: %s", expr, exc)
            return expr

    @classmethod
    def _build_canonical_skeleton(
        cls,
        expr: sp.Expr,
        variables: set[str],
        parameters: set[str],
    ) -> tuple[sp.Expr, list[str]]:
        """Lifts numeric constants and parameters into ordered canonical symbols C_0, C_1, ..."""
        {sp.Symbol(v) for v in variables}

        # Step 1: Replace all known parameters and numeric floats with a temporary token
        def abstract_node(node: sp.Expr) -> sp.Expr:
            if isinstance(node, sp.Pow):
                base, exp = node.args
                # Preserve structural integer/rational exponents of variables (e.g. x**2, x**3, 1/T**2)
                if isinstance(exp, (sp.Integer, int)) or (isinstance(exp, sp.Rational) and exp.q in (1, 2)):
                    return sp.Pow(abstract_node(base), exp)
                return sp.Pow(abstract_node(base), abstract_node(exp))
            if isinstance(node, sp.Float):
                return sp.Symbol("__C__")
            if isinstance(node, sp.Rational) and not isinstance(node, sp.Integer):
                return sp.Symbol("__C__")
            if isinstance(node, sp.Symbol) and str(node) in parameters:
                return sp.Symbol("__C__")
            if node.args:
                return node.func(*[abstract_node(arg) for arg in node.args])
            return node

        abstracted = abstract_node(expr)

        # Step 2: Combine like terms (e.g. __C__*x^2 + __C__*x^2 -> __C__*x^2)
        try:
            expanded = sp.expand(abstracted)
            # Collect additive terms
            terms = sp.Add.make_args(expanded)
            unique_variable_parts: dict[sp.Expr, sp.Expr] = {}
            for term in terms:
                # Factor out __C__
                non_c_factors = []
                for factor in sp.Mul.make_args(term):
                    if factor != sp.Symbol("__C__") and factor != -sp.Symbol("__C__"):
                        non_c_factors.append(factor)
                var_core = sp.Mul(*non_c_factors) if non_c_factors else sp.Integer(1)
                unique_variable_parts[var_core] = sp.Symbol("__C__")

            recombined = sp.Add(*[sp.Mul(sp.Symbol("__C__"), core) for core in unique_variable_parts])
        except Exception:  # noqa: BLE001 - sympy internals; falls back to the abstracted form
            logger.debug("Skeleton recombination fallback", exc_info=True)
            recombined = abstracted

        # Step 3: Assign deterministic parameter indices C_0, C_1, C_2 in canonical order
        counter = [0]
        assigned_params: list[str] = []

        def assign_indices(node: sp.Expr) -> sp.Expr:
            if node == sp.Symbol("__C__"):
                c_name = f"C_{counter[0]}"
                counter[0] += 1
                assigned_params.append(c_name)
                return sp.Symbol(c_name)
            if node.args:
                # Sort args deterministically before traversal if Add/Mul
                args = list(node.args)
                if isinstance(node, (sp.Add, sp.Mul)):
                    args = sorted(args, key=lambda a: (len(a.free_symbols), str(a)))
                return node.func(*[assign_indices(arg) for arg in args])
            return node

        skeleton = assign_indices(recombined)
        return skeleton, assigned_params

    @classmethod
    def _classify_family(cls, skeleton: sp.Expr, variables: set[str]) -> str:
        """Classifies the structural equation family."""
        skel_str = str(skeleton)
        var_symbols = [sp.Symbol(v) for v in variables]

        # 1. Check for Shomate-class (mix of T, T^2, T^3, 1/T^2)
        if any(v in ("T", "t") for v in variables):
            has_inv_sq = "/T**2" in skel_str or "*T**(-2)" in skel_str or "/t**2" in skel_str
            has_cube = "T**3" in skel_str or "t**3" in skel_str
            if has_inv_sq and (has_cube or "T**2" in skel_str):
                return "shomate_thermodynamic"

        # 2. Check for Exponential
        if "exp(" in skel_str:
            return "exponential_relaxation"

        # 3. Check for Trigonometric / Oscillatory
        if any(fn in skel_str for fn in ("sin(", "cos(", "tan(")):
            if skel_str.count("cos(") > 1 or skel_str.count("sin(") > 1 or "cos(cos" in skel_str:
                return "nested_trigonometric"
            return "trigonometric_oscillatory"

        # 4. Check for Polynomial
        if var_symbols:
            try:
                poly = sp.Poly(skeleton, var_symbols)
                deg = poly.total_degree()
                if deg == 1:
                    return "linear"
                if deg == 2:
                    return "quadratic"
                if deg == 3:
                    return "cubic"
                return f"polynomial_deg_{deg}"
            except Exception:  # noqa: BLE001 - sympy internals; falls through to the string probe
                logger.debug("Poly classification fallback", exc_info=True)

        # 5. Check for Power law / Rational
        if "/" in skel_str or "**(-" in skel_str:
            return "rational_or_power_law"

        return "transcendental_composite"

    @classmethod
    def canonicalize(
        cls,
        expression: str | sp.Expr,
        known_variables: list[str] | set[str] | None = None,
    ) -> CanonicalForm:
        """Transforms an expression into its simplified canonical structural form."""
        if isinstance(expression, str):
            raw_str = expression
            parsed = safe_parse_expr(expression)
        elif isinstance(expression, sp.Expr):
            parsed = expression
            raw_str = str(expression)
        else:
            raise TypeError(f"Expected str or sp.Expr, got {type(expression)}")

        # 1. Identify variables and parameters
        var_set, param_set = cls._extract_variables(parsed, known_variables)

        # 2. Simplify identities
        simplified = cls._simplify_identities(parsed)

        # 3. Build canonical skeleton
        skeleton, assigned_params = cls._build_canonical_skeleton(simplified, var_set, param_set)

        # 4. Classify family
        family = cls._classify_family(skeleton, var_set)

        # 5. Compute complexity (count of AST nodes)
        complexity = float(sp.count_ops(simplified) + len(var_set) + len(param_set))

        # 6. Cryptographic fingerprint over canonical skeleton string
        skel_str = str(skeleton)
        fingerprint = hashlib.sha256(skel_str.encode("utf-8")).hexdigest()

        return CanonicalForm(
            raw_expression=raw_str,
            simplified_expression=str(simplified),
            canonical_skeleton=skel_str,
            structural_family=family,
            fingerprint=fingerprint,
            complexity=complexity,
            variables=tuple(sorted(var_set)),
            parameters=tuple(assigned_params),
        )

    @classmethod
    def are_algebraically_equivalent(
        cls,
        expr_a: str | sp.Expr,
        expr_b: str | sp.Expr,
        known_variables: list[str] | set[str] | None = None,
    ) -> bool:
        """Determines if two expressions are mathematically identical modulo algebraic rewrite."""
        pa = safe_parse_expr(expr_a) if isinstance(expr_a, str) else expr_a
        pb = safe_parse_expr(expr_b) if isinstance(expr_b, str) else expr_b

        diff = pa - pb
        if diff == 0:
            return True

        simplified_diff = sp.simplify(sp.cancel(sp.expand_log(diff, force=True)))
        return bool(simplified_diff == 0)

    @classmethod
    def certify_identity(
        cls,
        expr_a: str | sp.Expr,
        expr_b: str | sp.Expr,
        domain: dict[str, tuple[float, float]] | None = None,
        n_points: int = 50,
        tol: float = 1e-8,
        seed: int = 0,
    ) -> str:
        """Certifies an algebraic identity with a proof-claim tier (P0/P1/P2).

        Scope is deliberately narrow: identities over an explicit finite domain,
        NOT universal conjectures (RH-class statements stay P0-only via
        MathematicalAssuranceEngine). Tiers:
        - P2: sympy proves equivalence AND independent numeric agreement on both
          the Python (numpy) and Rust evaluation backends.
        - P1: sympy proves equivalence but dual-backend numeric check is
          unavailable or disagrees above tolerance.
        - P0: sympy cannot establish equivalence.
        Deterministic given seed.
        """
        from apedp.mathematical.mathematical_layer import ProofClaimStatus

        try:
            pa = safe_parse_expr(expr_a) if isinstance(expr_a, str) else expr_a
            pb = safe_parse_expr(expr_b) if isinstance(expr_b, str) else expr_b
            diff = pa - pb
            from apedp.special_functions import SpecialFunctionLibrary
            diff_spec = SpecialFunctionLibrary.apply_special_identities(diff)
            proved = (
                diff == 0
                or diff_spec == 0
                or bool(sp.simplify(diff_spec) == 0)
                or bool(sp.simplify(sp.cancel(sp.expand_log(diff, force=True))) == 0)
            )
        except (sp.SympifyError, TypeError, ValueError, ArithmeticError):
            return ProofClaimStatus.P0_NO_PROOF.value
        if not proved:
            return ProofClaimStatus.P0_NO_PROOF.value

        variables = sorted(s for s in (pa.free_symbols | pb.free_symbols))
        if not variables:
            return ProofClaimStatus.P1_CANDIDATE.value
        domain = domain or {str(v): (0.5, 4.0) for v in variables}
        rng = np.random.default_rng(seed)
        pts: dict[str, np.ndarray] = {}
        for v in variables:
            lo, hi = domain.get(str(v), (0.5, 4.0))
            pts[str(v)] = rng.uniform(lo, hi, n_points)
        order = [str(v) for v in variables]
        try:
            from apedp.special_functions import SpecialFunctionLibrary

            safety_a = SpecialFunctionLibrary.check_domain_safety(pa, domain)
            safety_b = SpecialFunctionLibrary.check_domain_safety(pb, domain)
            if not safety_a.is_domain_safe or not safety_b.is_domain_safe:
                conflicts = safety_a.pole_conflicts + safety_b.pole_conflicts
                logger.debug("Identity certification rejected: pole conflict in domain: %s", conflicts)
                return ProofClaimStatus.P0_NO_PROOF.value

            f = sp.lambdify(variables, pa - pb, modules=["scipy", "numpy"])
            with np.errstate(all="ignore"):
                gap = np.asarray(f(*[pts[k] for k in order]), dtype=float)
            if not np.all(np.isfinite(gap)):
                return ProofClaimStatus.P1_CANDIDATE.value
            scale = np.maximum(
                np.abs(np.asarray(sp.lambdify(variables, pa, modules=["scipy", "numpy"])(*[pts[k] for k in order]), dtype=float)),
                1e-12,
            )
            if not bool(np.all(np.abs(gap) <= tol * scale)):
                return ProofClaimStatus.P1_CANDIDATE.value
        except (TypeError, ValueError, ArithmeticError):
            return ProofClaimStatus.P1_CANDIDATE.value

        try:
            from apedp.rust_bridge import RustKernelBridge

            bridge = RustKernelBridge()
            if not bridge.use_rust:
                return ProofClaimStatus.P1_CANDIDATE.value
            roda = bridge.evaluate_expression(f"({pa})-({pb})", order, [pts[k] for k in order])
            roda = np.asarray(roda, dtype=float)
            if bool(np.all(np.isfinite(roda)) and np.all(np.abs(roda) <= tol * scale)):
                return ProofClaimStatus.P2_VERIFIED.value
            return ProofClaimStatus.P1_CANDIDATE.value
        except (ImportError, TypeError, ValueError, ArithmeticError):
            logger.debug("Rust dual-backend identity check unavailable", exc_info=True)
            return ProofClaimStatus.P1_CANDIDATE.value

    @classmethod
    def same_structural_family(
        cls,
        expr_a: str | sp.Expr,
        expr_b: str | sp.Expr,
        known_variables: list[str] | set[str] | None = None,
    ) -> bool:
        """Determines if two expressions belong to the identical structural parameter class."""
        canon_a = cls.canonicalize(expr_a, known_variables)
        canon_b = cls.canonicalize(expr_b, known_variables)
        return canon_a.structural_family == canon_b.structural_family and canon_a.fingerprint == canon_b.fingerprint

    @classmethod
    def group_by_structural_family(
        cls,
        expressions: list[str | sp.Expr],
        known_variables: list[str] | set[str] | None = None,
    ) -> dict[str, list[CanonicalForm]]:
        """Partitions an ensemble of candidate expressions into unique structural families."""
        families: dict[str, list[CanonicalForm]] = {}
        for expr in expressions:
            canon = cls.canonicalize(expr, known_variables)
            key = f"{canon.structural_family}::{canon.fingerprint[:12]}"
            families.setdefault(key, []).append(canon)
        return families


__all__ = [
    "CanonicalForm",
    "SymbolicEquivalenceEngine",
]
