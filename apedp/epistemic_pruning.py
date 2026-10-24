"""In-loop epistemic guard for fast analytical pruning during symbolic discovery.

Vetoes unphysical candidates before running expensive numerical parameter optimization:
1. Interior singularities: Denominator poles inside the experimental domain.
2. Domain violations: Logarithms/square roots of negative values on the domain.
3. Asymptotic explosions: Property-specific divergence where physical bounds apply.
4. Prior art classification: Transparent labeling and provenance tracking (never vetoed unless explicitly configured).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from apedp.epistemic import NoveltyStatus, PriorArtAuditor

logger = logging.getLogger(__name__)

_PROBE_ERRORS = (
    sp.SympifyError,
    TypeError,
    ValueError,
    ArithmeticError,
    ZeroDivisionError,
    OverflowError,
)


@dataclass(frozen=True)
class EpistemicVetoResult:
    admitted: bool
    reason: str
    details: dict[str, Any]


class InLoopEpistemicGuard:
    """Pre-optimization analytical filter preventing wasted search cycles on invalid forms."""

    def __init__(self, veto_known_prior_art: bool = False):
        self.veto_known_prior_art = veto_known_prior_art

    @staticmethod
    def check_interior_singularities(
        expr: sp.Expr, bounds: dict[str, tuple[float, float]], num_samples: int = 21
    ) -> tuple[bool, str]:
        """Detect real denominator poles inside the variable domains."""
        denoms: set[sp.Expr] = set()
        top_denom = sp.denom(expr)
        if top_denom != 1:
            denoms.add(top_denom)
        for term in sp.Add.make_args(expr):
            d = sp.denom(term)
            if d != 1:
                denoms.add(d)
        for node in sp.preorder_traversal(expr):
            if isinstance(node, sp.Pow) and node.exp.is_negative:
                denoms.add(node.base ** (-node.exp))
        if not denoms:
            try:
                tog_denom = sp.denom(sp.together(expr))
                if tog_denom != 1:
                    denoms.add(tog_denom)
            except _PROBE_ERRORS:
                pass

        if not denoms:
            return True, "NO_DENOMINATOR"

        free_syms = {s.name: s for s in expr.free_symbols}
        for denom in denoms:
            for var_name, (v_min, v_max) in bounds.items():
                if var_name not in free_syms:
                    continue
                sym = free_syms[var_name]

                # 1. Algebraic zero check if single variable in denominator
                denom_syms = denom.free_symbols
                if denom_syms == {sym}:
                    try:
                        roots = sp.solveset(denom, sym, domain=sp.S.Reals)
                        if isinstance(roots, sp.FiniteSet):
                            for r in roots:
                                if r.is_real and (v_min < float(r) < v_max):
                                    return False, f"POLE_AT_{var_name}_{float(r):.4f}"
                    except _PROBE_ERRORS as exc:
                        logger.debug("Algebraic pole check failed: %s", exc)

                # 2. Dense numerical sampling check for denominator sign change / vanishing
                test_grid = np.linspace(v_min, v_max, num_samples)
                f_denom = sp.lambdify(
                    [sym],
                    denom.subs({s: 1.0 for s in denom.free_symbols if s != sym}),
                    modules=["numpy"],
                )
                try:
                    vals = np.asarray(f_denom(test_grid), dtype=float).ravel()
                    if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
                        return False, f"SINGULARITY_IN_DOMAIN_{var_name}"
                    if np.any(np.isclose(vals, 0.0, atol=1e-7)):
                        return False, f"POLE_AT_GRID_IN_{var_name}"
                    finite_vals = vals[np.isfinite(vals)]
                    signs = np.sign(finite_vals)
                    non_zero_signs = signs[signs != 0]
                    if len(non_zero_signs) > 1 and np.any(non_zero_signs[:-1] * non_zero_signs[1:] < 0):
                        return False, f"POLE_ZERO_CROSSING_IN_{var_name}"
                except _PROBE_ERRORS as exc:
                    logger.debug("Numerical pole scan failed: %s", exc)

        return True, "NO_INTERIOR_SINGULARITY"

    @staticmethod
    def check_asymptotics(
        expr: sp.Expr,
        bounds: dict[str, tuple[float, float]],
        bounded_vars: set[str] | None = None,
    ) -> tuple[bool, str]:
        """Verify that expressions do not blow up to +/- infinity if bounded behavior is required."""
        if not bounded_vars:
            return True, "NO_BOUNDED_CONSTRAINTS"

        free_syms = {s.name: s for s in expr.free_symbols}
        for var_name in bounded_vars:
            if var_name not in free_syms:
                continue
            sym = free_syms[var_name]
            try:
                lim_inf = sp.limit(expr, sym, sp.oo)
                if lim_inf.is_infinite:
                    return False, f"ASYMPTOTIC_EXPLOSION_AT_INF_{var_name}"
            except _PROBE_ERRORS as exc:
                logger.debug("Asymptotic limit check failed: %s", exc)

        return True, "ASYMPTOTICS_WELL_BEHAVED"

    def evaluate(
        self,
        skeleton: sp.Expr,
        bounds: dict[str, tuple[float, float]],
        bounded_vars: set[str] | None = None,
        check_prior_art: bool = False,
    ) -> EpistemicVetoResult:
        """Run all analytical checks on candidate skeleton, labeling prior art and vetoing unphysical forms."""
        # 1. Interior singularity check
        admit_sing, reason_sing = self.check_interior_singularities(skeleton, bounds)
        if not admit_sing:
            return EpistemicVetoResult(
                admitted=False,
                reason=reason_sing,
                details={"check": "singularity"},
            )

        # 2. Property-specific asymptotics check
        admit_asymp, reason_asymp = self.check_asymptotics(skeleton, bounds, bounded_vars)
        if not admit_asymp:
            return EpistemicVetoResult(
                admitted=False,
                reason=reason_asymp,
                details={"check": "asymptotics"},
            )

        # 3. Prior art provenance & identification (Labels by default; vetoes only if explicitly requested)
        prior_art_id = None
        novelty_status = "UNEVALUATED"
        if check_prior_art:
            try:
                audit = PriorArtAuditor.audit_novelty(skeleton)
                novelty_status = audit.status.value
                if audit.status == NoveltyStatus.KNOWN_MODEL:
                    prior_art_id = audit.closest_reference_id
                    if self.veto_known_prior_art:
                        return EpistemicVetoResult(
                            admitted=False,
                            reason=f"REFUSED_KNOWN_MODEL_{prior_art_id}",
                            details={"status": novelty_status, "reference": prior_art_id},
                        )
            except _PROBE_ERRORS as exc:
                logger.debug("Prior art audit check failed: %s", exc)

        return EpistemicVetoResult(
            admitted=True,
            reason="ADMITTED",
            details={"prior_art": prior_art_id, "novelty_status": novelty_status},
        )


__all__ = ["EpistemicVetoResult", "InLoopEpistemicGuard"]
