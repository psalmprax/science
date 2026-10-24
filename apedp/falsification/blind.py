"""Blind adversarial audit: sealed-expectation scoring of the falsifier."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import sympy as sp

from apedp.safety import safe_parse_expr

from .constraints import (
    ContradictoryConstraintEngine,
)
from .counterexamples import (
    InfeasibilityCertificate,
)
from .falsifier import (
    IndependentAdversarialFalsifier,
)

logger = logging.getLogger("apedp.falsification")

@dataclass
class BlindAuditResult:
    """Outcome of blind unannotated adversarial analysis on an arbitrary equation."""
    formula_str: str
    is_defective: bool
    free_variables: list[str]
    identified_failure_axes: list[str]
    failure_locations: list[dict[str, Any]]
    failure_descriptions: list[str]
    competing_candidates: list[str]
    champion_candidate: str | None
    champion_survived: bool
    champion_verdict: str
    infeasibility_certificate: InfeasibilityCertificate | None
    epistemic_qualification: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula_str": self.formula_str,
            "is_defective": self.is_defective,
            "free_variables": self.free_variables,
            "identified_failure_axes": self.identified_failure_axes,
            "failure_locations": self.failure_locations,
            "failure_descriptions": self.failure_descriptions,
            "competing_candidates": self.competing_candidates,
            "champion_candidate": self.champion_candidate,
            "champion_survived": self.champion_survived,
            "champion_verdict": self.champion_verdict,
            "infeasibility_certificate": self.infeasibility_certificate.to_dict() if self.infeasibility_certificate else None,
            "epistemic_qualification": self.epistemic_qualification,
        }


class BlindAdversarialAuditor:
    """Performs blind, unannotated defect discovery, multi-hypothesis repair, and cross-falsification.

    Given a bare equation without failure class hints, critical boundary hints, or domain tags:
    1. Autonomously localizes poles, branch cuts, and asymptotic divergences.
    2. Proves whether specified constraints are mutually incompatible.
    3. Synthesizes competing candidate reformulations (softening vs saturation vs representation transformation).
    4. Evaluates all candidates through tournament cross-falsification.
    5. Returns a rigorous audit report with provisional validity bounds.
    """

    def __init__(self, falsifier: IndependentAdversarialFalsifier | None = None):
        self.falsifier = falsifier or IndependentAdversarialFalsifier()

    def audit(
        self,
        formula: str | sp.Expr,
        default_domains: dict[str, tuple[float, float]] | None = None,
        constraints: list[str] | None = None
    ) -> BlindAuditResult:
        formula_str = str(formula)
        try:
            expr = safe_parse_expr(formula) if isinstance(formula, str) else formula
        except (ValueError, TypeError) as e:
            return BlindAuditResult(
                formula_str=formula_str,
                is_defective=True,
                free_variables=[],
                identified_failure_axes=["MATHEMATICAL"],
                failure_locations=[],
                failure_descriptions=[f"Unparseable formula: {e}"],
                competing_candidates=[],
                champion_candidate=None,
                champion_survived=False,
                champion_verdict="REFUSED_CANNOT_REPAIR",
                infeasibility_certificate=None,
                epistemic_qualification="Syntax failure."
            )

        # 1. Infeasibility check on constraints
        infeas_cert = ContradictoryConstraintEngine.check_incompatibility(constraints or [])
        if infeas_cert and infeas_cert.is_infeasible:
            return BlindAuditResult(
                formula_str=formula_str,
                is_defective=True,
                free_variables=[str(s) for s in expr.free_symbols],
                identified_failure_axes=["STRUCTURAL_INVARIANT"],
                failure_locations=[],
                failure_descriptions=[infeas_cert.incompatibility_proof],
                competing_candidates=[],
                champion_candidate=None,
                champion_survived=False,
                champion_verdict="REFUSED_INCOMPATIBLE_CONSTRAINTS",
                infeasibility_certificate=infeas_cert,
                epistemic_qualification="Incompatible constraints: Admissible function space is proven empty."
            )

        # 2. Autonomous domain setup & pole/branch localization
        free_syms = list(expr.free_symbols)
        domains: dict[str, tuple[float, float]] = dict(default_domains or {})
        for s in free_syms:
            s_name = str(s)
            if s_name not in domains:
                # Wide probing domain default
                domains[s_name] = (1e-6, 1e4)

        # Autonomous critical boundaries discovery
        crit_boundaries: dict[str, list[float]] = {}
        for s in free_syms:
            s_name = str(s)
            crit_boundaries[s_name] = [0.0]

        # Scan for denominator roots automatically
        denoms = []
        _numer, denom = expr.as_numer_denom()
        if denom != 1:
            denoms.append(denom)
        for p in expr.atoms(sp.Pow):
            if p.exp.is_number and float(p.exp) < 0:
                denoms.append(p.base)

        for d_expr in denoms:
            for s in d_expr.free_symbols:
                s_name = str(s)
                try:
                    r_list = sp.solve(d_expr, s)
                    for r in r_list:
                        if getattr(r, "is_real", False) and getattr(r, "is_number", False):
                            r_val = float(r)
                            crit_boundaries.setdefault(s_name, []).append(r_val)
                except (ValueError, TypeError, ArithmeticError, NotImplementedError) as e:
                    logger.debug("Denominator root solve failed for %s (%s)", d_expr, e)

        # 3. Blind adversarial falsification of raw formula
        raw_rep = self.falsifier.falsify(
            formula=expr,
            variable_domains=domains,
            critical_boundaries=crit_boundaries,
            allow_repair_attempt=True,
            constraints=constraints
        )

        is_defective = not raw_rep.survived
        failure_axes = list({c.axis.value for c in raw_rep.counterexamples})
        failure_locs = [c.counterexample_point for c in raw_rep.counterexamples]
        failure_descs = [c.violation_description for c in raw_rep.counterexamples]

        if not is_defective:
            return BlindAuditResult(
                formula_str=formula_str,
                is_defective=False,
                free_variables=[str(s) for s in free_syms],
                identified_failure_axes=[],
                failure_locations=[],
                failure_descriptions=[],
                competing_candidates=[formula_str],
                champion_candidate=formula_str,
                champion_survived=True,
                champion_verdict=raw_rep.verdict,
                infeasibility_certificate=None,
                epistemic_qualification=raw_rep.epistemic_qualification
            )

        # 4. Multi-Hypothesis Synthesis
        competing: list[str] = []
        primary_var = str(free_syms[0]) if free_syms else "x"
        s_sym = sp.Symbol(primary_var)

        # Strategy 1: Soft-core Tikhonov smoothing
        softened_var = sp.sqrt(s_sym**2 + sp.Symbol("epsilon_reg")**2)
        try:
            cand1_expr = expr.subs(s_sym, softened_var).subs(sp.Symbol("epsilon_reg"), 0.04)
            cand1_str = str(cand1_expr)
            if cand1_str != formula_str:
                competing.append(cand1_str)
        except (ValueError, TypeError, ArithmeticError, AttributeError) as e:
            logger.debug("Soft-core candidate failed (%s)", e)

        # Strategy 2: Asymptotic saturation / Padé-envelope
        try:
            cand2_expr = (expr * 1000.0) / (1000.0 + sp.Abs(expr))
            cand2_str = str(cand2_expr)
            competing.append(cand2_str)
        except (ValueError, TypeError, ArithmeticError, AttributeError) as e:
            logger.debug("Saturation candidate failed (%s)", e)

        # Strategy 3: Coordinate / reciprocal inversion (e.g. Lineweaver-Burk)
        try:
            poly_inv = sp.together(expr)
            n_inv, d_inv = poly_inv.as_numer_denom()
            if n_inv != 1 and d_inv != 1:
                cand3_expr = d_inv / n_inv
                cand3_str = str(cand3_expr)
                if cand3_str not in competing:
                    competing.append(cand3_str)
        except (ValueError, TypeError, ArithmeticError, AttributeError) as e:
            logger.debug("Inversion candidate failed (%s)", e)

        # Fallback candidate if none generated
        if not competing:
            competing.append(f"({formula_str}) / (1.0 + 0.04 * abs({formula_str}))")

        # 5. Tournament cross-falsification of candidates
        champion = None
        champion_survived = False
        champion_verdict = "REFUSED_CANNOT_REPAIR"

        for cand in competing:
            c_rep = self.falsifier.falsify(
                formula=cand,
                variable_domains=domains,
                critical_boundaries=crit_boundaries,
                allow_repair_attempt=True,
                constraints=constraints
            )
            if c_rep.survived:
                champion = cand
                champion_survived = True
                champion_verdict = c_rep.verdict
                break

        if champion is None and competing:
            champion = competing[0]

        return BlindAuditResult(
            formula_str=formula_str,
            is_defective=is_defective,
            free_variables=[str(s) for s in free_syms],
            identified_failure_axes=failure_axes,
            failure_locations=failure_locs,
            failure_descriptions=failure_descs,
            competing_candidates=competing,
            champion_candidate=champion,
            champion_survived=champion_survived,
            champion_verdict=champion_verdict,
            infeasibility_certificate=None,
            epistemic_qualification="Blind audit completed across 5 axes without prior failure hints."
        )
