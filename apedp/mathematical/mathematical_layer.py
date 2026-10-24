"""Mathematical Assurance & Proof Boundary Layer for APEDP.

Provides domain constraints, precision boundary certification, and formal proof guards
for pure mathematical discovery tasks (e.g. Riemann Hypothesis, prime distribution, recurrences),
ensuring that empirical numerical support is never conflated with mathematical proof.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import sympy as sp

from apedp.safety import safe_parse_expr

logger = logging.getLogger(__name__)


def _unverified(check: str, exc: Exception) -> str:
    """Wording for a check that raised rather than returning a verdict.

    Same shape as ``apedp/physics_constraints.py`` and
    ``apedp/singularity_probing.py``: name the check and the exception, and
    let the report say so. A probe that logged and fell through left its
    verdict at the initial value, so a crashed check reached the caller
    looking exactly like a passed one.
    """
    return f"{check} could not be evaluated ({type(exc).__name__}: {exc})"


class ProofBoundaryStatus(str, Enum):
    """Epistemic status regarding the boundary between empirical evidence and formal proof."""

    EMPIRICALLY_SUPPORTED = "EMPIRICALLY_SUPPORTED"
    PROOF_NOT_ESTABLISHED = "PROOF_NOT_ESTABLISHED"
    UNIVERSAL_STATEMENT_UNRESOLVED = "UNIVERSAL_STATEMENT_UNRESOLVED"
    FALSIFIED_BY_COUNTEREXAMPLE = "FALSIFIED_BY_COUNTEREXAMPLE"
    INDISTINGUISHABLE_BELOW_RESOLUTION = "INDISTINGUISHABLE_BELOW_RESOLUTION"


class ProofClaimStatus(str, Enum):
    """Proof-claim tier, independent of empirical confidence (C-classes).

    C-classes (C1..C5) report empirical support within the tested domain.
    P-tiers report proof standing: P0 = no proof, P1 = proof candidate
    (deductive argument proposed, unverified), P2 = formally verified proof.
    No finite-sample pipeline in this repo emits above P0; P1/P2 are reserved
    for a future proof-candidate/verification pipeline. RH therefore reports
    (C1,P0): empirically supported on N tested, proof not established.
    """

    P0_NO_PROOF = "P0"
    P1_CANDIDATE = "P1"
    P2_VERIFIED = "P2"


@dataclass(frozen=True)
class MathematicalDomainAudit:
    """Outcome of mathematical domain and boundary consistency audit."""

    is_admissible: bool
    domain_status: str
    in_critical_strip: bool
    singularities_encountered: list[str] = field(default_factory=list)
    proof_status: ProofBoundaryStatus = ProofBoundaryStatus.PROOF_NOT_ESTABLISHED
    proof_claim: ProofClaimStatus = ProofClaimStatus.P0_NO_PROOF
    resolution_limit: float = 1e-12
    rationale: str = ""
    # Checks that raised rather than returning a verdict. Non-empty means this
    # audit is not a clearance even where is_admissible is True, and the
    # field exists so "we found no pole" can be told from "we could not look".
    unverified: list[str] = field(default_factory=list)

    @property
    def cp_label(self) -> str:
        """Joint empirical/proof label stub (empirical part filled by caller)."""
        return self.proof_claim.value


class MathematicalAssuranceEngine:
    """Audits mathematical conjectures, precision boundaries, and enforces proof boundary guards."""

    def __init__(self, default_precision: float = 1e-12):
        self.default_precision = default_precision

    def audit_mathematical_candidate(
        self,
        candidate_expr: sp.Expr | str,
        variable_domains: Mapping[str, tuple[float, float]],
        sample_observations: dict[str, np.ndarray] | None = None,
        claimed_proof: bool = False,
    ) -> MathematicalDomainAudit:
        """Audits a candidate mathematical expression for analytic sanity and proof boundaries.

        Args:
            candidate_expr: Expression or string representation.
            variable_domains: Expected mathematical bounds for variables.
            sample_observations: Empirical data samples used during evaluation.
            claimed_proof: Whether the candidate generator asserted a mathematical proof.

        Returns:
            MathematicalDomainAudit containing admissibility, domain consistency, and proof status.
        """
        if isinstance(candidate_expr, str):
            try:
                expr = safe_parse_expr(candidate_expr)
            except Exception as exc:  # noqa: BLE001 - best-effort probe must not crash
                return MathematicalDomainAudit(
                    is_admissible=False,
                    domain_status="SYNTAX_ERROR",
                    in_critical_strip=False,
                    singularities_encountered=[str(exc)],
                    proof_status=ProofBoundaryStatus.PROOF_NOT_ESTABLISHED,
                    rationale=f"Failed to parse candidate expression: {exc}",
                )
        else:
            expr = candidate_expr

        # 1. Critical strip check for Riemann zeta candidates (0 < Re(s) < 1)
        in_strip = True
        singularities: list[str] = []
        # Checks that raised. Both probes below used to end in logger.debug
        # and fall through with their verdict at its initial value, which made
        # two failures indistinguishable from two passes:
        #   - in_strip stayed True, asserting the candidate sits in the RH
        #     critical strip on the strength of an evaluation that raised.
        #   - singularities stayed empty, and is_admissible is computed as
        #     len(singularities) == 0, so a sp.solve that raised certified the
        #     domain "VALID" and the candidate admissible. A candidate with a
        #     pole in the evaluated domain passed the audit because the search
        #     for poles crashed.
        # sp.solve and .evalf() raise on ordinary input, so these are expected
        # paths. The record carries what did not run.
        unverified: list[str] = []
        # safe_parse_expr("gamma") returns sympy's gamma *FunctionClass*, not
        # an Expr. On a class, `.is_number` is an unbound property object --
        # which is truthy -- so the numeric branch was always taken and
        # `.evalf()` then raised TypeError("missing 1 required positional
        # argument: 'self'"). Every bare-function-name candidate, including
        # the Euler gamma this engine exists to reason about, was crashing
        # here and the crash was swallowed: in_strip kept its initial True and
        # the audit reported an unverified candidate as admissible. Found by
        # the disclosure change above, which turned a silently passing test
        # into a visible failure.
        #
        # A FunctionClass is not a failed check, though. It has no free
        # symbols, so there is no domain in which it could be singular and no
        # value whose strip membership could be tested. Both checks are
        # vacuous, not unrun, and vacuous is not the same as unknown.
        is_expr = isinstance(expr, sp.Expr)
        try:
            if is_expr and expr.is_number:
                # Check if candidate evaluates to a constant or variable in [0, 1]
                val = float(expr.evalf())
                if not (0.0 < val < 1.0):
                    in_strip = False
            elif is_expr and "beta" in [str(s) for s in expr.free_symbols]:
                # If evaluating real part expression
                pass
        except Exception as e:  # noqa: BLE001 - sympy raises broadly on
            # expressions with no limit or a non-numeric real part; the strip
            # membership is unknown, not established.
            unverified.append(_unverified("Critical-strip probe", e))

        # 2. Singularity detection on domain
        for var, (vmin, vmax) in variable_domains.items():
            sym = sp.Symbol(var)
            if is_expr and sym in expr.free_symbols:
                denom = sp.denom(sp.together(expr))
                if denom != 1:
                    try:
                        roots = sp.solve(denom, sym)
                        for r in roots:
                            if r.is_real:
                                r_val = float(r.evalf())
                                if vmin <= r_val <= vmax:
                                    singularities.append(f"Pole at {var} = {r_val}")
                    except Exception as e:  # noqa: BLE001 - sp.solve raises
                    # broadly on transcendentals and multivariate denominators.
                    # A root that was never found is not an absent root.
                        unverified.append(
                            _unverified(f"Singularity detection for '{var}'", e))

        # 3. Proof Boundary Enforcement:
        # A finite empirical sample N can NEVER prove a universal mathematical conjecture.
        # Orthogonality invariant: low MSE / good extrapolation / high C-confidence
        # must NEVER promote P0->P1. P1 requires an explicit formal argument for a
        # proof system; P2 requires its verified acceptance. This audit emits P0 only.
        n_obs = len(next(iter(sample_observations.values()))) if sample_observations else 0
        if claimed_proof:
            proof_status = ProofBoundaryStatus.PROOF_NOT_ESTABLISHED
            rationale = (
                f"Proof boundary guard enforced: candidate generator claimed universal proof based on "
                f"N = {n_obs} finite numerical observations. Numerical evidence cannot establish universal mathematical proof."
            )
        else:
            proof_status = (
                ProofBoundaryStatus.EMPIRICALLY_SUPPORTED
                if n_obs > 0
                else ProofBoundaryStatus.UNIVERSAL_STATEMENT_UNRESOLVED
            )
            rationale = (
                f"Empirical numerical support verified over N = {n_obs} observations. "
                f"Formal proof remains unestablished (finite sample != universal proof)."
            )

        # A pole search that did not complete cannot support a clearance. The
        # direction is toward withholding, matching how every other failed
        # check in this codebase is now handled.
        is_admissible = len(singularities) == 0 and not unverified

        if unverified and not singularities:
            domain_status = "SINGULARITY_CHECK_INCOMPLETE"
        elif unverified:
            domain_status = "SINGULARITY_IN_DOMAIN"
        else:
            domain_status = "VALID"

        if unverified:
            rationale = (
                rationale + " Not a full clearance: " + "; ".join(unverified)
            ).strip()

        return MathematicalDomainAudit(
            is_admissible=is_admissible,
            domain_status=domain_status,
            in_critical_strip=in_strip,
            singularities_encountered=singularities,
            proof_status=proof_status,
            proof_claim=ProofClaimStatus.P0_NO_PROOF,
            resolution_limit=self.default_precision,
            rationale=rationale,
            unverified=unverified,
        )

    def assess_identifiability_resolution(
        self,
        candidate_errors: dict[str, float],
        numerical_resolution: float,
    ) -> tuple[bool, str, list[str]]:
        """Determines if competing mathematical hypotheses are distinguishable above numerical resolution.

        If |E(H_1) - E(H_2)| <= resolution, the hypotheses are mathematically indistinguishable
        at the current floating-point precision, necessitating MULTI_HYPOTHESIS state.

        Returns:
            (is_distinguishable, winning_candidate, indistinguishable_pool)
        """
        sorted_cands = sorted(candidate_errors.items(), key=lambda x: x[1])
        if len(sorted_cands) <= 1:
            return True, sorted_cands[0][0] if sorted_cands else "", []

        best_id, best_err = sorted_cands[0]
        _second_id, second_err = sorted_cands[1]

        diff = abs(second_err - best_err)
        if diff <= numerical_resolution:
            # Below numerical resolution: cannot separate
            tied = [cid for cid, err in sorted_cands if abs(err - best_err) <= numerical_resolution]
            return False, best_id, tied

        return True, best_id, [best_id]
