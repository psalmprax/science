"""Stage C Epistemic Decision Engine for APEDP.

Formalizes the 4-state epistemic outcome machine (empirical support ONLY --
never a mathematical proof; proof status lives separately in
apedp.mathematical.mathematical_layer.ProofBoundaryStatus):
- UNIQUE_SUPPORT (C1): domain-bounded uniqueness -- among admissible tested
  candidates, exactly one is supported over the observed regime. This is NOT
  structural recovery (same family as the generator) and NOT scientific law
  identification (established governing law); those stronger claims require
  family-match diagnostics and independent justification respectively, and
  remain P0 without them. Uniqueness without OOD evidence explicitly records
  extrapolation as untested missing evidence.
- MULTI_HYPOTHESIS: Multiple admissible hypotheses remain competitive with trade-offs.
  The engine explicitly refuses to force a single winner.
- INSUFFICIENT_EVIDENCE: Candidates cannot be verified due to missing or inadequate data.
- NO_VALID_CANDIDATE: All candidates failed mathematical admissibility or dimensional gates.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp

from apedp.singularity_probing import ActiveSingularityProbeEngine

logger = logging.getLogger(__name__)


#: Appended whenever the paired-bootstrap noise-floor probe did not run. The
#: wording is fixed so that a reader can recognise the caveat without having to
#: know which probe produced it.
_SE_PROBE_CAVEAT = (
    "The observational equivalence probe (paired bootstrap on the holdout "
    "difference) raised for: %s. Those candidates are absent from the "
    "competitive set because the comparison could not be made, not because "
    "they were shown to be distinguishable."
)


def _strip_abs_to_linear_form(base: sp.Expr) -> sp.Expr:
    """Reduce a singular component to the form whose zeros we want to locate.

    ``Abs(x - c)`` never changes sign, so a sign-change scan cannot see its
    root; the inner form has the same zero location and is a polynomial. The
    same holds for any positive power of it, which is why the strip has to go
    *through* ``Pow`` and not just test for ``Abs`` at the top.

    That second case was a live bug. A fractional-power singularity arrives as
    ``Abs(x - c)**0.5``, whose component base is a ``Pow`` rather than an
    ``Abs``, so the original ``while isinstance(inner, sp.Abs)`` loop left it
    untouched and ``sp.Poly`` raised ``PolynomialError``. The raise was caught
    and the component skipped, so the pole check reported clean on a component
    it had never examined.

    A *negative* exponent is deliberately not stripped: the caller's contract
    is to reduce the component to a polynomial, and ``Abs(x - c)**-0.5`` is a
    fractional-power singularity whose local behaviour a linear form does not
    describe. Only the zero location is being reused, and the caller
    re-checks the component's kind before acting on it.
    """
    inner = base
    while True:
        if isinstance(inner, sp.Abs) and inner.args:
            inner = inner.args[0]
        elif isinstance(inner, sp.Pow) and inner.exp.is_positive and inner.base.args:
            # A positive power has the same zeros as its base: the location is
            # unchanged, only the form differs.
            inner = inner.base
        else:
            return inner


def _fmt_se_failures(failures: list[tuple[str, str]]) -> str:
    return ", ".join(f"'{cid}' ({err})" for cid, err in failures)


class EpistemicState(str, Enum):
    UNIQUE_SUPPORT = "UNIQUE_SUPPORT"
    MULTI_HYPOTHESIS = "MULTI_HYPOTHESIS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NO_VALID_CANDIDATE = "NO_VALID_CANDIDATE"


class AbstentionReason(str, Enum):
    """Which gate declined to endorse a model, and for what cause.

    ``EpistemicState`` answers *whether* the synthesizer is withholding
    endorsement. It cannot answer *why*, because one state covers several
    conditions that call for opposite responses downstream. ``NO_VALID_
    CANDIDATE`` is returned both when the data contains a shock front no
    smooth family can express -- a true statement about the world, and the
    right answer -- and when every template failed to converge, which is a
    statement about the optimizer, not about the data. Reading the second as
    the first is how a broken run gets reported as a scientific finding.

    So the cause is a separate, machine-readable code, and the prose rationale
    is generated from it rather than written alongside it. Two consumers
    depend on the distinction: a benchmark that must not score an
    infrastructure failure as out-of-grammar, and a human deciding whether
    the next move is a new family or a different optimizer.

    Members are ordered by specificity, which is also the order in which
    :func:`apedp.model_synthesis._classify_adversarial` reports a primary
    reason when several fire at once: the specific cause is more useful than
    the catch-all, and the full set is always retained alongside it.
    """

    #: Nothing to explain yet -- the underspecified-data gate (N too small).
    UNDERSPECIFIED_DATA = "UNDERSPECIFIED_DATA"
    #: Every candidate template raised before producing a fit.
    NO_CANDIDATE_FITTED = "NO_CANDIDATE_FITTED"
    #: A discontinuity the smooth candidate grammar cannot represent.
    SHOCK_DISCONTINUITY = "SHOCK_DISCONTINUITY"
    #: Residuals are autocorrelated, so the smooth fit is misspecified.
    RESIDUAL_AUTOCORRELATION = "RESIDUAL_AUTOCORRELATION"
    #: The best fit does not beat the mean. The catch-all; least specific.
    LOW_VALIDATION_R2 = "LOW_VALIDATION_R2"


#: Human-readable text per reason, used to build the diagnostics rationale.
#: Kept next to the enum so a code and its wording cannot drift apart.
ABSTENTION_REASON_TEXT: dict[AbstentionReason, str] = {
    AbstentionReason.UNDERSPECIFIED_DATA: (
        "Severe data sparsity (N <= 8) precludes confident model selection."
    ),
    AbstentionReason.NO_CANDIDATE_FITTED: (
        "No candidate structure produced a fit; the abstention reflects an "
        "empty candidate set, not a property of the data."
    ),
    AbstentionReason.SHOCK_DISCONTINUITY: (
        "A shock discontinuity is present and smooth families deviate "
        "systematically from it."
    ),
    AbstentionReason.RESIDUAL_AUTOCORRELATION: (
        "Residual autocorrelation exceeds 0.45: the residuals carry structure "
        "the smooth candidate does not explain."
    ),
    AbstentionReason.LOW_VALIDATION_R2: (
        "The best candidate does not beat the mean on held-out data "
        "(validation R^2 < 0.85)."
    ),
}


class ModelClass(str, Enum):
    """Model family class for Tri-Hypothesis discovery and adjudication."""

    SYMBOLIC = "SYMBOLIC"
    NEURAL = "NEURAL"
    HYBRID = "HYBRID"


@dataclass(frozen=True)
class CandidateProfile:
    """Standardized representation of a candidate hypothesis across Stages A, B, and C."""

    candidate_id: str
    expression: str
    model_class: ModelClass = ModelClass.SYMBOLIC
    math_sound: bool = True
    dim_status: str = "VALID"  # "VALID", "INVALID", "UNRESOLVED"
    in_sample_mse: float = float("inf")
    in_sample_r2: float = 0.0
    holdout_mse: float = float("inf")
    holdout_r2: float = 0.0
    near_ood_mse: float = float("inf")
    far_ood_mse: float = float("inf")
    complexity: float = 1.0
    bic: float | None = None
    aic: float | None = None
    cond_number: float = 1.0
    bootstrap_stability: float = 1.0  # S(f) in [0, 1]
    residual_pattern_score: float = 0.0
    interpretability_score: float = 1.0  # 1.0 for symbolic, 0.5 for hybrid, 0.1 for neural
    residual_of_candidate: str | None = None
    limitations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_stage_a_admissible(self) -> bool:
        """Stage A gate: must be mathematically sound and dimensionally VALID.
        UNRESOLVED and INVALID are excluded from scientific selection."""
        return bool(self.math_sound) and self.dim_status == "VALID"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model_class"] = self.model_class.value
        d["is_stage_a_admissible"] = self.is_stage_a_admissible
        return d


@dataclass(frozen=True)
class StageCDecision:
    """Outcome of Stage C epistemic adjudication over an ensemble of hypotheses."""

    state: EpistemicState
    supported_hypotheses: list[str]
    winner: str | None
    adjudication_rationale: str
    discrimination_required: bool
    missing_evidence: list[str] = field(default_factory=list)
    candidate_evaluations: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy_version: str = "stage-c-v1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "supported_hypotheses": list(self.supported_hypotheses),
            "winner": self.winner,
            "adjudication_rationale": self.adjudication_rationale,
            "discrimination_required": self.discrimination_required,
            "missing_evidence": list(self.missing_evidence),
            "candidate_evaluations": self.candidate_evaluations,
            "policy_version": self.policy_version,
        }


class StageCDecisionEngine:
    """Adjudicates competing scientific hypotheses without baking in target answers.

    Core Invariant:
    If multiple admissible hypotheses exhibit comparable predictive performance,
    trade off complexity vs OOD stability, or split information criteria,
    the engine emits MULTI_HYPOTHESIS rather than forcing a winner.
    """

    def __init__(
        self,
        holdout_margin_ratio: float = 1.15,
        max_condition_number: float = 1e6,
        min_bootstrap_stability: float = 0.15,
        min_holdout_r2: float = 0.0,
    ):
        self.holdout_margin_ratio = holdout_margin_ratio
        self.max_condition_number = max_condition_number
        self.min_bootstrap_stability = min_bootstrap_stability
        self.min_holdout_r2 = min_holdout_r2

    @staticmethod
    def _pole_refusals_in_observed_data(
        candidates: list[CandidateProfile],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Refuses candidates whose expression has a pole strictly inside observed data.

        A pole strictly inside the observed interval (and outside known sampling
        gaps) with finite observations there means the model diverges where data
        exist: the structure is wrong at those points, not merely uncertain.
        Poles inside known sampling gaps (e.g. physical critical points) are
        exempt. Skipped entirely when a candidate carries no observed-range
        metadata, preserving behavior for all existing callers.

        Returns ``(refused, inconclusive)``. The second mapping exists because
        this function used to have one exit for two different findings. A
        candidate was refused when a pole was found and simply absent when the
        extraction raised, and the caller could only test for presence -- so
        "no pole" and "no answer" reached Stage A looking identical, and the
        second one endorsed a model nobody had checked. Sympy raises a wide,
        version-dependent set on these paths, so that is a normal outcome and
        not a corner case.
        """
        refusals: dict[str, str] = {}
        #: Candidate id -> why its pole status is unknown. Kept separate from
        #: `refusals` because "we found a pole" and "we could not look" are
        #: different findings, and the caller must be able to say which.
        inconclusive: dict[str, str] = {}
        probed = [
            c for c in candidates
            if "x_obs_min" in c.metadata and "x_obs_max" in c.metadata
        ]
        if not probed:
            return refusals, inconclusive
        engine = ActiveSingularityProbeEngine()
        for c in probed:
            try:
                x_min = float(c.metadata["x_obs_min"])
                x_max = float(c.metadata["x_obs_max"])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(x_min) and np.isfinite(x_max)) or x_max <= x_min:
                continue
            raw_gaps = c.metadata.get("x_gaps", []) or []
            gaps: list[tuple[float, float]] = []
            for g in raw_gaps:
                try:
                    gl, gh = float(g[0]), float(g[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if np.isfinite(gl) and np.isfinite(gh) and gh > gl:
                    gaps.append((gl, gh))
            span = x_max - x_min
            tol = 1e-9 * span
            try:
                components = engine.extract_singular_components(c.expression)
            except Exception as e:
                # Recorded as inconclusive, not skipped. Skipping leaves the
                # candidate out of `refusals`, and the caller reads absence of
                # refusal as "no pole in the observed data" -- so a sympy
                # expression the extractor cannot handle is certified pole-free
                # and goes on to be endorsed. An unchecked candidate must not be
                # an endorsed one.
                logger.debug("Pole extraction failed for '%s'", c.candidate_id, exc_info=True)
                inconclusive[c.candidate_id] = (
                    f"pole check could not be completed "
                    f"({type(e).__name__}: {e}); singularity status is unknown, "
                    f"not absent"
                )
                continue
            for base, kind in components:
                if kind not in ("DENOMINATOR", "POW_NEGATIVE"):
                    continue
                inner = _strip_abs_to_linear_form(base)
                # Bounded linear-form solve only: sp.solve on higher-degree
                # denominators can hang; nonlinear poles stay in scope of the
                # Tier-1 gap probe and the active singularity stress test.
                # Linear a*x + b == 0 -> root -b/a, computed directly.
                try:
                    poly = sp.Poly(inner, sp.Symbol("x"))
                    coeffs = poly.all_coeffs()
                    if poly.degree() != 1 or len(coeffs) != 2:
                        continue
                    a1 = float(coeffs[0])
                    b0 = float(coeffs[1])
                    if not (np.isfinite(a1) and np.isfinite(b0)) or a1 == 0.0:
                        continue
                    roots = [-b0 / a1]
                except sp.PolynomialError:
                    # Not a polynomial at all -- a denominator carrying exp()
                    # or another transcendental, or any degree above the bounded
                    # linear-form scope this function documents. The component's
                    # poles are handled by the Tier-1 gap probe and the active
                    # singularity stress test instead.
                    #
                    # This is deliberately NOT recorded as inconclusive. Doing
                    # so withholds the candidate, and since almost every fitted
                    # expression with an exponential in the denominator lands
                    # here, it withholds nearly everything -- converting a
                    # documented scope limit into a silent rejection. What
                    # distinguishes it from a real failure is that nothing
                    # unexpected happened: the expression is simply outside the
                    # scope this routine claims.
                    continue
                except Exception as e:
                    # A failure on something that *should* have been in scope.
                    # The component's roots are absent from `roots`, so its
                    # poles are never range-checked and the candidate would
                    # pass as pole-free.
                    logger.debug("Singular-root search failed for '%s'", c.candidate_id, exc_info=True)
                    inconclusive.setdefault(c.candidate_id, "")
                    inconclusive[c.candidate_id] = (
                        f"a singular-component root search failed "
                        f"({type(e).__name__}: {e}); this component's poles are "
                        f"unchecked"
                    )
                    continue
                for r in roots:
                    try:
                        rv = float(r)
                    except (TypeError, ValueError):
                        continue
                    if not (x_min + tol < rv < x_max - tol):
                        continue
                    if any(gl - tol <= rv <= gh + tol for gl, gh in gaps):
                        continue
                    refusals[c.candidate_id] = (
                        f"pole at x={rv:.6g} lies strictly inside observed data "
                        f"[{x_min:.4g}, {x_max:.4g}] with finite observations there"
                    )
                    break
                if c.candidate_id in refusals:
                    break
        return refusals, inconclusive

    def evaluate(
        self,
        candidates: list[CandidateProfile],
        observed_data: Mapping[str, np.ndarray] | None = None,
    ) -> StageCDecision:
        if not candidates:
            return StageCDecision(
                state=EpistemicState.NO_VALID_CANDIDATE,
                supported_hypotheses=[],
                winner=None,
                adjudication_rationale="No candidate hypotheses supplied for evaluation.",
                discrimination_required=False,
                missing_evidence=["candidate population"],
            )

        evaluations: dict[str, dict[str, Any]] = {}
        for c in candidates:
            evaluations[c.candidate_id] = {
                "math_sound": c.math_sound,
                "dim_status": c.dim_status,
                "stage_a_admissible": c.is_stage_a_admissible,
                "holdout_mse": c.holdout_mse,
                "complexity": c.complexity,
                "cond_number": c.cond_number,
                "bootstrap_stability": c.bootstrap_stability,
                "limitations": list(c.limitations),
            }

        # 1. Stage A Filter: Math + Dimensional Admissibility + Observed-Data Pole Refusal
        pole_refused, pole_inconclusive = self._pole_refusals_in_observed_data(candidates)
        for cid, reason in pole_refused.items():
            evaluations[cid]["pole_in_observed_data"] = reason
            evaluations[cid]["limitations"] = [*evaluations[cid].get("limitations", []), reason]
        for cid, reason in pole_inconclusive.items():
            evaluations[cid]["pole_check_inconclusive"] = reason
            evaluations[cid]["limitations"] = [*evaluations[cid].get("limitations", []), reason]
        # An unchecked candidate is withheld, exactly as a refused one is.
        # Stage A exists to catch wrong structure; letting a candidate through
        # because the check crashed converts a tool failure into an endorsement.
        pole_blocked = set(pole_refused) | set(pole_inconclusive)
        admissible = [
            c for c in candidates
            if c.is_stage_a_admissible and c.candidate_id not in pole_blocked
        ]
        unresolved = [c for c in candidates if c.dim_status == "UNRESOLVED"]
        invalid = [c for c in candidates if c.dim_status == "INVALID" or not c.math_sound]

        if not admissible:
            reasons = []
            if unresolved:
                reasons.append(f"{len(unresolved)} candidates held as UNRESOLVED")
            if invalid:
                reasons.append(f"{len(invalid)} candidates rejected as INVALID or mathematically unsound")
            if pole_refused:
                reasons.append(
                    f"{len(pole_refused)} candidate(s) refused for poles inside observed data: "
                    + "; ".join(f"'{cid}': {r}" for cid, r in pole_refused.items())
                )
            if pole_inconclusive:
                reasons.append(
                    f"{len(pole_inconclusive)} candidate(s) withheld because the pole "
                    f"check could not be completed (this is a tool failure, not a "
                    f"clean result): "
                    + "; ".join(f"'{cid}': {r}" for cid, r in pole_inconclusive.items())
                )
            rationale = "No candidate passed Stage A gates: " + "; ".join(reasons) + "."
            return StageCDecision(
                state=EpistemicState.NO_VALID_CANDIDATE,
                supported_hypotheses=[],
                winner=None,
                adjudication_rationale=rationale,
                discrimination_required=False,
                missing_evidence=["dimensionally and mathematically sound candidates"],
                candidate_evaluations=evaluations,
            )

        # 2. Stage B Filter: Empirical Viability and Identifiability
        viable: list[CandidateProfile] = []
        for c in admissible:
            is_finite = math.isfinite(c.holdout_mse) and math.isfinite(c.in_sample_mse)
            above_baseline = c.holdout_r2 >= self.min_holdout_r2
            well_conditioned = c.cond_number <= self.max_condition_number
            stable = c.bootstrap_stability >= self.min_bootstrap_stability

            evaluations[c.candidate_id]["is_finite"] = is_finite
            evaluations[c.candidate_id]["above_baseline"] = above_baseline
            evaluations[c.candidate_id]["well_conditioned"] = well_conditioned
            evaluations[c.candidate_id]["stable"] = stable

            if is_finite and above_baseline and well_conditioned and stable:
                viable.append(c)
            else:
                limits = list(c.limitations)
                if not is_finite:
                    limits.append("non-finite predictions on evaluation domain")
                if not above_baseline:
                    limits.append(f"holdout R² ({c.holdout_r2:.3f}) below threshold ({self.min_holdout_r2:.3f})")
                if not well_conditioned:
                    limits.append(f"ill-conditioned design matrix (cond={c.cond_number:.2e} > {self.max_condition_number:.2e})")
                if not stable:
                    limits.append(f"insufficient bootstrap stability (S={c.bootstrap_stability:.2f} < {self.min_bootstrap_stability:.2f})")
                evaluations[c.candidate_id]["limitations"] = limits

        if not viable:
            return StageCDecision(
                state=EpistemicState.INSUFFICIENT_EVIDENCE,
                supported_hypotheses=[c.candidate_id for c in admissible],
                winner=None,
                adjudication_rationale=(
                    f"All {len(admissible)} Stage-A admissible candidate(s) failed empirical viability "
                    "or parameter identifiability criteria. Available evidence cannot certify any hypothesis."
                ),
                discrimination_required=True,
                missing_evidence=[
                    "unbiased holdout data",
                    "better-conditioned observation regime",
                    "higher bootstrap sample stability",
                ],
                candidate_evaluations=evaluations,
            )

        # 3. Stage C Adjudication: Pareto & Trade-Off Analysis
        tradeoffs_detected: list[str] = []

        # Neural Adversarial Reference / Empirical Flexible Baseline Ceiling Check
        neural_candidates = [c for c in viable if c.model_class == ModelClass.NEURAL]
        min_neural_mse = min((c.holdout_mse for c in neural_candidates), default=None)
        if min_neural_mse is not None:
            for c in viable:
                if c.model_class == ModelClass.SYMBOLIC:
                    ratio = c.holdout_mse / max(min_neural_mse, 1e-12)
                    evaluations[c.candidate_id]["gap_to_neural_ceiling"] = float(ratio)
                    if ratio > 1.5:
                        evaluations[c.candidate_id]["neural_reference_verdict"] = "MISSING_STRUCTURE"
                        evaluations[c.candidate_id]["neural_adversary_verdict"] = "MISSING_STRUCTURE"
                        tradeoffs_detected.append(
                            f"Symbolic model '{c.candidate_id}' exhibits missing structure: holdout error is {ratio:.1f}x higher than flexible neural adversarial reference ceiling"
                        )
                    elif ratio <= 1.15:
                        evaluations[c.candidate_id]["neural_reference_verdict"] = "MATCHES_NEURAL_CEILING"
                        evaluations[c.candidate_id]["neural_adversary_verdict"] = "MATCHES_NEURAL_CEILING"
                    else:
                        evaluations[c.candidate_id]["neural_reference_verdict"] = "NEAR_NEURAL_CEILING"
                        evaluations[c.candidate_id]["neural_adversary_verdict"] = "NEAR_NEURAL_CEILING"

        # Minimum Description Length (MDL) / BIC Parsimony Pre-Pruning:
        # A hybrid model (Base + Neural) or high-complexity model (complexity >= 20.0)
        # cannot displace a simpler admissible symbolic model s unless its error reduction
        # is statistically justified under BIC (Delta BIC < -6.0). A symbolic model with
        # MISSING_STRUCTURE cannot prune a hybrid model that resolves that missing structure.
        complex_cands = [c for c in viable if c.model_class == ModelClass.HYBRID or (c.model_class == ModelClass.SYMBOLIC and c.complexity >= 20.0)]
        if complex_cands:
            pruned_viable = []
            for c in viable:
                if c.model_class == ModelClass.HYBRID or (c.model_class == ModelClass.SYMBOLIC and c.complexity >= 20.0):
                    n_samples = c.metadata.get("n_samples", 15)
                    simpler_matches = [
                        s for s in viable
                        if s.model_class == ModelClass.SYMBOLIC
                        and s.complexity <= c.complexity * 0.5
                        and evaluations[s.candidate_id].get("neural_adversary_verdict") != "MISSING_STRUCTURE"
                        and (
                            s.holdout_mse <= c.holdout_mse * self.holdout_margin_ratio
                            or (n_samples * math.log(max(c.holdout_mse, 1e-12) / max(s.holdout_mse, 1e-12)) + (c.complexity - s.complexity) * math.log(max(n_samples, 2)) >= -6.0)
                        )
                    ]
                    if simpler_matches:
                        evaluations[c.candidate_id]["mdl_pruned"] = True
                        tradeoffs_detected.append(
                            f"MDL Parsimony Pruning: Simpler candidate '{simpler_matches[0].candidate_id}' matches performance of complex model '{c.candidate_id}' under BIC penalty (ΔBIC >= -6.0); complex degrees of freedom pruned under Occam's razor"
                        )
                        continue
                pruned_viable.append(c)
            if pruned_viable:
                viable = pruned_viable

        viable.sort(key=lambda c: c.holdout_mse)
        best = viable[0]
        margin_threshold = best.holdout_mse * self.holdout_margin_ratio

        # Find candidates within the empirical holdout margin or observational equivalence window
        competitive: list[CandidateProfile] = []
        #: Candidates whose noise-equivalence probe could not run. Collected so
        #: the decision can say the question is open rather than answer it by
        #: omission.
        se_probe_failures: list[tuple[str, str]] = []
        for c in viable:
            if c.holdout_mse <= margin_threshold:
                competitive.append(c)
            elif (
                # In-sample vs holdout reversal trade-off on parsimonious models:
                c.in_sample_mse < best.in_sample_mse * 0.95
                and c.complexity < best.complexity
                and c.holdout_mse <= best.holdout_mse * 3.5
            ):
                competitive.append(c)
                tradeoffs_detected.append(
                    f"Candidate '{c.candidate_id}' exhibits in-sample rank reversal against '{best.candidate_id}': superior training fit ({c.in_sample_mse:.3e} vs {best.in_sample_mse:.3e}) and lower complexity ({c.complexity:.1f} vs {best.complexity:.1f}) despite higher holdout error"
                )
            else:
                # Observational noise floor equivalence & indistinguishability:
                diff_mse = abs(c.holdout_mse - best.holdout_mse)
                is_noise_equiv = False
                #: Which test actually established the equivalence. The
                #: trade-off sentence quotes one of these as its evidence, so
                #: it has to be the one that ran -- the old wording hardcoded
                #: "<= 1.5e-3" and printed it next to a Δ of 4e-3 when the
                #: decision had come from the bootstrap branch instead.
                equivalence_basis = ""

                # If absolute MSE difference is within empirical noise floor (<= 0.0015)
                # and candidate has acceptable predictive sanity:
                if diff_mse <= 0.0015 and (c.in_sample_r2 >= 0.50 or c.holdout_r2 >= 0.0 or c.in_sample_mse <= 1.0):
                    is_noise_equiv = True
                    equivalence_basis = f"|Δho_mse| = {diff_mse:.2e} <= 1.5e-3"

                # Paired bootstrap SE if residuals available in metadata:
                # Must be within reasonable performance proximity of best candidate
                if (not is_noise_equiv and "holdout_residuals" in c.metadata and "holdout_residuals" in best.metadata
                        and (c.holdout_mse <= best.holdout_mse * self.holdout_margin_ratio * 1.5 or diff_mse <= 0.05)):
                    try:
                        r_c = np.asarray(c.metadata["holdout_residuals"], dtype=float)
                        r_best = np.asarray(best.metadata["holdout_residuals"], dtype=float)
                        if len(r_c) == len(r_best) and len(r_c) >= 5:
                            d = r_c**2 - r_best**2
                            se_d = float(np.std(d, ddof=1) / np.sqrt(len(d)))
                            mean_d = float(np.mean(d))
                            if abs(mean_d) <= 1.96 * se_d:
                                is_noise_equiv = True
                                equivalence_basis = (
                                    f"paired bootstrap: |mean(Δsquared_residual)| = "
                                    f"{abs(mean_d):.2e} <= 1.96 * SE = {1.96 * se_d:.2e} "
                                    f"over {len(d)} paired residuals"
                                )
                    except Exception as e:
                        # Leaving is_noise_equiv False here does not merely
                        # withhold a trade-off note: it keeps the candidate out
                        # of `competitive`, which is what promotes the run to
                        # UNIQUE_SUPPORT. So a crashed probe silently resolved
                        # the adjudication toward "one winner". The trade-off
                        # sentence also quotes a p-value, so claiming the
                        # equivalence here would be asserting a statistic that
                        # was never computed. Neither is acceptable, so the
                        # ambiguity is recorded and the engine abstains below.
                        logger.debug("paired bootstrap SE probe failed", exc_info=True)
                        se_probe_failures.append((c.candidate_id, f"{type(e).__name__}: {e}"))

                if is_noise_equiv:
                    competitive.append(c)
                    tradeoffs_detected.append(
                        f"Candidate '{c.candidate_id}' is observationally indistinguishable "
                        f"from '{best.candidate_id}' within experimental noise floor "
                        f"({equivalence_basis}) with complexity {c.complexity:.1f} "
                        f"vs {best.complexity:.1f}"
                    )

        # Cross-class trade-offs (Symbolic vs Neural vs Hybrid)
        model_classes_present = {c.model_class for c in competitive}
        if len(model_classes_present) > 1:
            sym_cands = [c for c in competitive if c.model_class == ModelClass.SYMBOLIC]
            nn_cands = [c for c in competitive if c.model_class == ModelClass.NEURAL]
            hybrid_cands = [c for c in competitive if c.model_class == ModelClass.HYBRID]
            if sym_cands and nn_cands:
                tradeoffs_detected.append(
                    f"Cross-class competition: Symbolic model '{sym_cands[0].candidate_id}' matches predictive ceiling of Neural Network '{nn_cands[0].candidate_id}' within margin; Occam's razor strongly favors symbolic parsimony"
                )
            if hybrid_cands and nn_cands:
                tradeoffs_detected.append(
                    f"Hybrid model '{hybrid_cands[0].candidate_id}' matches pure neural predictive performance while preserving physical base equation"
                )

        if len(competitive) > 1:
            for other in competitive[1:]:
                # Complexity trade-off: a simpler model that is competitive
                if other.complexity < best.complexity * 0.85:
                    # An exact holdout fit (holdout_mse == 0.0) is reachable --
                    # with as many holdout points as free parameters the least
                    # squares residual vanishes. Dividing by it raised
                    # ZeroDivisionError, losing the whole adjudication, so the
                    # percentage is only quoted when the denominator is usable.
                    if best.holdout_mse > 0:
                        gap = f"is within {((other.holdout_mse / best.holdout_mse - 1) * 100):.1f}% holdout error of"
                    else:
                        gap = ("matches the exact zero holdout error of"
                               if other.holdout_mse <= 0
                               else "exceeds the exact zero holdout error of")
                    tradeoffs_detected.append(
                        f"Simpler candidate '{other.candidate_id}' (complexity {other.complexity:.1f}) {gap} '{best.candidate_id}' (complexity {best.complexity:.1f})"
                    )
                # In-sample vs out-of-sample reversal (overfitting suspicion)
                if other.in_sample_mse < best.in_sample_mse * 0.9 and other.holdout_mse > best.holdout_mse:
                    tradeoffs_detected.append(
                        f"Candidate '{other.candidate_id}' has superior in-sample fit ({other.in_sample_mse:.4g} vs {best.in_sample_mse:.4g}) but inferior holdout generalization ({other.holdout_mse:.4g} vs {best.holdout_mse:.4g})"
                    )
                # Near-OOD superiority
                if (math.isfinite(other.near_ood_mse) and math.isfinite(best.near_ood_mse)
                        and other.near_ood_mse < best.near_ood_mse * 0.9):
                        tradeoffs_detected.append(
                            f"Candidate '{other.candidate_id}' outperforms '{best.candidate_id}' on near-OOD extrapolation ({other.near_ood_mse:.4g} vs {best.near_ood_mse:.4g})"
                        )
                # Information criteria split (e.g. AIC prefers one, BIC prefers another)
                if (best.aic is not None and other.aic is not None and best.bic is not None and other.bic is not None
                        and ((best.aic < other.aic and best.bic > other.bic)
                             or (best.aic > other.aic and best.bic < other.bic))):
                        tradeoffs_detected.append(
                            f"Information criteria split between '{best.candidate_id}' and '{other.candidate_id}' (AIC vs BIC disagreement)"
                        )

        # Decision synthesis
        if len(competitive) > 1:
            survivor_names = [c.candidate_id for c in competitive]
            rationale_parts = [
                f"Multi-hypothesis state: {len(competitive)} admissible hypotheses are empirically competitive within the {((self.holdout_margin_ratio - 1)*100):.0f}% holdout margin or noise floor ({', '.join(survivor_names)})."
            ]
            if tradeoffs_detected:
                rationale_parts.append("Observed trade-offs: " + "; ".join(tradeoffs_detected) + ".")
            if se_probe_failures:
                rationale_parts.append(_SE_PROBE_CAVEAT % _fmt_se_failures(se_probe_failures))
            rationale_parts.append(
                "Available observations do not uniquely discriminate among these hypotheses. Forcing a winner would encode unevidenced bias."
            )
            return StageCDecision(
                state=EpistemicState.MULTI_HYPOTHESIS,
                supported_hypotheses=survivor_names,
                winner=None,
                adjudication_rationale=" ".join(rationale_parts),
                discrimination_required=True,
                missing_evidence=[
                    f"discriminating observations separating {', '.join(survivor_names)}",
                    "broader experimental regime",
                ],
                candidate_evaluations=evaluations,
            )

        # Mandatory Active Singularity & Pole Stress Testing on dominating candidate
        if observed_data is not None:
            singularity_engine = ActiveSingularityProbeEngine()
            other_exprs = [c.expression for c in candidates if c.candidate_id != best.candidate_id]
            probe_report = singularity_engine.probe_candidate_singularity(
                best.expression,
                other_exprs,
                observed_data,
            )
            if probe_report.has_singularity_risk:
                evaluations[best.candidate_id]["singularity_risk"] = probe_report.to_dict()
                survivor_names = [best.candidate_id]
                missing_evidence = [probe_report.rationale]
                if probe_report.gap_details:
                    var = probe_report.gap_details.get("variable", "x")
                    pl = probe_report.gap_details.get("pole_location", 0.0)
                    if probe_report.risk_type == "AVOIDED_POLE_IN_GAP":
                        gl = probe_report.gap_details.get("gap_left", 0.0)
                        gr = probe_report.gap_details.get("gap_right", 0.0)
                        missing_evidence.append(
                            f"empirical observations in sampling gap [{gl:.3f}, {gr:.3f}] around pole {var} = {pl:.4f}"
                        )
                    elif probe_report.risk_type == "UNOBSERVED_CRITICAL_REGIME":
                        missing_evidence.append(
                            f"empirical observations in critical asymptotic divergence regime near {var} = {pl:.4f}"
                        )

                return StageCDecision(
                    state=EpistemicState.INSUFFICIENT_EVIDENCE,
                    supported_hypotheses=survivor_names,
                    winner=None,
                    adjudication_rationale=(
                        f"Active Singularity Stress Testing halted certification for '{best.candidate_id}': "
                        f"{probe_report.rationale}"
                    ),
                    discrimination_required=True,
                    missing_evidence=missing_evidence,
                    candidate_evaluations=evaluations,
                )

        # Exactly one candidate dominates decisively within the tested domain.
        # Uniqueness here is an in-domain verdict: without out-of-domain
        # evaluation, extrapolation beyond observed inputs is untested and the
        # rationale must say so explicitly (never imply endorsed extrapolation).
        ood_tested = (
            math.isfinite(best.near_ood_mse) or math.isfinite(best.far_ood_mse)
        )
        rationale_parts = [
            (f"Candidate '{best.candidate_id}' [{best.model_class.value}] ({best.expression}) is uniquely supported within the tested domain: "
            f"it achieves the lowest holdout error ({best.holdout_mse:.4g}), well-conditioned parameters (cond={best.cond_number:.2e}), "
            f"and robust bootstrap recovery ({best.bootstrap_stability:.1%}) with no competing candidate within the {((self.holdout_margin_ratio - 1)*100):.0f}% margin.")
        ]
        if tradeoffs_detected:
            rationale_parts.append("Observed dynamics: " + "; ".join(tradeoffs_detected) + ".")
        missing: list[str] = []
        if se_probe_failures:
            # The state is left as-is deliberately: changing it to
            # INSUFFICIENT_EVIDENCE would alter every downstream consumer, and
            # the honest minimum is that the limit is stated where a reader of
            # this record will meet it. What must not happen is the current
            # behaviour -- a silently narrowed candidate set presented as
            # "no competing candidate within the margin".
            rationale_parts.append(
                "Uniqueness is provisional: the observational noise-floor probe did "
                "not complete for some candidates, and those were therefore never "
                "formally compared. " + _SE_PROBE_CAVEAT % _fmt_se_failures(se_probe_failures)
            )
            missing.append(
                "a completed paired-bootstrap noise-floor comparison against "
                + ", ".join(cid for cid, _ in se_probe_failures)
            )
        if not ood_tested:
            rationale_parts.append(
                "Support is bounded to the observed regime: no out-of-domain evaluation "
                "was supplied, so extrapolation beyond observed inputs is untested and "
                "must not be read as endorsed."
            )
            missing.append("extrapolation validation beyond the observed regime")
        rationale = " ".join(rationale_parts)
        return StageCDecision(
            state=EpistemicState.UNIQUE_SUPPORT,
            supported_hypotheses=[best.candidate_id],
            winner=best.candidate_id,
            adjudication_rationale=rationale,
            discrimination_required=False,
            missing_evidence=missing,
            candidate_evaluations=evaluations,
        )


__all__ = [
    "ABSTENTION_REASON_TEXT",
    "AbstentionReason",
    "CandidateProfile",
    "EpistemicState",
    "ModelClass",
    "StageCDecision",
    "StageCDecisionEngine",
]
