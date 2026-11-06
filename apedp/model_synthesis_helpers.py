"""Types, constants and scoring helpers for ``apedp.model_synthesis``.

Extracted from ``apedp/model_synthesis.py`` when it passed 2,100 lines. The
bodies are byte-identical; ``model_synthesis`` imports them back and re-exports
the public names, so ``from apedp.model_synthesis import SynthesizedModel``
still resolves and nothing downstream changed.

``SynthesizedModel`` and ``SynthesisDiagnostics`` are the public half of this
module. The rest is the scoring machinery the synthesizer calls: BIC re-ranking
(disabled pending the frozen-protocol decision recorded beside it), amplitude
caps, adversarial classification, abstention wording, input normalisation, the
noise floor, and regime identifiability.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
import sympy as sp

from apedp.decision.stage_c import (
    ABSTENTION_REASON_TEXT,
    AbstentionReason,
    EpistemicState,
)

logger = logging.getLogger("apedp.model_synthesis")



# Bayesian Information Criterion penalty per free parameter, as a multiplier
# on val_mse: BIC ranking is equivalent to ranking on
#     val_mse + k * ln(n) / n
# with k the free-parameter count and n the validation sample count. Expressed
# as a multiplier so it composes with the existing val_mse-sorted lists
# without changing their units.
#
# Provenance: the frozen EXT-BENCH-60 protocol diagnosed both genuine
# RECOVER misses (E16, P07) as rank-by-raw-val-MSE artifacts, where a
# 5-parameter polynomial edged out the true 2-parameter exponential by
# ~1e-04 MSE on noisy validation data, and prescribes BIC re-ranking as the
# next-round fix. See docs/PREREGISTRATION_FROZEN_EXTERNAL_BENCHMARK.md §9.5.
# Not applied to the locked round.
_BIC_RERANK_ENABLED = False
_BIC_RERANK_TOP_K = 8


def _bic_rerank(models: list[SynthesizedModel], n_val: int,
                top_k: int = _BIC_RERANK_TOP_K) -> SynthesizedModel | None:
    """Returns the model that wins a BIC-penalized contest, or None.

    Extracted as a module-level function so the criterion is testable on its
    own, without a full synthesis run per assertion.

    Ranking is on ``val_mse * (1 + k*ln(n_val)/n_val)``, which is monotone in
    the usual BIC ``n*ln(RSS/n) + k*ln(n)`` for fixed n. Three guards keep this
    from doing harm, each of which is pinned by a test:

    - **top-K only.** A candidate outside the val-MSE top-K has not earned the
      right to displace the incumbent on a penalized criterion.
    - **incumbent competes.** The current best is scored alongside the others
      rather than assumed to win, so a penalty too coarse to separate a pair
      leaves the incumbent in place.
    - **strict margin.** A tie must not flip the result. Without this the
      re-rank would be a coin toss dressed up as a criterion.
    """
    if not models:
        return None
    k_max = min(top_k, len(models))

    def _bic(m: SynthesizedModel) -> float:
        # float() around the product, not decoration: numpy's stubs type
        # np.log(int) as Any, and an Any-valued return silently propagates
        # into every comparison that sorts on it.
        k = max(len(m.parameters), 1)
        return float(m.val_mse * (1.0 + k * np.log(max(n_val, 1)) / max(n_val, 1)))

    head = models[:k_max]
    best_bic, best_idx, best_m = min(
        ((_bic(m), i, m) for i, m in enumerate(head)), key=lambda t: t[0],
    )
    if best_idx != 0 and best_bic < _bic(head[0]) * (1.0 - 1e-9):
        return best_m
    return head[0]


class SynthesisDiagnostics(TypedDict, total=False):
    """Typed diagnostics payload for SynthesizedModel (replaces untyped dict)."""

    template: str
    family: str
    val_r2: float
    residual_autocorr: float
    rationale: str
    stage_d_refined: bool
    stage_d_form: str
    stage_d_params: dict[str, float]
    coarse_val_mse: float
    coarse_expression: str
    stage_d2_evidence: dict[str, Any]
    physics_compliance: dict[str, Any]
    parameter_uncertainty: dict[str, Any]
    #: Primary abstention code, duplicated from the field of the same name so
    #: a consumer reading a serialized diagnostics dict -- which is what the
    #: benchmark writes -- does not have to know the dataclass exists.
    abstention_reason: str
    #: Every cause that fired, not just the primary. The gate is an OR over
    #: three criteria, so reporting one of them loses the others.
    abstention_reasons: list[str]


@dataclass(frozen=True)
class SynthesizedModel:
    """Represents an autonomously synthesized physical equation candidate.

    Empirical confidence (epistemic_class C1..C5) is independent of proof
    standing (proof_claim P0..P2). Finite-sample synthesis never exceeds P0;
    P1/P2 are reserved for a future proof pipeline. Joint label cp_label
    (e.g. "C1,P0") must be used in math reports to prevent C1=proven misread.
    """
    expression_str: str
    sympy_expr: sp.Expr
    parameters: dict[str, float]
    train_mse: float
    val_mse: float
    near_ood_mse: float | None = None
    far_ood_mse: float | None = None
    condition_number: float = 1.0
    has_singularity_risk: bool = False
    epistemic_state: EpistemicState | None = None
    structural_family: str = "unknown"
    diagnostics: SynthesisDiagnostics = field(default_factory=SynthesisDiagnostics)
    proof_claim: str = "P0"
    #: Which gate withheld endorsement, and why. ``None`` means the model was
    #: not withheld -- a cleared model has no reason, and defaulting to a
    #: placeholder here would make "cleared" and "abstained for an unknown
    #: reason" indistinguishable to every consumer downstream.
    abstention_reason: AbstentionReason | None = None

    @property
    def epistemic_class(self) -> str:
        """Maps Stage C EpistemicState to formal tournament epistemic class C1..C5."""
        if self.epistemic_state == EpistemicState.UNIQUE_SUPPORT:
            return "C1"
        elif self.epistemic_state == EpistemicState.MULTI_HYPOTHESIS:
            return "C2"
        elif self.epistemic_state == EpistemicState.INSUFFICIENT_EVIDENCE:
            return "C3"
        elif self.epistemic_state == EpistemicState.NO_VALID_CANDIDATE:
            return "C5"
        return "C3"

    @property
    def cp_label(self) -> str:
        """Joint empirical/proof label, e.g. "C1,P0"."""
        return f"{self.epistemic_class},{self.proof_claim}"


# Families whose leading amplitude is an asymptote that may lie far above
# observed y (Arrhenius pre-exponential on the early foot, logistic
# saturation level, envelope onsets). The generic amplitude cap would deny
# identifiable truth (observed: A=1e5 excluded by cap 6350 at y_max=127).
_ASYMPTOTE_AMPLITUDE_FAMILIES = frozenset({
    "exponential_decay", "logistic_saturation", "exponential_envelope",
    "rational_exponential_envelope", "arrhenius_decay",
})


def _amplitude_cap(y_max: float, family_label: str) -> float:
    """Upper bound for amplitude-like parameters, asymptote-aware."""
    cap = max(100.0, 50.0 * y_max)
    if family_label in _ASYMPTOTE_AMPLITUDE_FAMILIES and y_max > 0.0:
        cap = max(cap, 1000.0 * y_max)
    return cap


# Adversarial-gate thresholds, named because they are now read in two places:
# the classifier below, and any consumer that wants to reason about the gate
# without re-deriving it. They were inline literals inside a single boolean,
# which is exactly the shape that lets a threshold change without anyone
# noticing the reason code stopped matching the decision.
_SHOCK_R2_GATE = 0.96
_AUTOCORR_GATE = 0.45
_AUTOCORR_R2_GATE = 0.99
_LOW_R2_GATE = 0.85

#: Most-specific-first. The primary reported reason is the first member of the
#: returned tuple that fired, so a shock front is reported as a shock front
#: rather than as the low-R-squared that also happened to be true.
_ABSTENTION_PRIORITY = (
    AbstentionReason.NO_CANDIDATE_FITTED,
    AbstentionReason.SHOCK_DISCONTINUITY,
    AbstentionReason.RESIDUAL_AUTOCORRELATION,
    AbstentionReason.LOW_VALIDATION_R2,
)


def _classify_adversarial(
    *,
    n_candidates: int,
    has_shock: bool,
    top_r2: float,
    top_autocorr: float,
) -> tuple[AbstentionReason | None, tuple[AbstentionReason, ...]]:
    """Decides *and* explains the adversarial abstention gate, in one place.

    Returns ``(primary, all_fired)``; ``primary is None`` means the gate
    clears. The gate asks this function rather than evaluating its own
    boolean, so the decision and the reported cause cannot disagree.

    The empty-candidate case is special and was previously invisible. The gate
    used to test ``not fitted_models`` alongside the R-squared criteria, and
    the criteria were fed the ``-1.0`` sentinel that stands in for "no top
    model" -- so a run in which every template failed to converge was
    recorded as ``LOW_VALIDATION_R2``, a confident claim that the data is
    chaotic, made by a model that does not exist. An empty candidate set is
    now its own reason and short-circuits the rest, because there is no
    residual to autocorrelate and no R-squared to be low.
    """
    if n_candidates == 0:
        return AbstentionReason.NO_CANDIDATE_FITTED, (AbstentionReason.NO_CANDIDATE_FITTED,)

    fired: set[AbstentionReason] = set()
    if has_shock and top_r2 < _SHOCK_R2_GATE:
        fired.add(AbstentionReason.SHOCK_DISCONTINUITY)
    if top_autocorr > _AUTOCORR_GATE and top_r2 < _AUTOCORR_R2_GATE:
        fired.add(AbstentionReason.RESIDUAL_AUTOCORRELATION)
    if top_r2 < _LOW_R2_GATE:
        fired.add(AbstentionReason.LOW_VALIDATION_R2)

    if not fired:
        return None, ()
    ordered = tuple(r for r in _ABSTENTION_PRIORITY if r in fired)
    return ordered[0], ordered


def _abstention_rationale(reasons: tuple[AbstentionReason, ...]) -> str:
    """Prose built from the codes, so a message cannot describe a different cause.

    Every cause that fired is named. Reporting only the primary would lose
    information the gate actually used -- a shock front that also blew the
    R-squared floor is one finding, not two, and a reader who saw only
    ``SHOCK_DISCONTINUITY`` would not know the fit was also bad.
    """
    if not reasons:
        return ""
    codes = ", ".join(r.value for r in reasons)
    body = " ".join(ABSTENTION_REASON_TEXT[r] for r in reasons)
    return f"Formal abstention [{codes}]: {body}"


# Stage D amplitude/rate roles for input-normalization map-back:
# P_true = P_tilde / x_scale**e, where e is either a constant (rates and
# frequencies, which carry 1/x dimension) or the name of the form's fitted
# exponent parameter holding e's value.
_XSCALE_ROLES: dict[str, dict[str, str | float]] = {
    "power": {"A": "p"},
    "exp": {"k": 1.0},
    "undamped": {"w": 1.0},
    "damped": {"k": 1.0, "w": 1.0},
    "rational": {"A": "p", "B": "q"},
    "rational_saturation": {"A": "p", "B": "q"},
    "envelope": {"A": "p", "B": "q", "C": "r"},
    "exp_env": {"A": "p", "C": "r"},
    "logbound": {"A": "a", "B": "b"},
}

_NORMALIZE_X_FAMILIES = frozenset({
    "power_law", "rational_transition", "rational_saturation",
    "exponential_envelope", "exponential_decay",
    "logarithmic_boundary_layer", "rational_exponential_envelope",
    "linear_baseline", "polynomial_baseline",
})


def _input_normalization_scale(
    family_label: str,
    struct_template: str,
    x_values: np.ndarray,
) -> float:
    """Choose an x scale equivariantly, excluding templates without map-back."""
    if (
        family_label not in _NORMALIZE_X_FAMILIES
        or "abs(" in struct_template
        or ("log(x)" in struct_template and "log(1" not in struct_template)
    ):
        return 1.0
    x_abs_max = float(np.max(np.abs(x_values))) if len(x_values) else 1.0
    return x_abs_max if np.isfinite(x_abs_max) and x_abs_max > 0.0 else 1.0


#: Minimum samples before :func:`_regime_identifiability` will let
#: ``detect_regimes`` classify at all.
#:
#: This is derived from the sharpest feature the detector claims to resolve,
#: not chosen by experiment. A turnover is reported only when an interior
#: extremum is bracketed by samples on *both* sides, and each branch is then
#: fitted with a 2-parameter local power law, which needs 3 points to identify
#: an intercept and a slope. That is 3 (rising branch) + 2 (bracketing the
#: extremum) + 3 (falling branch) = 8. Below 8 the detector cannot tell "no
#: turnover" apart from "the turnover fell between two samples", so every
#: other verdict it returns is a coin flip dressed as a measurement.
MIN_RESOLVABLE_REGIME_SAMPLES = 8

#: Signal span, in multiples of the local noise floor, required before the
#: detector will report the *absence* of a regime feature.
_NOISE_FLOOR_SPAN_RATIO = 3.0

#: MAD-to-sigma scale factor for a Gaussian.
_MAD_TO_SIGMA = 1.4826

#: ``sqrt(6)``: the standard deviation of a second difference of iid noise with
#: unit variance is ``sqrt(1 + 4 + 1)``.
_SECOND_DIFF_SCALE = np.sqrt(6.0)

#: Oscillation frequencies proposed as *periods across the observed window*
#: rather than in the caller's x units. See the S4 branch of
#: :meth:`AutonomousModelSynthesizer.generate_candidate_structures` for why an
#: absolute grid alone is a prior on the caller's units.
#:
#: Range: 0.4 periods across the window (below this a single trough is not
#: distinguishable from a peak) to 4 (past which a smooth trend is not separable
#: from a fast oscillation without more resolution than these grids provide).
#: Spacing is geometric at a ~1.24 ratio, coarser than the absolute grid's ~1.12
#: over 1.5-4.0, because the same 12 slots have to cover a wider range: the
#: absolute grid's range is fixed by the corpus, this one's is not.
_SPAN_RELATIVE_OMEGA_PERIODS = (
    0.4, 0.5, 0.6, 0.75, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.2, 4.0,
)

#: Stage D's frequency box, *widened* to these many periods swept across the
#: observed window and then unioned with the legacy absolute range. The lower
#: end is well below the coarsest coarse candidate and the upper end above the
#: finest, so widening to reach them cannot exclude anything the pool proposed.
#: Both are in periods, not in 1/x, so they mean the same thing at every scale.
_W_PERIODS_LO = 0.02
_W_PERIODS_HI = 8.0

#: Whether a challenger that is much better on the near-OOD split may displace
#: the incumbent *even when it is more complex*.
#:
#: This is the engine default for callers with no protocol of their own. It
#: used to be documented as though one global constant settled the matter,
#: because the benchmark adapter's contract for the near-OOD signal is that it
#: "can only disqualify a more complex model (parsimony tiebreak) ... It never
#: promotes anything" (benchmarks/external_60_adapters.py).
#:
#: That conflated a protocol constraint with an engine capability, and the
#: conflation had a measurable cost. SYNTH-50 S11 asserts recovery of
#: ``logarithmic_boundary_layer`` and does not get it: the truth sits at rank 2
#: behind a surrogate, separated by 1.4% relative val_mse, and the parsimony
#: tiebreak prefers the surrogate. But S11 is a SYNTH-50 problem -- no adapter
#: contract governs it, and nothing about it is scored by a round whose
#: protocol says "val-MSE selection only". A rule that belongs to one frozen
#: protocol does not belong in the engine, so it moved to where that protocol
#: is: ``synthesize(allow_near_ood_promote=...)``, which APEDPAdapter pins to
#: False. The frozen round is now protected by the adapter rather than by a
#: constant any caller could flip.
#:
#: Nothing here says promotion is bad. Refereed on all 60 EXT-BENCH problems
#: with far-OOD MSE (which the selector never sees) as the judge: it changes 5
#: families (C41, C43, P08, R25, X52) and across the whole round is mildly
#: *positive* -- far-OOD better on 13, worse on 5, unchanged on 34, median
#: 2.19e-6 against 2.23e-6. But C41's far-OOD NRMSE regresses 0.0153 -> 0.347
#: and X52 becomes a new abstention, so enabling it globally is a change to
#: every caller in the repo and still wants the fresh preregistered round
#: protocol §6.3 requires. It is also not the cause of the P05/P08 family
#: regressions; the unfloored veto below is. Full account in
#: docs/NEAR_OOD_PROMOTION.md.
_NEAR_OOD_MAY_PROMOTE = False

#: Additive floor on the near-OOD veto, in near-OOD MSE units. ``"auto"`` means
#: the measured train noise floor, ``best.train_mse / n_val`` -- the same
#: quantity the parsimony margin already uses. ``0.0`` restores the bare ratio.
#:
#: The veto this floors disqualifies a candidate when
#: ``m.near_ood_mse > best.near_ood_mse * 1.5``. A ratio has no absolute scale,
#: so when the incumbent's near-OOD MSE is small the threshold is small too, and
#: the veto can be decided by a difference no measurement would support.
#:
#: EXT-BENCH P05 is the measured case. The two near-OOD MSEs the veto compares
#: are 3.9136e-3 (a four-parameter ``rational_transition``) and 5.8819e-3 (the
#: correct one-parameter ``power_law``, for ``power-1.5``). The threshold is
#: 5.8704e-3. **The veto fires by 0.20%**, and both MSEs sit ~25x below the
#: data's own train noise floor (~0.097), so the difference is not resolvable.
#: The power law's validation MSE is 0.085677 against the incumbent's 0.083419,
#: a gap of 0.00226 that is well inside the 0.01251 tie margin -- so the parsimony
#: rule is entitled to prefer it and is overruled by a sub-noise ratio.
#:
#: The parsimony margin immediately above this veto is already floored, by the
#: same noise estimate, for exactly this reason. Two adjacent decisions in one
#: loop, one treating a sub-noise difference as evidence and the other not, is
#: the defect.
#:
#: Measured on all 60: this changes exactly one recorded family (P05, back to
#: ``power_law``), taking named-family accuracy 23/27 -> 24/27, with no
#: abstention change. See docs/NEAR_OOD_PROMOTION.md.
_NEAR_OOD_VETO_FLOOR: float | str = "auto"


def _robust_noise_floor(ys: np.ndarray) -> float:
    """Noise scale in **y units**, invariant to any rescaling of ``x``.

    Estimated from second differences: ``1.4826 * median(|diff(y, 2)|) / sqrt(6)``.
    For iid noise the second difference has variance ``6 * sigma^2``, so this is
    a consistent noise scale; for anything locally quadratic the second
    difference annihilates it, leaving only the high-frequency content.

    Why this estimator, given three rejected ones:

    * ``median(|dy/dx|)`` (the original) is a *slope*, in y/x units. Compared
      against a y-unit span it is dimensionally meaningless, and its verdict
      depended on the units of x: rescaling x by 10 divided the "noise floor"
      by 10 and flipped the answer on 11 of the 60 EXT-BENCH problems -- the
      same curve abstaining at x*1 and passing at x*10, including a damped
      oscillation that lost its UNIQUE_SUPPORT.
    * A local-quadratic residual reads a *coarse oscillation* as noise, because
      a quadratic tracks only part of a half-cycle. That inflated the floor for
      exactly the damped/undamped block and required an ad-hoc "is this
      oscillatory?" carve-out to undo.
    * Second differences need no carve-out: a smooth oscillation is smooth at
      the sampling scale, so it cancels. The same third-difference idea is
      already used for the same purpose in ``apedp_dyn.gates.intervention``.

    The median (not the mean) makes it robust to isolated spikes.
    """
    d2 = np.diff(np.asarray(ys, dtype=float), n=2)
    d2 = d2[np.isfinite(d2)]
    if d2.size == 0:
        return 0.0
    return float(_MAD_TO_SIGMA * np.median(np.abs(d2)) / _SECOND_DIFF_SCALE)


def _regime_identifiability(xs: np.ndarray, ys: np.ndarray) -> tuple[bool, str]:
    """Decide whether qualitative regime structure is estimable from ``(xs, ys)``.

    Returns ``(is_underspecified, reason)``.

    This deliberately replaces an earlier ``len(x) <= 8`` row-count gate. That
    gate was a bare constant with no relation to the estimand: it abstained on
    clean, well-separated data for exactly the same inputs it would classify
    one sample longer, and it had been tuned to line up with the three
    sparse problems in the benchmark corpus. Identifiability here is a
    property of the *design and the noise*, so it is measured:

    1. At least three distinct sampling points, or no local model is estimable
       at all.
    2. The signal must vary by more than the local noise floor, measured as a
       robust span (IQR) against a robust floor. Below that, "there is no
       turnover" is indistinguishable from "the turnover is under the noise".
       Both terms are in y units (see :func:`_robust_noise_floor`), so this
       comparison does not depend on the units of x.
    3. Enough samples to resolve the sharpest feature (see
       :data:`MIN_RESOLVABLE_REGIME_SAMPLES`).
    """
    n = xs.size
    if n < 3 or np.unique(xs).size < 3:
        return True, (
            f"{n} samples over {np.unique(xs).size} distinct x values: "
            "no local model is estimable"
        )

    noise = _robust_noise_floor(ys)
    # Robust span: the interquartile range, not peak-to-peak. ptp(ys) is set
    # by the two most extreme samples, which on noisy data are noise: a signal
    # buried under its own noise floor then "passed" a span that the noise had
    # already widened for it. The IQR ignores the tails, so it measures the
    # bulk of the variation and is comparable to a MAD-based floor.
    #
    # The 3x ratio now carries real meaning: for a noise-dominated record the
    # IQR is ~1.35 sigma while the floor is ~1.0 sigma, so 3x cannot be
    # cleared by noise alone. It also makes a constant record abstain
    # deterministically (IQR is exactly 0.0, never a round-off coin flip).
    #
    # No oscillation carve-out is needed. An earlier local-model estimator read
    # a coarse oscillation as noise and needed one; second differences cancel
    # any locally quadratic trend, so an oscillation no longer inflates its own
    # floor and the exemption would only mask genuinely unresolvable records.
    span = float(np.subtract(*np.percentile(ys, [75, 25])))
    if span <= 0.0:
        # A record with no variation at all is unidentifiable by definition,
        # and independently of the noise estimate. Guarding it explicitly
        # matters because the noise rule is skipped when noise == 0 (a
        # perfectly constant record yields a second difference of exactly
        # 0.0), which would otherwise let a flat signal through to synthesis
        # and produce a confident expression for a record that says nothing.
        # This is the adv-trig-identity shape in EXT-BENCH block A.
        return True, "signal span is 0: no variation to identify a regime from"
    if noise > 0.0 and span < _NOISE_FLOOR_SPAN_RATIO * noise:
        return True, (
            f"signal span {span:.3g} is under {_NOISE_FLOOR_SPAN_RATIO:g}x the local "
            f"noise floor {noise:.3g}: regime features are not resolvable"
        )

    if n < MIN_RESOLVABLE_REGIME_SAMPLES:
        return True, (
            f"{n} samples cannot resolve the {MIN_RESOLVABLE_REGIME_SAMPLES} needed to "
            "bracket an interior extremum and fit a power law to both branches"
        )
    return False, ""
