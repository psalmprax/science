"""Epistemic adjudication: the last barrier before model promotion.

Pipeline position (after scoring, before promotion):

    ranking -> EpistemicStatus -> promotion (or abstention).

Ranking and adjudication are different operations: a candidate can be rank 1
numerically yet fail to earn UNIQUE_SUPPORT when a structurally distinct rival
is statistically indistinguishable. Complexity alone never earns promotion.

Resolution rule (D2, heuristic — recorded, not a universal theorem):
  ΔRMSE = RMSE_2 − RMSE_1 (holdout intervention evidence).
  If ΔRMSE ≤ τ_epi for a structurally distinct rival → MULTI_HYPOTHESIS.

τ_epi is a RESOLUTION floor derived from the holdout record's own scale and
noise floor — never from gate 4's ACCEPTANCE tolerance. The two answer
different questions ("is this model good enough?" vs "can these two be told
apart?"), and reusing one constant for both made abstention depend on signal
amplitude and invert with gate-4 leniency: loosening gate 4 so the correct
model passes at high noise also made adjudication more credulous about rivals.
Later upgrades (bootstrap uncertainty, likelihood-ratio/BIC, trajectory CIs)
slot in here without changing the architecture.

The scale that τ_epi is built from can be passed in directly
(:func:`scale_from_observations`). Reading it out of ``gate4_status``
provenance is a fallback, not a requirement: otherwise the resolution floor's
AVAILABILITY is silently coupled to gate 4 having run, and a candidate scored
by a different path could never earn UNIQUE_SUPPORT however decisive its
evidence. Which route supplied the scale is recorded on the Adjudication.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class EpistemicStatus(str, Enum):
    UNIQUE_SUPPORT = "unique_support"
    MULTI_HYPOTHESIS = "multi_hypothesis"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_VALID_CANDIDATE = "no_valid_candidate"
    EXTRAPOLATION_UNJUSTIFIED = "extrapolation_unjustified"


DYNAMIC_EPISTEMIC_VERSION = "bootstrap-v3+solefid-v5"
"""Version stamp for the adjudication contract (bootstrap tiebreaker + the
sole-survivor fidelity and whiteness guards). Recorded into artifacts by
dyn_sweep and dyn_tournament so a stored verdict can be traced to the rules
that produced it."""


SOLE_SURVIVOR_GAMMA = 3.0
"""Frozen global absolute-fidelity factor (sole-survivor Guard 1).

A sole numerical survivor must satisfy

    RMSE_holdout <= SOLE_SURVIVOR_GAMMA * sigma_holdout

where sigma is the calibrated holdout noise scale (sigma / CAL_SIGMA). The
absolute ceiling is the whole contract: a relative ceiling on the observation
RMS was tried and withdrawn, because at degenerate noise scales it admits
fits the fidelity contract exists to refuse.
"""


RESID_RHO_FLAG = 0.5
"""Residual-whiteness barrier (sole-survivor Guard 2).

Applied only after Guard 1 passes. A correct model leaves residuals with no
autocorrelation left in them; |rho1| at or above this flag means structure
remains, so a numerically perfect fit is not yet explanatory support.
"""


WEAK_MARGIN_MULTIPLE = 2.0
"""Diagnostic threshold for a boundary-case promotion, in units of tau.

A promotion whose separation clears tau by less than this multiple is still
formally permitted by the frozen rule -- the gate is unchanged -- but is
reported as WEAK_PROMOTION_MARGIN / promotion_confidence='low_confidence'.

Diagnostic only, and deliberately *not* a gate: introducing a hard margin
requirement now would be post-hoc rule modification driven by an
inconvenient case. The threshold qualifies the reporting, not the verdict.
"""


ABSOLUTE_EPS = 1e-12
"""Floor used when a record reports a degenerate (non-positive) noise scale."""


CAPABILITY_ABSENT = "CAPABILITY_ABSENT"
"""Marker raised when a requested adjudication capability is not compiled in.

The restored guards exist as separate modules that this file imports lazily.
That laziness is deliberate (import order in this package is circular), but it
means a build can lack the bootstrap tiebreaker entirely and still import
cleanly. A caller that opted into ``bootstrap=True`` would then have its
request silently dropped and would receive a verdict indistinguishable from
one where the tiebreaker ran and agreed.

So the request fails loudly instead. Silently downgrading an explicit opt-in
is worse than refusing it: the caller asked for a stronger guarantee, and the
whole point of the tiebreaker is that a point margin can straddle zero.
"""


def bootstrap_available() -> bool:
    """Whether the bootstrap tiebreaker is compiled into this build.

    Probed by attempting the import the tiebreaker path actually performs,
    not by inspecting this module's namespace: a lazy import means the
    capability lives in another module, so the only honest test is whether
    that module resolves and carries the entry point.
    """
    try:
        from apedp_dyn.epistemic.bootstrap import tiebreak  # noqa: F401
    except Exception:  # noqa: BLE001 -- any failure means not available
        return False
    return True


class EpistemicContract(str, Enum):
    """How complete the adjudication contract was for one verdict.

    Separate from CAPABILITY_ABSENT, which is about whether code is present;
    this is about what the caller actually asked for and got.

    FULL        every guard the caller asked for ran and was applied.
    INCOMPLETE  a verdict was returned through a deliberately weakened layer,
                so it must not be read as the full contract's answer.
    UNKNOWN     the contract's state could not be determined.
    """

    FULL = "full"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


class DataOrigin(str, Enum):
    """Standardized provenance taxonomy for data records in APEDP-DYN.

    Separates observational and experimental sources:
      SYNTHETIC:          analytical formula evaluations with synthetic noise.
      SIMULATION:         forward-integrated numerical ODE/PDE or analog quantum simulations.
      REAL_OBSERVATIONAL: passive observational records from the field or markets without intervention.
      EXPERIMENTAL:       active laboratory or physical interventions with controlled parameters.
      UNKNOWN:            unannotated / legacy fallback.
    """

    SYNTHETIC = "synthetic"
    SIMULATION = "simulation"
    REAL_OBSERVATIONAL = "real_observational"
    EXPERIMENTAL = "experimental"
    UNKNOWN = "unknown"


class BenchmarkLevel(str, Enum):
    """Formal capability taxonomy for APEDP benchmark suites.

    Explicitly declares which capability tier a benchmark is exercising:
      L0_SELECTION:            Human/predefined candidates. Tests comparative adjudication without overclaiming.
      L1_CONSTRAINED_DISCOVERY: Restricted grammar over known vocabulary. Tests candidate generation & supported term identification.
      L2_OPEN_DISCOVERY:       Broad grammar + physical/dimensional invariance gates. Tests unconstrained equation discovery.
      L3_ACTIVE_DISCOVERY:     L2 + optimal experimental intervention design to actively discriminate remaining hypotheses.
      UNKNOWN:                 Unannotated / legacy fallback.
    """

    L0_SELECTION = "L0_selection"
    L1_CONSTRAINED_DISCOVERY = "L1_constrained_discovery"
    L2_OPEN_DISCOVERY = "L2_open_discovery"
    L3_ACTIVE_DISCOVERY = "L3_active_discovery"
    UNKNOWN = "unknown"


class EpistemicAttribution(str, Enum):
    """Diagnostic meta-evidence attributing WHY an epistemic verdict occurred.

    Separates hypothesis-space insufficiency from intrinsic physical/observational
    indistinguishability, measurement noise swamping, and misspecification:
      HYPOTHESIS_SPACE_LIMITED:
        Abstention occurred because the admissible candidate space was starved,
        saturated, or lacked necessary functional expressiveness (model-space starvation).
      INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY:
        Multiple distinct mechanisms or parameterizations fit the data equally well
        within resolution floor tau_epi because the observational regime lacks distinguishing power.
      MEASUREMENT_NOISE_SWAMPED:
        The observational noise scale sigma / sqrt(n) or amplitude floor overwhelms
        the structural signal, rendering hypotheses unresolvable under available data.
      OUT_OF_GRAMMAR_OR_MISSPECIFIED:
        All candidates failed numerical admissibility or predictive adequacy ceilings.
      DECISIVE_EVIDENTIAL_SUPPORT:
        The winning candidate was decisively separated (Delta > tau_epi) and confirmed
        by validation and bootstrap checks.
      BOOTSTRAP_INSTABILITY:
        Point estimate cleared tau_epi, but paired block-bootstrap confidence interval
        straddled zero, indicating sampling instability.
      UNKNOWN:
        Unattributed fallback.
    """

    HYPOTHESIS_SPACE_LIMITED = "hypothesis_space_limited"
    INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY = "intrinsic_observational_nonidentifiability"
    MEASUREMENT_NOISE_SWAMPED = "measurement_noise_swamped"
    OUT_OF_GRAMMAR_OR_MISSPECIFIED = "out_of_grammar_or_misspecified"
    DECISIVE_EVIDENTIAL_SUPPORT = "decisive_evidential_support"
    BOOTSTRAP_INSTABILITY = "bootstrap_instability"
    EXTRAPOLATION_DIVERGENCE = "extrapolation_divergence"
    UNKNOWN = "unknown"


REL_FLOOR = 0.05
"""Amplitude resolution floor: a rival whose holdout RMSE differs by less than
REL_FLOOR of the observed signal RMS is not resolved by the intervention
available. Dimensionless by construction, so it does not track problem units.

Calibrated, not chosen. Two recorded constraints bracket it:

  lower  Two fits of the *same* structure that differ only by estimator choice
         differ by estimator noise, not evidence. Across the D2a A/B grid
         (outputs/d2a_ab_report.json, 2 problems x 6 noise levels) the largest
         holdout-RMSE gap between the spline and weak-form arms, relative to
         the holdout RMS, is 0.0121. A stricter live floor comes from the
         small-amplitude Duffing pair (linear vs cubic, ratio 0.0245), which
         must stay MULTI_HYPOTHESIS — the floor has to exceed it to abstain.
  upper  The decisive Duffing pair (ratio 0.700) must stay UNIQUE_SUPPORT, so
         the floor has to stay below it.

Admissible window (0.0245, 0.700). 0.05 sits 2.0x above the binding lower edge
and 14x below the upper. tests/test_dyn_epistemic.py re-derives the lower and
upper edges from the live problems, so this constant cannot drift out of its
window silently.
"""


def resolution_floor(obs_rms: float | None, noise_sigma: float | None,
                     n_samples: int | None,
                     rel_floor: float = REL_FLOOR) -> float | None:
    """Scale-free resolution floor for holdout RMSE differences.

    Components:
      - amplitude floor ``rel_floor * obs_rms``: resolution cannot beat the
        size of the signal the intervention actually excited.
      - noise floor ``noise_sigma / sqrt(n_samples)``: the standard error of an
        RMSE estimate taken against noisy truth.

    Returns None when neither component is available. Callers MUST NOT treat
    that as 0.0: a missing scale means no rival can be resolved, which is an
    abstention, not a promotion.
    """
    parts: list[float] = []
    if obs_rms is not None and np.isfinite(obs_rms) and obs_rms > 0:
        parts.append(rel_floor * float(obs_rms))
    if noise_sigma is not None and n_samples:
        parts.append(float(noise_sigma) / np.sqrt(float(n_samples)))
    return max(parts) if parts else None


@dataclass(frozen=True)
class Adjudication:
    status: EpistemicStatus
    promoted: Any = None
    rival: Any = None
    ranking: tuple = ()
    best_rmse: float | None = None
    second_rmse: float | None = None
    delta_rmse: float | None = None
    holdout_noise_scale: float | None = None
    holdout_scale_source: str = ""
    epistemic_margin: float | None = None
    tau_epi: float | None = None
    tau_epi_source: str = ""
    bootstrap_branch: str = ""
    bootstrap_lo: float | None = None
    bootstrap_hi: float | None = None
    contract: EpistemicContract = EpistemicContract.UNKNOWN
    data_origin: str = "unknown"
    benchmark_level: str = "unknown"
    epistemic_attribution: str = "unknown"
    promotion_confidence: str = ""
    diagnostic_flags: tuple = ()
    admissible_equivalence_class: tuple[str, ...] = ()
    reason: str = ""



def scale_from_observations(obs: np.ndarray) -> dict[str, Any]:
    """Scale of a holdout observation record, in the shape ``adjudicate`` takes.

    Imported lazily for the same reason as :func:`_identity`: the module import
    order in this package is deliberate.
    """
    from apedp_dyn.gates.intervention import observation_scale

    return observation_scale(obs)


def _identity(sc) -> Any:
    """Structural identity: canonical form + parameter-binding signature.

    Same term sets with different sharing (reciprocal kc vs independent
    k12/k21) are distinct hypotheses, not duplicates. Dispatch on the IR type,
    not on which attributes happen to exist: a hasattr probe silently
    misclassifies any IR that grows the other shape, and the coupled branch
    reaches ``.equations``/``.terms`` that must exist by contract.
    """
    # Imported lazily: apedp_dyn.grammar imports the fitter, which imports this
    # package's siblings, so the module-level import order is deliberate.
    from apedp_dyn.grammar.candidate_ir import CandidateDynamics, SystemCandidate
    from apedp_dyn.grammar.canonicalize import canonical_form
    from apedp_dyn.grammar.coupled import canonical_system

    c = sc.candidate
    if isinstance(c, SystemCandidate):
        sharing = tuple(sorted(
            (i, t.name, t.parameter) for i, e in enumerate(c.equations)
            for t in e.terms))
        return (canonical_system(c), sharing)
    if not isinstance(c, CandidateDynamics):
        raise TypeError(f"unknown candidate IR type: {type(c).__name__}")
    return (canonical_form(c),
            tuple(sorted((t.name, t.parameter) for t in c.terms)))


def _sole_survivor_fidelity(best, scale: dict[str, Any] | None,
                            gamma: float) -> tuple[bool, str]:
    """Guard 1: absolute holdout-fidelity ceiling for a sole numerical survivor.

    The sole contract is ``rmse <= gamma * (sigma / CAL_SIGMA)``. A relative
    alternative (a ceiling on a fraction of the observation RMS) was in force
    briefly and withdrawn: it is looser exactly where the noise estimate is
    degenerate, so it waved through mid-quality fits on near-noiseless records
    that the absolute contract is meant to refuse. Absolute only.

    Missing evidence never passes: no finite RMSE, or no usable noise scale,
    each independently force a refusal. Returns (passes, rationale).
    """
    rmse = getattr(best, "holdout_rmse", None)
    if rmse is None or not np.isfinite(rmse):
        return False, "no finite holdout RMSE on the sole survivor"
    scale = scale or {}
    sigma = scale.get("noise_sigma")
    if sigma is None or not np.isfinite(float(sigma)):
        return False, "no holdout noise scale; adequacy is unassessable"
    sigma = float(sigma)
    if sigma <= 0:
        ok = bool(rmse <= ABSOLUTE_EPS)
        return ok, f"degenerate noise scale; RMSE {rmse:.3g} vs eps {ABSOLUTE_EPS:.0e}"
    from apedp_dyn.epistemic.calibration import CAL_SIGMA, calibrated_sigma

    sig_cal = calibrated_sigma(sigma)
    ceiling = gamma * sig_cal
    ok = bool(rmse <= ceiling)
    return ok, (f"RMSE {rmse:.3g} vs absolute ceiling {ceiling:.3g} "
                f"({gamma:g}*sigma_cal [sigma_hat {sigma:.3g} / CAL "
                f"{CAL_SIGMA:g}])")


def _sole_survivor_whiteness(resid: Any, threshold: float) -> tuple[bool, str]:
    """Guard 2: residual-structure check, applied only after Guard 1 passes.

    Compares the lag-1 autocorrelation of the per-timepoint squared-error
    series against ``threshold``. Accuracy without whiteness is not
    explanatory support: a model can match the observation on average while
    its error remains structured, which means something is still unmodelled.

    On heteroskedastic series (volatility clustering), evaluates standardized
    innovations if raw residuals fail threshold.
    """
    if resid is None:
        return False, "no residual series; whiteness unassessable"
    arr = np.asarray(resid, dtype=float).ravel()
    if arr.size < 4:
        return False, f"residual series too short ({arr.size} points) to test"
    if not np.all(np.isfinite(arr)):
        return False, "residual series contains non-finite values"
    centred = arr - arr.mean()
    denom = float(np.dot(centred, centred))
    if denom <= 0.0:
        return False, ("residual series is constant: |rho1| undefined, "
                       "whiteness unassessable")
    rho1 = float(np.dot(centred[:-1], centred[1:]) / denom)
    ok = bool(abs(rho1) < threshold)
    if ok:
        return ok, f"|rho1|={abs(rho1):.3f} vs flag {threshold:g}"

    if arr.size >= 8:
        alpha = 0.8
        curr_var = float(np.var(centred))
        if curr_var > 1e-12:
            var_est = np.empty_like(centred)
            for t in range(len(centred)):
                curr_var = alpha * curr_var + (1.0 - alpha) * (centred[t] ** 2)
                var_est[t] = max(curr_var, 1e-12)
            z = centred / np.sqrt(var_est)
            z_centred = z - z.mean()
            z_denom = float(np.dot(z_centred, z_centred))
            if z_denom > 0.0:
                rho1_z = float(np.dot(z_centred[:-1], z_centred[1:]) / z_denom)
                if abs(rho1_z) < threshold:
                    return True, f"|rho1_innov|={abs(rho1_z):.3f} vs flag {threshold:g} (standardized innovations)"

    return False, f"|rho1|={abs(rho1):.3f} vs flag {threshold:g}"


def _residual_series(sc: Any) -> Any:
    """Per-timepoint holdout residual series gate 4 recorded for ``sc``.

    The single route by which a caller that never computes residuals can still
    reach the whiteness check. Callers that fit through fit_candidate get this
    for free; the conf benchmarks hand-build their provenance and populate the
    same key, so both kinds of caller are covered by one lookup.

    Signed residuals only. An earlier version fell back to ``holdout_sq_error``
    when no signed series existed, but lag-1 autocorrelation on squared errors
    measures volatility clustering, not residual whiteness -- a model with
    perfectly white but heteroskedastic residuals fails that test for the wrong
    reason. A candidate with no signed series therefore reports whiteness as
    unassessable rather than as failed; silence must not read as structure.
    """
    try:
        g4 = sc.provenance.get("gate4_status", {})
        return g4.get("holdout_residuals")
    except AttributeError:
        return None


def _note_series(sc: Any, resid_best: Any) -> Any:
    """Series for the diagnostic note: explicit argument, else gate 4's record.

    The note reads the explicit argument first, then the series gate 4 recorded.
    Callers that never compute residuals (the conf benchmarks hand-build their
    provenance) would otherwise be silently excluded from a diagnostic that
    exists to be read. Explicit None test, not truthiness: a numpy array has no
    single boolean reading and ``or`` would misbehave on one.
    """
    return resid_best if resid_best is not None else _residual_series(sc)


def _promotion_diagnostics(
    delta: float,
    tau: float | None,
    resid: Any,
    rho_flag: float,
    weak_margin_multiple: float = WEAK_MARGIN_MULTIPLE,
) -> tuple[str, tuple[str, ...]]:
    """Diagnostic-only qualification of a would-be promotion. Never changes status.

    Separates two questions the holdout comparison conflates:

        predictive arbitration -- did this candidate beat that rival on unseen
        data? (what ``delta > tau`` answers)

        residual completeness  -- has it captured all systematically
        recoverable structure? (what residual whiteness asks)

    Holdout superiority does not imply completeness: RMSE_A < RMSE_B can hold
    while A's residuals stay strongly autocorrelated. The frozen gate answers
    the first question only, so this records where the second is unresolved.

    Returns ``(promotion_confidence, diagnostic_flags)``:
      WEAK_PROMOTION_MARGIN        -- separation clears tau by less than
                                      ``weak_margin_multiple`` (a boundary-case
                                      promotion: formally permitted, evidentially thin)
      RESIDUAL_STRUCTURE_UNRESOLVED -- winner's lag-1 |rho1| exceeds the whiteness
                                      flag, so a predictive win is not a complete
                                      dynamical explanation
    """
    flags: list[str] = []
    confidence = ""
    if tau is not None and np.isfinite(tau) and tau > 0 and np.isfinite(delta):
        if (delta / tau) < weak_margin_multiple:
            flags.append("WEAK_PROMOTION_MARGIN")
            confidence = "low_confidence"
    if resid is not None:
        arr = np.asarray(resid, dtype=float).ravel()
        if arr.size >= 4 and np.all(np.isfinite(arr)):
            centred = arr - arr.mean()
            denom = float(np.dot(centred, centred))
            if denom > 0.0:
                rho1 = abs(float(np.dot(centred[:-1], centred[1:]) / denom))
                if rho1 > rho_flag:
                    flags.append("RESIDUAL_STRUCTURE_UNRESOLVED")
    return confidence, tuple(flags)


def _winner_structure_note(resid: Any, threshold: float) -> str:
    """Diagnostic-only annotation of residual structure on a promoted winner.

    The whiteness barrier (Guard 2) is part of the *sole*-survivor contract: it
    answers "is an unopposed survivor explaining the record?". On the pair path
    a rival exists and was beaten on holdout RMSE, which is direct evidence the
    residual pattern did not stop the rival from scoring worse. Demoting there
    would let a well-separated winner be held back by a residual shape the
    data already arbitrated. So the structure is reported, never enforced.

    Returns "" when there is nothing to report, or when the series cannot be
    assessed (no series, too short, constant): an unassessable series is not a
    finding, and on this path abstaining over silence would be a regression.
    """
    if resid is None:
        return ""
    arr = np.asarray(resid, dtype=float).ravel()
    if arr.size < 4 or not np.all(np.isfinite(arr)):
        return ""
    centred = arr - arr.mean()
    denom = float(np.dot(centred, centred))
    if denom <= 0.0:
        return ""
    rho1 = float(np.dot(centred[:-1], centred[1:]) / denom)
    if abs(rho1) < threshold:
        return ""
    return (f"; note: winner holdout residuals are structured (|rho1|="
            f"{abs(rho1):.3f} exceeds the {threshold:g} whiteness flag) — "
            "reported for the record, not enforced, since a structurally "
            "distinct rival was separated on this holdout")


def _bootstrap_refusal_detail(tb: dict) -> str:
    """Reason detail for a bootstrap overturn, by branch.

    A 'multi' CI straddles zero (no separation); a 'rival-favored' CI lies
    entirely below zero (resampled evidence favors the rival), so promoting
    the point-winner would contradict the bootstrap. Rank revision (promoting
    the rival instead) is out of scope: both cases refuse promotion.
    """
    if tb.get("branch") == "rival-favored":
        return (f"paired block bootstrap favors the rival (CI "
                f"[{tb.get('lo'):.3g}, {tb.get('hi'):.3g}] entirely below 0); "
                f"promoting the point-winner would contradict the resampled "
                f"evidence (rank revision is out of scope)")
    return (f"paired block bootstrap (CI [{tb.get('lo'):.3g}, "
            f"{tb.get('hi'):.3g}] straddles 0); the holdout does "
            f"not separate the two structures")


def adjudicate(fitted: list,
               tau_epi: float | None = None,
               holdout_noise_scale: float | None = None,
               holdout_scale: dict[str, Any] | None = None,
               resid_best: Any = None,
               resid_rival: Any = None,
               resid_sole: Any = None,
               bootstrap: bool = False,
               bootstrap_n_boot: int = 2000,
               bootstrap_seed: int = 0,
               sole_survivor_rule: bool = True,
               sole_gamma: float = SOLE_SURVIVOR_GAMMA,
               sole_rho_flag: float = RESID_RHO_FLAG,
               data_origin: str = "unknown",
               benchmark_level: str = "unknown",
               epistemic_attribution: str = "unknown",
               extrapolation_gate: dict[str, Any] | None = None) -> Adjudication:
    """Rank by train loss, then adjudicate on holdout evidence. Deterministic.

    sole_survivor_rule=True (default) enforces the frozen absolute-fidelity
    contract on unopposed survivors: Guard 1 (SOLE_SURVIVOR_GAMMA) followed by
    Guard 2 (residual whiteness, RESID_RHO_FLAG). Pass
    sole_survivor_rule=False to reproduce the pre-hardening baseline (sole
    numerical survivor promoted unopposed).

    Contract accounting (see :class:`EpistemicContract`): every returned
    Adjudication carries how complete the contract was. sole_survivor_rule=False
    returns a real verdict but through a deliberately weakened layer, so it is
    recorded INCOMPLETE -- the pre-hardening baseline is a comparison tool, not
    an answer the pipeline should cite as the full contract.

    bootstrap=True is different in kind: the caller explicitly asked for a
    stronger guarantee, so if the tiebreaker is not compiled in this raises
    :data:`CAPABILITY_ABSENT` rather than quietly returning the weaker verdict.

    data_origin labels the provenance source (e.g. "simulation", "experimental",
    "synthetic", "real_observational"). Can be passed directly or read from candidate provenance.

    benchmark_level declares the capability tier under test (L0_selection,
    L1_constrained_discovery, L2_open_discovery, L3_active_discovery).

    epistemic_attribution attributes why an epistemic verdict occurred (e.g.
    "hypothesis_space_limited", "intrinsic_observational_nonidentifiability",
    "measurement_noise_swamped", "out_of_grammar_or_misspecified", "decisive_evidential_support",
    "bootstrap_instability"). Can be passed directly or read from candidate provenance.
    """
    # Capability gate first: an explicit opt-in that cannot be honored must
    # fail before any verdict is computed, so no caller can mistake the
    # weaker result for the one they asked for.
    if bootstrap and not bootstrap_available():
        raise RuntimeError(
            f"{CAPABILITY_ABSENT}: bootstrap tiebreaker not compiled into "
            "this build; adjudicate(bootstrap=True) cannot be honored")

    # Every exit below reports INCOMPLETE, never FULL, when the caller
    # disabled the sole-survivor guards. A deliberately weakened layer must not
    # be readable as the full contract's answer.
    contract = (EpistemicContract.FULL if sole_survivor_rule
                else EpistemicContract.INCOMPLETE)

    ranked = sorted(
        [s for s in fitted
         if s.train_loss is not None and np.isfinite(s.train_loss)],
        key=lambda s: s.train_loss)

    # Canonicalization: duplicate structures are not distinct hypotheses.
    # Group by structural identity (canonical form + parameter-binding
    # signature, so e.g. reciprocal-kc vs independent-k12/k21 stay
    # distinct) and keep the best-fitting instance per group. Uniqueness
    # is assessed over canonical hypotheses only; dropped duplicates can
    # neither promote nor serve as rivals.
    canonical: list = []
    seen: dict = {}
    dropped_duplicate_ids: list[str] = []
    for s in ranked:
        try:
            key = _identity(s)
        except TypeError:
            key = ("untyped", s.candidate.candidate_id)
        if key in seen:
            dropped_duplicate_ids.append(s.candidate.candidate_id)
            continue
        seen[key] = True
        canonical.append(s)
    ranked = canonical
    dup_flags = (tuple(f"canonicalized_duplicate:{i}" for i in dropped_duplicate_ids)
                 if dropped_duplicate_ids else ())

    if not ranked:
        attr = (epistemic_attribution if epistemic_attribution != "unknown"
                else (EpistemicAttribution.HYPOTHESIS_SPACE_LIMITED.value if len(fitted) == 0
                      else EpistemicAttribution.OUT_OF_GRAMMAR_OR_MISSPECIFIED.value))
        return Adjudication(EpistemicStatus.NO_VALID_CANDIDATE,
                            contract=contract,
                            data_origin=data_origin,
                            benchmark_level=benchmark_level,
                            epistemic_attribution=attr,
                            reason="no scored candidates")
    best = ranked[0]

    # If data_origin was not explicitly supplied, check best candidate provenance
    if data_origin == "unknown":
        prov_origin = best.provenance.get("data_origin")
        if prov_origin:
            data_origin = str(prov_origin)

    # If benchmark_level was not explicitly supplied, check best candidate provenance
    if benchmark_level == "unknown":
        prov_lvl = best.provenance.get("benchmark_level")
        if prov_lvl:
            benchmark_level = str(prov_lvl)

    # If epistemic_attribution was not explicitly supplied, check best candidate provenance
    if epistemic_attribution == "unknown":
        prov_attr = best.provenance.get("epistemic_attribution")
        if prov_attr:
            epistemic_attribution = str(prov_attr)

    g4 = best.provenance.get("gate4_status", {})
    if holdout_noise_scale is None:
        holdout_noise_scale = g4.get("holdout_noise_sigma")

    scale = holdout_scale
    scale_source = "argument" if scale is not None else ""
    if scale is None:
        scale = {"obs_rms": g4.get("holdout_obs_rms"),
                 "noise_sigma": g4.get("holdout_noise_sigma"),
                 "n_samples": g4.get("holdout_n_samples")}
        if any(v is not None for v in scale.values()):
            scale_source = "gate4_provenance"

    if not g4.get("passed", False):
        attr = (epistemic_attribution if epistemic_attribution != "unknown"
                else EpistemicAttribution.OUT_OF_GRAMMAR_OR_MISSPECIFIED.value)
        return Adjudication(
            EpistemicStatus.INSUFFICIENT_EVIDENCE, ranking=tuple(ranked),
            best_rmse=best.holdout_rmse,
            holdout_noise_scale=holdout_noise_scale,
            holdout_scale_source=scale_source,
            contract=contract,
            data_origin=data_origin,
            benchmark_level=benchmark_level,
            epistemic_attribution=attr,
            reason=f"rank-1 {best.candidate.candidate_id} fails gate 4: "
                   f"{g4.get('detail', 'no holdout evidence')}; "
                   "no promotion of least-bad models")
    tau: float | None = None
    if tau_epi is not None:
        tau = float(tau_epi)
        tau_source = "explicit"
    else:
        tau = resolution_floor(scale.get("obs_rms"), scale.get("noise_sigma"),
                               scale.get("n_samples"))
        tau_source = "holdout-scale"
    if tau is None:
        # No scale means no resolution. Abstain; never fall back to a constant
        # (a missing key must not read as "zero tolerance").
        attr = (epistemic_attribution if epistemic_attribution != "unknown"
                else EpistemicAttribution.MEASUREMENT_NOISE_SWAMPED.value)
        return Adjudication(
            EpistemicStatus.INSUFFICIENT_EVIDENCE, ranking=tuple(ranked),
            best_rmse=best.holdout_rmse,
            holdout_noise_scale=holdout_noise_scale,
            holdout_scale_source=scale_source,
            contract=contract,
            data_origin=data_origin,
            benchmark_level=benchmark_level,
            epistemic_attribution=attr,
            reason="rank-1 passes gate 4 but the holdout record carries no "
                   "scale (obs_rms/noise_sigma), so no rival can be resolved; "
                   "pass holdout_scale= or tau_epi= to supply one")
    # Finding 14 (inverted pair anchor): the pair evaluation must be separated
    # on holdout evidence, never on the training-loss ordering. The train-loss
    # winner can be the most overfitted candidate in the pool; anchoring the
    # pair on it made delta = rival_ho - best_ho negative whenever a
    # generalizing rival existed, which trivially tripped `delta <= tau` for
    # any tau > 0 and denied UNIQUE_SUPPORT to genuine discoveries
    # (misclassified as MULTI_HYPOTHESIS with a contradictory diagnostic).
    # Re-anchor the promotion head on the holdout-lowest-RMSE candidate when
    # that candidate is itself gate-4 certified; otherwise the gate-4-certified
    # train-loss anchor stays.
    ho_scoring = [s for s in ranked
                  if s.holdout_rmse is not None and np.isfinite(s.holdout_rmse)]
    if ho_scoring:
        ho_head = min(ho_scoring, key=lambda s: s.holdout_rmse)
        if ho_head is not best and ho_head.provenance.get(
                "gate4_status", {}).get("passed", False):
            # FLAW-08: the holdout head takes over the pair. The caller-supplied
            # residual pair belonged to the ORIGINAL pair and cannot be reused
            # blindly after the re-anchor — but when both the re-anchored head
            # and the re-anchored nearest rival preserve their own gate-4
            # holdout residual series in provenance, the paired block-bootstrap
            # tiebreak can and must run; silently downgrading to point
            # estimates awarded unverified UNIQUE_SUPPORT (Winner's Curse).
            best = ho_head
            ranked = [ho_head] + [s for s in ranked if s is not ho_head]
            _anchor_resid = _residual_series(best)
            _runner = next((s for s in ranked[1:]
                            if s.holdout_rmse is not None and np.isfinite(s.holdout_rmse)), None)
            _runner_resid = _residual_series(_runner) if _runner is not None else None
            if (_anchor_resid is not None and _runner_resid is not None
                    and len(_anchor_resid) == len(_runner_resid)
                    and len(_anchor_resid) > 1):
                resid_best = np.asarray(_anchor_resid, dtype=float)
                resid_rival = np.asarray(_runner_resid, dtype=float)
                bootstrap = True
            else:
                # Series are unpairable — fall back to the honest
                # point-estimate contract (disclosed via pair_contract=
                # INCOMPLETE below) instead of asserting a bootstrap
                # confirmation that cannot be attributed.
                bootstrap = False
    best_rmse = best.holdout_rmse
    rivals = [s for s in ranked[1:]
              if _identity(s) != _identity(best) and s.holdout_rmse is not None
              and np.isfinite(s.holdout_rmse)]
    if not rivals:
        if sole_survivor_rule:
            # Guard 1: absolute predictive adequacy (ceiling first).
            ok1, why1 = _sole_survivor_fidelity(best, scale, sole_gamma)
            if not ok1:
                attr = (epistemic_attribution if epistemic_attribution != "unknown"
                        else EpistemicAttribution.OUT_OF_GRAMMAR_OR_MISSPECIFIED.value)
                return Adjudication(
                    EpistemicStatus.INSUFFICIENT_EVIDENCE, ranking=tuple(ranked),
                    best_rmse=best_rmse, delta_rmse=None, tau_epi=tau,
                    tau_epi_source=tau_source,
                    holdout_noise_scale=holdout_noise_scale,
                    holdout_scale_source=scale_source, epistemic_margin=None,
                    contract=contract,
                    data_origin=data_origin,
                    benchmark_level=benchmark_level,
                    epistemic_attribution=attr,
                    reason=f"sole survivor {best.candidate.candidate_id} passed "
                           f"numerical admissibility but failed the pre-specified "
                           f"absolute holdout-fidelity contract ({why1}); numerical "
                           f"survival is not explanatory support")
            # Guard 2: residual structure (whiteness second, never first).
            series = resid_sole if resid_sole is not None else resid_best
            if series is None:
                # Fall back to the series gate 4 records in provenance, so a
                # caller that forwards its fits is not silently denied the
                # evidence this guard needs.
                series = _residual_series(best)
            ok2, why2 = _sole_survivor_whiteness(series, sole_rho_flag)
            if not ok2:
                attr = (epistemic_attribution if epistemic_attribution != "unknown"
                        else EpistemicAttribution.HYPOTHESIS_SPACE_LIMITED.value)
                return Adjudication(
                    EpistemicStatus.INSUFFICIENT_EVIDENCE, ranking=tuple(ranked),
                    best_rmse=best_rmse, delta_rmse=None, tau_epi=tau,
                    tau_epi_source=tau_source,
                    holdout_noise_scale=holdout_noise_scale,
                    holdout_scale_source=scale_source, epistemic_margin=None,
                    contract=contract,
                    data_origin=data_origin,
                    benchmark_level=benchmark_level,
                    epistemic_attribution=attr,
                    reason=f"sole survivor {best.candidate.candidate_id} passed "
                           f"the fidelity ceiling ({why1}) but carries structured "
                           f"holdout residuals ({why2}); accuracy without whiteness "
                           f"is not explanatory support")
        if extrapolation_gate is not None and not extrapolation_gate.get("passed", True):
            reason_extrap = extrapolation_gate.get("reason", "divergence outside evidence regime")
            return Adjudication(
                EpistemicStatus.EXTRAPOLATION_UNJUSTIFIED, promoted=None, ranking=tuple(ranked),
                best_rmse=best_rmse, delta_rmse=None, tau_epi=tau,
                tau_epi_source=tau_source,
                holdout_noise_scale=holdout_noise_scale,
                holdout_scale_source=scale_source, epistemic_margin=None,
                contract=contract,
                data_origin=data_origin,
                benchmark_level=benchmark_level,
                epistemic_attribution=EpistemicAttribution.EXTRAPOLATION_DIVERGENCE.value,
                reason=f"in-sample evidence does not justify extrapolation ({reason_extrap})")
        attr = (epistemic_attribution if epistemic_attribution != "unknown"
                else EpistemicAttribution.DECISIVE_EVIDENTIAL_SUPPORT.value)
        return Adjudication(
            EpistemicStatus.UNIQUE_SUPPORT, promoted=best, ranking=tuple(ranked),
            best_rmse=best_rmse, delta_rmse=None, tau_epi=tau,
            tau_epi_source=tau_source,
            holdout_noise_scale=holdout_noise_scale,
            holdout_scale_source=scale_source, epistemic_margin=None,
            contract=contract,
            data_origin=data_origin,
            benchmark_level=benchmark_level,
            epistemic_attribution=attr,
            reason="no structurally distinct rival survived scoring")
    # Pair contract accounting: a pair path that runs without bootstrap
    # tiebreaking runs on point estimates alone and must not masquerade as FULL.
    pair_contract = (contract if (bootstrap and resid_best is not None and resid_rival is not None)
                     else EpistemicContract.INCOMPLETE)
    # External-review follow-through: the pair verdict decides the winner on the
    # SAME holdout slice that later certifies it (promotion head = holdout-lowest
    # RMSE), so that holdout is a selection set, not an untouched confirmation
    # set (review finding on F14). The referee cannot mint new confirmation
    # data, so every pair-path verdict DISCLOSES the reuse and its pool size;
    # the calling layer must charge it to the DSR trial ledger via
    # ``search_id``/``selection_pool_size`` (the standing multiple-testing
    # correction — see apedp_dyn/epistemic/trial_ledger.py) or obtain an
    # independent confirmation split. FULL contract claims are therefore only
    # valid when the caller's ledger separation is honored.
    selection_reuse_flag = (
        f"selection_reused_confirmation_holdout(n_candidates={len(ho_scoring)})"
        if ho_scoring else "")

    second = min(rivals, key=lambda s: s.holdout_rmse)
    # delta is measured against the holdout-anchored head, so a genuinely
    # generalizing candidate now earns a nonnegative separation instead of a
    # spurious refusal (Finding 14). delta IS delta_ho; the `delta <= tau`
    # comparison below therefore verifies the holdout criterion directly
    # before any MULTI_HYPOTHESIS verdict is issued.
    delta = float(second.holdout_rmse - best_rmse)
    margin = float(delta - tau)
    if delta <= tau:
        attr = (epistemic_attribution if epistemic_attribution != "unknown"
                else EpistemicAttribution.INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY.value)
        # Pair-consistent refusal (Finding 14 follow-through): the tie verdict
        # and its diagnostics are reported against the ANCHORED pair
        # (best vs nearest distinct rival). Recomputing a global holdout
        # minimum here contradicted the decision and printed diagnostics like
        # "Δ=1.93 ≤ τ=0.025": a huge holdout separation quoted next to a
        # refusal. delta below (possibly negative when the anchored head's
        # rival appears better but is not gate-4 certified) IS delta_ho.
        equiv = tuple(
            s.candidate.candidate_id for s in ranked
            if s.holdout_rmse is not None and (s.holdout_rmse - best_rmse) <= tau
        )
        return Adjudication(
            EpistemicStatus.MULTI_HYPOTHESIS, rival=second, ranking=tuple(ranked),
            best_rmse=best_rmse, second_rmse=float(second.holdout_rmse),
            delta_rmse=delta, tau_epi=tau, tau_epi_source=tau_source,
            holdout_noise_scale=holdout_noise_scale,
            holdout_scale_source=scale_source, epistemic_margin=margin,
            contract=pair_contract,
            data_origin=data_origin,
            benchmark_level=benchmark_level,
            epistemic_attribution=attr,
            admissible_equivalence_class=equiv,
            reason=f"nearest distinct rival {second.candidate.candidate_id} within τ "
                   f"(Δ={delta:.3g} ≤ τ={tau:.3g} [{tau_source}] on holdout RMSE); "
                   "observationally indistinguishable under the prespecified APEDP "
                   "resolution criterion; APEDP abstains from unique promotion")
    # Bootstrap-v2 tiebreaker, on the would-be-UNIQUE branch only. A point
    # margin that clears tau is not by itself evidence: the per-timepoint
    # error difference can straddle zero blockwise, which means the two
    # structures are not separated by this holdout after all. Off by default
    # so the v1 point-estimate path is untouched.
    if bootstrap and resid_best is not None and resid_rival is not None:
        from apedp_dyn.epistemic.bootstrap import tiebreak

        # resid_best/resid_rival are *signed* residuals (the whiteness guard
        # needs the signs); tiebreak differences per-timepoint squared errors.
        # Callers were inconsistent -- some passed ((obs - pred) ** 2)
        # explicitly, most passed holdout_residuals raw -- so squaring happened
        # here instead. Differencing signed residuals bootstraps their *mean*,
        # which tests nothing about accuracy and could return 'rival-favored'
        # against a large point margin (M1 Hall-Petch: d.mean -68 while
        # ΔRMSE = +70.8). The whiteness guard above is unaffected.
        tb = tiebreak(delta, float(tau),
                      np.asarray(resid_best, dtype=float) ** 2,
                      np.asarray(resid_rival, dtype=float) ** 2,
                      n_boot=bootstrap_n_boot, seed=bootstrap_seed)
        if tb.get("branch") in ("multi", "rival-favored"):
            attr = (epistemic_attribution if epistemic_attribution != "unknown"
                    else EpistemicAttribution.BOOTSTRAP_INSTABILITY.value)
            min_ho_rmse = min((s.holdout_rmse for s in ranked if s.holdout_rmse is not None), default=best_rmse)
            equiv = tuple(
                s.candidate.candidate_id for s in ranked
                if s.holdout_rmse is not None and (s.holdout_rmse - min_ho_rmse) <= tau
            )
            return Adjudication(
                EpistemicStatus.MULTI_HYPOTHESIS, rival=second, ranking=tuple(ranked),
                best_rmse=best_rmse, second_rmse=float(second.holdout_rmse),
                delta_rmse=delta, tau_epi=tau, tau_epi_source=tau_source,
                holdout_noise_scale=holdout_noise_scale,
                holdout_scale_source=scale_source, epistemic_margin=margin,
                bootstrap_branch=tb.get("branch", ""), bootstrap_lo=tb.get("lo"),
                bootstrap_hi=tb.get("hi"),
                contract=contract,
                data_origin=data_origin,
                benchmark_level=benchmark_level,
                epistemic_attribution=attr,
                admissible_equivalence_class=equiv,
                reason=f"point margin over {second.candidate.candidate_id} "
                       f"(Δ={delta:.3g} > τ={tau:.3g}) does not survive the "
                       f"{_bootstrap_refusal_detail(tb)}")
        if extrapolation_gate is not None and not extrapolation_gate.get("passed", True):
            reason_extrap = extrapolation_gate.get("reason", "divergence outside evidence regime")
            return Adjudication(
                EpistemicStatus.EXTRAPOLATION_UNJUSTIFIED, promoted=None, rival=second, ranking=tuple(ranked),
                best_rmse=best_rmse, second_rmse=float(second.holdout_rmse),
                delta_rmse=delta, tau_epi=tau, tau_epi_source=tau_source,
                holdout_noise_scale=holdout_noise_scale,
                holdout_scale_source=scale_source, epistemic_margin=margin,
                contract=contract,
                data_origin=data_origin,
                benchmark_level=benchmark_level,
                epistemic_attribution=EpistemicAttribution.EXTRAPOLATION_DIVERGENCE.value,
                reason=f"in-sample evidence does not justify extrapolation ({reason_extrap})")
        attr = (epistemic_attribution if epistemic_attribution != "unknown"
                else EpistemicAttribution.DECISIVE_EVIDENTIAL_SUPPORT.value)
        boot_conf, boot_flags = _promotion_diagnostics(
            delta, tau, _note_series(best, resid_best), sole_rho_flag)
        return Adjudication(
            EpistemicStatus.UNIQUE_SUPPORT, promoted=best, rival=second, ranking=tuple(ranked),
            best_rmse=best_rmse, second_rmse=float(second.holdout_rmse),
            delta_rmse=delta, tau_epi=tau, tau_epi_source=tau_source,
            holdout_noise_scale=holdout_noise_scale,
            holdout_scale_source=scale_source, epistemic_margin=margin,
            bootstrap_branch=tb.get("branch", ""), bootstrap_lo=tb.get("lo"),
            bootstrap_hi=tb.get("hi"),
            contract=contract,
            data_origin=data_origin,
            benchmark_level=benchmark_level,
            epistemic_attribution=attr,
            promotion_confidence=boot_conf,
            diagnostic_flags=tuple(boot_flags) + dup_flags
            + ((selection_reuse_flag,) if selection_reuse_flag else ()),
            reason=f"rival {second.candidate.candidate_id} separated by "
                   f"Δ={delta:.3g} > τ={tau:.3g} [{tau_source}] and confirmed "
                   f"by the paired block bootstrap (branch "
                   f"{tb.get('branch')!r})"
                   f"{_winner_structure_note(_note_series(best, resid_best), sole_rho_flag)}")
    note = _winner_structure_note(_note_series(best, resid_best),
                                  sole_rho_flag)
    diag_conf, diag_flags = _promotion_diagnostics(
        delta, tau, _note_series(best, resid_best), sole_rho_flag)
    if extrapolation_gate is not None and not extrapolation_gate.get("passed", True):
        reason_extrap = extrapolation_gate.get("reason", "divergence outside evidence regime")
        return Adjudication(
            EpistemicStatus.EXTRAPOLATION_UNJUSTIFIED, promoted=None, rival=second, ranking=tuple(ranked),
            best_rmse=best_rmse, second_rmse=float(second.holdout_rmse),
            delta_rmse=delta, tau_epi=tau, tau_epi_source=tau_source,
            holdout_noise_scale=holdout_noise_scale,
            holdout_scale_source=scale_source, epistemic_margin=margin,
            contract=pair_contract,
            data_origin=data_origin,
            benchmark_level=benchmark_level,
            epistemic_attribution=EpistemicAttribution.EXTRAPOLATION_DIVERGENCE.value,
            reason=f"in-sample evidence does not justify extrapolation ({reason_extrap})")
    attr = (epistemic_attribution if epistemic_attribution != "unknown"
            else EpistemicAttribution.DECISIVE_EVIDENTIAL_SUPPORT.value)
    return Adjudication(
        EpistemicStatus.UNIQUE_SUPPORT, promoted=best, rival=second, ranking=tuple(ranked),
        best_rmse=best_rmse, second_rmse=float(second.holdout_rmse),
        delta_rmse=delta, tau_epi=tau, tau_epi_source=tau_source,
        holdout_noise_scale=holdout_noise_scale,
        holdout_scale_source=scale_source, epistemic_margin=margin,
        contract=pair_contract,
        data_origin=data_origin,
        benchmark_level=benchmark_level,
        epistemic_attribution=attr,
            promotion_confidence=diag_conf,
            diagnostic_flags=tuple(diag_flags) + dup_flags
            + ((selection_reuse_flag,) if selection_reuse_flag else ()),
            reason=f"nearest distinct rival separated by Δ={delta:.3g} > "
        f"τ={tau:.3g} [{tau_source}]{note}")
