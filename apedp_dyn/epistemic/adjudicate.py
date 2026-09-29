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


DYNAMIC_EPISTEMIC_VERSION = "bootstrap-v2+solefid-v4"
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


ABSOLUTE_EPS = 1e-12
"""Floor used when a record reports a degenerate (non-positive) noise scale."""


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
    return ok, f"|rho1|={abs(rho1):.3f} vs flag {threshold:g}"


def _residual_series(sc: Any) -> Any:
    """Per-timepoint holdout error series gate 4 recorded for ``sc``.

    The single route by which a caller that never computes residuals can still
    reach the whiteness check. Callers that fit through fit_candidate get this
    for free; the conf benchmarks hand-build their provenance and populate the
    same key, so both kinds of caller are covered by one lookup.
    """
    try:
        return sc.provenance.get("gate4_status", {}).get("holdout_sq_error")
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
               sole_rho_flag: float = RESID_RHO_FLAG) -> Adjudication:
    """Rank by train loss, then adjudicate on holdout evidence. Deterministic.

    sole_survivor_rule=True (default) enforces the frozen absolute-fidelity
    contract on unopposed survivors: Guard 1 (SOLE_SURVIVOR_GAMMA) followed by
    Guard 2 (residual whiteness, RESID_RHO_FLAG). Pass
    sole_survivor_rule=False to reproduce the pre-hardening baseline (sole
    numerical survivor promoted unopposed).
    """
    ranked = sorted(
        [s for s in fitted
         if s.train_loss is not None and np.isfinite(s.train_loss)],
        key=lambda s: s.train_loss)

    if not ranked:
        return Adjudication(EpistemicStatus.NO_VALID_CANDIDATE, reason="no scored candidates")
    best = ranked[0]
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
        return Adjudication(
            EpistemicStatus.INSUFFICIENT_EVIDENCE, ranking=tuple(ranked),
            best_rmse=best.holdout_rmse,
            holdout_noise_scale=holdout_noise_scale,
            holdout_scale_source=scale_source,
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
        return Adjudication(
            EpistemicStatus.INSUFFICIENT_EVIDENCE, ranking=tuple(ranked),
            best_rmse=best.holdout_rmse,
            holdout_noise_scale=holdout_noise_scale,
            holdout_scale_source=scale_source,
            reason="rank-1 passes gate 4 but the holdout record carries no "
                   "scale (obs_rms/noise_sigma), so no rival can be resolved; "
                   "pass holdout_scale= or tau_epi= to supply one")
    best_rmse = best.holdout_rmse
    rivals = [s for s in ranked[1:]
              if _identity(s) != _identity(best) and s.holdout_rmse is not None
              and np.isfinite(s.holdout_rmse)]
    if not rivals:
        if sole_survivor_rule:
            # Guard 1: absolute predictive adequacy (ceiling first).
            ok1, why1 = _sole_survivor_fidelity(best, scale, sole_gamma)
            if not ok1:
                return Adjudication(
                    EpistemicStatus.INSUFFICIENT_EVIDENCE, ranking=tuple(ranked),
                    best_rmse=best_rmse, delta_rmse=None, tau_epi=tau,
                    tau_epi_source=tau_source,
                    holdout_noise_scale=holdout_noise_scale,
                    holdout_scale_source=scale_source, epistemic_margin=None,
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
                return Adjudication(
                    EpistemicStatus.INSUFFICIENT_EVIDENCE, ranking=tuple(ranked),
                    best_rmse=best_rmse, delta_rmse=None, tau_epi=tau,
                    tau_epi_source=tau_source,
                    holdout_noise_scale=holdout_noise_scale,
                    holdout_scale_source=scale_source, epistemic_margin=None,
                    reason=f"sole survivor {best.candidate.candidate_id} passed "
                           f"the fidelity ceiling ({why1}) but carries structured "
                           f"holdout residuals ({why2}); accuracy without whiteness "
                           f"is not explanatory support")
        return Adjudication(
            EpistemicStatus.UNIQUE_SUPPORT, promoted=best, ranking=tuple(ranked),
            best_rmse=best_rmse, delta_rmse=None, tau_epi=tau,
            tau_epi_source=tau_source,
            holdout_noise_scale=holdout_noise_scale,
            holdout_scale_source=scale_source, epistemic_margin=None,
            reason="no structurally distinct rival survived scoring")
    second = min(rivals, key=lambda s: s.holdout_rmse)
    delta = float(second.holdout_rmse - best_rmse)
    margin = float(delta - tau)
    if delta <= tau:
        return Adjudication(
            EpistemicStatus.MULTI_HYPOTHESIS, ranking=tuple(ranked),
            best_rmse=best_rmse, second_rmse=float(second.holdout_rmse),
            delta_rmse=delta, tau_epi=tau, tau_epi_source=tau_source,
            holdout_noise_scale=holdout_noise_scale,
            holdout_scale_source=scale_source, epistemic_margin=margin,
            reason=f"rival {second.candidate.candidate_id} within τ "
                   f"(Δ={delta:.3g} ≤ τ={tau:.3g} [{tau_source}]); structures "
                   "observationally indistinguishable under available "
                   "interventions")
    # Bootstrap-v2 tiebreaker, on the would-be-UNIQUE branch only. A point
    # margin that clears tau is not by itself evidence: the per-timepoint
    # error difference can straddle zero blockwise, which means the two
    # structures are not separated by this holdout after all. Off by default
    # so the v1 point-estimate path is untouched.
    if bootstrap and resid_best is not None and resid_rival is not None:
        from apedp_dyn.epistemic.bootstrap import tiebreak

        tb = tiebreak(delta, float(tau), np.asarray(resid_best, dtype=float),
                      np.asarray(resid_rival, dtype=float),
                      n_boot=bootstrap_n_boot, seed=bootstrap_seed)
        if tb.get("branch") == "multi":
            return Adjudication(
                EpistemicStatus.MULTI_HYPOTHESIS, ranking=tuple(ranked),
                best_rmse=best_rmse, second_rmse=float(second.holdout_rmse),
                delta_rmse=delta, tau_epi=tau, tau_epi_source=tau_source,
                holdout_noise_scale=holdout_noise_scale,
                holdout_scale_source=scale_source, epistemic_margin=margin,
                bootstrap_branch="multi", bootstrap_lo=tb.get("lo"),
                bootstrap_hi=tb.get("hi"),
                reason=f"point margin over {second.candidate.candidate_id} "
                       f"(Δ={delta:.3g} > τ={tau:.3g}) does not survive the "
                       f"paired block bootstrap (CI [{tb.get('lo'):.3g}, "
                       f"{tb.get('hi'):.3g}] straddles 0); the holdout does "
                       "not separate the two structures")
        return Adjudication(
            EpistemicStatus.UNIQUE_SUPPORT, promoted=best, ranking=tuple(ranked),
            best_rmse=best_rmse, second_rmse=float(second.holdout_rmse),
            delta_rmse=delta, tau_epi=tau, tau_epi_source=tau_source,
            holdout_noise_scale=holdout_noise_scale,
            holdout_scale_source=scale_source, epistemic_margin=margin,
            bootstrap_branch=tb.get("branch", ""), bootstrap_lo=tb.get("lo"),
            bootstrap_hi=tb.get("hi"),
            reason=f"rival {second.candidate.candidate_id} separated by "
                   f"Δ={delta:.3g} > τ={tau:.3g} [{tau_source}] and confirmed "
                   f"by the paired block bootstrap (branch "
                   f"{tb.get('branch')!r})"
                   f"{_winner_structure_note(_note_series(best, resid_best), sole_rho_flag)}")
    note = _winner_structure_note(_note_series(best, resid_best),
                                  sole_rho_flag)
    return Adjudication(
        EpistemicStatus.UNIQUE_SUPPORT, promoted=best, ranking=tuple(ranked),
        best_rmse=best_rmse, second_rmse=float(second.holdout_rmse),
        delta_rmse=delta, tau_epi=tau, tau_epi_source=tau_source,
        holdout_noise_scale=holdout_noise_scale,
        holdout_scale_source=scale_source, epistemic_margin=margin,
        reason=f"nearest distinct rival separated by Δ={delta:.3g} > "
               f"τ={tau:.3g} [{tau_source}]{note}")
