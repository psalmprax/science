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


REL_FLOOR = 0.05
"""Amplitude resolution floor: a rival whose holdout RMSE differs by less than
REL_FLOOR of the observed signal RMS is not resolved by the intervention
available. Dimensionless by construction, so it does not track problem units."""


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
    epistemic_margin: float | None = None
    tau_epi: float | None = None
    tau_epi_source: str = ""
    reason: str = ""


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


def adjudicate(fitted: list,
               tau_epi: float | None = None,
               holdout_noise_scale: float | None = None) -> Adjudication:
    """Rank by train loss, then adjudicate on holdout evidence. Deterministic."""
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
    if not g4.get("passed", False):
        return Adjudication(
            EpistemicStatus.INSUFFICIENT_EVIDENCE, ranking=tuple(ranked),
            best_rmse=best.holdout_rmse,
            holdout_noise_scale=holdout_noise_scale,
            reason=f"rank-1 {best.candidate.candidate_id} fails gate 4: "
                   f"{g4.get('detail', 'no holdout evidence')}; "
                   "no promotion of least-bad models")
    tau: float | None = None
    if tau_epi is not None:
        tau = float(tau_epi)
        tau_source = "explicit"
    else:
        tau = resolution_floor(g4.get("holdout_obs_rms"),
                               g4.get("holdout_noise_sigma"),
                               g4.get("holdout_n_samples"))
        tau_source = "holdout-scale"
    if tau is None:
        # No scale means no resolution. Abstain; never fall back to a constant
        # (a missing key must not read as "zero tolerance").
        return Adjudication(
            EpistemicStatus.INSUFFICIENT_EVIDENCE, ranking=tuple(ranked),
            best_rmse=best.holdout_rmse,
            holdout_noise_scale=holdout_noise_scale,
            reason="rank-1 passes gate 4 but the holdout record carries no "
                   "scale (obs_rms/noise_sigma), so no rival can be resolved; "
                   "pass tau_epi to override")
    best_rmse = best.holdout_rmse
    rivals = [s for s in ranked[1:]
              if _identity(s) != _identity(best) and s.holdout_rmse is not None
              and np.isfinite(s.holdout_rmse)]
    if not rivals:
        return Adjudication(
            EpistemicStatus.UNIQUE_SUPPORT, promoted=best, ranking=tuple(ranked),
            best_rmse=best_rmse, delta_rmse=None, tau_epi=tau,
            tau_epi_source=tau_source,
            holdout_noise_scale=holdout_noise_scale, epistemic_margin=None,
            reason="no structurally distinct rival survived scoring")
    second = min(rivals, key=lambda s: s.holdout_rmse)
    delta = float(second.holdout_rmse - best_rmse)
    margin = float(delta - tau)
    if delta <= tau:
        return Adjudication(
            EpistemicStatus.MULTI_HYPOTHESIS, ranking=tuple(ranked),
            best_rmse=best_rmse, second_rmse=float(second.holdout_rmse),
            delta_rmse=delta, tau_epi=tau, tau_epi_source=tau_source,
            holdout_noise_scale=holdout_noise_scale, epistemic_margin=margin,
            reason=f"rival {second.candidate.candidate_id} within τ "
                   f"(Δ={delta:.3g} ≤ τ={tau:.3g} [{tau_source}]); structures "
                   "observationally indistinguishable under available "
                   "interventions")
    return Adjudication(
        EpistemicStatus.UNIQUE_SUPPORT, promoted=best, ranking=tuple(ranked),
        best_rmse=best_rmse, second_rmse=float(second.holdout_rmse),
        delta_rmse=delta, tau_epi=tau, tau_epi_source=tau_source,
        holdout_noise_scale=holdout_noise_scale, epistemic_margin=margin,
        reason=f"nearest distinct rival separated by Δ={delta:.3g} > "
               f"τ={tau:.3g} [{tau_source}]")
