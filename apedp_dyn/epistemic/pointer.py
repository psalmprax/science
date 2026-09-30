"""Informative-abstention pointer (v1, evidence-only).

Position in the architecture (frozen):

    [gates 1-4, admissibility, bootstrap adjudication, statuses]
                          |
                          v (read-only inspection)
                 POINTER (this module)
                          |
                          v
              Recommendation (never a status change)

Contract:
  - Pure functions of already-generated evidence. The pointer receives plain
    data (status strings, margins, measured kappa, residual correlations,
    drive amplitudes). It never imports the fitter, simulator, or gates, and
    it cannot promote, demote, or re-route any candidate.
  - Output is a Recommendation: the echoed input status plus
    {evidence_deficit, suggested_intervention, diagnostic_basis, confidence}.
  - When the evidence does not support a specific deficit, the pointer says
    "unclassified_deficit" rather than inventing one. Restraint is tested.
  - Confidence is a documented v1 heuristic (relative ordering signal only),
    not a calibrated probability.

Routing (frozen precedence; thresholds frozen pre-outcome):
  1. status MULTI_HYPOTHESIS + measured kappa >= KAPPA_SEVERE (1e3)
     -> collinear_predictors / excite variable space.
  2. max |residual correlation vs omitted-channel| >= RESID_CORR_FLAG (0.5)
     -> missing_structure_or_regime / probe the structured slice.
  3. status UNIQUE_SUPPORT + structured winner residuals (|rho1| >= 0.5)
     -> qualified_support / trust with caution (promotion stands; the
     ledger records the qualification).
  4. status in (INSUFFICIENT_EVIDENCE, NO_VALID_CANDIDATE) + structured
     remainder (|rho1| >= 0.5) with no measured channel explaining it
     -> hypothesis_space / the grammar may not contain the truth; consult
     an external proposer or widen the vocabulary.
  5. status in (MULTI_HYPOTHESIS, INSUFFICIENT_EVIDENCE, NO_VALID_CANDIDATE)
     + drive_ratio (holdout/train excitation) < DRIVE_RATIO_WEAK (2.0)
     -> weak_intervention / stronger drive across the suspected boundary.
  6. otherwise -> unclassified_deficit / broaden evidence (generic).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

POINTER_VERSION = "1.1.0-evidence-only"

KAPPA_SEVERE = 1e3
RESID_CORR_FLAG = 0.5
DRIVE_RATIO_WEAK = 2.0

DEFICIT_VOCABULARY = (
    "collinear_predictors",
    "missing_structure_or_regime",
    "qualified_support",
    "hypothesis_space",
    "weak_intervention",
    "unclassified_deficit",
)


@dataclass(frozen=True)
class Recommendation:
    """Explanation of an abstention/refusal. Never a promotion decision."""

    status: str
    evidence_deficit: str
    suggested_intervention: str
    diagnostic_basis: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    pointer_version: str = POINTER_VERSION


def _confidence_collinear(kappa: float) -> float:
    import math

    return round(min(0.95, 0.5 + math.log10(max(kappa, 1.0) / 1e2) / 3.0), 3)


def recommend(
    status: str,
    delta_rmse: float | None = None,
    tau_epi: float | None = None,
    kappa_measured: float | None = None,
    resid_corr: float | None = None,
    resid_corr_label: str = "",
    drive_ratio: float | None = None,
    resid_rho1: float | None = None,
    winner_rho1: float | None = None,
) -> Recommendation:
    """Recommend the discriminating next experiment for a non-promoted outcome
    (or qualify a promoted one).

    All arguments are already-measured evidence. Returns a Recommendation;
    the input status is echoed unchanged (asserted by tests). resid_rho1 is
    the lag-1 autocorrelation of an unexplained residual remainder;
    winner_rho1 the same for a promoted winner's residuals.
    """
    base: dict[str, Any] = {}
    if delta_rmse is not None:
        base["delta_rmse"] = float(delta_rmse)
    if tau_epi is not None:
        base["tau_epi"] = float(tau_epi)

    if status == "multi_hypothesis" and kappa_measured is not None \
            and kappa_measured >= KAPPA_SEVERE:
        return Recommendation(
            status=status,
            evidence_deficit="collinear_predictors",
            suggested_intervention=(
                "excite variable space: add experiments spanning the weakly "
                "identified predictor directions (reduce kappa below 1e2)"),
            diagnostic_basis={**base, "kappa_measured": float(kappa_measured),
                              "kappa_band": "severe"},
            confidence=_confidence_collinear(float(kappa_measured)),
        )
    if resid_corr is not None and abs(float(resid_corr)) >= RESID_CORR_FLAG:
        return Recommendation(
            status=status,
            evidence_deficit="missing_structure_or_regime",
            suggested_intervention=(
                "probe the regime slice where residuals structure peaks; add a "
                "measurement channel for the suspected omitted loss/structure"),
            diagnostic_basis={**base, "resid_corr": float(resid_corr),
                              "resid_corr_label": resid_corr_label},
            confidence=round(min(0.95, abs(float(resid_corr))), 3),
        )
    if status == "unique_support" and winner_rho1 is not None \
            and abs(float(winner_rho1)) >= RESID_CORR_FLAG:
        return Recommendation(
            status=status,
            evidence_deficit="qualified_support",
            suggested_intervention=(
                "promotion stands but residuals are structured: widen "
                "prediction intervals and probe the structured slice before "
                "acting on the model"),
            diagnostic_basis={**base, "winner_rho1": float(winner_rho1)},
            confidence=round(min(0.95, abs(float(winner_rho1))), 3),
        )
    if status in ("insufficient_evidence", "no_valid_candidate") \
            and resid_rho1 is not None and abs(float(resid_rho1)) >= RESID_CORR_FLAG \
            and not (resid_corr is not None and abs(float(resid_corr)) >= RESID_CORR_FLAG):
        return Recommendation(
            status=status,
            evidence_deficit="hypothesis_space",
            suggested_intervention=(
                "no grammar term explains the structured remainder: widen the "
                "vocabulary or consult an external proposer over this regime"),
            diagnostic_basis={**base, "resid_rho1": float(resid_rho1)},
            confidence=round(min(0.95, abs(float(resid_rho1))), 3),
        )
    if status in ("multi_hypothesis", "insufficient_evidence", "no_valid_candidate") \
            and drive_ratio is not None and drive_ratio < DRIVE_RATIO_WEAK:
        return Recommendation(
            status=status,
            evidence_deficit="weak_intervention",
            suggested_intervention=(
                "apply a stronger intervention: increase drive amplitude across "
                "the suspected transition boundary (target holdout/train > 2)"),
            diagnostic_basis={**base, "drive_ratio": float(drive_ratio)},
            confidence=round(max(0.1, 1.0 - float(drive_ratio) / DRIVE_RATIO_WEAK), 3),
        )
    return Recommendation(
        status=status,
        evidence_deficit="unclassified_deficit",
        suggested_intervention=(
            "broaden evidence: no single deficit is supported by the available "
            "diagnostics; add independent variation before re-adjudicating"),
        diagnostic_basis=base,
        confidence=0.2,
    )
