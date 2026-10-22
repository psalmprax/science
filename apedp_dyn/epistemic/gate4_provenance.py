"""Gate-4 provenance for proposer-emitted candidates (core-side producer).

:func:`apedp_dyn.epistemic.adjudicate.adjudicate` ranks candidates by train loss
and then adjudicates on holdout evidence read from
``ScoredCandidate.provenance["gate4_status"]``. Until this module existed, every
non-test producer of that key lived under ``apedp_dyn/sandbox/`` (the demonstration
harnesses), so the core referee was reachable only from those harnesses or by
hand-assembling gate evidence — the exact shape a fabrication risk takes.

This module is the production-side seam:

* :func:`measure_proposal` scores one :class:`CandidateProposal` on a frame and
  returns its error metrics (MSE / RMSE / MAE / R² / signed residuals). Proposal
  objects carry a ``compute_score`` callable but no error metric of their own, so
  callers previously had to reach for a key that belongs to a different proposer.
* :func:`gate4_status` emits the same key set the sandbox harnesses emit
  (``passed``, ``detail``, ``holdout_obs_rms``, ``holdout_noise_sigma``,
  ``holdout_n_samples``, ``rmse``, ``holdout_residuals``) from measured holdout
  evidence, and refuses to mark the gate passed when incrementality was never
  tested.
* :func:`scored_candidate` wraps a proposal into the
  :class:`~apedp_dyn.grammar.candidate_ir.ScoredCandidate` the referee consumes.

Honest boundaries (no silent upgrades):

* ``passed`` means *predictive incrementality was demonstrated at the canonical
  alpha* against a supplied incumbent prediction. Without an incumbent the gate is
  recorded as NOT passed with the reason stated — an untested claim is not a pass.
* The noise scale is never invented. Omitting ``noise_sigma`` leaves
  ``holdout_noise_sigma`` as ``None``, which makes the referee's resolution floor
  unavailable and therefore makes it abstain; :func:`noise_sigma_from_residuals`
  exists as an explicit, documented estimator the caller may choose to use.
* The referee's structural duplicate detection needs the typed grammar IR
  (``CandidateDynamics``). A proposal carries a free-form formula, so the bridge
  passes a reference object that the referee's own ``_identity`` fallback labels
  ``untyped``; canonicalisation is then unavailable by construction and is
  disclosed in the provenance rather than faked.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from apedp_dyn.conf.canonical_spec import DIEBOLD_MARIANO_P_VALUE_THRESHOLD
from apedp_dyn.epistemic.novelty_adjudication import evaluate_predictive_incrementality
from apedp_dyn.grammar.candidate_ir import ScoredCandidate
from apedp_dyn.grammar.proposer_interface import CandidateProposal

__all__ = [
    "ProposalErrorMetrics",
    "ProposalNotEvaluableError",
    "gate4_status",
    "measure_proposal",
    "noise_sigma_from_residuals",
    "scored_candidate",
]


class ProposalNotEvaluableError(ValueError):
    """Raised when a proposal cannot be scored on the frame it was handed.

    A missing column, a wrong output length, or fewer than two finite pairs all
    land here. Failing closed is deliberate: a silent zero-filled prediction
    would enter the holdout record as if it were evidence.
    """


@dataclass(frozen=True)
class ProposalErrorMetrics:
    """Error metrics for one (proposal, frame, target) evaluation."""

    candidate_id: str
    n: int
    mse: float
    rmse: float
    mae: float
    r2: float
    residuals: np.ndarray = None  # type: ignore[assignment]  # signed, observed - predicted

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe record (residuals excluded: they travel in gate 4 separately)."""
        return {
            "candidate_id": self.candidate_id,
            "n": self.n,
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
        }


def measure_proposal(
    proposal: CandidateProposal,
    frame: pd.DataFrame,
    y_true: np.ndarray,
) -> ProposalErrorMetrics:
    """Score ``proposal`` on ``frame`` against ``y_true``.

    Non-finite pairs are dropped jointly (a prediction that is undefined at a
    point carries no evidence about it); everything else is computed on the
    surviving rows. Raises :class:`ProposalNotEvaluableError` when fewer than two
    finite pairs survive.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    try:
        pred = np.asarray(proposal.compute_score(frame), dtype=float).ravel()
    except Exception as exc:  # noqa: BLE001 - reframed as a typed refusal below
        raise ProposalNotEvaluableError(
            f"{proposal.candidate_id}: compute_score raised "
            f"{type(exc).__name__}: {exc}") from exc
    if pred.shape != y.shape:
        raise ProposalNotEvaluableError(
            f"{proposal.candidate_id}: prediction length {pred.shape} does not match "
            f"target length {y.shape}")
    mask = np.isfinite(y) & np.isfinite(pred)
    n = int(mask.sum())
    if n < 2:
        raise ProposalNotEvaluableError(
            f"{proposal.candidate_id}: only {n} finite (target, prediction) pair(s); "
            "an error metric needs at least 2")
    resid = y[mask] - pred[mask]
    mse = float(np.mean(resid ** 2))
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    if sst <= 1e-18:
        r2 = 1.0 if sse <= 1e-18 else float("-inf")
    else:
        r2 = 1.0 - sse / sst
    return ProposalErrorMetrics(
        candidate_id=proposal.candidate_id,
        n=n,
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mae=float(np.mean(np.abs(resid))),
        r2=float(r2),
        residuals=resid,
    )


def noise_sigma_from_residuals(residuals: np.ndarray, scale: float = 1.4826) -> float:
    """Robust noise-scale estimator (MAD), offered as an explicit opt-in.

    Never applied implicitly: a gate that quietly estimated its own noise floor
    could manufacture the resolution it then reports. Callers that have a
    measurement-noise figure from the instrument should pass that instead.
    """
    arr = np.asarray(residuals, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ProposalNotEvaluableError("cannot estimate a noise scale from no finite residuals")
    return float(scale * np.median(np.abs(arr - np.median(arr))))


def gate4_status(
    proposal: CandidateProposal,
    *,
    train_frame: pd.DataFrame,
    y_train: np.ndarray,
    holdout_frame: pd.DataFrame,
    y_holdout: np.ndarray,
    incumbent_pred: np.ndarray | None = None,
    incumbent_id: str = "incumbent",
    noise_sigma: float | None = None,
    alpha: float = DIEBOLD_MARIANO_P_VALUE_THRESHOLD,
) -> dict[str, Any]:
    """Build the ``gate4_status`` provenance dict for one proposal.

    Emits the key set ``adjudicate`` reads. ``passed`` is True only when
    ``incumbent_pred`` was supplied AND the Diebold-Mariano test at ``alpha``
    reports ``PREDICTIVELY_INCREMENTAL``; otherwise the reason is recorded in
    ``detail`` and the gate stays closed.
    """
    train_metrics = measure_proposal(proposal, train_frame, y_train)
    holdout_metrics = measure_proposal(proposal, holdout_frame, y_holdout)

    y_ho = np.asarray(y_holdout, dtype=float).ravel()
    finite_ho = y_ho[np.isfinite(y_ho)]
    obs_rms = float(np.sqrt(np.mean(finite_ho ** 2))) if finite_ho.size else float("nan")

    status: dict[str, Any] = {
        "passed": False,
        "holdout_obs_rms": obs_rms,
        "holdout_noise_sigma": None if noise_sigma is None else float(noise_sigma),
        "holdout_n_samples": holdout_metrics.n,
        "rmse": holdout_metrics.rmse,
        "holdout_residuals": holdout_metrics.residuals.tolist(),
        "alpha": float(alpha),
        "candidate_id": proposal.candidate_id,
        "proposal_metrics": {
            "train": train_metrics.to_dict(),
            "holdout": holdout_metrics.to_dict(),
        },
        "canonicalisation": "untyped (proposal formula is not grammar-IR)",
    }

    if incumbent_pred is None:
        status["detail"] = (
            "no incumbent prediction supplied: predictive incrementality was not "
            "tested, so gate 4 is not passed")
        return status

    try:
        pred_ho = np.asarray(proposal.compute_score(holdout_frame), dtype=float).ravel()
    except Exception as exc:  # noqa: BLE001 - reframed as a closed gate below
        status["detail"] = (
            f"compute_score raised on the holdout frame: {type(exc).__name__}: {exc}")
        return status

    res = evaluate_predictive_incrementality(
        y_ho, pred_ho, incumbent_pred, alpha=alpha)
    status.update({
        "incumbent": incumbent_id,
        "dm_statistic": float(res.dm_statistic),
        "dm_p_value": float(res.p_value),
        "loss_reduction": float(res.loss_reduction),
        "passed": bool(res.verdict == "PREDICTIVELY_INCREMENTAL"),
    })
    refusal = res.details.get("error")
    status["detail"] = (
        f"Diebold-Mariano vs {incumbent_id} at alpha={alpha:g}: "
        f"dm={res.dm_statistic:.4f}, p={res.p_value:.4g}, "
        f"loss_reduction={res.loss_reduction:.4g} -> {res.verdict}"
        + (f" ({refusal})" if refusal else ""))
    return status


@dataclass(frozen=True)
class _ProposalReference:
    """Minimal stand-in for the referee's typed candidate IR.

    Carries what ``_identity`` needs to fall back on (a stable ``candidate_id``)
    and what provenance writers read. Deliberately NOT a ``CandidateDynamics``:
    a free-form proposal cannot be expressed in the closed grammar vocabulary,
    and pretending otherwise would let the referee "canonicalise" a term set the
    proposal never had.
    """

    candidate_id: str
    formula: str


def scored_candidate(
    proposal: CandidateProposal,
    *,
    train_frame: pd.DataFrame,
    y_train: np.ndarray,
    holdout_frame: pd.DataFrame,
    y_holdout: np.ndarray,
    incumbent_pred: np.ndarray | None = None,
    incumbent_id: str = "incumbent",
    noise_sigma: float | None = None,
    alpha: float = DIEBOLD_MARIANO_P_VALUE_THRESHOLD,
    extra_provenance: dict[str, Any] | None = None,
) -> ScoredCandidate:
    """Bridge one :class:`CandidateProposal` into the referee's input.

    ``train_loss`` is the train MSE (the referee ranks by it), ``holdout_rmse`` is
    the measured holdout RMSE, and ``provenance["gate4_status"]`` is the dict from
    :func:`gate4_status` — so ``adjudicate([scored_candidate(...), ...])`` needs no
    sandbox harness and no hand-built evidence.
    """
    status = gate4_status(
        proposal,
        train_frame=train_frame, y_train=y_train,
        holdout_frame=holdout_frame, y_holdout=y_holdout,
        incumbent_pred=incumbent_pred, incumbent_id=incumbent_id,
        noise_sigma=noise_sigma, alpha=alpha)

    provenance: dict[str, Any] = {
        "gate4_status": status,
        "data_origin": str((extra_provenance or {}).get("data_origin", "unknown")),
        "proposal_formula": proposal.formula,
        "proposal_proposer_id": proposal.proposer_id,
    }
    if extra_provenance:
        provenance.update({k: v for k, v in extra_provenance.items() if k != "gate4_status"})

    return ScoredCandidate(
        candidate=_ProposalReference(
            candidate_id=proposal.candidate_id, formula=proposal.formula),
        provenance=provenance,
        train_loss=float(status["proposal_metrics"]["train"]["mse"]),
        holdout_rmse=float(status["rmse"]),
    )
