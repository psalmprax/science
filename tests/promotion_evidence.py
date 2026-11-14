"""Synthetic evidence for driving ``PromotionTracker`` through its derived path.

The ``verify_stage_N_from_*`` entry points take evidence (series, ledgers, a
Lean result) rather than numbers. Tests that need a candidate which has
genuinely passed a stage build that evidence here, so they exercise the same
computation a real caller does instead of opting into reported metrics.

Nothing here is a fixture file: every series is generated from a seed.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np

from apedp_dyn.epistemic.agnostic_adapter import InvariantKind
from apedp_dyn.epistemic.experiment_ledger import FullLineageExperimentLedger
from apedp_dyn.epistemic.lean_universal_bridge import (
    LeanTheoremVerification,
    LeanVerificationResult,
)
from apedp_dyn.epistemic.promotion_lifecycle import PromotionTracker
from apedp_dyn.epistemic.proof_claim import ProofClaimStatus
from apedp_dyn.epistemic.trial_ledger import CandidateTrialLedger
from apedp_dyn.market.universe import UniverseAudit

UTC = datetime.timezone.utc


def lean_result(
    candidate_id: str,
    *,
    status: ProofClaimStatus = ProofClaimStatus.P2_VERIFIED,
    kinds: tuple[InvariantKind, ...] = (
        InvariantKind.SCALE_HOMOGENEITY,
        InvariantKind.BOUND_POSITIVITY,
    ),
    proven: bool = True,
    axiom_dependent: bool = False,
    kernel_exit_code: int | None = 0,
    source: str = "theorem scale (x : Nat) : x = x := rfl",
) -> LeanVerificationResult:
    """A Lean verification result as the bridge would return it."""
    theorems = [
        LeanTheoremVerification(
            invariant_name=f"inv_{kind.value}",
            invariant_kind=kind,
            theorem_identifier=f"thm_{kind.value}",
            lean_theorem_statement="x = x",
            lean_proof_script="rfl",
            is_formally_proven=proven,
            proof_method="lean4_kernel",
            diagnostic_message="",
            axiom_dependent=axiom_dependent,
        )
        for kind in kinds
    ]
    return LeanVerificationResult(
        candidate_id=candidate_id,
        formula_repr="f(x)",
        all_theorems_proven=proven,
        proof_claim_status=status,
        theorems=theorems,
        lean_source_code=source,
        kernel_exit_code=kernel_exit_code,
    )


def correlated_pair(n: int, strength: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(predictions, targets)`` whose rank correlation is roughly ``strength``."""
    rng = np.random.default_rng(seed)
    predictions = rng.normal(size=n)
    targets = strength * predictions + np.sqrt(max(1.0 - strength**2, 0.0)) * rng.normal(size=n)
    return predictions, targets


def business_days(start: str, periods: int) -> np.ndarray:
    """``periods`` consecutive weekdays from ``start`` as ``datetime64[D]``."""
    return np.busday_offset(np.datetime64(start, "D"), np.arange(periods), roll="forward")


def forecasting_evidence(n: int = 600, seed: int = 11) -> dict[str, Any]:
    """Stage-3 evidence: a candidate that out-forecasts two incumbents and is
    not a relabelling of either literature feature."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    literature = {"KYLE_LAMBDA": rng.normal(size=n), "AMIHUD_ILLIQUIDITY": rng.normal(size=n)}
    y_true = signal + 0.5 * rng.normal(size=n)
    return {
        "y_true": y_true,
        "candidate_forecast": signal,
        "incumbent_forecasts": {
            "KYLE_LAMBDA": 0.1 * literature["KYLE_LAMBDA"],
            "AMIHUD_ILLIQUIDITY": 0.1 * literature["AMIHUD_ILLIQUIDITY"],
        },
        "candidate_signal": signal,
        "literature_features": literature,
    }


def strategy_returns(
    periods: int = 504, seed: int = 5, start: str = "2024-01-02"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(gross_returns, turnover, index)`` of a strong daily strategy."""
    rng = np.random.default_rng(seed)
    gross = rng.normal(0.0025, 0.008, periods)
    turnover = np.full(periods, 0.4)
    return gross, turnover, business_days(start, periods)


def persistent_trial_ledger(path: Path, n_trials: int = 3, seed: int = 3) -> CandidateTrialLedger:
    """A journaled search of ``n_trials`` similar candidates."""
    ledger = CandidateTrialLedger(persistence_path=path)
    rng = np.random.default_rng(seed)
    for i in range(n_trials):
        ledger.record_trial(
            f"trial_{i}", "proposer", f"f_{i}", "TACTICAL", float(1.5 + 0.05 * rng.normal()), n_bars=504
        )
    return ledger


def evidence_backed_tracker(
    candidate_id: str,
    workdir: Path,
    *,
    through: int = 4,
    experiment_ledger: FullLineageExperimentLedger | None = None,
    state_path: Path | None = None,
) -> PromotionTracker:
    """A tracker that passed stages ``1..through`` (at most 5) on derived evidence.

    ``experiment_ledger`` must be persistent; one is created under ``workdir``
    and the candidate registered in it when none is given. Stage 6 needs the
    clock to move past the stage-5 record, so tests drive it themselves.
    """
    if not 1 <= through <= 5:
        raise ValueError("through must be between 1 and 5")
    tracker = PromotionTracker(candidate_id, state_path=state_path)
    tracker.verify_stage_1_from_lean(lean_result(candidate_id))
    if through >= 2:
        tracker.verify_stage_2_from_predictions(*correlated_pair(600, 0.30, seed=1))
    if through >= 3:
        tracker.verify_stage_3_from_forecasts(**forecasting_evidence())
    if through >= 4:
        if experiment_ledger is None:
            experiment_ledger = FullLineageExperimentLedger(
                persistence_path=workdir / f"{candidate_id}_experiments.jsonl"
            )
        try:
            experiment_ledger.get_candidate(candidate_id)
        except KeyError:
            experiment_ledger.register_candidate(candidate_id, "proposer", "f(x)")
        predictions, targets = correlated_pair(400, 0.28, seed=2)
        tracker.verify_stage_4_from_holdout(
            predictions,
            targets,
            experiment_ledger=experiment_ledger,
            dataset_id=f"{candidate_id}_data",
            slice_identifier="confirmation_2025",
            index=business_days("2025-01-02", 400),
        )
    if through >= 5:
        gross, turnover, index = strategy_returns()
        tracker.verify_stage_5_from_returns(
            gross,
            turnover,
            index,
            trial_ledger=persistent_trial_ledger(workdir / f"{candidate_id}_trials.jsonl"),
            universe=UniverseAudit.single_instrument("SYNTHETIC"),
        )
    return tracker
