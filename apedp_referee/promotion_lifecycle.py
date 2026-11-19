"""Six-Stage Scoped Promotion State Machine.

Enforces separate, non-fungible promotion stages per the reviewer's specification:
  Stage 1: MATHEMATICAL_CONSISTENCY (Lean 4 kernel, dimensional homogeneity, bounds)
  Stage 2: PREDICTIVE_RECOVERY (In-sample error reduction, cross-validation stability)
  Stage 3: UNIQUE_IDENTIFICATION (Diebold-Mariano incrementality & structural novelty)
  Stage 4: OUT_OF_SAMPLE_TRANSFER (Untouched, embargoed holdout transfer)
  Stage 5: ECONOMIC_VALIDITY (Net return after friction, fee breakeven, DSR >= 0.95)
  Stage 6: REAL_MARKET_PROMOTION (Prospective execution meeting pre-registered hurdles)

Passing one stage never implies or confers promotion through subsequent stages.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from apedp_referee.canonical_spec import (
    DIEBOLD_MARIANO_P_VALUE_THRESHOLD,
    DSR_THRESHOLD,
    MAX_LITERATURE_R2,
    MIN_NET_SHARPE,
    MIN_OOS_TRANSFER_RETENTION,
    STAGE2_MIN_IC,
    STAGE4_MIN_OOS_IC,
    STAGE4_OOS_ALPHA,
    STAGE5_MIN_FEE_BPS,
    STAGE6_MAX_DRAWDOWN_CEILING,
    STAGE6_PROSPECTIVE_SHARPE_HURDLE,
)

if TYPE_CHECKING:
    from apedp_referee.experiment_ledger import FullLineageExperimentLedger
    from apedp_referee.proof_evidence import LeanVerificationResult
    from apedp_referee.trial_ledger import CandidateTrialLedger
    from apedp_referee.universe import UniverseAudit

#: Where a stage's metrics came from. ``derived`` means this module computed
#: them from the evidence it was handed (series, ledgers, a Lean result);
#: ``caller_reported`` means the caller typed the numbers in.
EVIDENCE_DERIVED = "derived"
EVIDENCE_REPORTED = "caller_reported"

#: Suffix on the status label of a stage that *passed* on reported numbers, so a
#: printed or logged label cannot be mistaken for an evidence-backed one.
REPORTED_LABEL_SUFFIX = "_ON_REPORTED_METRICS"

#: Procedure constants of the ``*_from_*`` variants. They are not acceptance
#: thresholds -- every threshold a stage is judged against still comes from the
#: canonical specification -- they only say how much evidence is enough to
#: compute a metric at all. Twenty observations is the floor the Diebold-Mariano
#: test in ``novelty_adjudication`` already applies.
MIN_EVIDENCE_OBSERVATIONS = 20
DEFAULT_CV_FOLDS = 5


def _utc_now() -> datetime.datetime:
    """Wall clock used to stamp stage records (patched by tests)."""
    return datetime.datetime.now(datetime.timezone.utc)


class PromotionStage(IntEnum):
    UNEVALUATED = 0
    STAGE_1_MATHEMATICAL_CONSISTENCY = 1
    STAGE_2_PREDICTIVE_RECOVERY = 2
    STAGE_3_UNIQUE_IDENTIFICATION = 3
    STAGE_4_OUT_OF_SAMPLE_TRANSFER = 4
    STAGE_5_ECONOMIC_VALIDITY = 5
    STAGE_6_REAL_MARKET_PROMOTION = 6


class PromotionPrerequisiteError(RuntimeError):
    """Raised when attempting to advance to a stage without passing all preceding stages."""
    pass


class PromotionViolationError(PromotionPrerequisiteError):
    """Raised when a promotion stage metric contradicts an earlier certified stage record."""
    pass


@dataclass
class StageVerificationRecord:
    stage: PromotionStage
    passed: bool
    status_label: str
    metrics: dict[str, Any]
    reason: str
    evidence_source: str = EVIDENCE_REPORTED
    recorded_at: str = ""


def information_coefficient(predictions: Any, targets: Any) -> tuple[float, int]:
    """Spearman rank information coefficient over the finite pairs.

    Returns ``(ic, n)``. ``ic`` is NaN when fewer than
    ``MIN_EVIDENCE_OBSERVATIONS`` finite pairs remain or either side is constant.
    """
    from scipy.stats import spearmanr

    pred = np.asarray(predictions, dtype=float).ravel()
    target = np.asarray(targets, dtype=float).ravel()
    if pred.shape != target.shape:
        raise ValueError(f"predictions {pred.shape} and targets {target.shape} differ in length")
    mask = np.isfinite(pred) & np.isfinite(target)
    pred, target = pred[mask], target[mask]
    n = int(pred.size)
    if n < MIN_EVIDENCE_OBSERVATIONS or np.ptp(pred) == 0.0 or np.ptp(target) == 0.0:
        return float("nan"), n
    return float(spearmanr(pred, target).statistic), n


def one_sided_ic_p_value(ic: float, n_effective: int) -> float:
    """P(IC >= observed | no association), Student-t with ``n_effective - 2`` dof.

    ``n_effective`` is the number of *independent* observations: with targets
    that overlap over ``h`` periods it is ``n / h``, not ``n``.
    """
    from scipy.stats import t as student_t

    if not math.isfinite(ic) or n_effective < 3:
        return 1.0
    if abs(ic) >= 1.0:
        return 0.0 if ic > 0 else 1.0
    statistic = ic * math.sqrt((n_effective - 2) / (1.0 - ic * ic))
    return float(student_t.sf(statistic, n_effective - 2))


def effective_sample_size(predictions: Any, targets: Any) -> int:
    """Independent observations in a pair of serially dependent series.

    When both series are autocorrelated -- a slow signal against returns that
    overlap from one bar to the next -- a correlation between them is far less
    certain than its ``n`` suggests. Bartlett's formula gives its variance as
    ``(1 + 2 * sum_k rho_x(k) * rho_y(k)) / n``; this returns ``n`` divided by
    that inflation factor, computed on ranks to match the Spearman IC, over lags
    up to ``n // 10``. The factor is never taken below one, so the result is at
    most ``n``. Measured in ``benchmarks/run_promotion_gate_calibration.py``: a
    signal with no relation to 21-bar overlapping targets passed stage 4 in
    about a third of trials when ``n`` was used as it stood.
    """
    from scipy.stats import rankdata

    pred = np.asarray(predictions, dtype=float).ravel()
    target = np.asarray(targets, dtype=float).ravel()
    mask = np.isfinite(pred) & np.isfinite(target)
    n = int(mask.sum())
    if n < MIN_EVIDENCE_OBSERVATIONS:
        return n
    x = rankdata(pred[mask])
    y = rankdata(target[mask])
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return n
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    inflation = 1.0
    for lag in range(1, n // 10 + 1):
        rho_x = float(np.dot(x[lag:], x[:-lag])) / n
        rho_y = float(np.dot(y[lag:], y[:-lag])) / n
        inflation += 2.0 * (1.0 - lag / n) * rho_x * rho_y
    return max(3, int(n / max(inflation, 1.0)))


def _periods_per_year(index: Any, n_rows: int) -> tuple[float, int, int]:
    """Observation frequency implied by a datetime index: (per year, first ns, last ns).

    Taken from the timestamps so the annualisation factor is a property of the
    data, not a number the caller chooses.
    """
    from apedp_referee.experiment_ledger import _canonical_index

    kind, _row_bytes, (first, last) = _canonical_index(index, n_rows)
    if kind != "datetime_ns":
        raise ValueError("a datetime index is required to annualise returns")
    span_years = (last - first) / 1e9 / (365.25 * 86400.0)
    if n_rows < 2 or span_years <= 0.0:
        raise ValueError("the index must span a positive length of time")
    return (n_rows - 1) / span_years, int(first), int(last)


def _no_looser_floor(name: str, supplied: float, canonical: float) -> float:
    """Return ``supplied`` for a minimum-style threshold, refusing a looser one.

    The canonical value is the preregistered floor. A caller may hold a
    candidate to a stricter (higher) floor; a lower or non-finite one would let
    the caller decide its own acceptance, so it raises instead of applying.
    """
    value = float(supplied)
    if not math.isfinite(value) or value < float(canonical):
        raise PromotionViolationError(
            f"{name}={supplied!r} is looser than the preregistered floor {canonical!r}; "
            "promotion thresholds may only be tightened by the caller."
        )
    return value


def _no_looser_ceiling(name: str, supplied: float, canonical: float) -> float:
    """Return ``supplied`` for a maximum-style threshold, refusing a looser one."""
    value = float(supplied)
    if not math.isfinite(value) or value > float(canonical):
        raise PromotionViolationError(
            f"{name}={supplied!r} is looser than the preregistered ceiling {canonical!r}; "
            "promotion thresholds may only be tightened by the caller."
        )
    return value


@dataclass
class PromotionTracker:
    """Per-candidate promotion state.

    ``state_path`` (optional) is an append-only JSONL journal of stage outcomes.
    When given, a tracker built for a ``candidate_id`` that already has journaled
    outcomes resumes from them, so a halt survives a new tracker instance or a
    new process instead of being forgotten with the object.

    Each stage has two entry points. ``verify_stage_N_from_*`` takes the
    evidence -- the series, the ledgers, the Lean result -- and computes the
    stage's metrics here. ``verify_stage_N`` takes the metrics as numbers; it is
    refused unless the tracker was built with ``allow_reported_metrics=True``,
    and a stage passed that way is recorded as ``caller_reported`` and labelled
    ``..._ON_REPORTED_METRICS``. ``is_evidence_backed()`` is True only when
    every stage passed so far was derived.
    """

    candidate_id: str
    highest_stage_passed: PromotionStage = PromotionStage.UNEVALUATED
    is_halted: bool = False
    halt_reason: str | None = None
    stage_records: dict[PromotionStage, StageVerificationRecord] = field(default_factory=dict)
    state_path: Path | str | None = None
    allow_reported_metrics: bool = False

    def __post_init__(self) -> None:
        if self.state_path is not None:
            self.state_path = Path(self.state_path)
            self._restore_from_journal()

    # ------------------------------------------------------------------
    # Entry points that compute the metrics from the evidence
    # ------------------------------------------------------------------

    def verify_stage_1_from_lean(self, lean_result: LeanVerificationResult) -> StageVerificationRecord:
        """Stage 1 from a Lean verification result, not from a status string.

        The stage passes only if the result is about this candidate, the Lean
        kernel accepted the module (exit code 0, every theorem proven), nothing
        in it rests on an axiom the module declared itself or on ``sorry``, and
        it contains a proven scale-homogeneity (dimensional) theorem and a
        proven bound/positivity theorem. A ``P2`` label alone is not enough.
        """
        self._assert_stage_reachable(PromotionStage.STAGE_1_MATHEMATICAL_CONSISTENCY)
        if lean_result.candidate_id != self.candidate_id:
            raise PromotionViolationError(
                f"Lean result is for candidate {lean_result.candidate_id!r}, "
                f"not {self.candidate_id!r}."
            )
        theorems = list(lean_result.theorems)
        axiom_dependent = sum(1 for t in theorems if t.axiom_dependent)
        uses_sorry = bool(re.search(r"\bsorry\b", lean_result.lean_source_code or ""))
        claimed = lean_result.proof_claim_status.name
        independently_checked = (
            claimed == "P2_VERIFIED"
            and bool(lean_result.all_theorems_proven)
            and lean_result.kernel_exit_code == 0
            and bool(theorems)
            and axiom_dependent == 0
            and not uses_sorry
        )

        def _proven(kind: str) -> bool:
            of_kind = [t for t in theorems if t.invariant_kind.value == kind]
            return bool(of_kind) and all(t.is_formally_proven and not t.axiom_dependent for t in of_kind)

        # A P2 label that does not survive the checks above is reported as the
        # P1 candidate it actually is.
        status = "P2_VERIFIED" if independently_checked else (
            "P1_CANDIDATE" if claimed == "P2_VERIFIED" else claimed
        )
        return self._decide_stage_1(
            status,
            _proven("scale_homogeneity"),
            _proven("bound_positivity"),
            source=EVIDENCE_DERIVED,
            derived={
                "lean_claimed_status": claimed,
                "lean_kernel_exit_code": lean_result.kernel_exit_code,
                "lean_theorem_count": len(theorems),
                "lean_axiom_dependent_theorems": axiom_dependent,
                "lean_uses_sorry": uses_sorry,
            },
        )

    def verify_stage_2_from_predictions(
        self,
        predictions: Any,
        targets: Any,
        n_folds: int = DEFAULT_CV_FOLDS,
    ) -> StageVerificationRecord:
        """Stage 2 from the development-sample predictions and realised targets.

        The information coefficient is the Spearman rank correlation over the
        finite pairs. Cross-validation stability is computed too: the sample is
        cut into ``n_folds`` contiguous blocks (order is kept, so a time series
        is not shuffled across its own future) and the IC must be positive in
        every block, each of which needs ``MIN_EVIDENCE_OBSERVATIONS`` pairs.
        """
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2; one fold is not a stability check")
        self._assert_stage_reachable(PromotionStage.STAGE_2_PREDICTIVE_RECOVERY)
        ic, n = information_coefficient(predictions, targets)
        pred = np.asarray(predictions, dtype=float).ravel()
        target = np.asarray(targets, dtype=float).ravel()
        mask = np.isfinite(pred) & np.isfinite(target)
        fold_ics = [
            information_coefficient(pred[mask][block], target[mask][block])[0]
            for block in np.array_split(np.arange(int(mask.sum())), n_folds)
        ]
        cv_stable = all(math.isfinite(value) and value > 0.0 for value in fold_ics)
        return self._decide_stage_2(
            ic,
            cv_stable,
            source=EVIDENCE_DERIVED,
            derived={
                "ic_method": "spearman",
                "n_observations": n,
                "n_folds": n_folds,
                "fold_ics": [float(value) for value in fold_ics],
            },
        )

    def verify_stage_3_from_forecasts(
        self,
        y_true: Any,
        candidate_forecast: Any,
        incumbent_forecasts: Mapping[str, Any],
        candidate_signal: Any,
        literature_features: Mapping[str, Any],
    ) -> StageVerificationRecord:
        """Stage 3 from the forecasts and the literature features themselves.

        The Diebold-Mariano test (HAC variance) is run against every incumbent
        forecast and the *largest* p-value is the one judged, so the candidate
        must beat each of them. The literature R-squared is the projection of
        the candidate signal on the supplied literature features, singly and
        jointly. Every supplied feature must be evaluable: a baseline that is
        constant or too short would otherwise silently drop out and leave an
        R-squared of zero.
        """
        self._assert_stage_reachable(PromotionStage.STAGE_3_UNIQUE_IDENTIFICATION)
        if not incumbent_forecasts or not literature_features:
            raise ValueError(
                "stage 3 needs at least one incumbent forecast and one literature feature"
            )
        from apedp_referee.novelty_adjudication import (
            evaluate_predictive_incrementality,
            evaluate_structural_novelty,
        )

        dm_results = {
            name: evaluate_predictive_incrementality(y_true, candidate_forecast, forecast)
            for name, forecast in incumbent_forecasts.items()
        }
        hardest = max(dm_results, key=lambda name: dm_results[name].p_value)
        novelty = evaluate_structural_novelty(candidate_signal, dict(literature_features))
        evaluated = [name for name in novelty.details if name != "JOINT_LITERATURE_SPAN"]
        all_evaluated = len(evaluated) == len(literature_features)
        literature_r2 = float(novelty.max_projection_r2) if all_evaluated else float("nan")
        return self._decide_stage_3(
            float(dm_results[hardest].p_value),
            literature_r2,
            source=EVIDENCE_DERIVED,
            derived={
                "dm_hardest_incumbent": hardest,
                "dm_p_values": {name: float(res.p_value) for name, res in dm_results.items()},
                "dm_loss_reduction": float(dm_results[hardest].loss_reduction),
                "literature_closest_baseline": novelty.closest_literature_baseline,
                "literature_features_evaluated": sorted(evaluated),
                "literature_features_supplied": sorted(literature_features),
            },
        )

    def verify_stage_4_from_holdout(
        self,
        predictions: Any,
        targets: Any,
        *,
        experiment_ledger: FullLineageExperimentLedger,
        dataset_id: str,
        slice_identifier: str,
        index: Any = None,
        horizon: int = 1,
    ) -> StageVerificationRecord:
        """Stage 4 from the holdout predictions and targets, through the ledger.

        Before anything is scored, the slice is claimed in ``experiment_ledger``
        as a confirmation holdout for this candidate, fingerprinted by its
        content (``targets`` and, if given, ``index``). A slice that is burned,
        already spent by another candidate, or the same data under another name
        raises ``HoldoutContaminationError`` and nothing is scored. The outcome
        is then written back, so a failed transfer burns the slice. "Untouched"
        is therefore the ledger's record, not the caller's assurance; the ledger
        must be persistent and the candidate registered in it.

        ``horizon`` is the number of periods each target spans. Overlapping
        targets are not independent, so the p-value never uses more than
        ``n / horizon`` observations. It does not rely on the caller declaring
        that, either: the serial dependence actually present in the two series
        is measured (:func:`effective_sample_size`) and the smaller of the two
        counts is the one used.
        """
        self._assert_stage_reachable(PromotionStage.STAGE_4_OUT_OF_SAMPLE_TRANSFER)
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if experiment_ledger.persistence_path is None:
            raise PromotionViolationError(
                "Stage 4 requires a persistent experiment ledger: an in-memory ledger forgets "
                "which holdouts were inspected or burned as soon as the process ends."
            )
        from apedp_referee.experiment_ledger import DatasetRole, fingerprint_holdout_slice

        fingerprint = fingerprint_holdout_slice(targets, index=index)
        try:
            experiment_ledger.record_data_exposure(
                self.candidate_id,
                dataset_id,
                slice_identifier,
                DatasetRole.CONFIRMATION_HOLDOUT,
                outcome_status="IN_PROGRESS",
                fingerprint=fingerprint,
            )
        except KeyError as exc:
            raise PromotionPrerequisiteError(
                f"Candidate {self.candidate_id} is not registered in the experiment ledger; "
                "its lineage must exist before it can claim a confirmation holdout."
            ) from exc
        ic, n = information_coefficient(predictions, targets)
        n_measured = effective_sample_size(predictions, targets)
        n_effective = min(n // horizon, n_measured)
        rec = self._decide_stage_4(
            None,
            ic,
            one_sided_ic_p_value(ic, n_effective),
            source=EVIDENCE_DERIVED,
            derived={
                "ic_method": "spearman",
                "n_observations": n,
                "n_effective": n_effective,
                "n_effective_measured": n_measured,
                "horizon": horizon,
                "holdout_dataset_id": dataset_id,
                "holdout_slice_identifier": slice_identifier,
                "holdout_content_hash": fingerprint.content_hash,
            },
        )
        experiment_ledger.record_data_exposure(
            self.candidate_id,
            dataset_id,
            slice_identifier,
            DatasetRole.CONFIRMATION_HOLDOUT,
            outcome_status="PASSED" if rec.passed else "FAILED",
            fingerprint=fingerprint,
        )
        return rec

    def verify_stage_5_from_returns(
        self,
        gross_returns: Any,
        turnover: Any,
        index: Any,
        *,
        trial_ledger: CandidateTrialLedger,
        universe: UniverseAudit,
        fee_bps: float = STAGE5_MIN_FEE_BPS,
    ) -> StageVerificationRecord:
        """Stage 5 from the strategy's per-period gross returns and turnover.

        ``universe`` says what the returns were earned on and has no default,
        so it cannot be left out by accident: pass
        ``apedp_referee.universe.audit_universe(...)`` for a panel, or
        ``UniverseAudit.single_instrument(name)`` for one named series. A
        universe chosen in hindsight, or one carrying forward-filled bars for
        assets that stopped trading, refuses the stage whatever the Sharpe
        ratio, because the returns themselves are then biased upward.

        Everything else the stage judges is computed here:

        * net returns are ``gross - turnover * fee_bps / 1e4``; ``fee_bps`` may
          be raised above the preregistered minimum but not lowered;
        * the annualisation factor comes from the timestamps in ``index``;
        * the net Sharpe ratio, skewness and kurtosis come from the net series;
        * the Deflated Sharpe Ratio uses ``trial_ledger``'s selection-adjusted
          trial count, and the ledger must be persistent and non-empty;
        * the breakeven fee is total gross return over total turnover.
        """
        self._assert_stage_reachable(PromotionStage.STAGE_5_ECONOMIC_VALIDITY)
        if not trial_ledger.is_persistent:
            raise PromotionViolationError(
                "Stage 5 requires a persistent trial ledger: an in-memory ledger forgets "
                "earlier searches, so its Deflated Sharpe Ratio understates multiplicity."
            )
        if trial_ledger.trial_count == 0:
            raise PromotionViolationError(
                "Stage 5 requires the search to be journaled: the trial ledger holds no trials, "
                "so there is no multiplicity to deflate by."
            )
        fee_bps = _no_looser_floor("fee_bps", fee_bps, STAGE5_MIN_FEE_BPS)
        gross = np.asarray(gross_returns, dtype=float).ravel()
        turn = np.asarray(turnover, dtype=float).ravel()
        if gross.shape != turn.shape:
            raise ValueError(f"gross_returns {gross.shape} and turnover {turn.shape} differ in length")
        if not (np.isfinite(gross).all() and np.isfinite(turn).all()) or (turn < 0).any():
            raise ValueError("gross_returns and turnover must be finite, and turnover non-negative")
        n = int(gross.size)
        periods_per_year, _first, _last = _periods_per_year(index, n)
        net = gross - turn * fee_bps / 1e4
        spread = float(np.std(net, ddof=1)) if n >= MIN_EVIDENCE_OBSERVATIONS else 0.0
        total_turnover = float(turn.sum())
        breakeven_fee_bps = float(gross.sum() / total_turnover * 1e4) if total_turnover > 0 else float("inf")
        universe_reasons = list(universe.reasons)
        refuse_reason = (
            "the universe does not support an economic claim: " + "; ".join(universe_reasons)
            if universe_reasons
            else None
        )
        derived: dict[str, Any] = {
            "n_observations": n,
            "periods_per_year": periods_per_year,
            "fee_bps_charged": fee_bps,
            "total_turnover": total_turnover,
            "dsr_source": "trial_ledger",
            "universe_selection_basis": universe.selection_basis,
            "universe_fit_for_economic_claims": not universe_reasons,
            "universe_reasons": universe_reasons,
        }
        if spread <= 0.0:
            return self._decide_stage_5(
                float("nan"), float("nan"), breakeven_fee_bps,
                source=EVIDENCE_DERIVED, derived=derived, refuse_reason=refuse_reason,
            )
        from scipy import stats

        net_sharpe = float(np.mean(net) / spread * math.sqrt(periods_per_year))
        dsr = trial_ledger.dsr_with_selection_adjustment(
            observed_sharpe=net_sharpe,
            n_bars=n,
            skewness=float(stats.skew(net)),
            kurtosis=float(stats.kurtosis(net, fisher=False)),
            annualization_factor=periods_per_year,
        )
        derived.update(
            {
                "dsr_n_trials": int(dsr["n_trials"]),
                "dsr_hidden_trials": int(dsr["hidden_trials"]),
                "dsr_expected_max_sr": float(dsr["expected_max_sr"]),
            }
        )
        return self._decide_stage_5(
            net_sharpe, float(dsr["dsr"]), breakeven_fee_bps,
            source=EVIDENCE_DERIVED, derived=derived, refuse_reason=refuse_reason,
        )

    def verify_stage_6_from_returns(self, prospective_returns: Any, index: Any) -> StageVerificationRecord:
        """Stage 6 from realised returns that postdate the stage-5 record.

        "Prospective" is checked against the clock, not asserted: every
        observation must be dated after stage 5 was recorded for this candidate
        and none may be dated in the future. The Sharpe ratio is annualised
        from the timestamps and the maximum drawdown is taken from the
        compounded equity curve. Naive timestamps are read as UTC.
        """
        self._assert_stage_reachable(PromotionStage.STAGE_6_REAL_MARKET_PROMOTION)
        returns = np.asarray(prospective_returns, dtype=float).ravel()
        if not np.isfinite(returns).all():
            raise ValueError("prospective_returns must be finite")
        n = int(returns.size)
        periods_per_year, first_ns, last_ns = _periods_per_year(index, n)
        frozen_at_text = self.stage_records[PromotionStage.STAGE_5_ECONOMIC_VALIDITY].recorded_at
        if not frozen_at_text:
            raise PromotionViolationError(
                "The stage-5 record carries no timestamp, so it cannot be shown that these "
                "returns were realised after the candidate was frozen."
            )
        frozen_at = datetime.datetime.fromisoformat(frozen_at_text)
        epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        first = epoch + datetime.timedelta(microseconds=first_ns // 1000)
        last = epoch + datetime.timedelta(microseconds=last_ns // 1000)
        if first <= frozen_at:
            raise PromotionViolationError(
                f"Returns dated from {first.isoformat()} are not prospective: stage 5 was "
                f"recorded at {frozen_at.isoformat()}, and stage 6 only counts what came after."
            )
        if last > _utc_now():
            raise PromotionViolationError(
                f"Returns dated up to {last.isoformat()} lie in the future."
            )
        spread = float(np.std(returns, ddof=1)) if n >= MIN_EVIDENCE_OBSERVATIONS else 0.0
        sharpe = (
            float(np.mean(returns) / spread * math.sqrt(periods_per_year)) if spread > 0.0 else float("nan")
        )
        equity: np.ndarray = np.cumprod(1.0 + returns)
        peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
        max_drawdown = float(max(np.max(1.0 - equity / peaks), 0.0))
        return self._decide_stage_6(
            sharpe,
            max_drawdown,
            source=EVIDENCE_DERIVED,
            derived={
                "n_observations": n,
                "periods_per_year": periods_per_year,
                "first_observation": first.isoformat(),
                "last_observation": last.isoformat(),
                "stage_5_recorded_at": frozen_at_text,
            },
        )

    # ------------------------------------------------------------------
    # Entry points on caller-reported numbers (need allow_reported_metrics)
    # ------------------------------------------------------------------

    def verify_stage_1(
        self,
        lean_status: str,
        dimensions_consistent: bool,
        bounded: bool,
    ) -> StageVerificationRecord:
        """Stage 1 on reported values; :meth:`verify_stage_1_from_lean` derives them."""
        self._require_reported_metrics("verify_stage_1", "verify_stage_1_from_lean")
        return self._decide_stage_1(lean_status, dimensions_consistent, bounded)

    def verify_stage_2(
        self,
        in_sample_ic: float,
        cv_stable: bool,
        min_ic: float = STAGE2_MIN_IC,
    ) -> StageVerificationRecord:
        """Stage 2 on reported values; :meth:`verify_stage_2_from_predictions` derives them."""
        self._require_reported_metrics("verify_stage_2", "verify_stage_2_from_predictions")
        return self._decide_stage_2(in_sample_ic, cv_stable, min_ic)

    def verify_stage_3(
        self,
        dm_p_value: float,
        max_literature_r2: float,
        dm_alpha: float = DIEBOLD_MARIANO_P_VALUE_THRESHOLD,
        max_r2_ceiling: float = MAX_LITERATURE_R2,
    ) -> StageVerificationRecord:
        """Stage 3 on reported values; :meth:`verify_stage_3_from_forecasts` derives them."""
        self._require_reported_metrics("verify_stage_3", "verify_stage_3_from_forecasts")
        return self._decide_stage_3(dm_p_value, max_literature_r2, dm_alpha, max_r2_ceiling)

    def verify_stage_4(
        self,
        in_sample_ic: float | None = None,
        oos_ic: float = 0.0,
        oos_p_value: float = 1.0,
        min_oos_ic: float = STAGE4_MIN_OOS_IC,
        alpha: float = STAGE4_OOS_ALPHA,
        min_retention: float | None = None,
    ) -> StageVerificationRecord:
        """Stage 4 on reported values; :meth:`verify_stage_4_from_holdout` derives them."""
        self._require_reported_metrics("verify_stage_4", "verify_stage_4_from_holdout")
        return self._decide_stage_4(in_sample_ic, oos_ic, oos_p_value, min_oos_ic, alpha, min_retention)

    def verify_stage_5(
        self,
        net_sharpe: float,
        dsr_metric: float,
        breakeven_fee_bps: float,
        min_sharpe_hurdle: float = MIN_NET_SHARPE,
        min_dsr_hurdle: float = DSR_THRESHOLD,
        min_fee_bps: float = STAGE5_MIN_FEE_BPS,
    ) -> StageVerificationRecord:
        """Stage 5 on reported values; :meth:`verify_stage_5_from_returns` derives them."""
        self._require_reported_metrics("verify_stage_5", "verify_stage_5_from_returns")
        return self._decide_stage_5(
            net_sharpe, dsr_metric, breakeven_fee_bps, min_sharpe_hurdle, min_dsr_hurdle, min_fee_bps
        )

    def verify_stage_6(
        self,
        prospective_sharpe: float,
        max_drawdown: float,
        prospective_hurdle: float = STAGE6_PROSPECTIVE_SHARPE_HURDLE,
        max_dd_ceiling: float = STAGE6_MAX_DRAWDOWN_CEILING,
    ) -> StageVerificationRecord:
        """Stage 6 on reported values; :meth:`verify_stage_6_from_returns` derives them."""
        self._require_reported_metrics("verify_stage_6", "verify_stage_6_from_returns")
        return self._decide_stage_6(prospective_sharpe, max_drawdown, prospective_hurdle, max_dd_ceiling)

    # ------------------------------------------------------------------
    # The decisions themselves: thresholds from the canonical specification
    # ------------------------------------------------------------------

    def _decide_stage_1(
        self,
        lean_status: str,
        dimensions_consistent: bool,
        bounded: bool,
        *,
        source: str = EVIDENCE_REPORTED,
        derived: Mapping[str, Any] | None = None,
    ) -> StageVerificationRecord:
        """Stage 1: Mathematical consistency verification."""
        self._assert_stage_reachable(PromotionStage.STAGE_1_MATHEMATICAL_CONSISTENCY)
        passed = (lean_status in ("P2_VERIFIED", "MATHEMATICALLY_CONSISTENT") and
                  dimensions_consistent and bounded)
        reason = "Mathematical consistency satisfied." if passed else \
                 f"Failed: lean={lean_status}, dims={dimensions_consistent}, bounded={bounded}."
        rec = StageVerificationRecord(
            stage=PromotionStage.STAGE_1_MATHEMATICAL_CONSISTENCY,
            passed=passed,
            status_label="MATHEMATICAL_CONSISTENCY_VERIFIED" if passed else "MATHEMATICAL_CONSISTENCY_REFUSED",
            metrics={"lean_status": lean_status, "dimensions_consistent": dimensions_consistent, "bounded": bounded},
            reason=reason,
        )
        self._record_outcome(rec, source, derived)
        return rec

    def _decide_stage_2(
        self,
        in_sample_ic: float,
        cv_stable: bool,
        min_ic: float = STAGE2_MIN_IC,
        *,
        source: str = EVIDENCE_REPORTED,
        derived: Mapping[str, Any] | None = None,
    ) -> StageVerificationRecord:
        """Stage 2: Predictive recovery in development."""
        self._assert_stage_reachable(PromotionStage.STAGE_2_PREDICTIVE_RECOVERY)
        min_ic = _no_looser_floor("min_ic", min_ic, STAGE2_MIN_IC)
        passed = (math.isfinite(in_sample_ic) and in_sample_ic >= min_ic and cv_stable)
        reason = f"Predictive recovery satisfied: IC={in_sample_ic:.4f}." if passed else \
                 f"Failed: IC={in_sample_ic:.4f} < {min_ic:.4f} or CV unstable."
        rec = StageVerificationRecord(
            stage=PromotionStage.STAGE_2_PREDICTIVE_RECOVERY,
            passed=passed,
            status_label="PREDICTIVE_RECOVERY_VERIFIED" if passed else "PREDICTIVE_RECOVERY_REFUSED",
            metrics={"in_sample_ic": in_sample_ic, "cv_stable": cv_stable},
            reason=reason,
        )
        self._record_outcome(rec, source, derived)
        return rec

    def _decide_stage_3(
        self,
        dm_p_value: float,
        max_literature_r2: float,
        dm_alpha: float = DIEBOLD_MARIANO_P_VALUE_THRESHOLD,
        max_r2_ceiling: float = MAX_LITERATURE_R2,
        *,
        source: str = EVIDENCE_REPORTED,
        derived: Mapping[str, Any] | None = None,
    ) -> StageVerificationRecord:
        """Stage 3: Unique identification against literature baselines."""
        self._assert_stage_reachable(PromotionStage.STAGE_3_UNIQUE_IDENTIFICATION)
        dm_alpha = _no_looser_ceiling("dm_alpha", dm_alpha, DIEBOLD_MARIANO_P_VALUE_THRESHOLD)
        max_r2_ceiling = _no_looser_ceiling("max_r2_ceiling", max_r2_ceiling, MAX_LITERATURE_R2)
        is_incremental = math.isfinite(dm_p_value) and dm_p_value < dm_alpha
        is_novel = math.isfinite(max_literature_r2) and max_literature_r2 < max_r2_ceiling
        passed = is_incremental and is_novel
        reason = "Unique identification confirmed." if passed else \
                 f"Failed: dm_p={dm_p_value:.4f} (want <{dm_alpha}) or lit_r2={max_literature_r2:.3f} (want <{max_r2_ceiling})."
        rec = StageVerificationRecord(
            stage=PromotionStage.STAGE_3_UNIQUE_IDENTIFICATION,
            passed=passed,
            status_label="UNIQUE_IDENTIFICATION_VERIFIED" if passed else "UNIQUE_IDENTIFICATION_REFUSED",
            metrics={"dm_p_value": dm_p_value, "max_literature_r2": max_literature_r2},
            reason=reason,
        )
        self._record_outcome(rec, source, derived)
        return rec

    def _decide_stage_4(
        self,
        in_sample_ic: float | None = None,
        oos_ic: float = 0.0,
        oos_p_value: float = 1.0,
        min_oos_ic: float = STAGE4_MIN_OOS_IC,
        alpha: float = STAGE4_OOS_ALPHA,
        min_retention: float | None = None,
        *,
        source: str = EVIDENCE_REPORTED,
        derived: Mapping[str, Any] | None = None,
    ) -> StageVerificationRecord:
        """Stage 4: Out-of-sample transfer to untouched holdout.

        Beyond the absolute OOS IC floor, the preregistered TRANSFER RETENTION
        gate is enforced: ``oos_ic / effective_in_sample_ic`` must reach
        ``MIN_OOS_TRANSFER_RETENTION`` (0.50) from the canonical spec. Callers
        may omit ``in_sample_ic`` to use Stage 2's recorded ``in_sample_ic``;
        passing a spoofed lower ``in_sample_ic`` than Stage 2's recorded value
        raises :class:`PromotionViolationError`.
        """
        self._assert_stage_reachable(PromotionStage.STAGE_4_OUT_OF_SAMPLE_TRANSFER)
        min_oos_ic = _no_looser_floor("min_oos_ic", min_oos_ic, STAGE4_MIN_OOS_IC)
        alpha = _no_looser_ceiling("alpha", alpha, STAGE4_OOS_ALPHA)
        stage2_rec = self.stage_records.get(PromotionStage.STAGE_2_PREDICTIVE_RECOVERY)
        stage2_ic = float(stage2_rec.metrics["in_sample_ic"]) if stage2_rec and "in_sample_ic" in stage2_rec.metrics else None

        if in_sample_ic is None:
            if stage2_ic is None:
                raise PromotionPrerequisiteError("Stage 2 in_sample_ic is missing from stage_records.")
            effective_in_sample_ic = stage2_ic
        else:
            effective_in_sample_ic = float(in_sample_ic)
            if stage2_ic is not None and math.isfinite(effective_in_sample_ic) and effective_in_sample_ic > 0:
                if effective_in_sample_ic + 1e-9 < stage2_ic:
                    raise PromotionViolationError(
                        f"Stage 4 in_sample_ic ({effective_in_sample_ic:.4f}) is lower than "
                        f"Stage 2 recorded in_sample_ic ({stage2_ic:.4f}); cannot deflate "
                        "in-sample baseline to inflate OOS retention."
                    )

        retention_floor = (
            MIN_OOS_TRANSFER_RETENTION
            if min_retention is None
            else _no_looser_floor("min_retention", min_retention, MIN_OOS_TRANSFER_RETENTION)
        )
        if (
            not math.isfinite(effective_in_sample_ic)
            or effective_in_sample_ic <= 0.0
            or not math.isfinite(oos_ic)
            or not math.isfinite(oos_p_value)
        ):
            retention = 0.0
            passed = False
        else:
            retention = oos_ic / effective_in_sample_ic
            passed = (
                oos_ic >= min_oos_ic
                and oos_p_value < alpha
                and retention >= retention_floor
            )
        reason = (
            f"Out-of-sample transfer verified: OOS IC={oos_ic:.4f} (p={oos_p_value:.4f}), "
            f"retention={retention:.2f} >= {retention_floor:.2f}."
            if passed
            else f"Failed OOS transfer: IC={oos_ic:.4f}, p={oos_p_value:.4f}, "
            f"retention={retention:.2f} (floor {retention_floor:.2f})."
        )
        rec = StageVerificationRecord(
            stage=PromotionStage.STAGE_4_OUT_OF_SAMPLE_TRANSFER,
            passed=passed,
            status_label="OUT_OF_SAMPLE_TRANSFER_VERIFIED" if passed else "OUT_OF_SAMPLE_TRANSFER_REFUSED",
            metrics={"oos_ic": oos_ic, "oos_p_value": oos_p_value, "in_sample_ic": effective_in_sample_ic, "retention": retention},
            reason=reason,
        )
        self._record_outcome(rec, source, derived)
        return rec

    def _decide_stage_5(
        self,
        net_sharpe: float,
        dsr_metric: float,
        breakeven_fee_bps: float,
        min_sharpe_hurdle: float = MIN_NET_SHARPE,
        min_dsr_hurdle: float = DSR_THRESHOLD,
        min_fee_bps: float = STAGE5_MIN_FEE_BPS,
        *,
        source: str = EVIDENCE_REPORTED,
        derived: Mapping[str, Any] | None = None,
        refuse_reason: str | None = None,
    ) -> StageVerificationRecord:
        """Stage 5: Economic validity after friction and multiple-testing deflation."""
        self._assert_stage_reachable(PromotionStage.STAGE_5_ECONOMIC_VALIDITY)
        min_sharpe_hurdle = _no_looser_floor("min_sharpe_hurdle", min_sharpe_hurdle, MIN_NET_SHARPE)
        min_dsr_hurdle = _no_looser_floor("min_dsr_hurdle", min_dsr_hurdle, DSR_THRESHOLD)
        min_fee_bps = _no_looser_floor("min_fee_bps", min_fee_bps, STAGE5_MIN_FEE_BPS)
        passed = (
            math.isfinite(net_sharpe)
            and math.isfinite(dsr_metric)
            and math.isfinite(breakeven_fee_bps)
            and net_sharpe >= min_sharpe_hurdle
            and dsr_metric >= min_dsr_hurdle
            and breakeven_fee_bps >= min_fee_bps
            and refuse_reason is None
        )
        reason = (
            f"Economic validity verified: SR={net_sharpe:.2f}, DSR={dsr_metric:.2f}, fee={breakeven_fee_bps:.2f} bps."
            if passed
            else f"Economic failure: SR={net_sharpe:.2f} (want >={min_sharpe_hurdle}), "
            f"DSR={dsr_metric:.2f} (want >={min_dsr_hurdle}), "
            f"fee={breakeven_fee_bps:.2f} bps (want >={min_fee_bps})."
        )
        if refuse_reason is not None:
            reason = f"Economic validity refused: {refuse_reason}"
        rec = StageVerificationRecord(
            stage=PromotionStage.STAGE_5_ECONOMIC_VALIDITY,
            passed=passed,
            status_label="ECONOMIC_VALIDITY_VERIFIED" if passed else "ECONOMIC_VALIDITY_REFUSED",
            metrics={"net_sharpe": net_sharpe, "dsr_metric": dsr_metric, "breakeven_fee_bps": breakeven_fee_bps},
            reason=reason,
        )
        self._record_outcome(rec, source, derived)
        return rec

    def verify_stage_5_from_ledger(
        self,
        net_sharpe: float,
        trial_ledger: CandidateTrialLedger,
        n_bars: int,
        breakeven_fee_bps: float,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        annualization_factor: float = 1.0,
    ) -> StageVerificationRecord:
        """Stage 5 with the Deflated Sharpe Ratio computed here, not reported.

        ``verify_stage_5`` accepts ``dsr_metric`` as a number, so the caller is
        trusted to have deflated by the true search multiplicity. This variant
        takes the trial ledger itself and derives the DSR from its
        selection-adjusted trial count. A ledger with no durable path is refused:
        its trial count restarts with every process, which is exactly the
        multiplicity the deflation exists to charge.

        ``net_sharpe`` and ``breakeven_fee_bps`` are still numbers here, so the
        record is ``caller_reported`` and the tracker must allow that;
        :meth:`verify_stage_5_from_returns` derives those too.
        """
        self._require_reported_metrics("verify_stage_5_from_ledger", "verify_stage_5_from_returns")
        self._assert_stage_reachable(PromotionStage.STAGE_5_ECONOMIC_VALIDITY)
        if not trial_ledger.is_persistent:
            raise PromotionViolationError(
                "Stage 5 requires a persistent trial ledger: an in-memory ledger forgets "
                "earlier searches, so its Deflated Sharpe Ratio understates multiplicity."
            )
        dsr = trial_ledger.dsr_with_selection_adjustment(
            observed_sharpe=net_sharpe,
            n_bars=n_bars,
            skewness=skewness,
            kurtosis=kurtosis,
            annualization_factor=annualization_factor,
        )
        return self._decide_stage_5(
            net_sharpe=net_sharpe,
            dsr_metric=float(dsr["dsr"]),
            breakeven_fee_bps=breakeven_fee_bps,
            source=EVIDENCE_REPORTED,
            derived={
                "dsr_source": "trial_ledger",
                "dsr_n_trials": int(dsr["n_trials"]),
                "dsr_hidden_trials": int(dsr["hidden_trials"]),
                "dsr_expected_max_sr": float(dsr["expected_max_sr"]),
            },
        )

    def _decide_stage_6(
        self,
        prospective_sharpe: float,
        max_drawdown: float,
        prospective_hurdle: float = STAGE6_PROSPECTIVE_SHARPE_HURDLE,
        max_dd_ceiling: float = STAGE6_MAX_DRAWDOWN_CEILING,
        *,
        source: str = EVIDENCE_REPORTED,
        derived: Mapping[str, Any] | None = None,
    ) -> StageVerificationRecord:
        """Stage 6: Real-market prospective promotion."""
        self._assert_stage_reachable(PromotionStage.STAGE_6_REAL_MARKET_PROMOTION)
        prospective_hurdle = _no_looser_floor(
            "prospective_hurdle", prospective_hurdle, STAGE6_PROSPECTIVE_SHARPE_HURDLE
        )
        max_dd_ceiling = _no_looser_ceiling(
            "max_dd_ceiling", max_dd_ceiling, STAGE6_MAX_DRAWDOWN_CEILING
        )
        passed = (
            math.isfinite(prospective_sharpe)
            and math.isfinite(max_drawdown)
            and prospective_sharpe >= prospective_hurdle
            and 0.0 <= max_drawdown <= max_dd_ceiling
        )
        reason = f"Prospective production promotion granted: SR={prospective_sharpe:.2f}, DD={max_drawdown:.1%}." if passed else \
                 f"Prospective production promotion refused: SR={prospective_sharpe:.2f} < {prospective_hurdle:.2f} or DD={max_drawdown:.1%} outside [0.0%, {max_dd_ceiling:.1%}]."
        rec = StageVerificationRecord(
            stage=PromotionStage.STAGE_6_REAL_MARKET_PROMOTION,
            passed=passed,
            status_label="REAL_MARKET_PROMOTION_GRANTED" if passed else "REAL_MARKET_PROMOTION_REFUSED",
            metrics={"prospective_sharpe": prospective_sharpe, "max_drawdown": max_drawdown},
            reason=reason,
        )
        self._record_outcome(rec, source, derived)
        return rec

    def _assert_state_matches_records(self) -> None:
        """Refuse to act on a tracker whose flags disagree with its stage records.

        ``highest_stage_passed`` and ``is_halted`` are plain attributes, so they
        can be assigned from outside. The stage records are what the gates
        actually wrote; the flags must be exactly what those records imply.
        """
        failed = [rec for rec in self.stage_records.values() if not rec.passed]
        if failed and not self.is_halted:
            raise PromotionViolationError(
                f"Candidate {self.candidate_id} has a refused stage on record "
                f"({failed[0].stage.name}) but is not marked halted."
            )
        for stage_value in range(1, int(self.highest_stage_passed) + 1):
            rec = self.stage_records.get(PromotionStage(stage_value))
            if rec is None or not rec.passed:
                raise PromotionViolationError(
                    f"Candidate {self.candidate_id} claims {self.highest_stage_passed.name} "
                    f"without a passing record for {PromotionStage(stage_value).name}."
                )

    def _assert_stage_reachable(self, target_stage: PromotionStage) -> None:
        self._assert_state_matches_records()
        if self.is_halted:
            raise PromotionPrerequisiteError(
                f"Candidate {self.candidate_id} promotion is HALTED due to prior failure: {self.halt_reason}"
            )
        if target_stage in self.stage_records:
            raise PromotionPrerequisiteError(
                f"{target_stage.name} is already on record for candidate {self.candidate_id}; "
                "a stage is evaluated once and cannot be re-run with different inputs."
            )
        expected_prev = PromotionStage(target_stage - 1)
        if self.highest_stage_passed < expected_prev:
            raise PromotionPrerequisiteError(
                f"Cannot evaluate {target_stage.name}: candidate has only passed up to {self.highest_stage_passed.name}."
            )

    def _record_outcome(
        self,
        rec: StageVerificationRecord,
        source: str = EVIDENCE_REPORTED,
        derived: Mapping[str, Any] | None = None,
    ) -> None:
        rec.evidence_source = source
        rec.recorded_at = _utc_now().isoformat()
        if derived:
            rec.metrics.update(derived)
        if rec.passed and source != EVIDENCE_DERIVED:
            rec.status_label += REPORTED_LABEL_SUFFIX
        self._apply_outcome(rec)
        self._journal_outcome(rec)

    def _require_reported_metrics(self, method: str, derived_alternative: str) -> None:
        if not self.allow_reported_metrics:
            raise PromotionViolationError(
                f"{method}() takes its metrics as numbers supplied by the caller, and this "
                f"tracker does not accept them. Use {derived_alternative}(), which computes "
                "them from the evidence, or build the tracker with allow_reported_metrics=True "
                "to record the stage as caller-reported."
            )

    # ------------------------------------------------------------------
    # Reading the tracker's standing
    # ------------------------------------------------------------------

    def assert_consistent(self) -> None:
        """Raise unless the flags are exactly what the stage records imply.

        Call this before trusting ``highest_stage_passed`` or ``is_halted``:
        both are plain attributes and can be assigned from outside.
        """
        self._assert_state_matches_records()

    @property
    def evidence_sources(self) -> dict[PromotionStage, str]:
        """Evidence source of every recorded stage, in stage order."""
        return {stage: self.stage_records[stage].evidence_source for stage in sorted(self.stage_records)}

    def is_evidence_backed(self, through: PromotionStage | None = None) -> bool:
        """Whether every stage up to ``through`` passed on derived evidence.

        ``through`` defaults to the highest stage passed. False for a tracker
        that has passed nothing, that is inconsistent, or that passed any of
        those stages on caller-reported numbers.
        """
        try:
            self._assert_state_matches_records()
        except PromotionViolationError:
            return False
        last = self.highest_stage_passed if through is None else through
        if int(last) < 1 or int(last) > int(self.highest_stage_passed):
            return False
        return all(
            self.stage_records[PromotionStage(value)].evidence_source == EVIDENCE_DERIVED
            for value in range(1, int(last) + 1)
        )

    def _apply_outcome(self, rec: StageVerificationRecord) -> None:
        self.stage_records[rec.stage] = rec
        if rec.passed:
            self.highest_stage_passed = rec.stage
        else:
            self.is_halted = True
            self.halt_reason = rec.reason

    def _journal_outcome(self, rec: StageVerificationRecord) -> None:
        if self.state_path is None:
            return
        path = Path(self.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "candidate_id": self.candidate_id,
            "stage": int(rec.stage),
            "passed": bool(rec.passed),
            "status_label": rec.status_label,
            "metrics": rec.metrics,
            "reason": rec.reason,
            "evidence_source": rec.evidence_source,
            "recorded_at": rec.recorded_at,
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _restore_from_journal(self) -> None:
        path = Path(self.state_path) if self.state_path is not None else None
        if path is None or not path.is_file():
            return
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("candidate_id") != self.candidate_id:
                        continue
                    rec = StageVerificationRecord(
                        stage=PromotionStage(int(entry["stage"])),
                        passed=bool(entry["passed"]),
                        status_label=str(entry["status_label"]),
                        metrics=dict(entry.get("metrics", {})),
                        reason=str(entry.get("reason", "")),
                        # A journal line from before sources were recorded can
                        # only be vouched for as caller-reported.
                        evidence_source=str(entry.get("evidence_source", EVIDENCE_REPORTED)),
                        recorded_at=str(entry.get("recorded_at", "")),
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    raise PromotionViolationError(
                        f"Promotion journal {path} is unreadable at line {line_number}; "
                        "refusing to resume from a damaged record."
                    ) from exc
                self._apply_outcome(rec)
