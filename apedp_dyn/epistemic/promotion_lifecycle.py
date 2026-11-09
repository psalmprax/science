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

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from apedp_dyn.conf.canonical_spec import (
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


@dataclass
class PromotionTracker:
    candidate_id: str
    highest_stage_passed: PromotionStage = PromotionStage.UNEVALUATED
    is_halted: bool = False
    halt_reason: str | None = None
    stage_records: dict[PromotionStage, StageVerificationRecord] = field(default_factory=dict)

    def verify_stage_1(
        self,
        lean_status: str,
        dimensions_consistent: bool,
        bounded: bool,
    ) -> StageVerificationRecord:
        """Stage 1: Mathematical consistency verification."""
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
        self._record_outcome(rec)
        return rec

    def verify_stage_2(
        self,
        in_sample_ic: float,
        cv_stable: bool,
        min_ic: float = STAGE2_MIN_IC,
    ) -> StageVerificationRecord:
        """Stage 2: Predictive recovery in development."""
        self._assert_stage_reachable(PromotionStage.STAGE_2_PREDICTIVE_RECOVERY)
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
        self._record_outcome(rec)
        return rec

    def verify_stage_3(
        self,
        dm_p_value: float,
        max_literature_r2: float,
        dm_alpha: float = DIEBOLD_MARIANO_P_VALUE_THRESHOLD,
        max_r2_ceiling: float = MAX_LITERATURE_R2,
    ) -> StageVerificationRecord:
        """Stage 3: Unique identification against literature baselines."""
        self._assert_stage_reachable(PromotionStage.STAGE_3_UNIQUE_IDENTIFICATION)
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
        self._record_outcome(rec)
        return rec

    def verify_stage_4(
        self,
        in_sample_ic: float | None = None,
        oos_ic: float = 0.0,
        oos_p_value: float = 1.0,
        min_oos_ic: float = STAGE4_MIN_OOS_IC,
        alpha: float = STAGE4_OOS_ALPHA,
        min_retention: float | None = None,
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
            MIN_OOS_TRANSFER_RETENTION if min_retention is None else min_retention
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
        self._record_outcome(rec)
        return rec

    def verify_stage_5(
        self,
        net_sharpe: float,
        dsr_metric: float,
        breakeven_fee_bps: float,
        min_sharpe_hurdle: float = MIN_NET_SHARPE,
        min_dsr_hurdle: float = DSR_THRESHOLD,
        min_fee_bps: float = STAGE5_MIN_FEE_BPS,
    ) -> StageVerificationRecord:
        """Stage 5: Economic validity after friction and multiple-testing deflation."""
        self._assert_stage_reachable(PromotionStage.STAGE_5_ECONOMIC_VALIDITY)
        passed = (
            math.isfinite(net_sharpe)
            and math.isfinite(dsr_metric)
            and math.isfinite(breakeven_fee_bps)
            and net_sharpe >= min_sharpe_hurdle
            and dsr_metric >= min_dsr_hurdle
            and breakeven_fee_bps >= min_fee_bps
        )
        reason = (
            f"Economic validity verified: SR={net_sharpe:.2f}, DSR={dsr_metric:.2f}, fee={breakeven_fee_bps:.2f} bps."
            if passed
            else f"Economic failure: SR={net_sharpe:.2f} (want >={min_sharpe_hurdle}), "
            f"DSR={dsr_metric:.2f} (want >={min_dsr_hurdle}), "
            f"fee={breakeven_fee_bps:.2f} bps (want >={min_fee_bps})."
        )
        rec = StageVerificationRecord(
            stage=PromotionStage.STAGE_5_ECONOMIC_VALIDITY,
            passed=passed,
            status_label="ECONOMIC_VALIDITY_VERIFIED" if passed else "ECONOMIC_VALIDITY_REFUSED",
            metrics={"net_sharpe": net_sharpe, "dsr_metric": dsr_metric, "breakeven_fee_bps": breakeven_fee_bps},
            reason=reason,
        )
        self._record_outcome(rec)
        return rec

    def verify_stage_6(
        self,
        prospective_sharpe: float,
        max_drawdown: float,
        prospective_hurdle: float = STAGE6_PROSPECTIVE_SHARPE_HURDLE,
        max_dd_ceiling: float = STAGE6_MAX_DRAWDOWN_CEILING,
    ) -> StageVerificationRecord:
        """Stage 6: Real-market prospective promotion."""
        self._assert_stage_reachable(PromotionStage.STAGE_6_REAL_MARKET_PROMOTION)
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
        self._record_outcome(rec)
        return rec

    def _assert_stage_reachable(self, target_stage: PromotionStage) -> None:
        if self.is_halted:
            raise PromotionPrerequisiteError(
                f"Candidate {self.candidate_id} promotion is HALTED due to prior failure: {self.halt_reason}"
            )
        expected_prev = PromotionStage(target_stage - 1)
        if self.highest_stage_passed < expected_prev:
            raise PromotionPrerequisiteError(
                f"Cannot evaluate {target_stage.name}: candidate has only passed up to {self.highest_stage_passed.name}."
            )

    def _record_outcome(self, rec: StageVerificationRecord) -> None:
        self.stage_records[rec.stage] = rec
        if rec.passed:
            self.highest_stage_passed = rec.stage
        else:
            self.is_halted = True
            self.halt_reason = rec.reason
