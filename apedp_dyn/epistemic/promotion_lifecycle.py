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

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from apedp_dyn.conf.canonical_spec import (
    DIEBOLD_MARIANO_P_VALUE_THRESHOLD,
    MAX_LITERATURE_R2,
    DSR_THRESHOLD,
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
        min_ic: float = 0.02,
    ) -> StageVerificationRecord:
        """Stage 2: Predictive recovery in development."""
        self._assert_stage_reachable(PromotionStage.STAGE_2_PREDICTIVE_RECOVERY)
        passed = (in_sample_ic >= min_ic and cv_stable)
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
        is_incremental = dm_p_value < dm_alpha
        is_novel = max_literature_r2 < max_r2_ceiling
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
        oos_ic: float,
        oos_p_value: float,
        min_oos_ic: float = 0.015,
        alpha: float = 0.05,
    ) -> StageVerificationRecord:
        """Stage 4: Out-of-sample transfer to untouched holdout."""
        self._assert_stage_reachable(PromotionStage.STAGE_4_OUT_OF_SAMPLE_TRANSFER)
        passed = (oos_ic >= min_oos_ic and oos_p_value < alpha)
        reason = f"Out-of-sample transfer verified: OOS IC={oos_ic:.4f} (p={oos_p_value:.4f})." if passed else \
                 f"Failed OOS transfer: IC={oos_ic:.4f} or p={oos_p_value:.4f}."
        rec = StageVerificationRecord(
            stage=PromotionStage.STAGE_4_OUT_OF_SAMPLE_TRANSFER,
            passed=passed,
            status_label="OUT_OF_SAMPLE_TRANSFER_VERIFIED" if passed else "OUT_OF_SAMPLE_TRANSFER_REFUSED",
            metrics={"oos_ic": oos_ic, "oos_p_value": oos_p_value},
            reason=reason,
        )
        self._record_outcome(rec)
        return rec

    def verify_stage_5(
        self,
        net_sharpe: float,
        dsr_metric: float,
        breakeven_fee_bps: float,
        min_sharpe_hurdle: float = 0.70,
        min_dsr_hurdle: float = DSR_THRESHOLD,
        min_fee_bps: float = 3.0,
    ) -> StageVerificationRecord:
        """Stage 5: Economic validity after friction and multiple-testing deflation."""
        self._assert_stage_reachable(PromotionStage.STAGE_5_ECONOMIC_VALIDITY)
        passed = (net_sharpe >= min_sharpe_hurdle and
                  dsr_metric >= min_dsr_hurdle and
                  breakeven_fee_bps >= min_fee_bps)
        reason = f"Economic validity verified: SR={net_sharpe:.2f}, DSR={dsr_metric:.2f}." if passed else \
                 f"Economic failure: SR={net_sharpe:.2f} (want >={min_sharpe_hurdle}), DSR={dsr_metric:.2f} (want >={min_dsr_hurdle})."
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
        prospective_hurdle: float = 0.70,
        max_dd_ceiling: float = 0.25,
    ) -> StageVerificationRecord:
        """Stage 6: Real-market prospective promotion."""
        self._assert_stage_reachable(PromotionStage.STAGE_6_REAL_MARKET_PROMOTION)
        passed = (prospective_sharpe >= prospective_hurdle and max_drawdown <= max_dd_ceiling)
        reason = f"Prospective production promotion granted: SR={prospective_sharpe:.2f}, DD={max_drawdown:.1%}." if passed else \
                 f"Prospective production promotion refused: SR={prospective_sharpe:.2f} < {prospective_hurdle:.2f} or DD={max_drawdown:.1%} > {max_dd_ceiling:.1%}."
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
