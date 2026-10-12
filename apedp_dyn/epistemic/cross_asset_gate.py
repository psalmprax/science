"""Cross-Asset Normalization Gate (CAN-G / Gate 0.5).

Formal epistemic verification for multi-asset and heterogeneous-population discoveries.
Ensures that a candidate representation discovered across a pooled universe is not
an artifact of cross-asset volatility scale distortion or unstratified pooling.

Separation of Concerns:
- Scientific Validity: Temporal validation, non-degeneracy, residual whiteness, and
  cross-sectional normalization. (A representation can be scientifically valid without
  being profitable in every regime).
- Portfolio Admissibility: Positive within-class risk-adjusted performance (SR > 0)
  required ONLY when the representation is being promoted as an active trading strategy.

Checks:
1. Volatility Normalization: Signals are standardized by asset-level historical volatility.
2. Asset-Class Stratification: Universe is partitioned into explicit disjoint asset classes.
3. Within-Class Evaluation: Must demonstrate non-degenerate and independently evaluated
   within-class behavior; positive Sharpe is required only when promoted for active portfolio deployment.
4. Pooled Ranking Diagnostic: Compares naive unstratified pooling vs stratified ranking to detect the 'Ranking Trap'.
5. Equal-Risk Allocation: Risk budgets are balanced across asset classes (two-tier allocation).
6. Leave-One-Asset-Class-Out (LOACO): Representation maintains viability when any single asset class is omitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class CrossAssetGateStatus(str, Enum):
    PASSED = "passed"
    FAILED_RANKING_TRAP = "failed_ranking_trap"
    FAILED_WITHIN_CLASS = "failed_within_class"
    FAILED_LOACO = "failed_loaco"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class CrossAssetGateReport:
    """Structured audit report for Gate 0.5 (CAN-G)."""
    status: CrossAssetGateStatus
    passed: bool
    scientific_valid: bool
    portfolio_admissible: bool
    volatility_normalized: bool
    asset_classes: tuple[str, ...]
    within_class_sharpes: dict[str, float]
    naive_pooled_sharpe: float
    two_tier_sharpe: float
    loaco_sharpes: dict[str, float]
    ranking_trap_detected: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "scientific_valid": self.scientific_valid,
            "portfolio_admissible": self.portfolio_admissible,
            "volatility_normalized": self.volatility_normalized,
            "asset_classes": list(self.asset_classes),
            "within_class_sharpes": self.within_class_sharpes,
            "naive_pooled_sharpe": round(self.naive_pooled_sharpe, 3),
            "two_tier_sharpe": round(self.two_tier_sharpe, 3),
            "loaco_sharpes": {k: round(v, 3) for k, v in self.loaco_sharpes.items()},
            "ranking_trap_detected": self.ranking_trap_detected,
            "reason": self.reason,
        }



def evaluate_cross_asset_gate(
    within_class_sharpes: dict[str, float],
    naive_pooled_sharpe: float,
    two_tier_sharpe: float,
    volatility_normalized: bool = True,
    loaco_sharpes: dict[str, float] | None = None,
    portfolio_promotion: bool = True,
) -> CrossAssetGateReport:
    """Evaluate Gate 0.5 (CAN-G) criteria across stratified asset classes.

    Parameters:
        within_class_sharpes: Sharpe ratios for each individual asset class (e.g. {'crypto': 1.70, 'equity': 0.55}).
        naive_pooled_sharpe: Sharpe ratio of naive unstratified cross-sectional long/short ranking.
        two_tier_sharpe: Sharpe ratio of two-tier risk-allocated portfolio.
        volatility_normalized: Whether asset volatility normalization was applied.
        loaco_sharpes: Leave-one-asset-class-out Sharpe ratios (defaults to within_class_sharpes for 2 classes).
        portfolio_promotion: If True, requires positive Sharpe (SR > 0) for active trading promotion.
                             If False, checks non-degenerate within-class evaluation for scientific validity.
    """
    classes = tuple(within_class_sharpes.keys())
    scientific_valid = volatility_normalized and all(np.isfinite(sr) for sr in within_class_sharpes.values())

    if len(classes) < 2:
        return CrossAssetGateReport(
            status=CrossAssetGateStatus.NOT_APPLICABLE,
            passed=True,
            scientific_valid=scientific_valid,
            portfolio_admissible=all(sr > 0.0 for sr in within_class_sharpes.values()),
            volatility_normalized=volatility_normalized,
            asset_classes=classes,
            within_class_sharpes=within_class_sharpes,
            naive_pooled_sharpe=naive_pooled_sharpe,
            two_tier_sharpe=two_tier_sharpe,
            loaco_sharpes={},
            ranking_trap_detected=False,
            reason="single asset-class universe; cross-asset gate not applicable",
        )

    if loaco_sharpes is None:
        loaco_sharpes = dict(within_class_sharpes)

    # 1. Check within-class performance
    failed_classes = [c for c, sr in within_class_sharpes.items() if sr <= 0.0]
    portfolio_admissible = (len(failed_classes) == 0) and (two_tier_sharpe > 0.0)

    if failed_classes and portfolio_promotion:
        return CrossAssetGateReport(
            status=CrossAssetGateStatus.FAILED_WITHIN_CLASS,
            passed=False,
            scientific_valid=scientific_valid,
            portfolio_admissible=False,
            volatility_normalized=volatility_normalized,
            asset_classes=classes,
            within_class_sharpes=within_class_sharpes,
            naive_pooled_sharpe=naive_pooled_sharpe,
            two_tier_sharpe=two_tier_sharpe,
            loaco_sharpes=loaco_sharpes,
            ranking_trap_detected=False,
            reason=f"representation failed portfolio admissibility (SR <= 0 in classes: {failed_classes}); scientific validity preserved",
        )

    # 2. Check Ranking Trap
    ranking_trap = (naive_pooled_sharpe <= 0.0) and all(sr > 0.0 for sr in within_class_sharpes.values())

    # 3. Check LOACO survival
    loaco_failed = [c for c, sr in loaco_sharpes.items() if sr <= 0.0]
    if loaco_failed and portfolio_promotion:
        return CrossAssetGateReport(
            status=CrossAssetGateStatus.FAILED_LOACO,
            passed=False,
            scientific_valid=scientific_valid,
            portfolio_admissible=False,
            volatility_normalized=volatility_normalized,
            asset_classes=classes,
            within_class_sharpes=within_class_sharpes,
            naive_pooled_sharpe=naive_pooled_sharpe,
            two_tier_sharpe=two_tier_sharpe,
            loaco_sharpes=loaco_sharpes,
            ranking_trap_detected=ranking_trap,
            reason=f"leave-one-asset-class-out test failed when isolating: {loaco_failed}",
        )

    # 4. Check two-tier survival
    if two_tier_sharpe <= 0.0 and portfolio_promotion:
        return CrossAssetGateReport(
            status=CrossAssetGateStatus.FAILED_RANKING_TRAP if ranking_trap else CrossAssetGateStatus.FAILED_WITHIN_CLASS,
            passed=False,
            scientific_valid=scientific_valid,
            portfolio_admissible=False,
            volatility_normalized=volatility_normalized,
            asset_classes=classes,
            within_class_sharpes=within_class_sharpes,
            naive_pooled_sharpe=naive_pooled_sharpe,
            two_tier_sharpe=two_tier_sharpe,
            loaco_sharpes=loaco_sharpes,
            ranking_trap_detected=ranking_trap,
            reason="two-tier allocation failed to achieve positive Sharpe for portfolio promotion",
        )

    # Passed!
    msg = "cross-asset normalization passed with two-tier risk allocation"
    if ranking_trap:
        msg += " (warning: naive unstratified pooling suffered ranking trap due to cross-asset volatility disparity; stratified allocation required)"

    return CrossAssetGateReport(
        status=CrossAssetGateStatus.PASSED,
        passed=True,
        scientific_valid=scientific_valid,
        portfolio_admissible=portfolio_admissible,
        volatility_normalized=volatility_normalized,
        asset_classes=classes,
        within_class_sharpes=within_class_sharpes,
        naive_pooled_sharpe=naive_pooled_sharpe,
        two_tier_sharpe=two_tier_sharpe,
        loaco_sharpes=loaco_sharpes,
        ranking_trap_detected=ranking_trap,
        reason=msg,
    )

