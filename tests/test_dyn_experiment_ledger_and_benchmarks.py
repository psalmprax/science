"""Test Suite for Full-Lineage Experiment Ledger, Anti-Snooping Guard,
Six-Stage Scoped Promotion Machine, and Calibrated Three-Tier Benchmarks.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.epistemic.experiment_ledger import (
    CandidateLifecycleState,
    DatasetRole,
    FullLineageExperimentLedger,
    HoldoutContaminationError,
)
from apedp_dyn.epistemic.promotion_lifecycle import (
    PromotionPrerequisiteError,
    PromotionStage,
    PromotionTracker,
)
from apedp_dyn.sandbox.calibrated_benchmark_suite import (
    run_calibrated_discovery_benchmark,
    run_calibrated_referee_benchmark,
    run_transfer_benchmark,
)

# ── 1. Experiment Ledger & Anti-Snooping Guard ───────────────────────────────

def test_experiment_ledger_lineage_and_persistence(tmp_path):
    """Ledger tracks parent-child lineage, versions, and persists to disk."""
    ledger_file = tmp_path / "experiment_ledger.jsonl"
    ledger = FullLineageExperimentLedger(persistence_path=ledger_file)

    # 1. Register parent candidate
    parent = ledger.register_candidate(
        candidate_id="cand_v1",
        proposer_id="sparse_linear",
        formula="0.5*mom5",
        feature_set=("mom5",),
        parameter_choices={"alpha": 0.01},
    )
    assert parent.version == 1
    assert parent.parent_candidate_id is None

    # 2. Register child candidate derived from parent
    child = ledger.register_candidate(
        candidate_id="cand_v2",
        proposer_id="sparse_linear",
        formula="0.5*mom5 + 0.3*cvd",
        parent_candidate_id="cand_v1",
        feature_set=("mom5", "cvd"),
        parameter_choices={"alpha": 0.005},
    )
    assert child.version == 2
    assert child.parent_candidate_id == "cand_v1"
    assert "0.5*mom5" in child.attempted_variants

    # 3. Reload ledger from disk and verify exact recovery
    reloaded = FullLineageExperimentLedger(persistence_path=ledger_file)
    rec_child = reloaded.get_candidate("cand_v2")
    assert rec_child.version == 2
    assert rec_child.parent_candidate_id == "cand_v1"


def test_holdout_anti_snooping_guard_burns_failed_confirmation_slice():
    """Failing a confirmation holdout permanently burns it, preventing data snooping by successor iterations."""
    ledger = FullLineageExperimentLedger()

    # Register candidate 1
    ledger.register_candidate("h1", "proposer_a", "formula_1")
    ledger.record_data_exposure("h1", "market_btc_2024", "block_b_q1", DatasetRole.TRAINING_DEVELOPMENT)

    # Candidate 1 inspects confirmation holdout
    ledger.record_data_exposure("h1", "market_btc_2024", "block_b_q2_holdout", DatasetRole.CONFIRMATION_HOLDOUT)

    # Candidate 1 FAILS confirmation
    ledger.update_lifecycle_state("h1", CandidateLifecycleState.CONFIRMATION_REJECTED, rejection_reason="OOS IC collapsed")
    assert ("market_btc_2024", "block_b_q2_holdout") in ledger.get_burned_holdouts()

    # Successor candidate 2 registers
    ledger.register_candidate("h2", "proposer_a", "formula_2_tweaked", parent_candidate_id="h1")

    # Candidate 2 attempts to use the SAME confirmation holdout -> MUST RAISE
    with pytest.raises(HoldoutContaminationError, match="DATA SNOOPING VIOLATION"):
        ledger.record_data_exposure("h2", "market_btc_2024", "block_b_q2_holdout", DatasetRole.CONFIRMATION_HOLDOUT)


# ── 2. Six-Stage Scoped Promotion State Machine ──────────────────────────────

def test_promotion_tracker_full_successful_lifecycle():
    """Candidate cleanly passing each stage advances step by step to prospective promotion."""
    tracker = PromotionTracker(candidate_id="alpha_star")

    # Stage 1: Mathematical Consistency
    s1 = tracker.verify_stage_1(lean_status="P2_VERIFIED", dimensions_consistent=True, bounded=True)
    assert s1.passed is True
    assert tracker.highest_stage_passed == PromotionStage.STAGE_1_MATHEMATICAL_CONSISTENCY

    # Stage 2: Predictive Recovery
    s2 = tracker.verify_stage_2(in_sample_ic=0.06, cv_stable=True)
    assert s2.passed is True
    assert tracker.highest_stage_passed == PromotionStage.STAGE_2_PREDICTIVE_RECOVERY

    # Stage 3: Unique Identification
    s3 = tracker.verify_stage_3(dm_p_value=0.002, max_literature_r2=0.45)
    assert s3.passed is True
    assert tracker.highest_stage_passed == PromotionStage.STAGE_3_UNIQUE_IDENTIFICATION

    # Stage 4: Out-of-Sample Transfer
    s4 = tracker.verify_stage_4(oos_ic=0.035, oos_p_value=0.01)
    assert s4.passed is True
    assert tracker.highest_stage_passed == PromotionStage.STAGE_4_OUT_OF_SAMPLE_TRANSFER

    # Stage 5: Economic Validity
    s5 = tracker.verify_stage_5(net_sharpe=1.10, dsr_metric=0.98, breakeven_fee_bps=6.5)
    assert s5.passed is True
    assert tracker.highest_stage_passed == PromotionStage.STAGE_5_ECONOMIC_VALIDITY

    # Stage 6: Real-Market Prospective Promotion
    s6 = tracker.verify_stage_6(prospective_sharpe=0.82, max_drawdown=0.18)
    assert s6.passed is True
    assert tracker.highest_stage_passed == PromotionStage.STAGE_6_REAL_MARKET_PROMOTION


def test_promotion_tracker_enforces_prerequisites_and_halts_on_failure():
    """Skipping a stage or failing an intermediate stage halts promotion immediately."""
    tracker = PromotionTracker(candidate_id="alpha_flawed")

    # Attempting Stage 3 without Stage 1 and 2 must raise
    with pytest.raises(PromotionPrerequisiteError, match="Cannot evaluate STAGE_3_UNIQUE_IDENTIFICATION"):
        tracker.verify_stage_3(dm_p_value=0.001, max_literature_r2=0.20)

    # Pass Stage 1
    tracker.verify_stage_1(lean_status="P2_VERIFIED", dimensions_consistent=True, bounded=True)

    # Fail Stage 2 (Predictive Recovery fails)
    s2_fail = tracker.verify_stage_2(in_sample_ic=0.005, cv_stable=False)
    assert s2_fail.passed is False
    assert tracker.is_halted is True

    # Any subsequent stage call must raise PromotionPrerequisiteError
    with pytest.raises(PromotionPrerequisiteError, match="promotion is HALTED"):
        tracker.verify_stage_3(dm_p_value=0.001, max_literature_r2=0.20)


# ── 3. Calibrated Three-Tier Benchmark Suite ─────────────────────────────────

def test_calibrated_referee_benchmark_reports_wilson_ci():
    """Referee benchmark evaluates nulls, ambiguous cases, and warranted signals with CIs."""
    # Mock referee
    def mock_referee(x, y, is_null=False, is_ambiguous=False, is_warranted=False):
        if is_null:
            return "REJECTED"
        if is_ambiguous:
            return "ABSTAINED"
        if is_warranted:
            return "PROMOTED"
        return "REJECTED"

    report = run_calibrated_referee_benchmark(mock_referee, n_trials=30, seed=42)
    assert report.fpr == 0.0
    assert report.car == 1.0
    assert report.vrr == 1.0
    assert report.passed_referee_contract is True
    assert report.fpr_ci[0] == 0.0
    assert report.vrr_ci[1] == 1.0


def test_calibrated_discovery_benchmark_measures_power_across_effect_sizes():
    """Discovery benchmark measures recovery power (1 - β) across effect size grid."""
    # Simple OLS correlation recovery proposer
    def mock_proposer(f, y):
        corr = float(np.corrcoef(f, y)[0, 1])
        t_stat = corr * np.sqrt((len(f) - 2) / max(1.0 - corr ** 2, 1e-8))
        success = bool(t_stat > 1.65)
        return success, corr

    report = run_calibrated_discovery_benchmark(mock_proposer, trials_per_condition=10, seed=42)
    assert report.strong_signal_power > report.weak_signal_power
    assert report.strong_signal_power >= 0.70  # strong signal (IC=0.15) recovered with power >= 70%
    assert len(report.conditions) == 5


def test_transfer_benchmark_evaluates_generalization_decay():
    """Transfer benchmark measures OOS IC retention across distinct test environments."""
    def mock_evaluator(X, y):
        corr = float(np.corrcoef(X, y)[0, 1])
        return 0.10, corr

    rng = np.random.RandomState(42)
    test_envs = {
        "regime_2024": (rng.randn(200), rng.randn(200) + 0.05 * rng.randn(200)),
        "regime_2025": (rng.randn(200), rng.randn(200) + 0.04 * rng.randn(200)),
    }
    report = run_transfer_benchmark(mock_evaluator, test_envs)
    assert "regime_2024" in report.details["per_env_oos_ic"]
    assert isinstance(report.retention_ratio, float)
