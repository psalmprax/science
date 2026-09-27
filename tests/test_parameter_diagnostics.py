"""Tests for parameter-recovery diagnostics in scientific discovery."""
# RECONSTRUCTION NOTE: deleted from the worktree by a concurrent session and
# restored here from bytecode forensics (tests/__pycache__/...pytest-9.1.1.pyc):
# all seeds/values/thresholds/messages below are recovered from compiled
# constants; verified by execution below.
from __future__ import annotations

from apedp.parameter_diagnostics import (
    RecoveryCategory,
    compute_parameter_recovery,
)


def test_category_a_clean_recovery():
    """Verify Category A classification when family matches and parameters are accurate."""
    target_params = {"p": 1.7, "A": 3.0}
    recovered_params = {"p": 1.70064, "A": 2.99823}
    rep = compute_parameter_recovery(
        recovered_family="power_law",
        target_family="power_law",
        recovered_params=recovered_params,
        target_params=target_params,
        val_mse=9.33e-6,
    )
    assert rep.is_category_a
    assert rep.category == RecoveryCategory.CATEGORY_A
    assert rep.max_relative_param_error < 0.01
    assert "Category A" in rep.verdict_summary


def test_category_b_impostor_surrogate_family():
    """Verify Category B classification when fit is predictive but family is wrong."""
    target_params = {"A": 2.0, "p": 1.3, "B": 0.7, "q": 2.1}
    recovered_params = {"A": 2.99, "p": 1.5, "B": 0.79, "q": 0.35, "C": 0.5}
    rep = compute_parameter_recovery(
        recovered_family="rational_exponential_envelope",
        target_family="rational_transition",
        recovered_params=recovered_params,
        target_params=target_params,
        val_mse=2.28e-05,
    )
    assert rep.is_category_b
    assert rep.category == RecoveryCategory.CATEGORY_B
    assert "impostor family" in rep.verdict_summary


def test_category_b_parameter_shift():
    """Verify Category B classification when family matches but parameters shifted outside tolerance."""
    target_params = {"A": 2.0, "p": 1.3, "B": 0.7, "q": 2.1}
    recovered_params = {"A": 1.84, "p": 1.21, "B": 0.57, "q": 2.08}
    rep = compute_parameter_recovery(
        recovered_family="rational_transition",
        target_family="rational_transition",
        recovered_params=recovered_params,
        target_params=target_params,
        val_mse=3.04e-05,
        param_rel_tolerance=0.05,
    )
    assert rep.is_category_b
    assert rep.category == RecoveryCategory.CATEGORY_B
    assert "shifted outside tolerance" in rep.verdict_summary


def test_category_c_failure():
    """Verify Category C classification when validation MSE exceeds predictive threshold."""
    target_params = {"A": 2.5, "p": 1.5}
    recovered_params = {"A": 1.0, "p": 0.5}
    rep = compute_parameter_recovery(
        recovered_family="power_law",
        target_family="power_law",
        recovered_params=recovered_params,
        target_params=target_params,
        val_mse=0.15,
        val_mse_threshold=0.0001,
    )
    assert rep.is_category_c
    assert rep.category == RecoveryCategory.CATEGORY_C
    assert "failure" in rep.verdict_summary


def test_stage_d2_evidence_report_decomposition():
    """Verify decomposed evidence report preserves distinct metrics without scalar collapsing."""
    from apedp.parameter_diagnostics import make_stage_d2_evidence_report

    rep = make_stage_d2_evidence_report(
        model_expression="2.0*x**1.3/(1.0 + 0.7*x**2.1)",
        structural_family="rational_transition",
        val_mse=0.00025,
        baseline_mse=1.2,
        n_samples=40,
        complexity=7.0,
        near_ood_mse=0.00031,
        condition_number=12.5,
        decision_verdict="ACCEPTED",
    )
    assert rep.final_decision == "ACCEPTED"
    assert rep.likelihood_evidence > 0.0
    assert rep.complexity_penalty > 0.0
    assert rep.ood_penalty >= 1.0
    d = rep.to_dict()
    assert "likelihood_evidence" in d
