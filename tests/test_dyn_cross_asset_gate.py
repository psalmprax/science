"""Tests for Cross-Asset Normalization Gate (CAN-G / Gate 0.5) and Epistemic Equivalence Class."""
from apedp_dyn.epistemic.adjudicate import (
    EpistemicStatus,
    adjudicate,
)
from apedp_dyn.epistemic.cross_asset_gate import (
    CrossAssetGateStatus,
    evaluate_cross_asset_gate,
)
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term


def test_cross_asset_gate_passes_stratified():
    """Stratified 30-asset system passes CAN-G despite naive ranking trap."""
    within_class = {"crypto": 1.70, "equity": 0.55}
    naive_pooled = -0.34
    two_tier = 1.35

    report = evaluate_cross_asset_gate(
        within_class_sharpes=within_class,
        naive_pooled_sharpe=naive_pooled,
        two_tier_sharpe=two_tier,
        volatility_normalized=True,
    )

    assert report.passed is True
    assert report.status == CrossAssetGateStatus.PASSED
    assert report.ranking_trap_detected is True
    assert "ranking trap" in report.reason.lower()


def test_cross_asset_gate_fails_on_negative_within_class():
    """Fails if representation fails in one asset class."""
    within_class = {"crypto": 1.70, "equity": -0.20}
    naive_pooled = -0.10
    two_tier = 0.50

    report = evaluate_cross_asset_gate(
        within_class_sharpes=within_class,
        naive_pooled_sharpe=naive_pooled,
        two_tier_sharpe=two_tier,
    )

    assert report.passed is False
    assert report.status == CrossAssetGateStatus.FAILED_WITHIN_CLASS
    assert "equity" in report.reason


def test_cross_asset_gate_single_class_not_applicable():
    """Single-class universe is marked NOT_APPLICABLE but passes."""
    within_class = {"crypto": 1.81}
    report = evaluate_cross_asset_gate(
        within_class_sharpes=within_class,
        naive_pooled_sharpe=1.81,
        two_tier_sharpe=1.81,
    )

    assert report.passed is True
    assert report.status == CrossAssetGateStatus.NOT_APPLICABLE


def test_adjudication_populates_equivalence_class():
    """Adjudication populates admissible_equivalence_class when MULTI_HYPOTHESIS occurs."""
    c1 = CandidateDynamics(lhs="y", terms=(Term("x1", "theta_0"),), candidate_id="cand_1")
    c2 = CandidateDynamics(lhs="y", terms=(Term("x2", "theta_0"),), candidate_id="cand_2")
    c3 = CandidateDynamics(lhs="y", terms=(Term("x3", "theta_0"),), candidate_id="cand_3")

    # c1 and c2 are within tau=0.05, c3 is far away
    sc1 = ScoredCandidate(c1, train_loss=0.01, holdout_rmse=0.10, provenance={"gate4_status": {"passed": True}})
    sc2 = ScoredCandidate(c2, train_loss=0.02, holdout_rmse=0.12, provenance={"gate4_status": {"passed": True}})
    sc3 = ScoredCandidate(c3, train_loss=0.03, holdout_rmse=0.30, provenance={"gate4_status": {"passed": True}})

    adj = adjudicate([sc1, sc2, sc3], tau_epi=0.05)

    assert adj.status == EpistemicStatus.MULTI_HYPOTHESIS
    assert adj.admissible_equivalence_class == ("cand_1", "cand_2")
    assert "cand_3" not in adj.admissible_equivalence_class
