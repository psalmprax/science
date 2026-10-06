"""Tests for APEDP Controlled Self-Improvement System and Constitutional Invariants."""
from __future__ import annotations

from apedp.constitution import ConstitutionalAuditor
from apedp.self_improvement import (
    FailureDetector,
    ImprovementLedger,
    ImprovementLevel,
    IndependentGatekeeper,
    ModificationProposal,
)


def test_constitutional_invariants_enforced():
    # 1. Test INV-1 violation (dimensional contradiction marked valid)
    proposal = {
        "id": "prop-bad-dim",
        "modifies_benchmark_data": False,
        "retrospective_criterion_change": False,
    }
    bench_bad_dim = {
        "classified_contradiction_as_valid": True,
        "unresolved_treated_as_valid": False,
        "selected_by_train_fit_alone": False,
    }
    verdict1 = ConstitutionalAuditor.audit(
        proposal, bench_bad_dim, regression_passed=True, provenance_trace={"author": "test", "timestamp": "now"}
    )
    assert verdict1.compliant is False
    assert any("INV-1" in r for r in verdict1.rejection_reasons)

    # 2. Test INV-2 violation (UNRESOLVED treated as VALID)
    bench_unresolved = {
        "classified_contradiction_as_valid": False,
        "unresolved_treated_as_valid": True,
        "selected_by_train_fit_alone": False,
    }
    verdict2 = ConstitutionalAuditor.audit(
        proposal, bench_unresolved, regression_passed=True, provenance_trace={"author": "test", "timestamp": "now"}
    )
    assert verdict2.compliant is False
    assert any("INV-2" in r for r in verdict2.rejection_reasons)

    # 3. Test INV-4 violation (tampering with benchmark data)
    prop_data_tamper = {
        "id": "prop-tamper",
        "modifies_benchmark_data": True,
        "retrospective_criterion_change": False,
    }
    bench_clean = {
        "classified_contradiction_as_valid": False,
        "unresolved_treated_as_valid": False,
        "selected_by_train_fit_alone": False,
    }
    verdict3 = ConstitutionalAuditor.audit(
        prop_data_tamper, bench_clean, regression_passed=True, provenance_trace={"author": "test", "timestamp": "now"}
    )
    assert verdict3.compliant is False
    assert any("INV-4" in r for r in verdict3.rejection_reasons)


def test_failure_detector_identifies_unresolved_transcendentals():
    histories = [
        {
            "candidates": [
                {"dim_status": "VALID", "reason_chain": "math OK -> dim VALID"},
                {"dim_status": "UNRESOLVED", "reason_chain": "dim UNRESOLVED (multi-coefficient transcendental argument)"},
                {"dim_status": "UNRESOLVED", "reason_chain": "dim UNRESOLVED (transcendental argument not evaluable)"},
            ]
        }
    ]
    signals = FailureDetector.detect_weaknesses(histories)
    assert len(signals) >= 1
    trans_signal = next((s for s in signals if s.signal_type == "HIGH_UNRESOLVED_TRANSCENDENTAL_RATE"), None)
    assert trans_signal is not None
    assert trans_signal.suggested_level == ImprovementLevel.LEVEL_3_PIPELINE


def test_independent_gatekeeper_promotes_valid_improvement():
    proposal = ModificationProposal(
        proposal_id="prop-001-coupling-thresh",
        level=ImprovementLevel.LEVEL_1_PARAMETER,
        target_component="separability_gate",
        modifications={"coupling_residual_threshold": 0.08},
        hypothesis="Lowering coupling residual threshold from 0.15 to 0.08 reduces false-decomposition rate",
        rationale="Empirical calibration sweep showed 80% false decomposition at 0.15 vs 47% at 0.08",
        author="OptimizationAgent",
    )

    benchmark_results = {
        "classified_contradiction_as_valid": False,
        "unresolved_treated_as_valid": False,
        "selected_by_train_fit_alone": False,
        "target_metric_improved": True,
        "metric_deltas": {"false_decomposition_rate": -0.33, "accuracy": +0.067},
        "reproducibility_verified": True,
        "supports_ambiguity_preservation": True,
    }

    decision = IndependentGatekeeper.evaluate_and_decide(
        proposal=proposal,
        benchmark_results=benchmark_results,
        regression_passed=True,
    )

    assert decision.accepted is True
    assert decision.status == "PROMOTED"
    assert decision.constitutional_verdict.compliant is True
    assert len(decision.provenance_hash) == 64


def test_improvement_ledger_maintains_audit_trail():
    ledger = ImprovementLedger()
    proposal = ModificationProposal(
        proposal_id="prop-002-grammar-guard",
        level=ImprovementLevel.LEVEL_3_PIPELINE,
        target_component="grammar",
        modifications={"filter_unresolved_transcendentals": True},
        hypothesis="Filter unresolvable transcendental forms at birth",
        rationale="Prevents wasted search evaluations",
        author="PipelineEngineerAgent",
    )

    decision = IndependentGatekeeper.evaluate_and_decide(
        proposal=proposal,
        benchmark_results={"target_metric_improved": True},
        regression_passed=True,
    )

    ledger.record_decision(proposal, decision)
    records = ledger.to_list()
    assert len(records) == 1
    assert records[0]["proposal"]["id"] == "prop-002-grammar-guard"
    assert records[0]["decision"]["status"] == "PROMOTED"
    assert records[0]["entry_hash"] == decision.provenance_hash
