"""Decision layer: hard gates, rule routing, calibration, logging."""
import pytest

from apedp.decision import (
    CalibrationPolicy,
    CandidateClass,
    CandidateEvidence,
    Decision,
    DecisionEngine,
    DecisionLog,
    EvidenceStrength,
    JevBackend,
    RuleBasedBackend,
    ValidationResult,
    run_hard_gates,
)


def _ev(**kw):
    base = {"candidate_id": "T-1", "candidate_class": CandidateClass.CANDIDATE,
                "mathematical": ValidationResult(True, 0.9),
                "physics": ValidationResult(True, 1.0),
                "numerical": ValidationResult(True, 0.9),
                "empirical": ValidationResult(False, 0.2),
                "evidence_strength": EvidenceStrength.MODERATE,
                "independent_datasets": 3, "benchmark_datasets": 1,
                "domain_valid": True, "evidence_ids": ("E-1",)}
    base.update(kw)
    return CandidateEvidence(**base)


def test_hard_gates_reject_unoverridable():
    bad = _ev(physics=ValidationResult(False, 0.1))
    assert run_hard_gates(bad) != []
    eng = DecisionEngine(RuleBasedBackend())
    out = eng.evaluate(bad)
    assert out.decision == Decision.REJECT
    assert out.confidence == 1.0


def test_rule_routing():
    eng = DecisionEngine(RuleBasedBackend())
    assert eng.evaluate(
        _ev(mathematical=ValidationResult(False))).decision == \
        Decision.REQUEST_DERIVATION
    assert eng.evaluate(
        _ev(numerical=ValidationResult(False))).decision == \
        Decision.RUN_MORE_TESTS
    assert eng.evaluate(_ev(independent_datasets=0)).decision == \
        Decision.SEEK_EMPIRICAL_VALIDATION
    assert eng.evaluate(_ev(empirical=ValidationResult(True, 0.9),
                            independent_datasets=2)).decision == \
        Decision.ACCEPT_FOR_BENCHMARK


def test_calibration_abstains_and_defers():
    pol = CalibrationPolicy(accept_threshold=0.95, review_threshold=0.70)
    assert pol.apply({"a": 0.5}) == "abstain"
    assert pol.apply({"a": 0.8}) == "human_review"
    assert pol.apply({"a": 0.97}) == "a"
    assert pol.apply({}) == "abstain"


def test_log_records_and_empty_calibration():
    log = DecisionLog()
    assert log.calibration_rate() is None
    eng = DecisionEngine(RuleBasedBackend())
    eng.evaluate(_ev())
    assert len(eng.log.records) == 1
    assert eng.log.records[0]["policy_version"] == "apedp-policy-v1"


def test_jev_stub_documents_boundary():
    with pytest.raises(NotImplementedError, match="no client"):
        JevBackend().decide({})
