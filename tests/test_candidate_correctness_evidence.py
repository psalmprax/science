"""Pins for the correctness-evidence gate on circuit candidates.

`docs/REPRESENTATION_AWARE_UNCOMPUTATION.md` gates 3, 4 and 7 say a candidate may not be
called correct unless value, junk and phase are each accounted for. This module makes
that a runtime rule: a correctness-claiming status without evidence raises, and the
ledger refuses to promote a receipt that does not carry it.

The DQI verifier's own rule folds the ancilla read into `value_errors`
(`val_err = reg_diff.any(axis=0) | junk.any(axis=0)`), so a DQI receipt's `junk` is
`covered_by` `value` with that stated reason — not silently absent. A cost-model
verifier (invsqrt, dfthcsos) reports `status` and `score` only and says so.
"""
from __future__ import annotations

import pytest

from apedp_dyn.quantum_compilation.candidates import (
    CORRECTNESS_CHECKS,
    CORRECTNESS_CLAIMING_STATUSES,
    CandidateStatus,
    CircuitCandidate,
    EvidenceIncompleteError,
    VerifierReceipt,
    candidate_id_for,
    evidence_from_receipt,
    missing_correctness_checks,
    normalise_correctness_evidence,
)


def _candidate(benchmark: str = "dqi") -> CircuitCandidate:
    parameters = {"depth": 3}
    return CircuitCandidate(
        candidate_id=candidate_id_for(
            benchmark, "test-generator", parameters, contract_sha256="e" * 64),
        benchmark=benchmark,
        generator_id="test-generator",
        parameters=parameters,
        artifact_path="",
        artifact_sha256="0" * 64,
        contract_sha256="e" * 64,
    )


def _receipt(metrics: dict, *, status: str = "PASS", score: float = 1.0) -> VerifierReceipt:
    return VerifierReceipt(
        status=status,
        score=score,
        toffoli_equivalent=1.0,
        num_qubits=1913,
        metrics=metrics,
        exit_code=0,
        command=("python", "verify_submission_dqi.py"),
        salt_id="a" * 64,
        verification_scope="full-domain",
        verifier_sha256="b" * 64,
        artifact_sha256="c" * 64,
        contract_sha256="d" * 64,
        duration_s=1.0,
    )


def test_the_three_checks_are_the_ones_the_verifier_measures():
    assert CORRECTNESS_CHECKS == ("value", "junk", "phase")


def test_a_per_shot_receipt_supplies_value_and_phase_and_covers_junk():
    evidence = evidence_from_receipt(
        _receipt({"value_errors": 0, "phase_errors": 0}), "dqi")
    assert evidence["value"]["measured"] is True and evidence["value"]["passed"] is True
    assert evidence["phase"]["measured"] is True and evidence["phase"]["passed"] is True
    # Junk is not a separate figure in this verifier: it is folded into `value_errors`.
    assert evidence["junk"]["covered_by"] == "value"
    assert "val_err = reg_diff.any(axis=0) | junk.any(axis=0)" in evidence["junk"]["reason"]


def test_a_receipt_without_phase_evidence_cannot_claim_correctness():
    receipt = _receipt({"value_errors": 0})  # Experiment D's defect, exactly.
    candidate = _candidate("dqi")
    assert missing_correctness_checks(
        evidence_from_receipt(receipt, "dqi"), benchmark="dqi")
    with pytest.raises(EvidenceIncompleteError):
        candidate.with_status(CandidateStatus.VERIFIED_IMPROVEMENT, receipt=receipt)


def test_a_failing_check_cannot_sit_under_a_correctness_claim():
    receipt = _receipt({"value_errors": 0, "phase_errors": 37})
    with pytest.raises(EvidenceIncompleteError):
        _candidate("dqi").with_status(CandidateStatus.VERIFIED_NONIMPROVING,
                                      receipt=receipt)


def test_a_passing_per_shot_receipt_is_accepted_and_recorded():
    receipt = _receipt({"value_errors": 0, "phase_errors": 0, "verified_shots": 102_400})
    updated = _candidate("dqi").with_status(CandidateStatus.VERIFIED_IMPROVEMENT,
                                           receipt=receipt)
    assert updated.status is CandidateStatus.VERIFIED_IMPROVEMENT
    assert set(updated.correctness_evidence) == set(CORRECTNESS_CHECKS)
    assert updated.to_dict()["correctness_evidence"]["phase"]["source"] == \
        "receipt.metrics['phase_errors']"


def test_a_per_shot_verifier_cannot_waive_all_three_checks():
    waiver = {check: {"not_applicable": "trust me"} for check in CORRECTNESS_CHECKS}
    with pytest.raises(EvidenceIncompleteError):
        normalise_correctness_evidence(waiver, benchmark="dqi")


def test_a_cost_model_verifier_states_that_it_does_not_measure_them():
    evidence = evidence_from_receipt(_receipt({"verified_shots": 200_000_000}),
                                     "invsqrt")
    assert set(evidence) == set(CORRECTNESS_CHECKS)
    assert all("not_applicable" in entry for entry in evidence.values())
    assert "reports `status` and `score` only" in evidence["phase"]["not_applicable"]
    updated = _candidate("invsqrt").with_status(
        CandidateStatus.VERIFIED_NONIMPROVING,
        receipt=_receipt({"verified_shots": 200_000_000}))
    assert updated.status is CandidateStatus.VERIFIED_NONIMPROVING


def test_an_unknown_benchmark_cannot_claim_correctness():
    receipt = _receipt({"verified_shots": 10})
    assert evidence_from_receipt(receipt, "unknown-benchmark") == {}
    with pytest.raises(EvidenceIncompleteError):
        _candidate("unknown-benchmark").with_status(
            CandidateStatus.VERIFIED_IMPROVEMENT, receipt=receipt)


def test_the_ledger_refuses_to_promote_a_receipt_without_phase_evidence(tmp_path):
    from apedp_dyn.quantum_compilation.ledger import QuantumCandidateLedger

    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    candidate = _candidate("dqi")
    ledger.register(candidate)
    # Value only: the candidate is not called correct, it is called unresolved.
    updated = ledger.record_verification(
        candidate,
        _receipt({"value_errors": 0}, score=1.0),
        incumbent_score=5.0,
    )
    assert updated.status is CandidateStatus.INSUFFICIENT_EVIDENCE
    assert "does not establish correctness" in ledger.node(candidate.candidate_id).rejection_reason


def test_the_ledger_promotes_a_full_evidence_receipt(tmp_path):
    from apedp_dyn.quantum_compilation.ledger import QuantumCandidateLedger

    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    candidate = _candidate("dqi")
    ledger.register(candidate)
    updated = ledger.record_verification(
        candidate,
        _receipt({"value_errors": 0, "phase_errors": 0}, score=1.0),
        incumbent_score=5.0,
    )
    assert updated.status is CandidateStatus.VERIFIED_IMPROVEMENT
    assert updated.correctness_evidence["junk"]["covered_by"] == "value"


def test_the_claiming_statuses_are_the_two_verified_ones():
    assert CORRECTNESS_CLAIMING_STATUSES == {CandidateStatus.VERIFIED_NONIMPROVING,
                                             CandidateStatus.VERIFIED_IMPROVEMENT}
