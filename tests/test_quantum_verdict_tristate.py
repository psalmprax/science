"""The APEDP-Q ledger's third verdict state: a run that learned nothing.

Status: audit-only

`QuantumCandidateLedger.record_verification` used to map every non-`PASS`
receipt to `VERIFIER_REJECTED`, which recorded a timeout, a crash or an
unparseable summary as a measured failure of the circuit. These tests pin the
split: `FAIL` is still a rejection, `ERROR` is `UNVERIFIABLE`, the confirmation
salt is spent either way, and `verdict_summary()` keeps a capability gap out of
the denominator that a rejection rate is read from.
"""
from __future__ import annotations

import pytest

from apedp_dyn.epistemic.experiment_ledger import (
    CandidateLifecycleState,
    HoldoutContaminationError,
)
from apedp_dyn.quantum_compilation.candidates import (
    CandidateStatus,
    CircuitCandidate,
    VerifierReceipt,
    candidate_id_for,
)
from apedp_dyn.quantum_compilation.experiment import ExperimentOutcome
from apedp_dyn.quantum_compilation.ledger import QuantumCandidateLedger

CONTRACT_SHA = "d" * 64
ARTIFACT_SHA = "c" * 64
VERIFIER_SHA = "b" * 64
SALT_ID = "a" * 64


def _candidate(benchmark: str = "dqi", *, seed: float = 1.0) -> CircuitCandidate:
    parameters = {"seed": seed}
    return CircuitCandidate(
        candidate_id=candidate_id_for(
            benchmark, "unit-proposer", parameters, contract_sha256=CONTRACT_SHA
        ),
        benchmark=benchmark,
        generator_id="unit-proposer",
        parameters=parameters,
        artifact_path="build/unit/submission.kmb",
        artifact_sha256=ARTIFACT_SHA,
        contract_sha256=CONTRACT_SHA,
    )


def _receipt(
    *,
    status: str,
    scope: str = "full-domain",
    score: float | None = 1.0,
    metrics: dict | None = None,
    salt_id: str = SALT_ID,
) -> VerifierReceipt:
    return VerifierReceipt(
        status=status,
        score=score,
        toffoli_equivalent=None,
        num_qubits=None,
        metrics=dict(metrics or {}),
        exit_code=0 if status == "PASS" else 1,
        command=("verifier", "submission.kmb"),
        salt_id=salt_id,
        verification_scope=scope,
        verifier_sha256=VERIFIER_SHA,
        artifact_sha256=ARTIFACT_SHA,
        contract_sha256=CONTRACT_SHA,
        duration_s=0.25,
    )


def test_an_unfinished_run_leaves_the_candidate_unverifiable(tmp_path):
    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    candidate = _candidate()
    ledger.register(candidate)

    updated = ledger.record_verification(
        candidate,
        _receipt(
            status="ERROR",
            score=None,
            metrics={"error": "official verifier produced no parseable JSON summary"},
        ),
        incumbent_score=3.0,
    )

    assert updated.status is CandidateStatus.UNVERIFIABLE
    node = ledger.node(candidate.candidate_id)
    assert node.state is CandidateLifecycleState.UNRESOLVED
    assert "capability gap" in (node.rejection_reason or "")
    # Nothing was measured, so nothing may become an incumbent.
    assert ledger.incumbent("dqi") == (None, None)


def test_a_failed_verification_is_still_a_rejection(tmp_path):
    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    candidate = _candidate()
    ledger.register(candidate)

    updated = ledger.record_verification(
        candidate,
        _receipt(status="FAIL", score=None, metrics={"error": "value mismatch"}),
        incumbent_score=3.0,
    )

    assert updated.status is CandidateStatus.VERIFIER_REJECTED
    assert ledger.node(candidate.candidate_id).state is (
        CandidateLifecycleState.CONFIRMATION_REJECTED
    )


def test_a_clean_payload_behind_an_error_exit_is_not_promoted(tmp_path):
    """A crashed run can still print a plausible summary; the exit code decides."""
    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    candidate = _candidate()
    ledger.register(candidate)

    updated = ledger.record_verification(
        candidate,
        _receipt(
            status="ERROR",
            score=0.5,  # cheaper than the incumbent, but from a run that did not finish
            metrics={"value_errors": 0, "phase_errors": 0},
        ),
        incumbent_score=5.0,
    )

    assert updated.status is CandidateStatus.UNVERIFIABLE
    assert ledger.incumbent("dqi") == (None, None)
    assert ledger.verdict_summary()["eligible"] == 0


def test_a_salt_that_reached_a_broken_verifier_is_still_spent(tmp_path):
    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    first = _candidate(seed=1.0)
    second = _candidate(seed=2.0)
    ledger.register(first)
    ledger.register(second)

    ledger.record_verification(
        first,
        _receipt(status="ERROR", score=None, metrics={"error": "timeout"}),
        incumbent_score=None,
    )
    # The slice is spent: reaching a verifier consumes a secret salt whatever the
    # run returned, and `UNRESOLVED` is a terminal state in the core ledger, so an
    # ERROR burns it exactly as a FAIL would. A gap never buys a retry.
    assert ledger.burned_holdouts() == {
        ("dqi:full-domain", f"{VERIFIER_SHA}:{SALT_ID}"): first.candidate_id
    }
    with pytest.raises(HoldoutContaminationError):
        ledger.record_verification(
            second,
            _receipt(status="ERROR", score=None, metrics={"error": "timeout"}),
            incumbent_score=None,
        )


def test_the_ledger_counts_capability_gaps_apart(tmp_path):
    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    failed = _candidate(seed=1.0)
    gap = _candidate(seed=2.0)
    improved = _candidate(seed=3.0)
    for candidate in (failed, gap, improved):
        ledger.register(candidate)

    ledger.record_verification(
        failed,
        _receipt(status="FAIL", score=None, salt_id="1" * 64),
        incumbent_score=5.0,
    )
    ledger.record_verification(
        gap,
        _receipt(status="ERROR", score=None, salt_id="2" * 64),
        incumbent_score=5.0,
    )
    ledger.record_verification(
        improved,
        _receipt(
            status="PASS",
            score=1.0,
            salt_id="3" * 64,
            metrics={"value_errors": 0, "phase_errors": 0},
        ),
        incumbent_score=5.0,
    )

    summary = ledger.verdict_summary()
    assert summary["candidates"] == 3
    assert summary["unverifiable"] == 1
    assert summary["eligible"] == 2
    assert summary["by_status"] == {
        CandidateStatus.UNVERIFIABLE.value: 1,
        CandidateStatus.VERIFIED_IMPROVEMENT.value: 1,
        CandidateStatus.VERIFIER_REJECTED.value: 1,
    }


def test_an_experiment_outcome_carries_the_verdict_summary(tmp_path):
    """A run report has to show the capability gap next to the rejections."""
    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    failed = _candidate(seed=1.0)
    gap = _candidate(seed=2.0)
    for candidate in (failed, gap):
        ledger.register(candidate)
    ledger.record_verification(
        failed,
        _receipt(status="FAIL", score=None, salt_id="1" * 64),
        incumbent_score=5.0,
    )
    ledger.record_verification(
        gap,
        _receipt(status="ERROR", score=None, salt_id="2" * 64),
        incumbent_score=5.0,
    )

    outcome = ExperimentOutcome(incumbent_score=None, results=[], ledger=ledger)
    assert outcome.verdict_summary == ledger.verdict_summary()
    assert outcome.verdict_summary["unverifiable"] == 1
    assert outcome.verdict_summary["eligible"] == 1
    # An outcome with no ledger cannot invent a denominator either.
    empty = ExperimentOutcome(incumbent_score=None, results=[])
    assert empty.verdict_summary == {
        "candidates": 0,
        "by_status": {},
        "unverifiable": 0,
        "eligible": 0,
    }


def test_summary_rows_keep_the_distinction_behind_the_core_state(tmp_path):
    """Unverifiable and evidence-incomplete both sit at UNRESOLVED in the core ledger."""
    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    gap = _candidate(seed=1.0)
    no_evidence = _candidate(seed=2.0)
    ledger.register(gap)
    ledger.register(no_evidence)

    ledger.record_verification(
        gap,
        _receipt(status="ERROR", score=None, salt_id="1" * 64),
        incumbent_score=None,
    )
    ledger.record_verification(
        no_evidence,
        _receipt(status="PASS", scope="development-prefix", salt_id="2" * 64),
        incumbent_score=None,
    )

    rows = {row["candidate_id"]: row for row in ledger.summary()}
    assert {row["state"] for row in rows.values()} == {
        CandidateLifecycleState.UNRESOLVED.value
    }
    assert rows[gap.candidate_id]["status"] == CandidateStatus.UNVERIFIABLE.value
    assert rows[no_evidence.candidate_id]["status"] == (
        CandidateStatus.INSUFFICIENT_EVIDENCE.value
    )
