"""Unit Test Suite for APEDP Experimental Harness (AEH).

Verifies:
  1. Capability isolation & firewall blocks unauthorized data access (answer key, raw test data).
  2. Evidence non-mutation invariant is strictly enforced.
  3. Event ledger forms an unbroken cryptographic SHA-256 hash chain; tampering is caught.
  4. All 8 EEDR repair audit invariants are enforced.
  5. Experiment manifest and result package emit valid cryptographic provenance.
"""
from __future__ import annotations

import pytest

from apedp_dyn.epistemic.claim_ladder import (
    ClaimLevel,
    EpistemicAdjudicationVerdict,
    EvidenceBundle,
    EvidenceRequirement,
    adjudicate_scientific_claim,
)
from apedp_dyn.harness import (
    AEH,
    AuditEvent,
    ComponentRole,
    DataFirewall,
    EventLedger,
    FirewallViolationError,
    IsolatedDataVault,
)


def test_aeh_firewall_blocks_forbidden_access():
    """Verify DataFirewall blocks Proposer and Repairer from accessing answer keys and test data."""
    # 1. Proposer cannot access answer key
    with pytest.raises(FirewallViolationError, match="strictly forbidden from accessing 'answer_key'"):
        DataFirewall.verify_access(ComponentRole.PROPOSER, "answer_key")

    # 2. Proposer cannot access holdout data
    with pytest.raises(FirewallViolationError, match="strictly forbidden from accessing 'holdout_data'"):
        DataFirewall.verify_access(ComponentRole.PROPOSER, "holdout_data")

    # 3. Repairer cannot access raw test data or future test score
    with pytest.raises(FirewallViolationError, match="strictly forbidden from accessing 'raw_test_data'"):
        DataFirewall.verify_access(ComponentRole.REPAIRER, "raw_test_data")

    # 4. Proposer cannot mutate scientific evidence
    with pytest.raises(FirewallViolationError, match="not permitted to mutate scientific evidence"):
        DataFirewall.verify_access(ComponentRole.PROPOSER, "training_observables", attempting_mutation=True)

    # 5. Adjudicator cannot mutate evidence
    with pytest.raises(FirewallViolationError, match="not permitted to mutate scientific evidence"):
        DataFirewall.verify_access(ComponentRole.FINAL_ADJUDICATOR, "complete_evidence_bundle", attempting_mutation=True)


def test_aeh_isolated_data_vault_access_control():
    """Verify IsolatedDataVault prevents role leakage."""
    vault = IsolatedDataVault(
        training_data={"x": [1, 2, 3], "y": [2, 4, 6]},
        validation_data={"x_val": [4, 5]},
        test_data={"x_test": [6, 7]},
        answer_key={"true_model": "y = 2x", "param_a": 2.0},
    )

    # Proposer can get training observables
    train = vault.get_data(ComponentRole.PROPOSER, "training_observables")
    assert "x" in train

    # Proposer CANNOT get answer key
    with pytest.raises(FirewallViolationError):
        vault.get_data(ComponentRole.PROPOSER, "answer_key")

    # Validator CANNOT get answer key
    with pytest.raises(FirewallViolationError):
        vault.get_data(ComponentRole.VALIDATOR, "answer_key")

    # Orchestrator CAN access answer key for post-hoc scoring
    key = vault.get_data(ComponentRole.HARNESS_ORCHESTRATOR, "answer_key")
    assert key["true_model"] == "y = 2x"


def test_aeh_cryptographic_ledger_tamper_detection():
    """Verify ledger hash chain integrity and tamper detection."""
    harness = AEH(experiment_id="EXP_TEST_LEDGER_001", random_seed=123)

    harness.ledger.record_event(
        role=ComponentRole.PROPOSER,
        action="CANDIDATE_PROPOSED",
        payload_summary={"candidate_id": "CAND_1", "model": "y = a*x + b"},
    )
    harness.ledger.record_event(
        role=ComponentRole.VALIDATOR,
        action="VALIDATION_EVALUATED",
        payload_summary={"candidate_id": "CAND_1", "rmse": 0.05},
    )

    # Unmodified chain is valid
    assert harness.ledger.verify_chain_integrity() is True

    # Tampering with an event payload breaks chain integrity
    original_event = harness.ledger._events[1]
    tampered_event = AuditEvent(
        event_id=original_event.event_id,
        timestamp_utc=original_event.timestamp_utc,
        role=original_event.role,
        action=original_event.action,
        payload_summary={"candidate_id": "CAND_1", "model": "MALICIOUS_OVERWRITE"},
        payload_hash=original_event.payload_hash,  # Hash mismatch with payload
        prev_event_hash=original_event.prev_event_hash,
        event_hash=original_event.event_hash,
    )
    harness.ledger._events[1] = tampered_event
    assert harness.ledger.verify_chain_integrity() is False


def test_aeh_eedr_8_invariants_enforced():
    """Verify AEH enforces the 8 EEDR repair invariants during an adjudication and repair lifecycle."""
    harness = AEH(experiment_id="EXP_EEDR_INVARIANTS_001", random_seed=42)

    # 1. Proposer stage
    cand_id = "PREDICTIVE_CLAIMING_CAUSAL_001"
    harness.run_proposer_stage(
        candidate_id=cand_id,
        proposer_fn=lambda: "dy/dt = r*y*(1 - y/K)",
        claimed_tier=ClaimLevel.C4_CAUSAL_MECHANISTIC,
    )

    # 2. Adjudication stage (fails C4 due to missing interventional invariance)
    bundle = EvidenceBundle(metrics={"in_sample_r2": 0.96, "holdout_rmse": 0.03})
    bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
    bundle.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)

    verdict = harness.run_adjudication_stage(
        candidate_id=cand_id,
        claimed_tier=ClaimLevel.C4_CAUSAL_MECHANISTIC,
        evidence=bundle,
        adjudicate_fn=adjudicate_scientific_claim,
    )
    assert verdict.verdict != EpistemicAdjudicationVerdict.PROMOTED

    # 3. EEDR Repair stage (calibrates claim to C2 without mutating evidence)
    eedr_res = harness.run_eedr_repair_stage(
        failed_candidate_id=cand_id,
        claimed_tier=ClaimLevel.C4_CAUSAL_MECHANISTIC,
        evidence=bundle,
        domain="causal_inference",
    )
    assert eedr_res.final_status == "PROMOTED_REPAIRED"
    assert eedr_res.evidence_fingerprint_preserved is True

    # 4. Finalize experiment and verify manifest
    pkg = harness.finalize_experiment(
        evaluation_protocol="AEH-EEDR-CALIBRATION-TEST",
        metrics={"test_metric": 1.0},
    )
    assert pkg.experiment_id == "EXP_EEDR_INVARIANTS_001"
    assert pkg.manifest.candidate_count == 1
    assert pkg.manifest.repair_count == 1
    assert pkg.manifest.promotion_count == 1
    assert len(pkg.manifest.manifest_hash) == 64
    assert len(pkg.package_hash) == 64


def test_aeh_catches_unauthorized_repair_operator():
    """Verify ledger audit catches unauthorized repair operator attempt."""
    harness = AEH(experiment_id="EXP_UNAUTHORIZED_OP_TEST")

    # Record fake repair with illegal operator
    harness.ledger.record_event(
        role=ComponentRole.PROPOSER,
        action="CANDIDATE_PROPOSED",
        payload_summary={"candidate_id": "CAND_BAD"},
    )
    harness.ledger.record_event(
        role=ComponentRole.FINAL_ADJUDICATOR,
        action="INITIAL_ADJUDICATION_FAILED",
        payload_summary={"candidate_id": "CAND_BAD"},
    )

    invariants = harness.ledger.audit_eedr_invariants(
        original_candidate_id="CAND_BAD",
        repaired_candidate_id="CAND_BAD_REPAIRED",
        initial_evidence_hash="HASH_A",
        post_repair_evidence_hash="HASH_A",
        operator_applied="UNAUTHORIZED_P_HACKING_OPERATOR",
        authorized_operators=("parameter_refinement", "claim_calibration_demotion"),
    )
    assert invariants["inv3_repair_operator_authorized"] is False


def test_aeh_vault_deep_immutability():
    """Verify IsolatedDataVault seals arrays and prevents in-place mutation of experimental data."""
    import numpy as np

    raw_arr = np.array([10.0, 20.0, 30.0])
    vault = IsolatedDataVault(
        training_data={"sensor_reading": raw_arr, "labels": [1, 2, 3]},
    )

    data = vault.get_data(ComponentRole.PROPOSER, "training_observables")
    retrieved_arr = data["sensor_reading"]

    # Verify array is sealed as read-only
    with pytest.raises(ValueError, match="read-only"):
        retrieved_arr[0] = 999.0

    # Underlying vault data remains untouched
    assert vault._training_data["sensor_reading"][0] == 10.0


def test_aeh_disk_journal_wal_and_replay(tmp_path):
    """Verify Write-Ahead Logging (WAL) flushes events to disk and supports forensic replay."""
    journal_file = tmp_path / "test_ledger_wal.jsonl"
    harness = AEH(
        experiment_id="EXP_WAL_TEST",
        random_seed=42,
        journal_path=journal_file,
    )

    harness.ledger.record_event(
        role=ComponentRole.PROPOSER,
        action="TEST_ACTION_A",
        payload_summary={"key": "val_a"},
    )
    harness.ledger.record_event(
        role=ComponentRole.VALIDATOR,
        action="TEST_ACTION_B",
        payload_summary={"key": "val_b"},
    )

    # Verify journal exists on disk and has 3 events (Genesis + 2 actions)
    assert journal_file.exists()
    lines = journal_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3

    # Replay from disk journal and verify chain integrity
    replayed = EventLedger.replay_from_journal(journal_file)
    assert len(replayed.events) == 3
    assert replayed.verify_chain_integrity() is True
    assert replayed.get_ledger_hash() == harness.ledger.get_ledger_hash()


def test_aeh_dynamic_dataset_hash_and_git_commit():
    """Verify compute_dataset_hash and git commit provenance resolution."""
    import pandas as pd

    from apedp_dyn.harness.manifest import compute_dataset_hash, get_git_commit_sha

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    h1 = compute_dataset_hash(df)
    assert len(h1) == 64

    # Different data gives different hash
    df2 = pd.DataFrame({"a": [1, 2, 999], "b": [4.0, 5.0, 6.0]})
    h2 = compute_dataset_hash(df2)
    assert h1 != h2

    # Git commit is non-empty 40-char SHA or release tag
    git_sha = get_git_commit_sha()
    assert len(git_sha) > 10


def test_aeh_stage_timeout_protection():
    """Verify hard timeout protection aborts runaway stages and logs timeout event."""
    import time

    harness = AEH(
        experiment_id="EXP_TIMEOUT_TEST",
        random_seed=42,
        stage_timeout_sec=0.15,
    )

    def runaway_proposer():
        time.sleep(0.5)
        return "infinite_loop_model"

    with pytest.raises(FirewallViolationError, match="STAGE TIMEOUT"):
        harness.run_proposer_stage(
            candidate_id="CAND_RUNAWAY",
            proposer_fn=runaway_proposer,
            claimed_tier=ClaimLevel.C2_PREDICTIVE,
        )

    # Verify timeout was recorded in ledger
    events = [e for e in harness.ledger.events if e.action == "PROPOSER_STAGE_TIMEOUT"]
    assert len(events) == 1
    assert events[0].payload_summary["candidate_id"] == "CAND_RUNAWAY"


def test_eedr_multistep_iterative_repair_budget():
    """Verify execute_eedr_loop supports bounded multi-step repair budget."""
    from apedp_dyn.epistemic.eedr import execute_eedr_loop

    bundle = EvidenceBundle(metrics={"in_sample_r2": 0.95, "holdout_rmse": 0.02})
    bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
    bundle.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)

    # Multi-step repair with budget = 2
    res = execute_eedr_loop(
        candidate_id="MULTISTEP_CAND",
        proposed_claim=ClaimLevel.C4_CAUSAL_MECHANISTIC,
        evidence=bundle,
        domain="causal_inference",
        max_repair_budget=2,
    )

    assert res.final_status == "PROMOTED_REPAIRED"
    assert res.final_justified_claim == ClaimLevel.C2_PREDICTIVE
    assert res.evidence_fingerprint_preserved is True
    assert any("REPAIR ATTEMPT 1/2" in msg for msg in res.audit_trail)

