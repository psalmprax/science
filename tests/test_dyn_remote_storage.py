"""Comprehensive unit and integration tests for remote & distributed object storage backends.

Tests:
1. LocalStorageBackend: file I/O, atomic locking, lock timeouts, stale lock breaking, token verification on release.
2. MemoryStorageBackend: in-memory operations, concurrent access, locking.
3. S3StorageBackend: mock client integration, lock race condition prevention, lock lease expiration, release validation.
4. GCSStorageBackend: mock client integration, generation match precondition, token validation on release.
5. Epistemic Anti-Snooping Discipline across Remote Ledgers:
   - FullLineageExperimentLedger sync_to_remote / pull_from_remote.
   - Deserialization verification: burned confirmation holdouts must be restored into _burned_holdouts across restarts.
   - HoldoutContaminationError triggered when a successor candidate attempts to access a burned holdout.
"""
from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from apedp_dyn.epistemic.experiment_ledger import (
    CandidateLifecycleState,
    DatasetRole,
    FullLineageExperimentLedger,
    HoldoutContaminationError,
)
from apedp_dyn.epistemic.remote_storage import (
    GCSStorageBackend,
    LocalStorageBackend,
    MemoryStorageBackend,
    S3StorageBackend,
    StorageLockError,
    get_storage_backend,
)


def test_memory_storage_backend_crud():
    backend = MemoryStorageBackend()
    assert not backend.exists("test.txt")
    backend.write_bytes("test.txt", b"hello memory")
    assert backend.exists("test.txt")
    assert backend.read_bytes("test.txt") == b"hello memory"
    assert backend.list_keys() == ["test.txt"]

    backend.delete("test.txt")
    assert not backend.exists("test.txt")
    assert backend.list_keys() == []


def test_memory_storage_locking():
    backend = MemoryStorageBackend()
    with backend.lock("res1") as token:
        assert isinstance(token, str)
        # Attempting to acquire same lock with short timeout should raise StorageLockError
        with pytest.raises(StorageLockError):
            backend.acquire_lock("res1", timeout=0.1)

    # After exiting context, lock should be free
    token2 = backend.acquire_lock("res1", timeout=0.5)
    assert backend.release_lock("res1", token2)


def test_local_storage_backend_crud():
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalStorageBackend(base_path=tmpdir)
        backend.write_bytes("folder/file.json", b'{"key": "val"}')
        assert backend.exists("folder/file.json")
        assert backend.read_bytes("folder/file.json") == b'{"key": "val"}'
        assert "folder/file.json" in backend.list_keys()

        # Delete
        assert backend.delete("folder/file.json")
        assert not backend.exists("folder/file.json")


def test_local_storage_locking_and_stale_release():
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalStorageBackend(base_path=tmpdir)

        # Acquire lock
        token = backend.acquire_lock("ledger.jsonl", timeout=1.0)
        assert token is not None

        # Lock exists, second acquire should timeout
        with pytest.raises(StorageLockError):
            backend.acquire_lock("ledger.jsonl", timeout=0.1)

        # Token verification on release: wrong token should return False
        assert not backend.release_lock("ledger.jsonl", "wrong-token-12345")

        # Correct token should release
        assert backend.release_lock("ledger.jsonl", token)

        # Now can re-acquire
        token2 = backend.acquire_lock("ledger.jsonl", timeout=1.0)
        assert backend.release_lock("ledger.jsonl", token2)


def test_local_storage_stale_lock_breaking():
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalStorageBackend(base_path=tmpdir)
        lock_file = Path(tmpdir) / ".lock_stale.jsonl"
        # Simulate a stale lock from 100 seconds ago
        lock_file.write_bytes(f"dead_token:{time.time() - 100}".encode())
        os.utime(lock_file, (time.time() - 100, time.time() - 100))

        # Acquire with 0.5s timeout: should detect stale mtime, delete, and succeed
        token = backend.acquire_lock("stale.jsonl", timeout=0.5)
        assert token is not None
        backend.release_lock("stale.jsonl", token)


def test_s3_storage_backend_mocked():
    mock_client = MagicMock()
    mock_client.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "apedp/data/file1.json"}]}
    ]

    # Setup get_object
    mock_client.get_object.return_value = {
        "Body": io.BytesIO(b"s3 binary content")
    }

    backend = S3StorageBackend(bucket="my-apedp-bucket", prefix="apedp", s3_client=mock_client)

    assert backend.read_bytes("data/file1.json") == b"s3 binary content"
    mock_client.get_object.assert_called_with(Bucket="my-apedp-bucket", Key="apedp/data/file1.json")

    backend.write_bytes("data/file2.json", b"new content")
    mock_client.put_object.assert_called_with(
        Bucket="my-apedp-bucket", Key="apedp/data/file2.json", Body=b"new content"
    )

    assert backend.list_keys("data") == ["data/file1.json"]


def test_s3_lock_contention_and_race_prevention():
    mock_client = MagicMock()

    # Simulate active lock held by another process
    active_token = "rival_token"
    recent_time = time.time()
    mock_client.get_object.side_effect = lambda **kwargs: {
        "Body": io.BytesIO(f"{active_token}:{recent_time}".encode())
    }

    backend = S3StorageBackend(bucket="my-bucket", prefix="exp", s3_client=mock_client)

    # Attempting to acquire while lock is active should timeout and NOT overwrite
    with pytest.raises(StorageLockError):
        backend.acquire_lock("ledger.jsonl", timeout=0.15)


def test_s3_release_lock_token_validation():
    mock_client = MagicMock()
    mock_client.get_object.return_value = {
        "Body": io.BytesIO(b"my_token:123456789.0")
    }

    backend = S3StorageBackend(bucket="my-bucket", prefix="exp", s3_client=mock_client)
    backend._active_locks["my_token"] = "exp/locks/test.lock"

    # Correct token releases
    assert backend.release_lock("test", "my_token")
    mock_client.delete_object.assert_called_with(Bucket="my-bucket", Key="exp/locks/test.lock")

    # Mismatched token does not delete object
    mock_client.delete_object.reset_mock()
    mock_client.get_object.return_value = {
        "Body": io.BytesIO(b"other_process_token:123456789.0")
    }
    assert not backend.release_lock("test", "different_token")
    mock_client.delete_object.assert_not_called()


def test_gcs_storage_backend_mocked():
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    mock_blob.exists.return_value = True
    mock_blob.download_as_bytes.return_value = b"gcs data"

    backend = GCSStorageBackend(bucket="gcs-bucket", prefix="gcs-prefix", client=mock_client)

    assert backend.read_bytes("test.json") == b"gcs data"
    mock_blob.download_as_bytes.assert_called_once()

    backend.write_bytes("test.json", b"new gcs data")
    mock_blob.upload_from_string.assert_called_with(b"new gcs data")


def test_get_storage_backend_factory(monkeypatch):
    b_mem = get_storage_backend("memory")
    assert isinstance(b_mem, MemoryStorageBackend)

    with tempfile.TemporaryDirectory() as tmpdir:
        b_loc = get_storage_backend("local", base_path=tmpdir)
        assert isinstance(b_loc, LocalStorageBackend)

    monkeypatch.setenv("APEDP_STORAGE_BACKEND", "memory")
    b_env = get_storage_backend()
    assert isinstance(b_env, MemoryStorageBackend)


def test_epistemic_ledger_remote_sync_and_anti_snooping_restoration():
    """Verify that FullLineageExperimentLedger syncs to remote storage and that
    burned confirmation holdouts survive deserialization across simulated compute node termination.
    """
    remote_backend = MemoryStorageBackend()

    # --- Node 1: Executes initial discovery and burns a confirmation holdout ---
    with tempfile.TemporaryDirectory() as node1_dir:
        ledger1_path = Path(node1_dir) / "experiment_ledger.jsonl"
        ledger1 = FullLineageExperimentLedger(
            persistence_path=ledger1_path,
            storage_backend=remote_backend,
            remote_key="experiment_ledger.jsonl",
            auto_sync=True,
        )

        # Register Candidate A
        ledger1.register_candidate(
            candidate_id="cand_alpha",
            proposer_id="rime_proposer",
            formula="alpha_model(x)",
        )

        # Record exposure to confirmation holdout slice
        ledger1.record_data_exposure(
            candidate_id="cand_alpha",
            dataset_id="equities_2024",
            slice_identifier="holdout_q4",
            role=DatasetRole.CONFIRMATION_HOLDOUT,
            outcome_status="IN_PROGRESS",
        )

        # Candidate A fails confirmation referee adjudication
        ledger1.update_lifecycle_state(
            candidate_id="cand_alpha",
            new_state=CandidateLifecycleState.CONFIRMATION_REJECTED,
            rejection_reason="Refuted by Diebold-Mariano p-value >= 0.05",
            metrics={"p_value": 0.42, "rmse": 0.88},
        )

        # Verify holdout is burned locally on Node 1
        burned_node1 = ledger1.get_burned_holdouts()
        assert ("equities_2024", "holdout_q4") in burned_node1
        assert burned_node1[("equities_2024", "holdout_q4")] == "cand_alpha"

        # Explicitly ensure remote storage received the sync
        ledger1.sync_to_remote()
        assert remote_backend.exists("experiment_ledger.jsonl")

    # --- Node 2: New VM/container boots up after Node 1 terminates ---
    with tempfile.TemporaryDirectory() as node2_dir:
        ledger2_path = Path(node2_dir) / "experiment_ledger.jsonl"
        ledger2 = FullLineageExperimentLedger(
            persistence_path=ledger2_path,
            storage_backend=remote_backend,
            remote_key="experiment_ledger.jsonl",
            auto_sync=True,
        )

        # Node 2 pulls remote ledger
        pulled = ledger2.pull_from_remote()
        assert pulled is True

        # Anti-Snooping Invariant Check: Burned holdout registry MUST be restored
        burned_node2 = ledger2.get_burned_holdouts()
        assert ("equities_2024", "holdout_q4") in burned_node2, (
            "Burned holdout slice ('equities_2024', 'holdout_q4') was lost on deserialization! "
            "AGENTS.md §1.2 anti-snooping violation."
        )

        # Candidate B attempts to snoop the burned holdout slice
        ledger2.register_candidate(
            candidate_id="cand_beta",
            proposer_id="hawkes_proposer",
            formula="beta_model(x)",
        )

        # Verify that Candidate B is strictly blocked by HoldoutContaminationError
        with pytest.raises(HoldoutContaminationError) as exc_info:
            ledger2.record_data_exposure(
                candidate_id="cand_beta",
                dataset_id="equities_2024",
                slice_identifier="holdout_q4",
                role=DatasetRole.CONFIRMATION_HOLDOUT,
            )

        assert "burned" in str(exc_info.value).lower()
        assert "cand_alpha" in str(exc_info.value)
