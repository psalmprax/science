"""Comprehensive unit & integration test suite for Cryptographic Multi-Party Preregistration Governance.

Verifies:
1. M-of-N Ed25519 quorum signature verification.
2. Tampered specification detection (raises GovernanceTamperingError).
3. Insufficient signature count refusal (< M).
4. Forged / invalid signature refusal.
5. Append-only transparency log hash-chain verification and tampering detection.
6. Full end-to-end enforcement on live repository specifications.
"""
from __future__ import annotations

import datetime
import tempfile
from pathlib import Path

import pytest

from apedp_dyn.epistemic.governance import (
    GovernanceManifest,
    GovernanceTamperingError,
    SignatureRecord,
    TransparencyLogEntry,
    TrusteeIdentity,
    compute_file_sha256,
    enforce_preregistration_governance,
    generate_trustee_keypair,
    sign_spec_sha256,
    verify_ed25519_signature,
    verify_governance_manifest,
)


def test_ed25519_keypair_generation_and_signature_verification():
    priv, pub = generate_trustee_keypair()
    assert len(priv) == 64
    assert len(pub) == 64

    test_hash = "abcdef1234567890" * 4
    sig = sign_spec_sha256(priv, test_hash)
    assert len(sig) == 128  # 64 bytes hex

    # Verify signature
    assert verify_ed25519_signature(pub, test_hash, sig) is True

    # Tampered message fails
    assert verify_ed25519_signature(pub, "0000000000000000" * 4, sig) is False

    # Corrupted signature fails
    corrupt_sig = "0" * 128
    assert verify_ed25519_signature(pub, test_hash, corrupt_sig) is False


def test_governance_quorum_and_tampering_detection():
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = Path(tmpdir) / "spec.json"
        spec_path.write_text('{"threshold": 0.95, "dm_p": 0.01}')
        spec_hash = compute_file_sha256(spec_path)

        # 3 trustees
        priv1, pub1 = generate_trustee_keypair()
        priv2, pub2 = generate_trustee_keypair()
        _priv3, pub3 = generate_trustee_keypair()

        t1 = TrusteeIdentity("t1", "Trustee 1", pub1)
        t2 = TrusteeIdentity("t2", "Trustee 2", pub2)
        t3 = TrusteeIdentity("t3", "Trustee 3", pub3)

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        sig1 = SignatureRecord("t1", sign_spec_sha256(priv1, spec_hash), now)
        sig2 = SignatureRecord("t2", sign_spec_sha256(priv2, spec_hash), now)

        # Transparency log
        h0 = TransparencyLogEntry.compute_hash(0, now, "GENESIS", spec_hash, "t1", "0" * 64)
        entry0 = TransparencyLogEntry(0, now, "GENESIS", spec_hash, "t1", "0" * 64, h0)

        # Manifest requiring 2-of-3
        manifest = GovernanceManifest(
            spec_id="test_spec",
            spec_sha256=spec_hash,
            quorum_threshold=2,
            trustees=[t1, t2, t3],
            signatures=[sig1, sig2],
            transparency_log=[entry0],
        )

        # Valid verification
        res = verify_governance_manifest(manifest, spec_path)
        assert res.is_valid is True
        assert res.valid_signature_count == 2
        assert res.has_quorum is True

        # Tampering attack 1: Modifying specification file on disk
        tampered_path = Path(tmpdir) / "tampered.json"
        tampered_path.write_text('{"threshold": 0.50, "dm_p": 0.05}')  # Relaxed thresholds
        res_tampered = verify_governance_manifest(manifest, tampered_path)
        assert res_tampered.is_valid is False
        assert res_tampered.spec_sha256_match is False

        # Tampering attack 2: Insufficient signatures (only 1 signature for threshold 2)
        insufficient_manifest = GovernanceManifest(
            spec_id="test_spec",
            spec_sha256=spec_hash,
            quorum_threshold=2,
            trustees=[t1, t2, t3],
            signatures=[sig1],
            transparency_log=[entry0],
        )
        res_insufficient = verify_governance_manifest(insufficient_manifest, spec_path)
        assert res_insufficient.is_valid is False
        assert res_insufficient.has_quorum is False

        # Tampering attack 3: Broken transparency hash chain
        bad_entry0 = TransparencyLogEntry(0, now, "GENESIS", spec_hash, "t1", "0" * 64, "badhash" * 8)
        bad_chain_manifest = GovernanceManifest(
            spec_id="test_spec",
            spec_sha256=spec_hash,
            quorum_threshold=2,
            trustees=[t1, t2, t3],
            signatures=[sig1, sig2],
            transparency_log=[bad_entry0],
        )
        res_bad_chain = verify_governance_manifest(bad_chain_manifest, spec_path)
        assert res_bad_chain.is_valid is False
        assert res_bad_chain.transparency_log_verified is False


def test_enforce_preregistration_governance_raises_on_violation():
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = Path(tmpdir) / "spec.json"
        spec_path.write_text('{"dm_p": 0.01}')
        manifest_path = Path(tmpdir) / "governance.json"

        # Manifest with mismatched hash
        manifest = GovernanceManifest(
            spec_id="spec",
            spec_sha256="0" * 64,
            quorum_threshold=1,
            trustees=[],
            signatures=[],
            transparency_log=[],
        )
        manifest.save_to_file(manifest_path)

        with pytest.raises(GovernanceTamperingError) as exc_info:
            enforce_preregistration_governance(spec_path, manifest_path)

        assert "EPISTEMIC GOVERNANCE VIOLATION" in str(exc_info.value)


def test_canonical_repository_governance_manifest_valid():
    """Verify that the repository's canonical governance manifest certifies the canonical spec."""
    spec_path = Path(__file__).resolve().parent.parent / "specifications" / "canonical_preregistered_spec_v1.json"
    manifest_path = Path(__file__).resolve().parent.parent / "specifications" / "governance_manifest.json"

    assert spec_path.is_file()
    assert manifest_path.is_file()

    result = enforce_preregistration_governance(spec_path, manifest_path)
    assert result.is_valid is True
    assert result.spec_sha256_match is True
    assert result.has_quorum is True
    assert result.valid_signature_count >= 2
    assert result.transparency_log_verified is True
