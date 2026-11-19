"""Pins for trustee signature intake (the last manual step in FLAW-01 closure).

The ceremony could mint a re-signed manifest, but a signature emailed back by a
trustee had to be pasted into the ceremony state by hand. These pins prove the
intake path only ever records a signature that verifies against the manifest's
REGISTERED public key over the current request digest, and that it refuses
everything else: wrong key, unregistered signer, malformed hex, naive timestamp,
stale request, duplicate trustee.

The sandbox builds its own keypairs and manifest; no repository signature is
generated or replayed, and the live manifest is never written.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from apedp_dyn.epistemic.governance import (
    GovernanceManifest,
    GovernanceTamperingError,
    TrusteeIdentity,
    generate_trustee_keypair,
    sign_spec_sha256,
)
from scripts.intake_governance_signature import (
    load_ledger,
    record_signature,
)

DIGEST_PLACEHOLDER = "{{digest}}"


@pytest.fixture()
def sandbox(tmp_path):
    """A registered 2-of-2 sandbox: current spec, manifest, request, empty ledger."""
    spec_path = tmp_path / "spec.json"
    spec_path.write_text('{"threshold": 0.95, "dm_p": 0.01}')
    digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()

    lead_priv, lead_pub = generate_trustee_keypair()
    stat_priv, stat_pub = generate_trustee_keypair()
    manifest = GovernanceManifest(
        spec_id="sandbox_spec_v1",
        spec_sha256=digest,
        quorum_threshold=2,
        trustees=[
            TrusteeIdentity(trustee_id="t_lead_01", name="Lead", public_key_hex=lead_pub,
                            role="Chief Referee"),
            TrusteeIdentity(trustee_id="t_stat_02", name="Stat", public_key_hex=stat_pub,
                            role="Statistical Referee"),
        ],
        signatures=[],
        transparency_log=[],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save_to_file(manifest_path)

    request = {
        "request_kind": "governance_remint_request",
        "spec_id": "sandbox_spec_v1",
        "spec_sha256": digest,
        "message_trustees_sign": digest,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    return {
        "spec_path": spec_path,
        "digest": digest,
        "manifest_path": manifest_path,
        "request_path": request_path,
        "ledger_path": tmp_path / "signatures.json",
        "lead_priv": lead_priv,
        "lead_pub": lead_pub,
        "stat_priv": stat_priv,
    }


def _record(env, trustee_id, signature_hex, signed_at="2026-11-07T14:05:00+00:00"):
    return record_signature(
        ledger_path=env["ledger_path"],
        manifest_path=env["manifest_path"],
        request_path=env["request_path"],
        trustee_id=trustee_id,
        signature_hex=signature_hex,
        signed_at=signed_at,
        spec_path=env["spec_path"],
    )


def test_a_verified_signature_is_recorded_with_its_registered_key(sandbox):
    signature = sign_spec_sha256(sandbox["lead_priv"], sandbox["digest"])
    record = _record(sandbox, "t_lead_01", signature)

    assert record["trustee_id"] == "t_lead_01"
    assert record["spec_sha256"] == sandbox["digest"]
    assert record["signature_hex"] == signature
    # The record names the key it was checked against, not the one it came with.
    assert record["verified_against"] == sandbox["lead_pub"]

    ledger = load_ledger(sandbox["ledger_path"])
    assert ledger["request_spec_sha256"] == sandbox["digest"]
    assert [s["trustee_id"] for s in ledger["signatures"]] == ["t_lead_01"]
    assert sandbox["ledger_path"].exists()


def test_intake_never_writes_the_manifest_or_the_spec(sandbox):
    manifest_before = sandbox["manifest_path"].read_bytes()
    spec_before = sandbox["spec_path"].read_bytes()
    signature = sign_spec_sha256(sandbox["lead_priv"], sandbox["digest"])
    _record(sandbox, "t_lead_01", signature)
    assert sandbox["manifest_path"].read_bytes() == manifest_before
    assert sandbox["spec_path"].read_bytes() == spec_before


def test_a_wrong_key_is_refused_and_nothing_is_recorded(sandbox):
    """Signed correctly, but by a key that is not the registered one."""
    forged = sign_spec_sha256(sandbox["stat_priv"], sandbox["digest"])
    with pytest.raises(GovernanceTamperingError, match="REGISTERED public key"):
        _record(sandbox, "t_lead_01", forged)
    assert not sandbox["ledger_path"].exists()


def test_an_unregistered_signer_is_refused(sandbox):
    stranger_priv, _ = generate_trustee_keypair()
    signature = sign_spec_sha256(stranger_priv, sandbox["digest"])
    with pytest.raises(GovernanceTamperingError, match="not REGISTERED"):
        _record(sandbox, "t_shadow_99", signature)
    assert not sandbox["ledger_path"].exists()


def test_a_malformed_signature_is_refused(sandbox):
    with pytest.raises(GovernanceTamperingError, match="128 lowercase hex"):
        _record(sandbox, "t_lead_01", "deadbeef")


def test_a_naive_timestamp_is_refused(sandbox):
    """An append-only log cannot later disambiguate a zone-less instant."""
    signature = sign_spec_sha256(sandbox["lead_priv"], sandbox["digest"])
    with pytest.raises(GovernanceTamperingError, match="timezone"):
        _record(sandbox, "t_lead_01", signature, signed_at="2026-11-07T14:05:00")


def test_a_stale_request_is_refused(sandbox):
    """The spec moved after the request was issued: signatures are void."""
    signature = sign_spec_sha256(sandbox["lead_priv"], sandbox["digest"])
    sandbox["spec_path"].write_text('{"threshold": 0.90}')
    with pytest.raises(GovernanceTamperingError, match="STALE"):
        _record(sandbox, "t_lead_01", signature)


def test_a_duplicate_trustee_is_refused(sandbox):
    signature = sign_spec_sha256(sandbox["lead_priv"], sandbox["digest"])
    _record(sandbox, "t_lead_01", signature)
    with pytest.raises(GovernanceTamperingError, match="Duplicate"):
        _record(sandbox, "t_lead_01", signature)


def test_two_trustees_fill_a_two_of_three_quorum_ledger(sandbox):
    lead = sign_spec_sha256(sandbox["lead_priv"], sandbox["digest"])
    stat = sign_spec_sha256(sandbox["stat_priv"], sandbox["digest"])
    _record(sandbox, "t_lead_01", lead)
    _record(sandbox, "t_stat_02", stat, signed_at="2026-11-07T14:12:00+00:00")
    ledger = load_ledger(sandbox["ledger_path"])
    assert sorted(s["trustee_id"] for s in ledger["signatures"]) == [
        "t_lead_01", "t_stat_02"]
