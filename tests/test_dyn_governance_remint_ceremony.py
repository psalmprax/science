"""Pins for the trustee re-certification ceremony (remint machinery).

The structural gap these pins address: the governance manifest seals the
GENESIS spec SHA while the disk spec (lawfully revised + re-pinned by
``canonical_spec.py``) hashes differently, and ONLY external trustee keys
can re-close the chain. What had NO formal mechanism at all was the
re-certification procedure itself — these pins prove the ceremony harness
(``scripts/run_governance_remint_ceremony.py``) is real, lawful, and
fail-closed:

1. END-TO-END (sandbox): genesis manifest (3 trustees, quorum 2) → spec
   lawfully revised → request anchored to the "import pin" → 2 of 3
   registered trustees sign → mint chains the transparency log and the
   production verifier passes the draft against the REVISED spec — same
   roster, same spec_id, nothing stronger invented.
2. FORGERY / UNREGISTERED: an unknown trustee ID, a wrong key, and a
   forged collected record are all refused — never recorded, never minted.
3. BELOW QUORUM: one signature alone mints nothing.
4. PIN DISAGREEMENT: a request whose "pin" disagrees with the disk bytes
   fail-closes — no ceremony can run on unsynced repo pins.
5. LIVE REPO GUARD: with no ceremony state at all, the real repository
   cannot be minted (quorum shortfall → refuse) — the red stays red
   until real trustees sign, exactly per AGENTS §1.2/§1.3.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from apedp_dyn.epistemic.governance import (
    GovernanceManifest,
    GovernanceTamperingError,
    SignatureRecord,
    TransparencyLogEntry,
    TrusteeIdentity,
    generate_trustee_keypair,
    verify_governance_manifest,
)
from scripts.run_governance_remint_ceremony import (
    MANIFEST_PATH as _MANIFEST,
)
from scripts.run_governance_remint_ceremony import (
    SPEC_PATH as _SPEC,
)
from scripts.run_governance_remint_ceremony import (
    build_request,
    collect_signature,
    mint_manifest,
    status_report,
)


def _log_entry(index: int, ts: str, action: str, sha: str, tid: str,
               prev: str) -> TransparencyLogEntry:
    return TransparencyLogEntry(
        index=index, timestamp=ts, action=action, spec_sha256=sha,
        trustee_id=tid, prev_entry_hash=prev,
        entry_hash=TransparencyLogEntry.compute_hash(
            index=index, timestamp=ts, action=action, spec_sha256=sha,
            trustee_id=tid, prev_entry_hash=prev),
    )


@pytest.fixture()
def sandbox_ceremony():
    """Genesis governance sandbox: 3 trustees (fresh keypairs), quorum 2,
    genesis manifest sealed to the ORIGINAL spec bytes; then the spec is
    lawfully revised. Returns everything a ceremony needs."""
    privkeys: dict[str, str] = {}
    trustees: list[TrusteeIdentity] = []
    for tid in ("t_referee_01", "t_stat_02", "t_auditor_03"):
        priv, pub = generate_trustee_keypair()
        privkeys[tid] = priv
        trustees.append(TrusteeIdentity(trustee_id=tid, name=tid,
                                        public_key_hex=pub, role="trustee"))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        spec_path = tmp / "spec.json"
        spec_path.write_text('{"threshold": 0.95, "dm_p": 0.01, "v": 1}')
        import hashlib
        genesis_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
        ts = "2026-10-18T00:00:00+00:00"
        manifest = GovernanceManifest(
            spec_id="sandbox_spec_v1",
            spec_sha256=genesis_sha,
            quorum_threshold=2,
            trustees=trustees,
            signatures=[SignatureRecord(
                trustee_id="t_referee_01", signature_hex="0" * 128,
                signed_at=ts)],  # placeholder; replaced below
            transparency_log=[],
        )
        # Build real genesis: 2 real signatures over the OLD hash + chained log.
        sigs = []
        for tid in ("t_referee_01", "t_stat_02"):
            from apedp_dyn.epistemic.governance import sign_spec_sha256
            sigs.append(SignatureRecord(
                trustee_id=tid,
                signature_hex=sign_spec_sha256(privkeys[tid], genesis_sha),
                signed_at=ts))
        chain: list[TransparencyLogEntry] = []
        prev = "0" * 64
        for index, (action, tid) in enumerate(
                [("PREREGISTRATION_GENESIS", "t_referee_01"),
                 ("QUORUM_CO_SIGNATURE", "t_stat_02")]):
            entry = _log_entry(index, ts, action, genesis_sha, tid, prev)
            chain.append(entry)
            prev = entry.entry_hash
        manifest.signatures = sigs
        manifest.transparency_log = chain
        manifest_path = tmp / "manifest.json"
        manifest.save_to_file(manifest_path)

        # Lawful post-certification revision (the desync event in miniature).
        spec_path.write_text('{"threshold": 0.90, "dm_p": 0.01, "v": 2}')
        yield {
            "spec_path": spec_path,
            "manifest_path": manifest_path,
            "privkeys": privkeys,
            "genesis_sha": genesis_sha,
        }


def test_ceremony_end_to_end_rechained_and_verifies(sandbox_ceremony):
    """2-of-3 registered trustees re-seal the REVISED spec; production
    verification passes the minted manifest end to end."""
    import hashlib
    env = sandbox_ceremony
    new_sha = hashlib.sha256(env["spec_path"].read_bytes()).hexdigest()

    request = build_request(env["spec_path"], env["manifest_path"],
                            pinned_sha=new_sha)
    assert request["message_trustees_sign"] == new_sha
    assert request["prior_spec_sha256"] == env["genesis_sha"]
    assert request["prior_manifest_transparency_log_intact"] is True

    collected = [
        collect_signature(request, env["manifest_path"],
                          "t_referee_01", env["privkeys"]["t_referee_01"]),
        collect_signature(request, env["manifest_path"],
                          "t_stat_02", env["privkeys"]["t_stat_02"]),
    ]
    for record in collected:
        assert record["spec_sha256"] == new_sha
        assert record["signature_hex"]

    draft = mint_manifest(request, env["manifest_path"], collected,
                          env["spec_path"])
    assert draft.spec_id == "sandbox_spec_v1"          # roster/spec_id carried
    assert [t.trustee_id for t in draft.trustees] == [
        "t_referee_01", "t_stat_02", "t_auditor_03"]
    assert len(draft.transparency_log) == 4            # 2 genesis + 2 recert
    actions = [e.action for e in draft.transparency_log]
    assert actions[:2] == ["PREREGISTRATION_GENESIS", "QUORUM_CO_SIGNATURE"]
    assert actions[2:] == ["PREREGISTRATION_RECERTIFICATION"] * 2

    result = verify_governance_manifest(draft, env["spec_path"])
    assert result.is_valid and result.has_quorum, result.errors
    assert result.valid_signers == ["t_referee_01", "t_stat_02"]
    assert result.spec_sha256_match is True


def test_forgery_unregistered_and_stale_records_refused(sandbox_ceremony):
    """An unregistered signer is refused outright; a WRONG key produces a
    signature that fails verification against the REGISTERED public key and
    is never recorded; a forged collected record cannot pass mint."""
    import hashlib
    env = sandbox_ceremony
    new_sha = hashlib.sha256(env["spec_path"].read_bytes()).hexdigest()
    request = build_request(env["spec_path"], env["manifest_path"],
                            pinned_sha=new_sha)

    with pytest.raises(GovernanceTamperingError, match="not REGISTERED"):
        collect_signature(request, env["manifest_path"],
                          "t_shadow_key_99", env["privkeys"]["t_referee_01"])

    _, wrong_pub_or_priv = generate_trustee_keypair()
    with pytest.raises(GovernanceTamperingError, match="forgery"):
        collect_signature(request, env["manifest_path"],
                          "t_referee_01", wrong_pub_or_priv)

    good = collect_signature(request, env["manifest_path"],
                             "t_referee_01", env["privkeys"]["t_referee_01"])
    forged = {**good, "trustee_id": "t_stat_02"}   # good sig, wrong claimant
    with pytest.raises(GovernanceTamperingError, match="FAILED re-verification"):
        mint_manifest(request, env["manifest_path"], [forged], env["spec_path"])


def test_below_quorum_mints_nothing(sandbox_ceremony):
    """One verified signature alone can NEVER re-certify — the quorum gate is
    the production verifier's own threshold, not the ceremony's opinion."""
    import hashlib
    env = sandbox_ceremony
    new_sha = hashlib.sha256(env["spec_path"].read_bytes()).hexdigest()
    request = build_request(env["spec_path"], env["manifest_path"],
                            pinned_sha=new_sha)
    only = [collect_signature(request, env["manifest_path"],
                              "t_referee_01", env["privkeys"]["t_referee_01"])]
    with pytest.raises(GovernanceTamperingError, match="quorum"):
        mint_manifest(request, env["manifest_path"], only, env["spec_path"])


def test_pin_disagreement_fail_closed(sandbox_ceremony):
    """If the repo's own pins disagree with the disk bytes, no ceremony can
    proceed — the request refuses to mint on unsynced pins."""
    import hashlib
    env = sandbox_ceremony
    wrong_pin = hashlib.sha256(b"not the spec").hexdigest()
    assert wrong_pin != hashlib.sha256(env["spec_path"].read_bytes()).hexdigest()
    with pytest.raises(GovernanceTamperingError, match="PIN DISAGREEMENT"):
        build_request(env["spec_path"], env["manifest_path"], pinned_sha=wrong_pin)


def test_live_repo_cannot_be_minted_without_real_trustees():
    """Against the LIVE repository, with no ceremony state at ALL, minting
    fails closed (quorum shortfall on the current desync) — the red stays
    red until real trustees sign. Nothing is written."""
    report = status_report()
    assert report["inline_pins_agree"] is True, (
        "spec disk sha and canonical_spec.py pin must agree for the ceremony "
        "to be applicable at all")
    assert report["desync"] is True, (
        "the live manifest still seals the genesis SHA; if this flips, the "
        "trustees have re-certified and this guard updates naturally")

    # The ceremony's own gate: its ledger holds ZERO signatures over the
    # CURRENT spec SHA (the manifest's two valid signatures cover the OLD
    # sha only), so no quorum can close for the new hash until real
    # trustees sign in.
    request = build_request(_SPEC, _MANIFEST)   # real repo, real pin anchor
    assert request["spec_sha256"] == request["message_trustees_sign"]
    empty_ledger = []
    assert len(empty_ledger) < report["quorum_threshold"], (
        "0 collected signature ceremonies can never close the live quorum")
    with pytest.raises(GovernanceTamperingError):
        mint_manifest(request, _MANIFEST, empty_ledger, _SPEC)

