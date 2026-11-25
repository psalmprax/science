"""One signature over two files: the manifest's further sealed files, and the ceremony that fills it in.

Keys made here are test keys in temporary folders. The repository's roster, quorum and signatures are
only read.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from apedp_dyn.epistemic.governance import (
    GovernanceManifest,
    SealedFile,
    SignatureRecord,
    TrusteeIdentity,
    combined_digest,
    compute_file_sha256,
    generate_trustee_keypair,
    sign_spec_sha256,
    verify_ed25519_signature,
    verify_governance_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import governance_ceremony as ceremony


def _folder(tmp_path: Path) -> tuple[Path, list[tuple[str, str]], GovernanceManifest]:
    """Two files to seal, three test trustees with a quorum of two, and a manifest with nothing signed yet."""
    (tmp_path / "canonical_preregistered_spec_v1.json").write_text('{"threshold": 0.95}')
    (tmp_path / "discovery_checks_v1.json").write_text('{"accuracy": 0.999}')
    keys = [generate_trustee_keypair() for _ in range(3)]
    trustees = [TrusteeIdentity(f"t{i}", f"Trustee {i}", public) for i, (_private, public) in enumerate(keys)]
    manifest = GovernanceManifest("canonical_preregistered_spec_v1", "0" * 64, 2, trustees=trustees)
    return tmp_path, keys, manifest


def test_a_manifest_that_seals_one_file_is_signed_and_written_as_before() -> None:
    repository = json.loads((ROOT / "specifications" / "governance_manifest.json").read_text(encoding="utf-8"))
    manifest = GovernanceManifest.from_dict(repository)
    if not manifest.also_sealed:                              # until the trustees have signed for two files
        assert manifest.signed_message() == manifest.spec_sha256
        assert manifest.to_dict() == repository               # nothing added on the way back out
    one = GovernanceManifest("spec", "ab" * 32, 2)
    assert one.signed_message() == "ab" * 32 and "also_sealed" not in one.to_dict()


def test_the_message_for_two_files_is_a_fixed_text_hashed() -> None:
    also = [SealedFile("discovery_checks_v1", "discovery_checks_v1.json", "22" * 32)]
    text = ("apedp-governance-manifest/2\ncanonical_preregistered_spec_v1:" + "11" * 32
            + "\ndiscovery_checks_v1:discovery_checks_v1.json:" + "22" * 32 + "\n")
    assert combined_digest("canonical_preregistered_spec_v1", "11" * 32, also) == hashlib.sha256(text.encode()).hexdigest()
    more = [*also, SealedFile("another", "another.json", "33" * 32)]
    assert combined_digest("s", "11" * 32, more) == combined_digest("s", "11" * 32, list(reversed(more)))   # by id, not by order
    assert combined_digest("s", "11" * 32, also) != combined_digest("s", "11" * 32, [SealedFile("discovery_checks_v1", "discovery_checks_v1.json", "23" * 32)])


def test_signatures_over_the_combined_message_verify_and_over_the_specification_alone_do_not(tmp_path: Path) -> None:
    folder, keys, manifest = _folder(tmp_path)
    spec_sha256, also = ceremony.to_be_sealed(folder)
    signed = combined_digest(manifest.spec_id, spec_sha256, also)
    sealed = GovernanceManifest(manifest.spec_id, spec_sha256, 2, trustees=manifest.trustees, also_sealed=also,
                                signatures=[SignatureRecord(f"t{i}", sign_spec_sha256(keys[i][0], signed), "2026-11-25T00:00:00+00:00")
                                            for i in (0, 1)])
    result = verify_governance_manifest(sealed, folder / "canonical_preregistered_spec_v1.json")
    assert result.is_valid and result.also_sealed_match and result.valid_signature_count == 2

    old_way = GovernanceManifest(manifest.spec_id, spec_sha256, 2, trustees=manifest.trustees, also_sealed=also,
                                 signatures=[SignatureRecord(f"t{i}", sign_spec_sha256(keys[i][0], spec_sha256), "2026-11-25T00:00:00+00:00")
                                             for i in (0, 1)])
    refused = verify_governance_manifest(old_way, folder / "canonical_preregistered_spec_v1.json")
    assert not refused.is_valid and refused.valid_signature_count == 0      # a signature for one file does not seal two


def test_a_change_to_the_second_file_breaks_the_seal(tmp_path: Path) -> None:
    folder, keys, manifest = _folder(tmp_path)
    signed = ceremony.message(manifest, folder)
    new = ceremony.assemble(manifest, [(f"t{i}", sign_spec_sha256(keys[i][0], signed), "2026-11-25T00:00:00+00:00") for i in (0, 2)],
                            folder)
    spec = folder / "canonical_preregistered_spec_v1.json"
    assert verify_governance_manifest(new, spec).is_valid
    (folder / "discovery_checks_v1.json").write_text('{"accuracy": 0.99}')            # a threshold loosened
    broken = verify_governance_manifest(new, spec)
    assert not broken.is_valid and broken.spec_sha256_match and not broken.also_sealed_match
    assert any("discovery_checks_v1.json" in error and "mismatch" in error for error in broken.errors)
    (folder / "discovery_checks_v1.json").unlink()
    assert any("missing" in error for error in verify_governance_manifest(new, spec).errors)


@pytest.mark.parametrize("entry, word", [
    (SealedFile("x", "../canonical_preregistered_spec_v1.json", "0" * 64), "plain file name"),
    (SealedFile("x", "sub/discovery_checks_v1.json", "0" * 64), "plain file name"),
    (SealedFile("canonical_preregistered_spec_v1", "discovery_checks_v1.json", "0" * 64), "more than once"),
])
def test_a_sealed_file_is_a_plain_name_listed_once(tmp_path: Path, entry: SealedFile, word: str) -> None:
    folder, _keys, manifest = _folder(tmp_path)
    manifest.spec_sha256 = compute_file_sha256(folder / "canonical_preregistered_spec_v1.json")
    manifest.also_sealed = [entry]
    result = verify_governance_manifest(manifest, folder / "canonical_preregistered_spec_v1.json")
    assert not result.is_valid and not result.also_sealed_match and any(word in error for error in result.errors)


def test_the_ceremony_keeps_the_roster_and_the_quorum_and_only_appends_to_the_log(tmp_path: Path) -> None:
    folder, keys, manifest = _folder(tmp_path)
    signed = ceremony.message(manifest, folder)
    first = ceremony.assemble(manifest, [(f"t{i}", sign_spec_sha256(keys[i][0], signed), f"2026-11-25T0{i}:00:00+00:00") for i in (0, 1)],
                              folder)
    assert first.trustees == manifest.trustees and first.quorum_threshold == manifest.quorum_threshold
    assert [e.action for e in first.transparency_log] == ["SPEC_REVISION", "QUORUM_CO_SIGNATURE"]
    assert all(e.spec_sha256 == signed for e in first.transparency_log)          # the log records what was signed

    (folder / "discovery_checks_v1.json").write_text('{"accuracy": 0.999, "share": 0.05}')   # a later revision
    again = ceremony.message(first, folder)
    assert again != signed
    second = ceremony.assemble(first, [(f"t{i}", sign_spec_sha256(keys[i][0], again), "2026-12-01T00:00:00+00:00") for i in (1, 2)],
                               folder)
    assert second.transparency_log[:2] == first.transparency_log                # never rewritten
    assert len(second.transparency_log) == 4
    assert verify_governance_manifest(second, folder / "canonical_preregistered_spec_v1.json").is_valid


def test_the_ceremony_refuses_what_is_not_a_registered_trustees_signature_over_the_message(tmp_path: Path) -> None:
    folder, keys, manifest = _folder(tmp_path)
    signed = ceremony.message(manifest, folder)
    good = ("t0", sign_spec_sha256(keys[0][0], signed), "2026-11-25T00:00:00+00:00")
    outsider_private, _public = generate_trustee_keypair()
    with pytest.raises(ValueError, match="not a registered trustee"):
        ceremony.assemble(manifest, [good, ("someone_else", sign_spec_sha256(outsider_private, signed), "t")], folder)
    with pytest.raises(ValueError, match="two signatures"):
        ceremony.assemble(manifest, [good, good], folder)
    with pytest.raises(ValueError, match="does not verify"):                     # a registered id, another key
        ceremony.assemble(manifest, [good, ("t1", sign_spec_sha256(outsider_private, signed), "t")], folder)
    with pytest.raises(ValueError, match="does not verify"):                     # the old message, for one file only
        ceremony.assemble(manifest, [good, ("t1", sign_spec_sha256(keys[1][0], manifest.spec_sha256), "t")], folder)


def test_the_request_to_the_trustees_names_the_message_the_code_computes() -> None:
    """The document a trustee reads must carry exactly what the ceremony will verify."""
    manifest = GovernanceManifest.from_file(ROOT / "specifications" / "governance_manifest.json")
    spec_sha256, also = ceremony.to_be_sealed()
    request = (ROOT / "docs" / "GOVERNANCE_RESIGNING_REQUEST.md").read_text(encoding="utf-8")
    if manifest.also_sealed and manifest.signed_message() == ceremony.message(manifest):
        return                                              # signed: the request is history
    assert ceremony.message(manifest) in request
    assert spec_sha256 in request and all(entry.sha256 in request for entry in also)


def test_the_repositorys_old_signatures_are_not_signatures_over_the_new_message() -> None:
    manifest = GovernanceManifest.from_file(ROOT / "specifications" / "governance_manifest.json")
    if manifest.also_sealed:
        return
    keys = {trustee.trustee_id: trustee.public_key_hex for trustee in manifest.trustees}
    new = ceremony.message(manifest)
    for signature in manifest.signatures:
        assert verify_ed25519_signature(keys[signature.trustee_id], manifest.spec_sha256, signature.signature_hex)
        assert not verify_ed25519_signature(keys[signature.trustee_id], new, signature.signature_hex)
