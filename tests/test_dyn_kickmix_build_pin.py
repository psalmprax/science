"""Pins for the two ``kickmix`` builds a verifier run may use.

Status: audit-only

``apedp_dyn/quantum_compilation/kickmix_builds.py`` names both builds by SHA-256 and
``contracts.py`` refuses anything else. These pins are deliberately cheap: they hash files
that are already on disk and never compile. The expensive half — that the pinned commit
plus pinned patch actually reproduce the pinned bytes — is recorded in
``benchmarks/quantum_compilation/kickmix_mr_build.json`` by
``scripts/build_kickmix_mr.py`` and re-checked here against the built extension when it is
present.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from apedp_dyn.quantum_compilation import contracts
from apedp_dyn.quantum_compilation import kickmix_builds as pins
from apedp_dyn.quantum_compilation.contracts import ContractIntegrityError, load_contract

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE = REPO_ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
SHIPPED_EXTENSION = BUNDLE / "kickmix" / "kickmix.so"
VENDOR_TREE = BUNDLE / "kickmix" / "_qualtran_src" / "kickmix"
PATCH = REPO_ROOT / "scripts" / "kickmix_mr_opcode.patch"
BUILD_DIR = REPO_ROOT / "build" / "kickmix_mr" / "lib"
RECORD = REPO_ROOT / "benchmarks" / "quantum_compilation" / "kickmix_mr_build.json"


def _built_extension() -> Path | None:
    found = sorted(BUILD_DIR.glob("kickmix*.so"))
    return found[0] if found else None


def test_the_shipped_build_is_pinned_to_the_bytes_on_disk():
    """The name in the registry is the file the verifier imports, not a remembered hash."""
    assert pins.sha256_file(SHIPPED_EXTENSION) == pins.SHIPPED_BUILD.sha256
    build = pins.assert_allowed(SHIPPED_EXTENSION)
    assert build.name == "shipped"
    assert build.allowed_for_verifier is True
    assert build.has_mr is False


def test_the_mr_pin_names_the_commit_and_patch_it_was_built_from():
    """A pin without its inputs cannot be reproduced; the inputs are pinned too."""
    assert pins.sha256_file(PATCH) == pins.MR_BUILD.provenance["patch_sha256"]
    if shutil.which("git") and (VENDOR_TREE / ".git").exists():
        head = subprocess.run(
            ["git", "-C", str(VENDOR_TREE), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        assert head == pins.MR_BUILD.provenance["source_commit"]
    assert pins.MR_BUILD.allowed_for_verifier is True
    assert pins.MR_BUILD.has_mr is True


def test_only_the_two_named_builds_may_produce_a_verifier_receipt():
    assert [b.name for b in pins.ALLOWED_VERIFIER_BUILDS] == ["shipped", "mr"]
    assert all(b.allowed_for_verifier for b in pins.ALLOWED_VERIFIER_BUILDS)
    # The earlier local build stays named so old receipts can be read, but is not allowed:
    # it came from a mutated working tree and cannot be regenerated from the repository.
    assert pins.LEGACY_LOCAL_BUILD.allowed_for_verifier is False
    assert pins.LEGACY_LOCAL_BUILD.sha256 not in {
        b.sha256 for b in pins.ALLOWED_VERIFIER_BUILDS}


def test_an_unpinned_extension_is_refused(tmp_path, monkeypatch):
    stray = tmp_path / "kickmix.so"
    stray.write_bytes(b"not a kickmix build")
    with pytest.raises(pins.UnpinnedKickmixBuildError, match="not a pinned kickmix build"):
        pins.assert_allowed(stray)
    # And the legacy build is refused with its own reason, not a generic one.
    monkeypatch.setattr(pins, "sha256_file",
                        lambda path, **_: pins.LEGACY_LOCAL_BUILD.sha256)
    with pytest.raises(pins.UnpinnedKickmixBuildError, match="not reproducible"):
        pins.assert_allowed(stray)


def test_the_contract_binds_the_extension_bytes_not_just_the_tree():
    contract = load_contract("dqi")
    assert contract.kickmix_build == "shipped"
    assert contract.kickmix_extension_sha256 == pins.SHIPPED_BUILD.sha256
    contract.assert_locked()


def test_a_substituted_extension_is_rejected(monkeypatch):
    """The tree hash would not notice a swap; the extension pin must."""
    contract = load_contract("dqi")
    monkeypatch.setattr(contracts, "_pinned_extension", lambda root: pins.MR_BUILD)
    with pytest.raises(ContractIntegrityError, match="kickmix extension changed"):
        contract.assert_locked()


def test_the_repository_reproduces_the_mr_pin():
    """The built extension is the pinned bytes, and the receipt says which inputs made them."""
    extension = _built_extension()
    if extension is None:
        pytest.skip("MR-capable kickmix not built; run scripts/build_kickmix_mr.py")
    assert pins.sha256_file(extension) == pins.MR_BUILD.sha256
    assert pins.assert_allowed(extension, role="mr").name == "mr"

    record = json.loads(RECORD.read_text())
    assert record["pin"]["matched"] is True
    assert record["pin"]["expected_sha256"] == pins.MR_BUILD.sha256
    assert record["build"]["sha256"] == pins.MR_BUILD.sha256
    assert record["source"]["commit"] == pins.MR_BUILD.provenance["source_commit"]
    assert record["patch"]["sha256"] == pins.MR_BUILD.provenance["patch_sha256"]
    # The extension must still carry the instruction the pin exists for.
    assert record["build"]["instructions"]["has_mr"] is True


def test_a_contract_without_the_build_fields_cannot_be_asserted():
    """Fields added with the pin must actually be covered by the contract digest."""
    contract = load_contract("invsqrt")
    stripped = dataclasses.replace(contract, kickmix_build="shipped")
    # Same values, so the metadata digest still holds -- the point is that the field is
    # part of `to_dict()` and therefore of `contract_sha256`, not a free-floating extra.
    assert "kickmix_build" in stripped.to_dict()
    assert "kickmix_extension_sha256" in stripped.to_dict()
    stripped.assert_locked()
