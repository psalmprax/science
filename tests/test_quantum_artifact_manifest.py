"""The artifact manifest must describe the bytes, not wish them into being.

The verified quantum circuits are compiled ``.kmb`` binaries and are never
committed: the dqi reference decoder is 886 MB and ``.gitignore`` excludes both
``artifacts/`` and ``sample_submission/dqi.kmb``. So a checkout has the code and
the record, and may have none of the bytes.

That record is the point. Without it, a missing artifact manifests as an empty
proposal list, which reads exactly like a healthy run with nothing to propose --
which is how `test_dqi_and_dfthcsos_proposers_screen_and_measure` failed with a
bare `0 == 1` and no hint that a file was absent.

These tests hold the manifest to what it claims: schema, a stated storage policy,
a hash for every row (including rows whose bytes are gone), and a hash match for
every row whose bytes are present.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "verify_quantum_artifacts.py"
MANIFEST = ROOT / "config" / "quantum_artifact_manifest.json"

#: The sha256 of the dqi reference decoder that passed the official verifier at
#: 102,400 shots (1,893 qubits, 5,725,461 Toffoli-like). Pinned so the circuit
#: cannot be silently swapped for a different one.
DQI_DECODER_SHA256 = "959471d05a26373b4f14ca0cae2d8639ea25f18170d9f6532139aa51ccc04995"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_quantum_artifacts", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify = _load_verifier()


def _manifest() -> dict:
    assert MANIFEST.is_file(), (
        f"{MANIFEST.relative_to(ROOT)} is missing; regenerate it with "
        "`python scripts/verify_quantum_artifacts.py --write`"
    )
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_declares_its_schema_and_policy():
    payload = _manifest()
    assert payload["schema"] == verify.SCHEMA
    policy = payload["policy"]
    assert policy["committed"] is False, "large build products must not be committed"
    assert policy["rationale"], "the policy needs a reason, not just a rule"
    assert policy["cache"], "the manifest must name where the bytes do live"


def test_every_row_is_identifiable():
    """A row without a role or a benchmark cannot be acted on."""
    for row in _manifest()["artifacts"]:
        assert row["path"] and row["benchmark"] and row["role"], row
        assert row["sha256"], (
            f"{row['path']} records no sha256; an absent artifact without a hash "
            "cannot be found in the cache or proven to be the right one"
        )


def test_present_artifacts_match_their_recorded_hash():
    """The check that catches a truncated copy, a partial download or a wrong file."""
    assert verify.check_manifest() == 0


def test_the_present_dqi_circuit_is_the_verified_decoder():
    """Pin the identity of the one circuit that is actually on disk here."""
    dqi = next(row for row in _manifest()["artifacts"] if row["benchmark"] == "dqi"
               and row["path"].endswith("sample_submission/dqi.kmb"))
    if not (ROOT / dqi["path"]).is_file():
        pytest.skip(f"{dqi['path']} is absent; nothing to identify")
    assert dqi["sha256"] == DQI_DECODER_SHA256
    assert verify.sha256_file(ROOT / dqi["path"]) == DQI_DECODER_SHA256


def test_check_tolerates_absence_but_not_corruption(tmp_path, monkeypatch):
    """A clean checkout has no artifacts; that must not read as a failure."""
    assert verify.check_manifest() == 0, "absent rows must not fail the check"

    # Point the verifier at a manifest whose recorded hash is wrong for a file
    # that is present, and confirm it is reported as corruption.
    fake_root = tmp_path
    (fake_root / "config").mkdir()
    (fake_root / "blob.kmb").write_bytes(b"not the circuit\n")
    manifest = {
        "schema": verify.SCHEMA,
        "policy": {"committed": False, "cache": str(tmp_path / "cache"), "rationale": "test"},
        "artifacts": [
            {"path": "blob.kmb", "benchmark": "dqi", "role": "test", "present": True,
             "bytes": 17, "sha256": "0" * 64},
        ],
    }
    (fake_root / "config" / "quantum_artifact_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(verify, "ROOT", fake_root)
    monkeypatch.setattr(verify, "MANIFEST_PATH", fake_root / "config" / "quantum_artifact_manifest.json")
    assert verify.check_manifest() == 1
