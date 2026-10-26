from __future__ import annotations

import importlib.util
import hashlib
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_official_receipts.py"
spec = importlib.util.spec_from_file_location("receipt_runner", SCRIPT)
rr = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(rr)


def test_unconfigured_templates_are_rejected_recursively():
    found = rr.find_placeholders({"root": "/ABSOLUTE/PATH/TO/science", "cmd": ["python", "<SET_BUILDER>"]})
    assert len(found) == 2
    assert "root" in found[0]


def test_protected_input_hash_pins_exact_file(tmp_path):
    repo = tmp_path / "repo"
    comp = tmp_path / "competition"
    repo.mkdir(); comp.mkdir()
    f = repo / "submission.npz"
    f.write_bytes(b"frozen-input")
    digest = hashlib.sha256(b"frozen-input").hexdigest()
    rows = rr.verify_protected_files(repo, [{"root": "repo", "path": "submission.npz", "sha256": digest}], comp)
    assert rows[0]["ok"] is True
    f.write_bytes(b"changed-input")
    rows = rr.verify_protected_files(repo, [{"root": "repo", "path": "submission.npz", "sha256": digest}], comp)
    assert rows[0]["ok"] is False


def test_protected_files_can_live_under_competition_root(tmp_path):
    repo = tmp_path / "repo"; comp = tmp_path / "competition"
    repo.mkdir(); comp.mkdir()
    f = comp / "verifiers" / "verify.py"
    f.parent.mkdir(); f.write_text("# pinned official source\n")
    digest = hashlib.sha256(f.read_bytes()).hexdigest()
    rows = rr.verify_protected_files(repo, [{"root": "competition", "path": "verifiers/verify.py", "sha256": digest}], comp)
    assert rows[0]["ok"] is True


def test_source_files_are_snapshotted_and_optional_pin_is_enforced(tmp_path):
    repo = tmp_path / "repo"; comp = tmp_path / "competition"
    repo.mkdir(); comp.mkdir()
    f = comp / "verify.py"; f.write_text("official verifier version A\n")
    digest = hashlib.sha256(f.read_bytes()).hexdigest()
    items = [{"root": "competition", "path": "verify.py", "sha256": digest}]
    before = rr.snapshot_source_files(repo, comp, items)
    assert before[0]["ok"] is True
    f.write_text("official verifier version B\n")
    after = rr.snapshot_source_files(repo, comp, items)
    assert after[0]["ok"] is False
    assert before[0]["sha256"] != hashlib.sha256(f.read_bytes()).hexdigest()


def test_command_preflight_checks_executable_not_just_string_presence():
    valid = {"components": [{"name": "test", "candidate_builder_command": [sys.executable, "build.py"], "official_verifier_command": [sys.executable, "verify.py"]}]}
    invalid = {"components": [{"name": "test", "candidate_builder_command": ["/missing/python"], "official_verifier_command": [sys.executable, "verify.py"]}]}
    assert rr.check_command_executables(valid) == []
    assert any("does not exist" in e for e in rr.check_command_executables(invalid))
