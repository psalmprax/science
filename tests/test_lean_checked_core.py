"""The machine-checked Lean core stays machine-checkable.

``lean_checked/`` is the small Lake project whose 39 theorems are the
*kernel-checked* part of the formal story (the 557-file transcription tree
under ``formal/`` is prose-shaped Lean that has never been elaborated).
``scripts/check_lean_checked.py`` is the gate: a static layer that forbids
proof-faking constructs and an (optional, toolchain-dependent) kernel layer
that rebuilds the project and re-measures every theorem's axioms.

These tests pin the parts that work without a Lean toolchain:

* the static layer passes on the committed tree;
* the static layer actually detects ``sorry``/``axiom``/``native_decide``
  (a checker that cannot fail is not a check);
* every theorem declared in the sources is listed in ``Audit.lean``;
* the committed ``axiom_audit.json`` names the declared theorems exactly,
  claims only the standard axioms, and names the pinned toolchain.

When ``lake`` is on PATH (CI installs Lean 4.24.0), the kernel layer is
exercised too, via the same entry point CI runs.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_lean_checked.py"
PROJECT = ROOT / "lean_checked"
RECORD = PROJECT / "axiom_audit.json"

sys.path.insert(0, str(CHECKER.parent))

check_lean_checked = __import__("check_lean_checked")


def test_static_layer_passes_on_the_committed_tree() -> None:
    problems, declared = check_lean_checked.static_check()
    assert problems == []
    # The project is only worth gating if it holds a real body of theorems.
    assert len(declared) >= 30


def test_every_source_theorem_is_audited() -> None:
    _, declared = check_lean_checked.static_check()
    audit_text = (PROJECT / "Audit.lean").read_text(encoding="utf-8")
    audited = {m.split(".")[-1] for m in re.findall(r"#print axioms\s+([\w.']+)", audit_text)}
    for name in declared:
        assert name.split(".")[-1] in audited, f"{name} missing from Audit.lean"


def test_static_layer_detects_proof_faking(tmp_path: Path) -> None:
    """A checker that cannot fail is not a check: plant each forbidden
    construct in a scratch copy and require the static layer to flag it."""
    for i, bad_line in enumerate(
        (
            "theorem planted : 1 = 1 := by sorry",
            "axiom planted_axiom : False",
            "theorem planted : 1 = 1 := by native_decide",
        )
    ):
        scratch = tmp_path / f"planted_{i}"
        shutil.copytree(PROJECT, scratch, ignore=shutil.ignore_patterns(".lake"))
        target = scratch / "ApedpChecked" / "Dimensions.lean"
        target.write_text(
            target.read_text(encoding="utf-8") + "\n" + bad_line + "\n", encoding="utf-8"
        )
        problems, _ = check_lean_checked.static_check(project=scratch)
        assert problems, f"static layer failed to flag: {bad_line}"


def test_committed_record_matches_the_declared_theorems() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    _, declared = check_lean_checked.static_check()
    by_theorem = record["axioms_by_theorem"]
    assert record["theorems"] == len(by_theorem) == len(declared)

    def short(name: str) -> str:
        return name.split(".")[-1]

    assert {short(k) for k in by_theorem} == {short(d) for d in declared}
    assert record["lean_version"].startswith("Lean (version 4.24.0")
    standard = set(check_lean_checked.STANDARD_AXIOMS)
    assert sorted(standard) == record["standard_axioms_allowed"]
    for name, axioms in by_theorem.items():
        extra = set(axioms) - standard
        assert not extra, f"{name} claims non-standard axioms: {sorted(extra)}"
    assert set(record["axiom_free"]) <= set(by_theorem)


def test_checker_entry_point_passes_as_ci_runs_it() -> None:
    """Run the gate through its command line. Without a toolchain this is the
    static layer only (the record is compared by the test above); with one it
    is the full gate CI runs, kernel layer and record comparison included."""
    argv = [sys.executable, "-I", str(CHECKER), "--check"]
    if shutil.which("lake") and shutil.which("lean"):
        argv.append("--require-lean")
    proc = subprocess.run(argv, capture_output=True, text=True, check=False, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_lean_project_has_no_vendored_build_output() -> None:
    """.lake/ holds toolchain-specific build products; committing it would
    let a stale .olean masquerade as a fresh check."""
    tracked = subprocess.run(
        ["git", "ls-files", "lean_checked/"], capture_output=True, text=True, check=False, cwd=ROOT
    ).stdout.splitlines()
    assert tracked, "lean_checked/ must be tracked"
    assert not [p for p in tracked if "/.lake/" in p or p.endswith(".olean")]


@pytest.mark.skipif(
    shutil.which("lake") is None or shutil.which("lean") is None,
    reason="Lean toolchain not on PATH; CI runs this with Lean 4.24.0 installed",
)
def test_kernel_layer_re_measures_the_record() -> None:
    problems, axioms, version = check_lean_checked.kernel_check()
    assert problems == []
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert version == record["lean_version"]
    assert {k: sorted(v) for k, v in axioms.items()} == record["axioms_by_theorem"]
