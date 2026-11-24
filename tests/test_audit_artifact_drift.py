"""Every generated audit artefact must be reproducible from the tree.

The pins in other test files prove the same thing for the artefacts they own,
but each only runs when that file is selected. This one is the gate: it rebuilds
all of them through ``scripts/check_audit_artifact_drift.py``, which CI runs on
its own. Two tests below exist because a drift gate that cannot report drift is
worse than none -- one injects a stale artefact, the other a failing command.
"""
from __future__ import annotations

import json
import sys

from scripts.check_audit_artifact_drift import REBUILT, drift


def test_every_generated_artefact_matches_its_generator():
    problems = drift()
    assert problems == [], "\n".join(problems)


def test_a_hand_edited_field_is_reported_by_name():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    relative = "benchmarks/formal/lean_axiom_audit.json"
    recorded = json.loads((root / relative).read_text(encoding="utf-8"))

    def stale() -> dict:
        return {**recorded, "totals": {**recorded["totals"], "files": -1}}

    problems = drift(rebuilt={relative: stale}, commands={})
    assert len(problems) == 1, problems
    assert "totals.files" in problems[0], problems[0]
    assert "@" not in problems[0]


def test_a_missing_artefact_is_named_rather_than_skipped():
    problems = drift(rebuilt={"benchmarks/absent.json": lambda: {"a": 1}}, commands={})
    assert problems == ["benchmarks/absent.json: missing"]


def test_a_generator_that_raises_is_a_stale_artefact_not_a_crash():
    def broken() -> dict:
        raise RuntimeError("no tree")

    problems = drift(rebuilt={"benchmarks/formal/lean_axiom_audit.json": broken}, commands={})
    assert len(problems) == 1
    assert "RuntimeError: no tree" in problems[0]


def test_a_failing_check_command_is_reported_with_its_exit_code():
    problems = drift(
        rebuilt={},
        commands={"scripts/static_table_census.json": [sys.executable, "-c", "raise SystemExit(3)"]},
    )
    assert len(problems) == 1, problems
    assert "exited 3" in problems[0], problems[0]


def test_the_registry_covers_the_reports_the_repository_generates():
    """A new generated report must be registered here, not just committed."""
    names = set(REBUILT)
    assert "benchmarks/formal/lean_axiom_audit.json" in names
    assert "benchmarks/applied_basis_report.json" in names
    assert "benchmarks/untested_modules_register.json" in names
