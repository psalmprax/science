"""Pins for the correctness-check audit over the DQI records.

Experiment D called a candidate correct on its syndromes alone; Experiment D2
measured the phase and found 37 of 64 shots wrong. These pins keep that shape
visible: they re-derive the audit from the records on disk and check that the
recorded counts still match, that the historical gap is still named, and that the
D2 record itself is fully checked.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "audit_record_correctness_checks.py"
RECORD = ROOT / "benchmarks" / "quantum_compilation" / "correctness_audit.json"
RECORDS = ROOT / "benchmarks" / "quantum_compilation"

#: Totals when the audit was recorded. A new record, or a record that starts storing
#: the phase it was missing, moves these — deliberately, and visibly.
TOTALS = {
    "PARTIAL_EVIDENCE_MISSING_JUNK_AND_PHASE": 7,
    "PARTIAL_EVIDENCE_MISSING_PHASE": 6,
    "checked": 12,
    "claims_correct_aggregate_only": 51,
    "resources-only": 5,
}

#: Rows whose correctness claim rests on partial evidence.
PARTIAL_ROWS = 13


def _load():
    spec = importlib.util.spec_from_file_location("correctness_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text())


def _totals() -> dict:
    module = _load()
    totals: dict[str, int] = {}
    for path in sorted(RECORDS.glob("*.json")):
        if path.name == RECORD.name:
            continue
        entry = module.audit_record(path)
        for status, count in entry["counts"].items():
            totals[status] = totals.get(status, 0) + count
    return totals


def test_the_recorded_totals_still_match_the_records_on_disk(record):
    live = _totals()
    assert live == record["totals"], (live, record["totals"])
    assert live == TOTALS


def test_the_historical_gap_is_still_named(record):
    gaps = {(gap["record"].split("/")[-1], gap["path"]) for gap in record["gaps"]}
    assert len(gaps) == PARTIAL_ROWS
    # Experiment D's candidate: syndromes recorded, phase not, called correct for
    # four hours of this project's history.
    key = ("dqi_official_compatibility.json", "builds.pinned.candidate")
    assert key in gaps
    row = next(gap for gap in record["gaps"]
               if gap["path"] == "builds.pinned.candidate")
    assert row["missing"] == ["junk", "phase"]
    assert row["toffoli"] == pytest.approx(5_433_399.609375, abs=1e-6)


def test_the_d2_record_has_no_partial_rows(record):
    d2 = next(entry for entry in record["per_record"]
              if entry["record"].endswith("dqi_official_candidate.json"))
    assert all(row["status"] == "checked" for row in d2["rows"]), \
        [row["path"] for row in d2["rows"] if row["status"] != "checked"]


def test_no_partial_row_in_the_audit_is_a_phase_safe_candidate(record):
    """The gap is in the *evidence*, not in the claim: the names are recorded."""
    for gap in record["gaps"]:
        assert gap["missing"], gap
        assert gap["record"].startswith("benchmarks/quantum_compilation/")


def test_the_audit_pins_its_script_and_scans_every_record(record):
    assert record["reproducibility"]["script_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()).hexdigest()
    scanned = {entry["record"] for entry in record["per_record"]}
    on_disk = {f"benchmarks/quantum_compilation/{path.name}"
               for path in RECORDS.glob("*.json") if path.name != RECORD.name}
    assert scanned == on_disk
