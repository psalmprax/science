"""Phase 0 pins: the inventory's claims are checkable, and the DQI baseline is frozen.

The inventory is only useful if it cannot drift into fiction. These tests check
three things:

* every repository path it names exists (a capability cannot be "reused" if the
  file it points at was renamed or never existed);
* every VQC module is listed in it (a new module cannot appear without the
  inventory being updated);
* the DQI evidence it claims not to disturb is byte-identical to the frozen
  hashes -- the "do not change the DQI baseline" rule, enforced by arithmetic
  rather than by intention.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "VQC_PHASE0_CAPABILITY_INVENTORY.md"
BASELINE = ROOT / "benchmarks" / "quantum_compilation" / "vqc_phase0_baseline.json"
VQC_PACKAGE = ROOT / "apedp_dyn" / "vqc"

_REPO_PATH = re.compile(r"`((?:apedp_dyn|apedp|scripts|benchmarks|docs|tests)/[A-Za-z0-9_./-]+\.(?:py|json|md))`")


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


def test_every_path_the_inventory_names_exists(text):
    named = sorted(set(_REPO_PATH.findall(text)))
    assert len(named) >= 15, f"the inventory should name real artifacts, found {len(named)}"
    missing = [path for path in named if not (ROOT / path).exists()]
    assert not missing, "inventory names paths that do not exist:\n" + "\n".join(missing)


def test_every_vqc_module_is_inventoried(text):
    modules = sorted(
        str(path.relative_to(ROOT)) for path in VQC_PACKAGE.glob("*.py")
        if not path.name.startswith("__")
    )
    assert modules, "the VQC package should have modules"
    missing = [module for module in modules if module not in text]
    assert not missing, "modules absent from the Phase 0 inventory:\n" + "\n".join(missing)


def test_the_inventory_covers_the_seven_capability_areas_and_the_decisions(text):
    lowered = text.lower()
    for area in ("representation", "simulation", "noise", "logical qubit",
                 "discovery", "optimis", "verification"):
        assert area in lowered, f"inventory does not discuss {area!r}"

    for decision in ("**reuse**", "**adapt**", "**build**", "**retain**",
                     "**not connected**"):
        assert decision in text, f"inventory does not define {decision}"

    for absent in ("stabilizer", "surface code", "mwpm", "qasm 3"):
        assert absent in lowered, f"inventory should name the missing capability {absent!r}"


def test_the_dqi_baseline_is_frozen_and_unchanged():
    baseline = json.loads(BASELINE.read_text())
    records = baseline["records"]
    assert len(records) >= 10
    for path, expected in records.items():
        target = ROOT / path
        assert target.exists(), f"frozen record disappeared: {path}"
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected, (
            f"{path} changed after the Phase 0 baseline was frozen; if the change is "
            f"intended, the baseline must be re-frozen deliberately, not silently")


def test_the_frozen_submission_record_still_carries_the_measured_result():
    """A hash proves the bytes; this proves they still mean what the inventory says."""
    record = json.loads(
        (ROOT / "benchmarks/quantum_compilation/dqi_submission_score.json").read_text())
    assert record["verified"]["reference"]["status"] == "PASS"
    assert record["verified"]["candidate"]["status"] == "PASS"
    assert record["score_ratio"]["score"] == pytest.approx(0.9490542129538326, rel=1e-12)
    assert record["failed_checks"] == []


def test_the_baseline_says_what_it_is_for():
    baseline = json.loads(BASELINE.read_text())
    assert baseline["artifact"] == "vqc_phase0_baseline"
    assert "must be byte-identical" in baseline["purpose"]
    assert baseline["frozen_on"] == "2026-10-30"
