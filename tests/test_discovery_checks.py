"""The discovery checks' thresholds: named, pinned, and the same everywhere they are still defined."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from apedp_dyn.conf import discovery_checks
from apedp_dyn.discovery_l2 import vouching, vouching_v2
from apedp_dyn.evolution import evolved_proposer

ROOT = Path(__file__).resolve().parents[1]


def test_the_file_is_the_one_pinned_and_names_each_threshold() -> None:
    assert discovery_checks.load()["name"] == "discovery_checks_v1"
    assert set(discovery_checks.THRESHOLDS) == {"DISCOVERY_ACCURACY_R2", "LAST_CHECKS_TEST_LEVEL", "LAST_CHECKS_MIN_POWER"}
    assert all(len(text) > 40 for text in discovery_checks.MEANING.values())


def test_a_changed_file_fails_loudly(tmp_path, monkeypatch) -> None:
    copy = tmp_path / "discovery_checks_v1.json"
    shutil.copy(discovery_checks.SPEC_PATH, copy)
    copy.write_text(copy.read_text().replace('"value": 0.999', '"value": 0.99'))
    monkeypatch.setattr(discovery_checks, "SPEC_PATH", copy)
    with pytest.raises(discovery_checks.DiscoveryChecksIntegrityError):
        discovery_checks.load()


def test_every_older_definition_still_equals_the_named_value() -> None:
    """Frozen modules keep their own definitions; they must not drift from the names."""
    assert vouching.ACCURACY_R2 == vouching_v2.ACCURACY_R2 == discovery_checks.DISCOVERY_ACCURACY_R2
    assert vouching.LEVEL == vouching_v2.LEVEL == discovery_checks.LAST_CHECKS_TEST_LEVEL
    assert vouching.POWER == vouching_v2.POWER == discovery_checks.LAST_CHECKS_MIN_POWER
    assert evolved_proposer.ACCURATE_R2 == discovery_checks.DISCOVERY_ACCURACY_R2
    baseline = re.search(r"^R2_ACCURACY = ([0-9.]+)$", (ROOT / "scripts" / "run_srbench_baseline.py").read_text(), re.M)
    assert baseline is not None and float(baseline.group(1)) == discovery_checks.DISCOVERY_ACCURACY_R2


def test_it_says_whether_the_trustees_have_signed_it() -> None:
    manifest = (ROOT / "specifications" / "governance_manifest.json").read_text()
    assert discovery_checks.SIGNED_BY_TRUSTEES == ("discovery_checks_v1.json" in manifest)
