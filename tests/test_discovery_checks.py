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
    assert set(discovery_checks.THRESHOLDS) == {
        "DISCOVERY_ACCURACY_R2", "LAST_CHECKS_TEST_LEVEL", "LAST_CHECKS_MIN_POWER",
        "REPORTED_WRONG_SHARE_MAX", "REPORTED_WRONG_SHARE_BOUND_CONFIDENCE", "REPORTED_WRONG_SHARE_BOUND_MAX"}
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


def test_the_named_share_of_wrong_reports_is_the_value_the_earlier_plans_borrowed() -> None:
    """Naming it changed no number: five plans used the referee's allowed share of false positives for it."""
    from apedp_dyn.conf import canonical_spec

    assert discovery_checks.REPORTED_WRONG_SHARE_MAX == canonical_spec.REFEREE_MAX_FPR == 0.05
    assert discovery_checks.REPORTED_WRONG_SHARE_BOUND_CONFIDENCE == 0.95
    assert discovery_checks.REPORTED_WRONG_SHARE_BOUND_MAX == 0.1


def test_it_says_whether_the_trustees_have_signed_it(tmp_path, monkeypatch) -> None:
    """True only when the governance manifest lists this file with the digest pinned here."""
    import json

    manifest = json.loads((ROOT / "specifications" / "governance_manifest.json").read_text())
    listed = [e for e in manifest.get("also_sealed", []) if e.get("file") == "discovery_checks_v1.json"]
    status = discovery_checks.manifest_status()
    assert status["listed"] == bool(listed)
    assert discovery_checks.SIGNED_BY_TRUSTEES == (bool(listed) and listed[0]["sha256"] == discovery_checks.PINNED_SHA256)

    def with_manifest(payload: dict) -> dict:
        path = tmp_path / "governance_manifest.json"
        path.write_text(json.dumps(payload))
        monkeypatch.setattr(discovery_checks, "MANIFEST_PATH", path)
        return discovery_checks.manifest_status()

    entry = {"spec_id": "discovery_checks_v1", "file": "discovery_checks_v1.json", "sha256": discovery_checks.PINNED_SHA256}
    assert with_manifest({"spec_sha256": "0" * 64}) == {**status, "manifest_path": str(tmp_path / "governance_manifest.json"),
                                                        "listed": False, "sealed_sha256": None, "drift": False}
    sealed = with_manifest({"also_sealed": [entry]})
    assert sealed["listed"] and not sealed["drift"]
    stale = with_manifest({"also_sealed": [{**entry, "sha256": "1" * 64}]})
    assert stale["listed"] and stale["drift"]                # listed with an older digest: not signed for this file
    monkeypatch.setattr(discovery_checks, "MANIFEST_PATH", tmp_path / "missing.json")
    assert discovery_checks.manifest_status()["listed"] is False
