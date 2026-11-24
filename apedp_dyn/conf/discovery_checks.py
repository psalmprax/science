"""The discovery checks' thresholds, named by what they mean and pinned by hash.

Status: experimental

``specifications/discovery_checks_v1.json`` holds three thresholds:

* ``DISCOVERY_ACCURACY_R2``, the line that defines a wrong law in every
  discovery benchmark;
* ``LAST_CHECKS_TEST_LEVEL``, the level at which the last checks refuse;
* ``LAST_CHECKS_MIN_POWER``, the power they need before they vouch.

Until 2026-11-23 the first was a bare literal in three files. The other two
were borrowed from canonical-spec entries for unrelated tests: the
Diebold-Mariano level and the temporal-rate calibration power. Their values
are unchanged.

This module checks the file's SHA-256 on import, as ``canonical_spec`` does
for the canonical specification. A changed file fails loudly. Modules frozen
by a preregistration keep their own definitions; ``tests/test_discovery_checks.py``
checks that each still equals the value named here. New code imports from here.

**Not yet binding the way the canonical specification is.** The signed
governance manifest does not cover this file. Until the trustees sign a
manifest that does, the pin below is only as good as the repository it lives in.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SPEC_PATH = Path(__file__).resolve().parents[2] / "specifications" / "discovery_checks_v1.json"
PINNED_SHA256 = "81ae13eb0dd9bd061f246026178d405a4fd7f41a825a7abfa7d90085796232c0"
SIGNED_BY_TRUSTEES = False


class DiscoveryChecksIntegrityError(RuntimeError):
    """The thresholds file is not the one pinned here."""


def load() -> dict[str, Any]:
    """The file's contents, after its hash is checked."""
    content = SPEC_PATH.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != PINNED_SHA256:
        raise DiscoveryChecksIntegrityError(
            f"{SPEC_PATH.name} has SHA-256 {digest}, pinned {PINNED_SHA256}: a threshold changes only with a new "
            "pin, a CHANGELOG entry and the owner's word (AGENTS.md 1.3 and 1.5)")
    loaded: dict[str, Any] = json.loads(content)
    return loaded


_SPEC = load()
THRESHOLDS: dict[str, float] = {name: float(entry["value"]) for name, entry in _SPEC["thresholds"].items()}
MEANING: dict[str, str] = {name: str(entry["meaning"]) for name, entry in _SPEC["thresholds"].items()}
DISCOVERY_ACCURACY_R2 = THRESHOLDS["DISCOVERY_ACCURACY_R2"]
LAST_CHECKS_TEST_LEVEL = THRESHOLDS["LAST_CHECKS_TEST_LEVEL"]
LAST_CHECKS_MIN_POWER = THRESHOLDS["LAST_CHECKS_MIN_POWER"]
