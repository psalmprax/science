"""The discovery checks' thresholds, named by what they mean and pinned by hash.

Status: experimental

``specifications/discovery_checks_v1.json`` holds six thresholds:

* ``DISCOVERY_ACCURACY_R2``, the line that defines a wrong law in every
  discovery benchmark;
* ``LAST_CHECKS_TEST_LEVEL``, the level at which the last checks refuse;
* ``LAST_CHECKS_MIN_POWER``, the power they need before they vouch;
* ``REPORTED_WRONG_SHARE_MAX``, the largest share of wrong laws among those a
  method reports in a test;
* ``REPORTED_WRONG_SHARE_BOUND_CONFIDENCE`` and
  ``REPORTED_WRONG_SHARE_BOUND_MAX``: the upper bound of that share, at that
  confidence, must be at most the second before the share counts as shown.

Until 2026-11-23 the first was a bare literal in three files. The next two
were borrowed from canonical-spec entries for unrelated tests: the
Diebold-Mariano level and the temporal-rate calibration power. Their values
are unchanged.

The last three were added on 2026-11-25 by the owner's decision. Five plans up
to PREREG-REAL-TABLES-2026-11 had borrowed the referee's allowed share of false
positives (``REFEREE_MAX_FPR``, the same 5 %) for the share of wrong reports,
and none asked anything of its upper bound: 2 wrong of 44 passed with a bound
of 13.6 %. Plans frozen from that date apply both conditions
(``apedp_dyn/discovery_l2/report_share.py``); recorded results are not judged
again.

This module checks the file's SHA-256 on import, as ``canonical_spec`` does
for the canonical specification. A changed file fails loudly. Modules frozen
by a preregistration keep their own definitions; ``tests/test_discovery_checks.py``
checks that each still equals the value named here. New code imports from here.

**Not yet binding the way the canonical specification is.** Until the
trustees sign a governance manifest that lists this file with the digest pinned
here (``also_sealed`` in ``specifications/governance_manifest.json``; the
owner decided on 2026-11-25 that they should), the pin below is only as good as
the repository it lives in. ``manifest_status`` says which it is;
``SIGNED_BY_TRUSTEES`` is true only when the manifest lists this file with this
digest. Whether the signatures on that manifest are valid is checked where the
manifest is verified (``apedp_dyn/epistemic/governance.py``), not here: this
module needs only the standard library.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SPEC_PATH = Path(__file__).resolve().parents[2] / "specifications" / "discovery_checks_v1.json"
MANIFEST_PATH = SPEC_PATH.parent / "governance_manifest.json"
SPEC_ID = "discovery_checks_v1"
# 2026-11-23: 81ae13eb0dd9bd061f246026178d405a4fd7f41a825a7abfa7d90085796232c0 (three thresholds)
# 2026-11-25: the three REPORTED_WRONG_SHARE thresholds added, owner's decision (CHANGELOG.md)
PINNED_SHA256 = "60eabf58a23c9f2d366d45875e427999bb34c4086cb74107caf9dabd7bb220d0"


def manifest_status() -> dict[str, Any]:
    """Whether the governance manifest lists this file, and with which digest."""
    sealed: str | None = None
    try:
        payload = json.loads(MANIFEST_PATH.read_bytes().decode("utf-8"))
        entries = payload.get("also_sealed") if isinstance(payload, dict) else None
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict) and entry.get("file") == SPEC_PATH.name and isinstance(entry.get("sha256"), str):
                sealed = entry["sha256"]
    except (OSError, ValueError):
        sealed = None
    return {"manifest_path": str(MANIFEST_PATH), "listed": sealed is not None, "sealed_sha256": sealed,
            "disk_sha256": PINNED_SHA256, "drift": sealed is not None and sealed != PINNED_SHA256}


MANIFEST_STATUS: dict[str, Any] = manifest_status()
SIGNED_BY_TRUSTEES: bool = bool(MANIFEST_STATUS["listed"] and not MANIFEST_STATUS["drift"])


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
REPORTED_WRONG_SHARE_MAX = THRESHOLDS["REPORTED_WRONG_SHARE_MAX"]
REPORTED_WRONG_SHARE_BOUND_CONFIDENCE = THRESHOLDS["REPORTED_WRONG_SHARE_BOUND_CONFIDENCE"]
REPORTED_WRONG_SHARE_BOUND_MAX = THRESHOLDS["REPORTED_WRONG_SHARE_BOUND_MAX"]
