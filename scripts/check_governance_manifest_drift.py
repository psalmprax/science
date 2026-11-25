#!/usr/bin/env python3
"""Fail when the signed governance manifest lags the canonical-spec pin, or does not seal the discovery checks.

The import-time hash check in ``apedp_referee/canonical_spec.py`` proves the
disk spec matches its Python pin. It deliberately does not prove that the
Ed25519 governance manifest seals the same revision; when the two disagree the
Python gate is green while the signed certification is stale (FLAW-01). This
script is the CI gate for that second question.

Since 2026-11-25, by the owner's decision, the manifest must also list the
second specification (``specifications/discovery_checks_v1.json``, the
thresholds that decide whether a found law is reported) with the digest pinned
in ``apedp_dyn/conf/discovery_checks.py``. Both are compared here; the
signatures themselves are verified by ``tests/test_dyn_governance.py``.

Exit codes:
    0  the manifest seals both pinned revisions (or ``--check`` was not given)
    1  drift, or the manifest is missing/unreadable
    2  usage error
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
_CANONICAL_SPEC = ROOT / "apedp_referee" / "canonical_spec.py"
_DISCOVERY_CHECKS = ROOT / "apedp_dyn" / "conf" / "discovery_checks.py"


def _load_governance_manifest_status() -> Callable[[], dict[str, Any]]:
    """Load ``canonical_spec`` by file path, without importing any package.

    ``canonical_spec`` itself needs only the standard library, but importing
    it through ``apedp_dyn.conf.canonical_spec`` once ran
    ``apedp_dyn/__init__.py``, which imported SymPy and the rest of the
    package. The CI job that runs this script installs nothing, so that import
    crashed with ``ModuleNotFoundError`` and the job failed without ever
    comparing the two hashes. Loading the one file keeps the gate
    dependency-free and makes its failure mean what it says. The file is
    ``apedp_referee/canonical_spec.py``; it reads the specification copy beside
    it, which ``tests/test_referee_core_boundary.py`` keeps identical to
    ``specifications/canonical_preregistered_spec_v1.json``.
    """
    spec = importlib.util.spec_from_file_location("_apedp_canonical_spec", _CANONICAL_SPEC)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {_CANONICAL_SPEC}")
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        # The module warns on drift at import; this script reports it itself.
        warnings.simplefilter("ignore", RuntimeWarning)
        spec.loader.exec_module(module)
    status_fn: Callable[[], dict[str, Any]] = module.governance_manifest_status
    return status_fn


governance_manifest_status = _load_governance_manifest_status()


def discovery_checks_status() -> dict[str, Any]:
    """Whether the manifest lists the discovery checks' file with its pinned digest.

    ``discovery_checks.py`` is loaded by file path for the same reason as above: it needs only the
    standard library, and this job installs nothing.
    """
    spec = importlib.util.spec_from_file_location("_apedp_discovery_checks", _DISCOVERY_CHECKS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {_DISCOVERY_CHECKS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    status: dict[str, Any] = module.manifest_status()
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when the manifest does not seal the pinned revision",
    )
    args = parser.parse_args(argv)

    status = governance_manifest_status()
    print(f"manifest: {status['manifest_path']}")
    print(f"sealed:   {status['sealed_sha256']}")
    print(f"disk:     {status['disk_sha256']}")

    if not status["manifest_present"]:
        print(
            "GOVERNANCE DRIFT: the governance manifest is missing or unreadable",
            file=sys.stderr,
        )
        return 1
    if status["drift"]:
        _second_file_sealed()
        print(
            "GOVERNANCE DRIFT: the signed manifest does not seal this spec "
            "revision; re-signing requires the registered trustees "
            "(AGENTS.md §1.3, FLAW-01)",
            file=sys.stderr,
        )
        return 1 if args.check else 0
    print("governance manifest covers the canonical-spec pin")
    return 0 if _second_file_sealed() else (1 if args.check else 0)


def _second_file_sealed() -> bool:
    second = discovery_checks_status()
    print(f"discovery checks, sealed: {second['sealed_sha256']}")
    print(f"discovery checks, disk:   {second['disk_sha256']}")
    if second["listed"] and not second["drift"]:
        print("governance manifest covers the discovery-checks pin")
        return True
    print("GOVERNANCE DRIFT: the signed manifest does not seal the discovery checks' thresholds "
          "(specifications/discovery_checks_v1.json); the trustees sign one message for both files "
          "(docs/GOVERNANCE_RESIGNING_REQUEST.md, section 8)", file=sys.stderr)
    return False


if __name__ == "__main__":
    raise SystemExit(main())
