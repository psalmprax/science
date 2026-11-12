#!/usr/bin/env python3
"""Fail when the signed governance manifest lags the canonical-spec pin.

The import-time hash check in ``apedp_dyn/conf/canonical_spec.py`` proves the
disk spec matches its Python pin. It deliberately does not prove that the
Ed25519 governance manifest seals the same revision; when the two disagree the
Python gate is green while the signed certification is stale (FLAW-01). This
script is the CI gate for that second question.

Exit codes:
    0  the manifest seals the pinned revision (or ``--check`` was not given)
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
_CANONICAL_SPEC = ROOT / "apedp_dyn" / "conf" / "canonical_spec.py"


def _load_governance_manifest_status() -> Callable[[], dict[str, Any]]:
    """Load ``canonical_spec`` by file path, without importing ``apedp_dyn``.

    ``canonical_spec`` itself needs only the standard library, but
    ``from apedp_dyn.conf.canonical_spec import ...`` first runs
    ``apedp_dyn/__init__.py``, which imports SymPy and the rest of the package.
    The CI job that runs this script installs nothing, so that import crashed
    with ``ModuleNotFoundError`` and the job failed without ever comparing the
    two hashes. Loading the one file keeps the gate dependency-free and makes
    its failure mean what it says.
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
        print(
            "GOVERNANCE DRIFT: the signed manifest does not seal this spec "
            "revision; re-signing requires the registered trustees "
            "(AGENTS.md §1.3, FLAW-01)",
            file=sys.stderr,
        )
        return 1 if args.check else 0
    print("governance manifest covers the canonical-spec pin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
