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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apedp_dyn.conf.canonical_spec import governance_manifest_status  # noqa: E402


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
