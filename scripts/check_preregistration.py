"""Check that the preregistered files still match the digests recorded when it was frozen.

    python scripts/check_preregistration.py            # verify; exit 1 on any mismatch
    python scripts/check_preregistration.py --write    # record the digests (freezing only)

The market-data workflow runs the check before it builds any data, so a
preregistration cannot be edited after the fact without the build refusing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIGEST = ROOT / "preregistration" / "DIGEST.json"
FROZEN = (
    "preregistration/PREREG-CRYPTO-PIT-2026-11.md",
    "scripts/run_prereg_crypto_pit.py",
    "scripts/build_point_in_time_crypto_panel.py",
)


def digests(root: Path = ROOT) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FROZEN}


def check(root: Path = ROOT, digest_path: Path = DIGEST) -> list[str]:
    """Every file whose bytes differ from the recorded digest (empty if none)."""
    recorded = json.loads(digest_path.read_text(encoding="utf-8"))["sha256"]
    if set(recorded) != set(FROZEN):
        return [f"recorded files {sorted(recorded)} are not the frozen set {sorted(FROZEN)}"]
    now = digests(root)
    return [f"{name}: recorded {recorded[name]}, now {now[name]}" for name in FROZEN if now[name] != recorded[name]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        DIGEST.write_text(json.dumps({"preregistration": "PREREG-CRYPTO-PIT-2026-11", "sha256": digests()},
                                     indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"recorded {len(FROZEN)} digests in {DIGEST.relative_to(ROOT)}")
        return 0
    problems = check()
    for problem in problems:
        print(f"PREREGISTRATION CHANGED: {problem}")
    if not problems:
        print(f"preregistration intact: {len(FROZEN)} files match {DIGEST.relative_to(ROOT)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
