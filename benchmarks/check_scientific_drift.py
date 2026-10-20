"""Check benchmark corpus identity and frozen known-truth metric guardrails."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "scientific_benchmark_manifest.json"


def _metric(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        value = value[part]
    return value


def check() -> list[str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    checksums_path = ROOT / manifest["corpus"]["checksums_file"]
    actual_hash = hashlib.sha256(checksums_path.read_bytes()).hexdigest()
    if actual_hash != manifest["corpus"]["checksums_sha256"]:
        failures.append(
            f"benchmark corpus checksum manifest drift: expected "
            f"{manifest['corpus']['checksums_sha256']}, got {actual_hash}"
        )

    results = json.loads((ROOT / manifest["results_file"]).read_text(encoding="utf-8"))
    for path, guard in manifest["metrics"].items():
        actual = _metric(results, path)
        drift = abs(actual - guard["baseline"])
        if drift > guard["max_absolute_drift"]:
            failures.append(
                f"{path} drifted by {drift:g}; allowed "
                f"{guard['max_absolute_drift']:g} from baseline {guard['baseline']:g}"
            )
    return failures


def main() -> int:
    failures = check()
    if failures:
        print("Failure category: scientific-regression / benchmark-drift", file=sys.stderr)
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print("Scientific benchmark drift guardrails passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
