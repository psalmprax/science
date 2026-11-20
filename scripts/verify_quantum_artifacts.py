#!/usr/bin/env python3
"""Where the large quantum circuit artifacts live, and whether you have them.

The verified circuits are compiled ``.kmb`` binaries. They are build products, not
source: the dqi reference decoder alone is 886 MB, and ``.gitignore`` already
excludes ``artifacts/`` wholesale and ``sample_submission/dqi.kmb`` by name. So a
checkout ships the manifest and not the bytes, and that is deliberate.

What a checkout *does* ship is this file's companion,
``config/quantum_artifact_manifest.json``: one row per circuit with its sha256,
byte count and the reason it exists. That record is what makes a missing artifact
diagnosable -- "the file is not here" instead of "the proposal list is empty".

Storage policy, recorded rather than assumed:

  * the canonical copy lives outside the working tree, in a content-addressed
    cache (``$APEDP_ARTIFACT_CACHE``, default ``~/.cache/apedp/artifacts``),
    keyed by sha256 so a copy is either the circuit or it is not;
  * a working tree materializes what it needs by copying or symlinking from that
    cache into the path the code expects;
  * nothing over the size ceiling is ever committed.

Usage::

    python scripts/verify_quantum_artifacts.py --write    # re-record from disk
    python scripts/verify_quantum_artifacts.py --check    # verify what is present
    python scripts/verify_quantum_artifacts.py --materialize  # cache -> working tree

``--check`` exits non-zero only for a *wrong* artifact (a hash mismatch, or a size
mismatch). An absent artifact is reported and does not fail the run: a clean
checkout has none of them, and that is not an error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "config" / "quantum_artifact_manifest.json"
SCHEMA = "apedp.quantum-artifacts.v1"

#: Circuits worth naming. Each is a compiled binary that decided something.
TRACKED: list[dict[str, str]] = [
    {
        "path": "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/sample_submission/invsqrt.kmb",
        "role": "Benchmark I sample; small enough to ship in git (579 KB)",
        "benchmark": "invsqrt",
    },
    {
        "path": "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/sample_submission/dqi.kmb",
        "role": "Benchmark II reference decoder, official PASS at 102,400 shots",
        "benchmark": "dqi",
    },
    {
        "path": "artifacts/quantum_compilation/invsqrt/invsqrt_89q_shared_pddd.kmb",
        "role": "Benchmark I champion (89 qubits, 1,466 static Toffolis, 157,605.52)",
        "benchmark": "invsqrt",
    },
    {
        "path": "artifacts/quantum_compilation/invsqrt/invsqrt_90q_affine_opt.kmb",
        "role": "Benchmark I sealed 90Q champion receipt",
        "benchmark": "invsqrt",
    },
    {
        "path": "artifacts/quantum_compilation/dqi/iteration_sweep_receipts/dqi_niters_222.kmb",
        "role": "Benchmark II iteration sweep winner (222 EEA iterations); same "
                "sha256 as sample_submission/dqi.kmb",
        "benchmark": "dqi",
        # Recorded in the sweep receipt (artifacts/.../iteration_sweep_receipts/
        # results.json) while the bytes existed. Kept here so an absent row is
        # still recoverable from the cache rather than anonymous.
        "known_sha256": "959471d05a26373b4f14ca0cae2d8639ea25f18170d9f6532139aa51ccc04995",
        "known_bytes": 886118180,
    },
]

DEFAULT_CACHE = Path(
    os.environ.get("APEDP_ARTIFACT_CACHE", Path.home() / ".cache" / "apedp" / "artifacts")
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_path(digest: str) -> Path:
    """Content-addressed location for one artifact."""
    return DEFAULT_CACHE / digest[:2] / digest


def describe(entry: dict[str, str]) -> dict[str, object]:
    """One manifest row, measured from disk when the file is there."""
    path = ROOT / entry["path"]
    row: dict[str, object] = {
        "path": entry["path"],
        "benchmark": entry["benchmark"],
        "role": entry["role"],
        "present": path.is_file(),
    }
    if path.is_file():
        row["bytes"] = path.stat().st_size
        row["sha256"] = sha256_file(path)
    else:
        row["bytes"] = entry.get("known_bytes")
        row["sha256"] = entry.get("known_sha256")
    return row


def write_manifest() -> int:
    rows = [describe(entry) for entry in TRACKED]
    payload = {
        "schema": SCHEMA,
        "note": (
            "Compiled .kmb circuits are build products and are never committed. "
            "This manifest is the record: sha256 and size for each one, so a "
            "'present: false' row explains an empty proposal list instead of "
            "leaving it to look like a healthy run. Restore with "
            "scripts/verify_quantum_artifacts.py --materialize, or recompile from "
            "the builder the row's benchmark names."
        ),
        "policy": {
            "committed": False,
            "cache": str(DEFAULT_CACHE),
            "rationale": (
                "The dqi reference decoder is 886 MB; the artifact set is "
                "multiple GB. Git is the wrong store for it, and a code checkout "
                "must not require it -- the code paths that need bytes are the "
                "ones that say so."
            ),
        },
        "artifacts": rows,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    present = sum(1 for row in rows if row["present"])
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}: {present}/{len(rows)} present")
    return 0


def check_manifest() -> int:
    if not MANIFEST_PATH.is_file():
        print(f"no manifest at {MANIFEST_PATH}; run --write", file=sys.stderr)
        return 1
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        print(f"unknown manifest schema {payload.get('schema')!r}", file=sys.stderr)
        return 1

    wrong: list[str] = []
    absent: list[str] = []
    ok = 0
    for row in payload["artifacts"]:
        path = ROOT / row["path"]
        if not path.is_file():
            absent.append(row["path"])
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        if row.get("bytes") is not None and size != row["bytes"]:
            wrong.append(f"{row['path']}: {size} bytes, manifest says {row['bytes']}")
        elif row.get("sha256") is not None and digest != row["sha256"]:
            wrong.append(f"{row['path']}: sha256 {digest[:16]}..., manifest says {row['sha256'][:16]}...")
        else:
            ok += 1

    for path in absent:
        print(f"  absent  {path}")
    print(f"verified {ok}, absent {len(absent)}, wrong {len(wrong)}")
    for problem in wrong:
        print(f"  WRONG   {problem}", file=sys.stderr)
    return 1 if wrong else 0


def materialize() -> int:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    restored = 0
    for row in payload["artifacts"]:
        target = ROOT / row["path"]
        if target.is_file():
            continue
        digest = row.get("sha256")
        if not digest:
            print(f"  no sha256 recorded for {row['path']}; cannot locate it")
            continue
        source = cache_path(digest)
        if not source.is_file():
            print(f"  not in cache: {row['path']} (expected {source})")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256_file(target) != digest:
            print(f"  COPY CORRUPTED {row['path']}", file=sys.stderr)
            return 1
        restored += 1
        print(f"  restored {row['path']}")
    print(f"materialized {restored}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help="record what is on disk")
    group.add_argument("--check", action="store_true", help="verify what is present (default)")
    group.add_argument("--materialize", action="store_true", help="copy from cache into the tree")
    args = parser.parse_args(argv)

    if args.write:
        return write_manifest()
    if args.materialize:
        return materialize()
    return check_manifest()


if __name__ == "__main__":
    raise SystemExit(main())
