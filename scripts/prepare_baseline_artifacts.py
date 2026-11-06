#!/usr/bin/env python3
"""Stage A Track 1 Baseline Candidate Preparation.

Copies the pinned reproducible baseline artifacts into the component run directory
and verifies their SHA-256 against canonical ground-truth values before verifier execution.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_HASHES = {
    "invsqrt": {
        "source": REPO_ROOT / "artifacts/quantum_compilation/invsqrt/full-verification/candidate.kmb",
        "sha256": "53479d65b192b99868275f8f26e166b0a95715eef742b7d493515d8c64ce3057",
        "target_name": "invsqrt.kmb",
    },
    "dqi": {
        "source": REPO_ROOT / "artifacts/quantum_compilation/dqi/iteration_sweep_receipts/dqi_niters_222.kmb",
        "sha256": "959471d05a26373b4f14ca0cae2d8639ea25f18170d9f6532139aa51ccc04995",
        "target_name": "dqi.kmb",
    },
    "dfthcsos": {
        "source_dir": REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/sample_submission/dfthcsos",
        "submission_npz_sha256": "14d5cc25eb3bf82a9ccb87c811db28d127bef6d28697dd190dc1e875de3bfe69",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Track 1 baseline artifacts")
    parser.add_argument("--benchmark", choices=["invsqrt", "dqi", "dfthcsos"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = CANONICAL_HASHES[args.benchmark]

    if args.benchmark in ("invsqrt", "dqi"):
        src = spec["source"]
        if not src.is_file():
            print(f"ERROR: Pinned baseline file not found: {src}", file=sys.stderr)
            return 1
        digest = sha256_file(src)
        if digest != spec["sha256"]:
            print(f"ERROR: Pinned baseline hash mismatch for {args.benchmark}: {digest} != {spec['sha256']}", file=sys.stderr)
            return 1
        dst = args.output_dir / spec["target_name"]
        shutil.copy2(src, dst)
        print(f"Prepared {args.benchmark} baseline candidate: {dst} (SHA256: {digest})")
        return 0

    if args.benchmark == "dfthcsos":
        src_dir = spec["source_dir"]
        if not src_dir.is_dir():
            print(f"ERROR: Pinned baseline directory not found: {src_dir}", file=sys.stderr)
            return 1
        npz_file = src_dir / "submission.npz"
        if not npz_file.is_file():
            print(f"ERROR: submission.npz not found in {src_dir}", file=sys.stderr)
            return 1
        digest = sha256_file(npz_file)
        if digest != spec["submission_npz_sha256"]:
            print(f"ERROR: submission.npz hash mismatch: {digest} != {spec['submission_npz_sha256']}", file=sys.stderr)
            return 1
        target_dir = args.output_dir / "dfthcsos"
        target_dir.mkdir(parents=True, exist_ok=True)
        # Copy submission.npz and angle_k_*.kmb
        for item in src_dir.glob("*.kmb"):
            shutil.copy2(item, target_dir / item.name)
        shutil.copy2(npz_file, target_dir / "submission.npz")
        print(f"Prepared dfthcsos baseline candidate in {target_dir} (submission.npz SHA256: {digest})")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
