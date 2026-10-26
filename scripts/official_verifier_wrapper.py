#!/usr/bin/env python3
"""Unified runner wrapper for official competition verifiers.

Executes the official verifier script using the CPython 3.13 Kickmix runtime,
captures the resulting stdout JSON receipt, and writes it atomically to the
path specified by --receipt.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KICKMIX_DIR = Path(
    os.environ.get(
        "APEDP_Q_KICKMIX_DIR",
        REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix",
    )
)
PYTHON_BIN = Path(os.environ.get("APEDP_Q_PYTHON", "/dev/shm/qvcc-venv313/bin/python"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Official verifier wrapper")
    parser.add_argument("--benchmark", choices=["invsqrt", "dqi", "dfthcsos"], required=True)
    parser.add_argument("--verifier-script", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate circuit file or directory")
    parser.add_argument("--receipt", type=Path, required=True, help="Target receipt JSON output path")
    parser.add_argument("--verifier-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    if not args.verifier_script.is_file():
        print(f"ERROR: Verifier script not found: {args.verifier_script}", file=sys.stderr)
        return 1

    if not args.candidate.exists():
        print(f"ERROR: Candidate target not found: {args.candidate}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{KICKMIX_DIR}:{current_pythonpath}" if current_pythonpath else str(KICKMIX_DIR)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [str(PYTHON_BIN), str(args.verifier_script), str(args.candidate)]
    if args.verifier_args:
        cmd.extend(args.verifier_args)

    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.stderr:
        sys.stderr.write(proc.stderr)

    if proc.returncode != 0:
        print(f"ERROR: Verifier exited with returncode {proc.returncode}", file=sys.stderr)
        return proc.returncode

    try:
        # The verifiers output JSON to stdout. Find the JSON object in stdout.
        stdout_text = proc.stdout.strip()
        json_start = stdout_text.find("{")
        json_end = stdout_text.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError(f"No JSON object detected in stdout:\n{stdout_text}")
        payload = json.loads(stdout_text[json_start:json_end])
    except Exception as exc:
        print(f"ERROR: Failed to parse verifier output as JSON: {exc}\nSTDOUT:\n{proc.stdout}", file=sys.stderr)
        return 1

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.receipt)

    print(f"Verified {args.benchmark}: receipt written to {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
