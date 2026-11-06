#!/usr/bin/env python3
"""Run the Benchmark II verifier on the full official instance.

The official circuit is 86M instructions, and the verifier's ``reaction_depth``
diagnostic peaks at 13.9 GB on it (136 s) — which is what killed the process the
first time this was measured, on a host with less headroom than this one. That
diagnostic is not part of the score; everything that is, runs unmodified.

So this applies exactly one substitution to a copy of the verifier and proves it:
``'reaction_depth': circuit.reaction_depth(),`` becomes ``'reaction_depth': None``.
Both SHA-256 digests are printed so the patch is auditable, and the copy is
written to ``build/`` (gitignored) rather than over the shipped file. The
competition's verifier is never modified in place.

    python3.13 scripts/run_dqi_full_instance_verifier.py \
        --circuit apedp_dyn/sandbox/.../sample_submission/dqi.kmb \
        --num-shots 102400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
VERIFIER = BUNDLE / "verifiers" / "dqi" / "verify_submission_dqi.py"
DEFAULT_CIRCUIT = BUNDLE / "sample_submission" / "dqi.kmb"
OUT = ROOT / "build" / "dqi_verify_full"

NEEDLE = "'reaction_depth': circuit.reaction_depth(),"
REPLACEMENT = "'reaction_depth': None,  # diagnostic only; 13.9 GB peak on 86M ops"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--circuit", type=Path, default=DEFAULT_CIRCUIT)
    parser.add_argument("--num-shots", type=int, default=102400)
    parser.add_argument("--salt", default="")
    parser.add_argument("--json", type=Path,
                        default=ROOT / "benchmarks" / "quantum_compilation"
                        / "dqi_full_instance_reverify.json")
    args = parser.parse_args()

    source = VERIFIER.read_text(encoding="utf-8")
    if source.count(NEEDLE) != 1:
        raise SystemExit(f"expected exactly one {NEEDLE!r} in {VERIFIER}, found {source.count(NEEDLE)}")
    OUT.mkdir(parents=True, exist_ok=True)
    patched = OUT / VERIFIER.name
    patched.write_text(source.replace(NEEDLE, REPLACEMENT), encoding="utf-8")

    print(json.dumps({"verifier": str(VERIFIER.relative_to(ROOT)),
                      "verifier_sha256": sha256(VERIFIER),
                      "patched_sha256": sha256(patched),
                      "circuit": str(args.circuit.relative_to(ROOT)),
                      "circuit_sha256": sha256(args.circuit),
                      "circuit_bytes": args.circuit.stat().st_size,
                      "num_shots": args.num_shots}, indent=2))

    command = [sys.executable, str(patched), str(args.circuit),
               "--num-shots", str(args.num_shots)]
    if args.salt:
        command += ["--salt", args.salt]
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        command, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(BUNDLE / "kickmix")},
    )
    print(completed.stdout[-4000:])
    if completed.stderr:
        print(completed.stderr[-2000:], file=sys.stderr)
    if completed.returncode != 0:
        return completed.returncode

    result = json.loads(completed.stdout)
    keeps = ("status", "score", "toffoli_count", "num_qubits", "verified_shots",
             "value_errors", "phase_errors", "effective_error_rate", "repetition_factor",
             "weight_errors", "circuit_sha256")
    record = {
        "experiment": "the full 4,095-position official instance, re-measured in this checkout",
        "circuit": str(args.circuit.relative_to(ROOT)),
        "circuit_sha256": sha256(args.circuit),
        "num_shots": args.num_shots,
        "verifier_sha256": sha256(VERIFIER),
        "patched_sha256": sha256(patched),
        "patch": {"replaced": NEEDLE, "with": REPLACEMENT,
                  "why": "reaction_depth is a reported diagnostic, not a scoring term; it peaks at "
                         "13.9 GB on the 86M-instruction circuit. Every scoring line is the shipped one."},
        "verified": {k: result[k] for k in keeps if k in result},
        "score_components": {k: v for k, v in result.items()
                             if k in ("resources", "metrics", "R", "eps_plus")},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"record: {args.json.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
