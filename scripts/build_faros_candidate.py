#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build FAROS and optionally run a fresh, artifact-bound official verifier.

The integer model certificate is not a quantum-circuit correctness certificate.
Calibration replays recorded deltas; no LP optimization is performed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apedp_dyn.quantum_compilation.faros_isq import (
    calibrate_faros_circuit,
    exhaustive_integer_certificate,
    load_chebyshev_minimax_tables,
    materialize_faros_circuit,
    verify_faros_candidate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "artifacts/quantum_compilation/invsqrt")
    parser.add_argument("--candidate-id", default="invsqrt_faros_r2_86q")
    parser.add_argument("--python-executable")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--max-x", type=int, help="Prefix verification; omitted uses full domain")
    parser.add_argument("--shots-per-x", type=int, default=64)
    args = parser.parse_args(argv)
    if Path(args.candidate_id).name != args.candidate_id or args.candidate_id in (".", ".."):
        parser.error("candidate-id must be a filename, not a path")
    if args.max_x is not None and not 1 <= args.max_x <= 3_133_452:
        parser.error("max-x must lie in [1, 3133452]")
    if args.shots_per_x < 1:
        parser.error("shots-per-x must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{args.candidate_id}.pipeline.json"
    # Replace any prior success before starting a new attempt.
    summary = {"status": "in_progress", "verification_requested": args.verify}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    cfg, tables, _ = load_chebyshev_minimax_tables()
    certificate = exhaustive_integer_certificate(
        cfg, tables, output_json=args.output_dir / f"{args.candidate_id}.integer-model.json"
    )
    if certificate["pass"] is not True or certificate["tested_inputs"] != 3_133_452:
        raise RuntimeError("Full-domain integer model certificate failed")
    circuit_path = args.output_dir / f"{args.candidate_id}.kmb"
    materialize_faros_circuit(
        cfg=cfg, tables=tables, output_path=circuit_path,
        candidate_id=args.candidate_id, python_executable=args.python_executable,
        force_rebuild=True,
    )
    calibration = calibrate_faros_circuit(
        circuit_path, python_executable=args.python_executable
    )
    # Remeasure the final artifact, not the pre-calibration circuit.
    resources = materialize_faros_circuit(
        output_path=circuit_path, candidate_id=args.candidate_id,
        python_executable=args.python_executable,
    )
    summary.update({
        "status": "built_not_verified",
        "integer_model_passed": True,
        "circuit_sha256": resources["sha256"],
        "resources": resources,
        "calibration": calibration,
        "verification": None,
    })
    exit_code = 0
    if args.verify:
        receipt = verify_faros_candidate(
            circuit_path, python_executable=args.python_executable,
            max_x=args.max_x, shots_per_x=args.shots_per_x,
            receipt_path=args.output_dir / f"{args.candidate_id}.verifier.json",
        )
        if receipt.artifact_sha256 != resources["sha256"]:
            raise RuntimeError("Verifier receipt is not bound to the final circuit")
        passed = receipt.status == "PASS" and receipt.exit_code == 0
        summary.update({
            "status": "verifier_passed" if passed else "verifier_failed",
            "verification": receipt.to_dict(),
        })
        exit_code = 0 if passed else 1
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
